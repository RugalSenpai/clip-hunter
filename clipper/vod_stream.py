"""Concurrent 3-worker streaming VOD pipeline with graceful cancellation.

Worker 1 (Producer): Streams 10-min blocks from VTT (if captions exist) or fetches
                     10-min audio-only chunks + Whisper transcription (if no captions).
Worker 2 (Judge):    Scores each 10-min block via pre-filter + LLM (local Ollama / API)
                     and immediately enqueues winning moments (>= min_score).
Worker 3 (Renderer): Downloads ONLY the winning 20-60s video window via --download-sections,
                     aligns word timestamps, detects facecam, renders 9:16 / 1:1 / 16:9
                     with optional subtitles, writes .txt + updates summary.json.
"""

from __future__ import annotations

import json
import math
import queue
import signal
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from .aligner import (
    rebase_words_for_clip,
    snap_to_sentence_boundaries,
    transcribe_media,
)
from .blocker import overlaps_existing, split_into_blocks
from .exporter import export_multi_ratio
from .facecam import detect_facecam
from .fetcher import (
    fetch_audio_section,
    fetch_captions,
    fetch_video_section,
    get_video_info,
)
from .judge import Judge, save_candidates, validate_candidates
from .models import Candidate, ClipArtifact, TranscriptBlock, format_timestamp, parse_timestamp
from .prefilter import score_block
from .reporter import (
    format_final_report,
    open_clips_folder,
    write_clip_metadata_files,
    write_summary_json,
)
from .vtt_parser import parse_vtt_file

_SENTINEL = None


def _candidate_key(cand: Candidate) -> str:
    return f"{cand.start}-{cand.end}"


def _load_existing_artifacts(summary_path: Path) -> list[ClipArtifact]:
    if not summary_path.exists():
        return []
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    loaded: list[ClipArtifact] = []
    for item in data.get("clips", []):
        try:
            cand = Candidate(
                start=item.get("start", "00:00:00"),
                end=item.get("end", "00:00:30"),
                category=item.get("category", "funny"),
                score=float(item.get("score", 7.0)),
                hook=item.get("hook", ""),
                title=item.get("title", "Clip"),
                caption=item.get("caption", ""),
                hashtags=item.get("hashtags", []),
                why=item.get("why", ""),
            )
            loaded.append(
                ClipArtifact(
                    index=int(item.get("index", len(loaded) + 1)),
                    candidate=cand,
                    layout=item.get("layout", "gameplay"),
                    start_sec=parse_timestamp(cand.start),
                    end_sec=parse_timestamp(cand.end),
                    duration=float(item.get("duration_sec", 0.0)),
                    files=item.get("files", {}),
                    metadata_files=item.get("metadata_files", {}),
                    preview_frame=item.get("preview_frame"),
                )
            )
        except Exception:
            continue
    return loaded


def render_single_candidate(
    url: str,
    candidate: Candidate,
    clip_index: int,
    run_dir: Path,
    ratios: list[str],
    enable_subtitles: bool = True,
    whisper_model: str = "small.en",
    yunet_model_path: str | Path | None = None,
    compute_mode: str = "auto",
    enable_hook: bool = True,
    profile: str = "profile1",
    on_clip_progress: Callable[[dict[str, Any]], None] | None = None,
) -> ClipArtifact | None:
    """Download a single winning window, align with Whisper, detect facecam, and export ratios."""
    work_dir = run_dir / "work" / f"clip_{clip_index:02d}"
    work_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = run_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    def _emit_progress(status: str, progress: int, status_text: str, task_step: int = 1, task_total: int = 3) -> None:
        if on_clip_progress:
            on_clip_progress(
                {
                    "index": clip_index,
                    "title": candidate.title,
                    "score": candidate.score,
                    "category": candidate.category,
                    "duration_sec": round(max(1.0, candidate.end_sec - candidate.start_sec), 2),
                    "hook": candidate.hook,
                    "caption": candidate.caption,
                    "hashtags": list(candidate.hashtags),
                    "ratios": ["9:16"] if profile == "profile2" else list(ratios),
                    "layout": candidate.layout_override or "auto",
                    "files": {},
                    "status": status,
                    "progress": progress,
                    "status_text": status_text,
                    "task_step": task_step,
                    "task_total": task_total,
                }
            )

    _emit_progress("downloading", 0, "Downloading", 1, 3)
    raw_video = work_dir / "window.mp4"
    downloaded_path, window_start = fetch_video_section(
        url=url,
        start_sec=candidate.start_sec,
        end_sec=candidate.end_sec,
        out_path=raw_video,
        pad_before_sec=4.0,
        pad_after_sec=4.0,
        on_progress=lambda p: _emit_progress("downloading", p, "Downloading", 1, 3),
    )
    _emit_progress("downloading", 100, "Downloading", 1, 3)

    _emit_progress("rendering", 0, "Transcribing", 2, 3)
    # Whisper word-level alignment to tighten cut points (and feed word-by-word captions)
    segs, words = transcribe_media(
        media_path=downloaded_path,
        model_size=whisper_model,
        device=compute_mode,
        time_offset=window_start,
        vad_filter=False,
        on_progress=lambda p: _emit_progress("rendering", p, "Transcribing", 2, 3),
    )
    snapped_start, snapped_end = snap_to_sentence_boundaries(
        target_start=candidate.start_sec,
        target_end=candidate.end_sec,
        segments=segs,
    )
    duration = snapped_end - snapped_start
    if duration < 12.0:
        return None
    if duration > 75.0:
        snapped_end = snapped_start + 75.0
        duration = 75.0

    local_start = max(0.0, snapped_start - window_start)
    rebased_words = rebase_words_for_clip(words, snapped_start, snapped_end) if enable_subtitles else []

    _emit_progress("rendering", 90, "Detecting facecam", 2, 3)
    preview_path = work_dir / "preview.jpg"
    facecam = detect_facecam(
        video_path=downloaded_path,
        yunet_model_path=yunet_model_path,
        layout_override=candidate.layout_override,
        preview_out=preview_path,
    )
    _emit_progress("rendering", 100, "Transcribing", 2, 3)

    _emit_progress("rendering", 0, "Rendering", 3, 3)
    files = export_multi_ratio(
        source_video=downloaded_path,
        clip_start_in_source=local_start,
        clip_duration=duration,
        clip_index=clip_index,
        title=candidate.title,
        hook_text=candidate.hook,
        facecam=facecam,
        words=rebased_words,
        clips_dir=clips_dir,
        ratios=ratios,
        enable_subtitles=enable_subtitles,
        work_dir=work_dir,
        compute_mode=compute_mode,
        enable_hook=enable_hook,
        profile=profile,
        on_progress=lambda p: _emit_progress("rendering", p, "Rendering", 3, 3),
    )

    artifact = ClipArtifact(
        index=clip_index,
        candidate=candidate,
        layout=facecam.layout,
        start_sec=snapped_start,
        end_sec=snapped_end,
        duration=duration,
        files=files,
        metadata_files={},
        preview_frame=str(preview_path) if preview_path.exists() else None,
    )
    write_clip_metadata_files(artifact)
    try:
        edit_state = {
            "window_video": str(downloaded_path),
            "window_start": round(window_start, 3),
            "local_start": round(local_start, 3),
            "duration": round(duration, 3),
            "enable_subtitles": enable_subtitles,
            "enable_hook": enable_hook,
            "hook_text": candidate.hook,
            "hook_y_pct": 8.0,
            "sub_y_pct": 76.0,
            "highlight_color": "#FFFF00",
            "profile": profile,
            "layout": facecam.layout,
            "facecam": {
                "x": facecam.x,
                "y": facecam.y,
                "w": facecam.w,
                "h": facecam.h,
                "frame_w": facecam.frame_w,
                "frame_h": facecam.frame_h,
            },
            "ratios": list(ratios),
            "words": [{"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3)} for w in rebased_words],
        }
        (work_dir / "edit_state.json").write_text(
            json.dumps(edit_state, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError:
        pass
    return artifact


def run_streaming_vod(
    url: str,
    run_dir: str | Path,
    judge: Judge,
    ratios: list[str] = ["9:16", "1:1", "16:9"],
    enable_subtitles: bool = True,
    block_duration_sec: float = 600.0,
    min_score: float = 7.0,
    max_clips: int = 10,
    prefilter_min_score: float = 0.0,
    whisper_model: str = "small.en",
    game_tag: str | None = None,
    open_folder: bool = False,
    stop_event: threading.Event | None = None,
    on_clip_ready: Callable[[ClipArtifact], None] | None = None,
    on_clip_progress: Callable[[dict[str, Any]], None] | None = None,
    compute_mode: str = "auto",
    resume: bool = True,
    enable_hook: bool = True,
    profile: str = "profile1",
) -> list[ClipArtifact]:
    """Run the 3-worker concurrent VOD pipeline until video ends or stop_event is set."""
    t0 = time.monotonic()
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    clips_dir = run_path / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    cancel = stop_event or threading.Event()
    prev_handler = None
    if threading.current_thread() is threading.main_thread():
        def _handle_sigint(_sig, _frame):
            print("\n[cancel] Stopping gracefully — keeping all finished clips...", flush=True)
            cancel.set()

        prev_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, _handle_sigint)

    info = get_video_info(url)
    streamer = info["uploader"]
    duration_sec = float(info["duration"])
    print(
        f"[vod-stream] {info['title']} ({format_timestamp(duration_sec)}) by {streamer}",
        flush=True,
    )

    block_queue: queue.Queue[TranscriptBlock | None] = queue.Queue(maxsize=4)
    winner_queue: queue.Queue[Candidate | None] = queue.Queue(maxsize=64)

    state_path = run_path / "state.json"
    candidates_path = run_path / "candidates.json"
    summary_path = clips_dir / "summary.json"

    artifacts: list[ClipArtifact] = []
    all_candidates: list[Candidate] = []
    picked_intervals: list[tuple[float, float]] = []
    rendered_keys: set[str] = set()
    last_judged_block = -1
    total_blocks = max(1, math.ceil(duration_sec / block_duration_sec)) if duration_sec > 0 else 0

    if resume and (state_path.exists() or candidates_path.exists()):
        if state_path.exists():
            try:
                prev_state = json.loads(state_path.read_text(encoding="utf-8"))
                last_judged_block = int(prev_state.get("last_judged_block", -1))
                total_blocks = int(prev_state.get("total_blocks", total_blocks))
                rendered_keys.update(prev_state.get("rendered_keys", []))
            except Exception:
                pass
        artifacts = _load_existing_artifacts(summary_path)
        for art in artifacts:
            rendered_keys.add(_candidate_key(art.candidate))
        if candidates_path.exists():
            try:
                raw_cands = json.loads(candidates_path.read_text(encoding="utf-8"))
                all_candidates = validate_candidates(raw_cands)
                for c in all_candidates:
                    picked_intervals.append((c.start_sec, c.end_sec))
            except Exception:
                pass

    state_lock = threading.Lock()
    errors: list[str] = []
    stop_judging = threading.Event()

    def _save_state(status: str = "running") -> None:
        payload = {
            "url": url,
            "title": info["title"],
            "uploader": streamer,
            "duration": duration_sec,
            "ratios": list(ratios),
            "enable_subtitles": enable_subtitles,
            "enable_hook": enable_hook,
            "profile": profile,
            "block_duration_sec": block_duration_sec,
            "min_score": min_score,
            "max_clips": max_clips,
            "whisper_model": whisper_model,
            "compute_mode": compute_mode,
            "status": status,
            "total_blocks": total_blocks,
            "last_judged_block": last_judged_block,
            "winners_count": len(all_candidates),
            "rendered_count": len(artifacts),
            "rendered_keys": sorted(rendered_keys),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            state_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass

    _save_state("running")

    # Queue any previously judged winners that were not yet rendered before a crash/stop
    pending_unrendered = [c for c in all_candidates if _candidate_key(c) not in rendered_keys]
    if resume and (last_judged_block >= 0 or pending_unrendered or artifacts):
        print(
            f"[resume] Resuming run: judged through block #{last_judged_block}, "
            f"{len(artifacts)} clip(s) already rendered, {len(pending_unrendered)} pending render.",
            flush=True,
        )
        for c in pending_unrendered:
            winner_queue.put(c)
        if len(all_candidates) >= max_clips:
            stop_judging.set()

    def producer() -> None:
        nonlocal total_blocks
        try:
            if stop_judging.is_set():
                print("[resume] Max clips already selected — skipping further block scanning.", flush=True)
                return

            captions_dir = run_path / "captions"
            existing_vtts = sorted(captions_dir.glob("*.vtt")) if captions_dir.exists() else []
            vtt_path = existing_vtts[0] if existing_vtts else fetch_captions(url, captions_dir)

            if vtt_path and vtt_path.exists():
                print(f"[producer] Using captions: {vtt_path.name}", flush=True)
                segments = parse_vtt_file(vtt_path)
                blocks = split_into_blocks(segments, block_duration_sec=block_duration_sec)
                with state_lock:
                    total_blocks = len(blocks)
                    _save_state("running")
                if last_judged_block + 1 >= len(blocks):
                    print("[resume] LLM already reviewed all blocks — rendering remaining clips only.", flush=True)
                    return
                for blk in blocks:
                    if blk.index <= last_judged_block:
                        continue
                    if cancel.is_set() or stop_judging.is_set():
                        break
                    block_queue.put(blk)
            else:
                # No platform CC: stream 10-minute audio-only chunks + Whisper one by one
                print("[producer] No auto-captions found — streaming 10-min audio chunks + Whisper", flush=True)
                audio_dir = run_path / "audio_chunks"
                audio_dir.mkdir(parents=True, exist_ok=True)
                idx = max(0, last_judged_block + 1)
                chunk_start = idx * block_duration_sec
                total = duration_sec if duration_sec > 0 else block_duration_sec
                if chunk_start >= total and total > 0:
                    print("[resume] LLM already reviewed all audio chunks — rendering remaining clips only.", flush=True)
                    return
                while chunk_start < total and not cancel.is_set() and not stop_judging.is_set():
                    chunk_end = min(total, chunk_start + block_duration_sec)
                    chunk_file = audio_dir / f"chunk_{idx:03d}.wav"
                    print(
                        f"[producer] Downloading audio {format_timestamp(chunk_start)}-{format_timestamp(chunk_end)}...",
                        flush=True,
                    )
                    wav = fetch_audio_section(url, chunk_start, chunk_end, chunk_file)
                    if cancel.is_set():
                        break
                    segs, _words = transcribe_media(
                        wav, model_size=whisper_model, device=compute_mode, time_offset=chunk_start
                    )
                    if segs:
                        block_queue.put(
                            TranscriptBlock(index=idx, start=chunk_start, end=chunk_end, segments=segs)
                        )
                    # Clean up audio chunk after transcribing to keep disk small
                    try:
                        wav.unlink(missing_ok=True)
                    except OSError:
                        pass
                    idx += 1
                    chunk_start = chunk_end
        except Exception as exc:
            msg = f"producer: {exc}"
            print(f"[producer ERROR] {exc}", flush=True)
            errors.append(msg)
            cancel.set()
        finally:
            block_queue.put(_SENTINEL)

    def judge_worker() -> None:
        nonlocal last_judged_block
        try:
            while True:
                blk = block_queue.get()
                if blk is _SENTINEL:
                    break
                if cancel.is_set() or stop_judging.is_set():
                    continue
                p_score = score_block(blk)
                if p_score < prefilter_min_score:
                    print(
                        f"[judge] Skipping block #{blk.index} (prefilter {p_score:.1f} < {prefilter_min_score})",
                        flush=True,
                    )
                    with state_lock:
                        last_judged_block = max(last_judged_block, blk.index)
                        _save_state("running")
                    continue
                print(
                    f"[judge] Evaluating block #{blk.index} "
                    f"({format_timestamp(blk.start)}-{format_timestamp(blk.end)}, arousal={p_score:.1f})...",
                    flush=True,
                )
                cands = judge.judge_block(blk, streamer=streamer, game_tag=game_tag)
                print(
                    f"[judge] Block #{blk.index} -> LLM returned {len(cands)} candidate(s)",
                    flush=True,
                )
                for cand in cands:
                    if cancel.is_set() or stop_judging.is_set():
                        break
                    if cand.score < min_score:
                        print(
                            f"[judge]   dropped \"{cand.title}\" (score {cand.score:.1f} < {min_score})",
                            flush=True,
                        )
                        continue
                    if cand.duration < 12.0:
                        print(
                            f"[judge]   dropped \"{cand.title}\" (duration {cand.duration:.1f}s < 12s)",
                            flush=True,
                        )
                        continue
                    with state_lock:
                        if overlaps_existing(cand.start_sec, cand.end_sec, picked_intervals):
                            print(
                                f"[judge]   dropped \"{cand.title}\" (overlaps existing pick)",
                                flush=True,
                            )
                            continue
                        if len(all_candidates) >= max_clips:
                            stop_judging.set()
                            break
                        picked_intervals.append((cand.start_sec, cand.end_sec))
                        all_candidates.append(cand)
                        save_candidates(all_candidates, candidates_path)
                        reached_cap = len(all_candidates) >= max_clips
                    print(
                        f"[judge] Winner ({cand.score:.1f}): {cand.title} "
                        f"[{cand.start}-{cand.end}] -> queued for render",
                        flush=True,
                    )
                    winner_queue.put(cand)
                    if reached_cap:
                        print(
                            f"[judge] Reached max_clips={max_clips}; stopping further block judging while renderer finishes queued clips.",
                            flush=True,
                        )
                        stop_judging.set()
                        break
                with state_lock:
                    last_judged_block = max(last_judged_block, blk.index)
                    _save_state("running")
        except Exception as exc:
            msg = f"judge: {exc}"
            print(f"[judge ERROR] {exc}", flush=True)
            errors.append(msg)
            cancel.set()
        finally:
            winner_queue.put(_SENTINEL)

    def render_worker() -> None:
        clip_idx = max((a.index for a in artifacts), default=0)
        while True:
            cand = winner_queue.get()
            if cand is _SENTINEL:
                break
            if cancel.is_set():
                continue
            key = _candidate_key(cand)
            if key in rendered_keys:
                continue
            clip_idx += 1
            print(f"[render] Clip {clip_idx:02d}: fetching & editing \"{cand.title}\"...", flush=True)
            try:
                art = render_single_candidate(
                    url=url,
                    candidate=cand,
                    clip_index=clip_idx,
                    run_dir=run_path,
                    ratios=ratios,
                    enable_subtitles=enable_subtitles,
                    whisper_model=whisper_model,
                    compute_mode=compute_mode,
                    enable_hook=enable_hook,
                    profile=profile,
                    on_clip_progress=on_clip_progress,
                )
            except Exception as exc:
                msg = f"render clip {clip_idx:02d}: {exc}"
                print(f"[render ERROR] {msg}", flush=True)
                errors.append(msg)
                if on_clip_progress:
                    on_clip_progress({"index": clip_idx, "remove": True})
                continue
            if art is None and on_clip_progress:
                on_clip_progress({"index": clip_idx, "remove": True})
            if art is not None:
                with state_lock:
                    artifacts.append(art)
                    rendered_keys.add(key)
                    write_summary_json(
                        clips_dir=clips_dir,
                        stream_info=info,
                        moments_found=len(all_candidates),
                        artifacts=artifacts,
                        elapsed_sec=time.monotonic() - t0,
                    )
                    _save_state("running")
                print(
                    f"[clip ready] #{art.index:02d} \"{art.candidate.title}\" "
                    f"({art.duration:.1f}s, {art.layout}) -> {art.files.get('9:16', '')}",
                    flush=True,
                )
                if on_clip_ready:
                    on_clip_ready(art)

    threads = [
        threading.Thread(target=producer, name="vod-producer", daemon=True),
        threading.Thread(target=judge_worker, name="vod-judge", daemon=True),
        threading.Thread(target=render_worker, name="vod-render", daemon=True),
    ]
    for th in threads:
        th.start()
    for th in threads:
        while th.is_alive():
            th.join(timeout=0.5)
            if cancel.is_set():
                th.join(timeout=2.0)
                break

    if prev_handler is not None:
        signal.signal(signal.SIGINT, prev_handler)

    if all_candidates:
        save_candidates(all_candidates, candidates_path)

    final_status = "stopped" if cancel.is_set() else ("error" if errors and not artifacts else "completed")
    _save_state(final_status)

    elapsed = time.monotonic() - t0
    write_summary_json(
        clips_dir=clips_dir,
        stream_info=info,
        moments_found=len(all_candidates),
        artifacts=artifacts,
        elapsed_sec=elapsed,
    )
    print("\n" + format_final_report(duration_sec, len(all_candidates), artifacts, elapsed, clips_dir))
    if errors:
        print(f"[warnings] {'; '.join(errors)}", flush=True)
    if open_folder:
        open_clips_folder(clips_dir)
    return artifacts
