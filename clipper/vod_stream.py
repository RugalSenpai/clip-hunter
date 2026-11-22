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

import queue
import signal
import threading
import time
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
from .judge import Judge, save_candidates
from .models import Candidate, ClipArtifact, TranscriptBlock, format_timestamp
from .prefilter import score_block
from .reporter import (
    format_final_report,
    open_clips_folder,
    write_clip_metadata_files,
    write_summary_json,
)
from .vtt_parser import parse_vtt_file

_SENTINEL = None


def render_single_candidate(
    url: str,
    candidate: Candidate,
    clip_index: int,
    run_dir: Path,
    ratios: list[str],
    enable_subtitles: bool = True,
    whisper_model: str = "small.en",
    yunet_model_path: str | Path | None = None,
) -> ClipArtifact | None:
    """Download a single winning window, align with Whisper, detect facecam, and export ratios."""
    work_dir = run_dir / "work" / f"clip_{clip_index:02d}"
    work_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = run_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    raw_video = work_dir / "window.mp4"
    downloaded_path, window_start = fetch_video_section(
        url=url,
        start_sec=candidate.start_sec,
        end_sec=candidate.end_sec,
        out_path=raw_video,
        pad_before_sec=4.0,
        pad_after_sec=4.0,
    )

    # Whisper word-level alignment to tighten cut points (and feed word-by-word captions)
    segs, words = transcribe_media(
        media_path=downloaded_path,
        model_size=whisper_model,
        device="cpu",
        compute_type="int8",
        time_offset=window_start,
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

    preview_path = work_dir / "preview.jpg"
    facecam = detect_facecam(
        video_path=downloaded_path,
        yunet_model_path=yunet_model_path,
        layout_override=candidate.layout_override,
        preview_out=preview_path,
    )

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
    winner_queue: queue.Queue[Candidate | None] = queue.Queue(maxsize=16)

    artifacts: list[ClipArtifact] = []
    all_candidates: list[Candidate] = []
    picked_intervals: list[tuple[float, float]] = []
    state_lock = threading.Lock()
    errors: list[str] = []
    stop_judging = threading.Event()

    def producer() -> None:
        try:
            vtt_path = fetch_captions(url, run_path / "captions")
            if vtt_path and vtt_path.exists():
                print(f"[producer] Using captions: {vtt_path.name}", flush=True)
                segments = parse_vtt_file(vtt_path)
                blocks = split_into_blocks(segments, block_duration_sec=block_duration_sec)
                for blk in blocks:
                    if cancel.is_set() or stop_judging.is_set():
                        break
                    block_queue.put(blk)
            else:
                # No platform CC: stream 10-minute audio-only chunks + Whisper one by one
                print("[producer] No auto-captions found — streaming 10-min audio chunks + Whisper", flush=True)
                audio_dir = run_path / "audio_chunks"
                audio_dir.mkdir(parents=True, exist_ok=True)
                chunk_start = 0.0
                idx = 0
                total = duration_sec if duration_sec > 0 else block_duration_sec
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
                        wav, model_size=whisper_model, time_offset=chunk_start
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
        except Exception as exc:
            msg = f"judge: {exc}"
            print(f"[judge ERROR] {exc}", flush=True)
            errors.append(msg)
            cancel.set()
        finally:
            winner_queue.put(_SENTINEL)

    def render_worker() -> None:
        clip_idx = 0
        while True:
            cand = winner_queue.get()
            if cand is _SENTINEL:
                break
            if cancel.is_set():
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
                )
            except Exception as exc:
                msg = f"render clip {clip_idx:02d}: {exc}"
                print(f"[render ERROR] {msg}", flush=True)
                errors.append(msg)
                continue
            if art is not None:
                with state_lock:
                    artifacts.append(art)
                    write_summary_json(
                        clips_dir=clips_dir,
                        stream_info=info,
                        moments_found=len(all_candidates),
                        artifacts=artifacts,
                        elapsed_sec=time.monotonic() - t0,
                    )
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
        th.join()

    if prev_handler is not None:
        signal.signal(signal.SIGINT, prev_handler)

    if all_candidates:
        save_candidates(all_candidates, run_path / "candidates.json")

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
