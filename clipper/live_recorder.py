"""Component 11: Mode B live stream recorder with rolling 15-min buffer & per-minute Whisper."""

from __future__ import annotations

import re
import signal
import socket
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .aligner import rebase_words_for_clip, snap_to_sentence_boundaries, transcribe_media
from .blocker import build_live_window, overlaps_existing
from .exporter import export_multi_ratio
from .facecam import detect_facecam
from .judge import Judge
from .models import ClipArtifact, TranscriptSegment, WordTimestamp, format_timestamp
from .reporter import write_clip_metadata_files, write_summary_json

_MEAN_VOL_RE = re.compile(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB")
_MAX_VOL_RE = re.compile(r"max_volume:\s*(-?\d+(?:\.\d+)?)\s*dB")


@dataclass
class LiveSegmentRecord:
    index: int
    path: Path
    start_sec: float
    end_sec: float
    mean_db: float
    max_db: float
    segments: list[TranscriptSegment]
    words: list[WordTimestamp]


class TwitchChatVelocityMonitor:
    """Lightweight anonymous Twitch IRC reader tracking messages/min (no API key needed)."""

    def __init__(self, channel: str) -> None:
        self.channel = channel.lstrip("#").lower()
        self._timestamps: deque[float] = deque(maxlen=5000)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="twitch-irc", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def count_in_window(self, window_sec: float = 60.0) -> int:
        cutoff = time.time() - window_sec
        return sum(1 for ts in list(self._timestamps) if ts >= cutoff)

    def _run(self) -> None:
        try:
            sock = socket.create_connection(("irc.chat.twitch.tv", 6667), timeout=15)
            sock.settimeout(5.0)
            sock.sendall(b"PASS SCHMOOPIIE\r\nNICK justinfan42424\r\n")
            sock.sendall(f"JOIN #{self.channel}\r\n".encode("utf-8"))
            buf = ""
            while not self._stop.is_set():
                try:
                    data = sock.recv(4096).decode("utf-8", errors="ignore")
                except socket.timeout:
                    continue
                if not data:
                    break
                buf += data
                while "\r\n" in buf:
                    line, buf = buf.split("\r\n", 1)
                    if line.startswith("PING"):
                        sock.sendall(b"PONG :tmi.twitch.tv\r\n")
                    elif " PRIVMSG " in line:
                        self._timestamps.append(time.time())
            sock.close()
        except OSError:
            pass


def measure_loudness(media_path: Path) -> tuple[float, float]:
    """Return (mean_db, max_db) for a 1-minute segment via ffmpeg volumedetect."""
    cmd = [
        "ffmpeg",
        "-i",
        str(media_path),
        "-af",
        "volumedetect",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    mean_match = _MEAN_VOL_RE.search(proc.stderr)
    max_match = _MAX_VOL_RE.search(proc.stderr)
    mean_db = float(mean_match.group(1)) if mean_match else -30.0
    max_db = float(max_match.group(1)) if max_match else -10.0
    return mean_db, max_db


def _resolve_live_stream_url(page_url: str) -> str:
    """Resolve the direct HLS/stream URL via yt-dlp -g so ffmpeg can segment it."""
    from .fetcher import get_ytdlp_cookie_args

    cookie_args = get_ytdlp_cookie_args()
    proc = subprocess.run(
        ["yt-dlp", *cookie_args, "-g", "-f", "best[height<=1080]/best", page_url],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        err = proc.stderr[-800:]
        if "Sign in to confirm" in err and not cookie_args:
            err += (
                "\n[hint] YouTube requested sign-in cookies. Select a browser in "
                "'YouTube Cookies' (or provide a cookies.txt path) in the left panel."
            )
        raise RuntimeError(f"Could not resolve live stream URL: {err}")
    return proc.stdout.strip().splitlines()[0]


def _concat_buffer_segments(seg_paths: list[Path], out_path: Path) -> Path:
    """Concatenate buffer .ts segments losslessly into a single clip source."""
    list_file = out_path.with_suffix(".concat.txt")
    lines = [f"file '{p.resolve().as_posix()}'" for p in seg_paths]
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c",
        "copy",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Buffer concat failed: {proc.stderr[-1000:]}")
    return out_path


def run_live_clipper(
    url: str,
    run_dir: str | Path,
    judge: Judge,
    buffer_min: int = 15,
    ratios: list[str] = ["9:16", "1:1", "16:9"],
    enable_subtitles: bool = True,
    min_score: float = 7.5,
    whisper_model: str = "small.en",
    twitch_channel: str | None = None,
    streamer: str = "Streamer",
    game_tag: str | None = None,
    stop_event: threading.Event | None = None,
    on_clip_ready: Callable[[ClipArtifact], None] | None = None,
    compute_mode: str = "auto",
    enable_hook: bool = True,
    profile: str = "profile1",
) -> list[ClipArtifact]:
    """Mode B: Record rolling 1-min segments, run local Whisper, judge 5-min overlapping windows."""
    t0 = time.monotonic()
    run_path = Path(run_dir)
    buffer_dir = run_path / "buffer"
    clips_dir = run_path / "clips"
    buffer_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    stop = stop_event or threading.Event()
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, lambda *_: stop.set())

    chat_monitor: TwitchChatVelocityMonitor | None = None
    if twitch_channel:
        chat_monitor = TwitchChatVelocityMonitor(twitch_channel)
        chat_monitor.start()

    stream_url = _resolve_live_stream_url(url)
    seg_pattern = str(buffer_dir / "seg_%05d.ts")
    recorder_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        stream_url,
        "-c",
        "copy",
        "-f",
        "segment",
        "-segment_time",
        "60",
        "-reset_timestamps",
        "1",
        seg_pattern,
    ]
    recorder = subprocess.Popen(
        recorder_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(
        f"[live] Recording 1-min segments into {buffer_dir} (rolling {buffer_min}-min buffer)...",
        flush=True,
    )

    records: deque[LiveSegmentRecord] = deque()
    processed_indices: set[int] = set()
    picked_intervals: list[tuple[float, float]] = []
    artifacts: list[ClipArtifact] = []
    last_judged_end = 0.0
    window_idx = 0
    clip_idx = 0
    baseline_db_history: deque[float] = deque(maxlen=15)
    baseline_chat_history: deque[int] = deque(maxlen=15)

    try:
        while not stop.is_set():
            seg_files = sorted(buffer_dir.glob("seg_*.ts"))
            # All segments except the newest one (still being written) are complete
            complete_files = seg_files[:-1] if recorder.poll() is None else seg_files

            for seg_file in complete_files:
                m = re.search(r"seg_(\d+)\.ts$", seg_file.name)
                if not m:
                    continue
                idx = int(m.group(1))
                if idx in processed_indices:
                    continue
                processed_indices.add(idx)

                seg_start = idx * 60.0
                seg_end = seg_start + 60.0
                mean_db, max_db = measure_loudness(seg_file)
                baseline_db_history.append(mean_db)

                segs, words = transcribe_media(
                    seg_file,
                    model_size=whisper_model,
                    device=compute_mode,
                    time_offset=seg_start,
                )
                records.append(
                    LiveSegmentRecord(
                        index=idx,
                        path=seg_file,
                        start_sec=seg_start,
                        end_sec=seg_end,
                        mean_db=mean_db,
                        max_db=max_db,
                        segments=segs,
                        words=words,
                    )
                )

                # Auto-prune segments older than buffer_min minutes
                while len(records) > buffer_min:
                    old = records.popleft()
                    try:
                        old.path.unlink(missing_ok=True)
                    except OSError:
                        pass

                # Judge every ~4m15s (5-min windows with 45s overlap)
                latest_end = records[-1].end_sec
                if latest_end >= 300.0 and (latest_end - last_judged_end) >= 255.0:
                    last_judged_end = latest_end
                    all_segs = [s for r in records for s in r.segments]
                    avg_db = sum(baseline_db_history) / max(1, len(baseline_db_history))
                    loud_spikes = sum(1 for r in records if r.mean_db > avg_db + 4.5)
                    signals: dict = {"loudness_spikes": loud_spikes, "avg_db": round(avg_db, 1)}
                    if chat_monitor is not None:
                        cpm = chat_monitor.count_in_window(60.0)
                        baseline_chat_history.append(cpm)
                        avg_cpm = sum(baseline_chat_history) / max(1, len(baseline_chat_history))
                        signals["chat_msgs_per_min"] = cpm
                        signals["chat_spikes"] = 1 if avg_cpm > 5 and cpm >= avg_cpm * 2.5 else 0

                    window = build_live_window(
                        segments=all_segs,
                        window_end=latest_end,
                        window_duration_sec=300.0,
                        block_index=window_idx,
                        signals=signals,
                    )
                    window_idx += 1
                    if window is None:
                        continue

                    cands = judge.judge_block(window, streamer=streamer, game_tag=game_tag)
                    for cand in cands:
                        if cand.score < min_score or cand.duration < 12.0:
                            continue
                        if overlaps_existing(cand.start_sec, cand.end_sec, picked_intervals):
                            continue
                        picked_intervals.append((cand.start_sec, cand.end_sec))

                        # Gather matching buffer segments and cut straight from local disk
                        matching = [
                            r
                            for r in records
                            if r.end_sec >= cand.start_sec - 4.0 and r.start_sec <= cand.end_sec + 4.0
                        ]
                        if not matching:
                            continue
                        clip_idx += 1
                        work_dir = run_path / "work" / f"live_clip_{clip_idx:02d}"
                        work_dir.mkdir(parents=True, exist_ok=True)
                        concat_mp4 = _concat_buffer_segments([r.path for r in matching], work_dir / "buf.ts")
                        buf_origin = matching[0].start_sec

                        win_segs = [s for r in matching for s in r.segments]
                        win_words = [w for r in matching for w in r.words]
                        snapped_start, snapped_end = snap_to_sentence_boundaries(
                            cand.start_sec, cand.end_sec, win_segs
                        )
                        dur = min(75.0, snapped_end - snapped_start)
                        if dur < 12.0:
                            continue

                        rebased = rebase_words_for_clip(win_words, snapped_start, snapped_start + dur)
                        preview = work_dir / "preview.jpg"
                        facecam = detect_facecam(concat_mp4, layout_override=cand.layout_override, preview_out=preview)
                        files = export_multi_ratio(
                            source_video=concat_mp4,
                            clip_start_in_source=max(0.0, snapped_start - buf_origin),
                            clip_duration=dur,
                            clip_index=clip_idx,
                            title=cand.title,
                            hook_text=cand.hook,
                            facecam=facecam,
                            words=rebased if enable_subtitles else [],
                            clips_dir=clips_dir,
                            ratios=ratios,
                            enable_subtitles=enable_subtitles,
                            work_dir=work_dir,
                            compute_mode=compute_mode,
                            enable_hook=enable_hook,
                            profile=profile,
                        )
                        art = ClipArtifact(
                            index=clip_idx,
                            candidate=cand,
                            layout=facecam.layout,
                            start_sec=snapped_start,
                            end_sec=snapped_start + dur,
                            duration=dur,
                            files=files,
                            metadata_files={},
                            preview_frame=str(preview) if preview.exists() else None,
                        )
                        write_clip_metadata_files(art)
                        artifacts.append(art)
                        write_summary_json(
                            clips_dir=clips_dir,
                            stream_info={"url": url, "streamer": streamer, "mode": "live"},
                            moments_found=len(picked_intervals),
                            artifacts=artifacts,
                            elapsed_sec=time.monotonic() - t0,
                        )
                        # One-line report per new clip (Section 4 Mode B)
                        print(
                            f"[NEW CLIP #{art.index:02d}] \"{art.candidate.title}\" | "
                            f"{format_timestamp(art.start_sec)} ({art.duration:.0f}s, {art.layout}) -> "
                            f"{art.files.get('9:16', '')}",
                            flush=True,
                        )
                        if on_clip_ready:
                            on_clip_ready(art)

            if recorder.poll() is not None:
                break
            time.sleep(3.0)
    finally:
        if recorder.poll() is None:
            recorder.terminate()
            try:
                recorder.wait(timeout=10)
            except subprocess.TimeoutExpired:
                recorder.kill()
        if chat_monitor is not None:
            chat_monitor.stop()

    return artifacts
