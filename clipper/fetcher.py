"""Component 5: yt-dlp caption fetcher, audio chunk fetcher, and --download-sections video fetcher."""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable

from .models import format_timestamp

_YTDLP_COOKIES_BROWSER = ""
_YTDLP_COOKIES_FILE = ""
_STOP_EVENT: threading.Event | None = None


def set_active_stop_event(ev: threading.Event | None) -> None:
    """Register the active pipeline stop_event so long yt-dlp/ffmpeg calls abort promptly."""
    global _STOP_EVENT
    _STOP_EVENT = ev


def is_stop_requested() -> bool:
    """Return True if the active pipeline stop_event has been triggered."""
    return _STOP_EVENT is not None and _STOP_EVENT.is_set()


_DL_PCT_RE = re.compile(r"\[download\]\s+(\d+(?:\.\d+)?)%")
_FF_TIME_RE = re.compile(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)")


def parse_progress_pct(line: str, total_dur: float = 0.0) -> int | None:
    """Extract 0-100% progress from a yt-dlp [download] line or ffmpeg time= line."""
    m = _DL_PCT_RE.search(line)
    if m:
        return max(0, min(99, int(float(m.group(1)))))
    if total_dur > 0.5:
        mt = _FF_TIME_RE.search(line)
        if mt:
            sec = int(mt.group(1)) * 3600 + int(mt.group(2)) * 60 + float(mt.group(3))
            return max(0, min(99, int((sec / total_dur) * 100)))
    return None


def _run_interruptible(
    cmd: list[str],
    timeout_sec: int = 600,
    on_line: Callable[[str], None] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess, draining pipes continuously and terminating promptly if _STOP_EVENT is set."""
    import time

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    end_time = time.monotonic() + timeout_sec if timeout_sec > 0 else None

    if on_line is None:
        while True:
            try:
                stdout, stderr = proc.communicate(timeout=0.2)
                return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
            except subprocess.TimeoutExpired:
                if _STOP_EVENT is not None and _STOP_EVENT.is_set():
                    proc.kill()
                    try:
                        proc.communicate(timeout=2)
                    except Exception:
                        pass
                    raise RuntimeError("Cancelled by user")
                if end_time is not None and time.monotonic() > end_time:
                    proc.kill()
                    try:
                        proc.communicate(timeout=2)
                    except Exception:
                        pass
                    raise subprocess.TimeoutExpired(cmd, timeout_sec)

    out_chunks: list[str] = []
    err_chunks: list[str] = []

    def _drain(stream, bucket: list[str]) -> None:
        try:
            for line in iter(stream.readline, ""):
                bucket.append(line)
                try:
                    on_line(line)
                except Exception:
                    pass
        finally:
            try:
                stream.close()
            except Exception:
                pass

    t_out = threading.Thread(target=_drain, args=(proc.stdout, out_chunks), daemon=True)
    t_err = threading.Thread(target=_drain, args=(proc.stderr, err_chunks), daemon=True)
    t_out.start()
    t_err.start()

    while proc.poll() is None:
        if _STOP_EVENT is not None and _STOP_EVENT.is_set():
            proc.kill()
            t_out.join(timeout=1)
            t_err.join(timeout=1)
            raise RuntimeError("Cancelled by user")
        if end_time is not None and time.monotonic() > end_time:
            proc.kill()
            t_out.join(timeout=1)
            t_err.join(timeout=1)
            raise subprocess.TimeoutExpired(cmd, timeout_sec)
        time.sleep(0.15)

    t_out.join(timeout=2)
    t_err.join(timeout=2)
    return subprocess.CompletedProcess(cmd, proc.returncode, "".join(out_chunks), "".join(err_chunks))


def set_ytdlp_cookies(browser: str = "", cookies_file: str = "") -> None:
    """Configure optional yt-dlp browser/file cookies for authenticated YouTube requests."""
    global _YTDLP_COOKIES_BROWSER, _YTDLP_COOKIES_FILE
    _YTDLP_COOKIES_BROWSER = (browser or "").strip().lower()
    _YTDLP_COOKIES_FILE = (cookies_file or "").strip()


def get_ytdlp_cookie_args() -> list[str]:
    """Return yt-dlp CLI flags for cookies if configured via UI or environment."""
    cookies_file = _YTDLP_COOKIES_FILE or os.environ.get("CLIPPER_COOKIES_FILE", "").strip()
    if cookies_file and Path(cookies_file).exists():
        return ["--cookies", cookies_file]
    browser = _YTDLP_COOKIES_BROWSER or os.environ.get("CLIPPER_COOKIES_BROWSER", "").strip().lower()
    if browser and browser != "none":
        return ["--cookies-from-browser", browser]
    return []


def _run_ytdlp(
    args: list[str],
    timeout_sec: int = 600,
    on_line: Callable[[str], None] | None = None,
) -> subprocess.CompletedProcess[str]:
    cmd = ["yt-dlp", "--no-playlist", *get_ytdlp_cookie_args(), *args]
    proc = _run_interruptible(cmd, timeout_sec=timeout_sec, on_line=on_line)
    if proc.returncode != 0:
        raise RuntimeError(
            f"yt-dlp failed (exit {proc.returncode}): {' '.join(cmd)}\n"
            f"stderr: {proc.stderr[-1500:]}"
        )
    return proc


def get_video_info(url: str) -> dict[str, Any]:
    """Fetch lightweight video metadata (id, title, uploader, duration) without downloading media."""
    proc = _run_ytdlp(["--dump-single-json", "--skip-download", url], timeout_sec=120)
    data = json.loads(proc.stdout)
    return {
        "id": data.get("id", "video"),
        "title": data.get("title", "Untitled"),
        "uploader": data.get("uploader") or data.get("channel") or "Streamer",
        "duration": float(data.get("duration") or 0.0),
        "webpage_url": data.get("webpage_url") or url,
    }


def fetch_captions(url: str, captions_dir: str | Path, lang: str = "en") -> Path | None:
    """Download auto-captions (.vtt) only — never video or live_chat."""
    out_dir = Path(captions_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    template = str(out_dir / "%(id)s.%(ext)s")

    try:
        _run_ytdlp(
            [
                "--skip-download",
                "--write-auto-sub",
                "--write-sub",
                "--sub-lang",
                lang,
                "--sub-format",
                "vtt",
                "-o",
                template,
                url,
            ],
            timeout_sec=180,
        )
    except RuntimeError:
        return None

    vtt_files = sorted(out_dir.glob("*.vtt"))
    if not vtt_files:
        return None
    # Prefer `<id>.en.vtt` over `<id>.en-orig.vtt` when both exist
    for f in vtt_files:
        if f.name.endswith(f".{lang}.vtt"):
            return f
    return vtt_files[0]


def fetch_audio_section(
    url: str,
    start_sec: float,
    end_sec: float,
    out_path: str | Path,
) -> Path:
    """Download an audio-only section (e.g. a 10-min chunk) in 16kHz mono WAV for Whisper."""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    wav_path = target.with_suffix(".wav")
    dur = max(1.0, end_sec - start_sec)

    # Fast path: resolve direct audio stream URL and seek/decode to 16kHz mono WAV in one ffmpeg pass
    try:
        proc = _run_ytdlp(
            [
                "--extractor-args",
                "youtube:player_client=android,web",
                "-g",
                "-f",
                "bestaudio[ext=m4a]/bestaudio/best",
                url,
            ],
            timeout_sec=60,
        )
        direct_url = proc.stdout.strip().splitlines()[0]
        if direct_url:
            ff = _run_interruptible(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{max(0.0, start_sec):.3f}",
                    "-i",
                    direct_url,
                    "-t",
                    f"{dur:.3f}",
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(wav_path),
                ],
                timeout_sec=180,
            )
            if ff.returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 1000:
                return wav_path
    except Exception:
        if _STOP_EVENT is not None and _STOP_EVENT.is_set():
            raise

    # Fallback: yt-dlp --download-sections
    section = f"*{format_timestamp(start_sec)}-{format_timestamp(end_sec)}"
    _run_ytdlp(
        [
            "--download-sections",
            section,
            "-f",
            "bestaudio/best",
            "-x",
            "--audio-format",
            "wav",
            "--force-overwrites",
            "-o",
            str(target.with_suffix(".%(ext)s")),
            url,
        ],
        timeout_sec=300,
    )
    if wav_path.exists():
        return wav_path
    matches = sorted(target.parent.glob(f"{target.stem}.*"))
    if not matches:
        raise FileNotFoundError(f"Expected audio section output at {wav_path}")
    return matches[0]


def _probe_video_height(path: Path) -> int:
    """Return video stream height via ffprobe, or 0 if unavailable."""
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=height",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return int(proc.stdout.strip().splitlines()[0]) if proc.stdout.strip() else 0
    except Exception:
        return 0


def fetch_video_section(
    url: str,
    start_sec: float,
    end_sec: float,
    out_path: str | Path,
    pad_before_sec: float = 4.0,
    pad_after_sec: float = 4.0,
    on_progress: Callable[[int], None] | None = None,
) -> tuple[Path, float]:
    """Download only a 20-60s video window (+ small padding) via --download-sections.

    Returns (downloaded_path, actual_window_start_sec).
    """
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fetch_start = max(0.0, start_sec - pad_before_sec)
    fetch_end = max(fetch_start + 5.0, end_sec + pad_after_sec)
    total_dur = max(1.0, fetch_end - fetch_start)
    if target.exists() and target.stat().st_size > 100_000:
        if on_progress:
            on_progress(100)
        return target, fetch_start
    section = f"*{format_timestamp(fetch_start)}-{format_timestamp(fetch_end)}"

    last_pct = -1

    def _on_line(line: str) -> None:
        nonlocal last_pct
        if not on_progress:
            return
        pct = parse_progress_pct(line, total_dur=total_dur)
        if pct is not None and pct > last_pct:
            last_pct = pct
            on_progress(pct)

    base_args = [
        "--newline",
        "--download-sections",
        section,
        "--force-keyframes-at-cuts",
        "-S",
        "res:1080,vcodec:h264,ext:mp4:m4a",
        "-f",
        "bestvideo[height>=1080]+bestaudio/bestvideo+bestaudio/best",
        "--merge-output-format",
        "mp4",
        "--force-overwrites",
        "-o",
        str(target),
    ]
    clients_to_try: list[list[str]] = [
        [],
        ["--extractor-args", "youtube:player_client=web_safari,web"],
        ["--extractor-args", "youtube:player_client=tv,web"],
    ]
    last_err: Exception | None = None
    best_height = 0
    best_backup: Path | None = None

    for extra_args in clients_to_try:
        try:
            _run_ytdlp(
                [*base_args, *extra_args, url],
                timeout_sec=300,
                on_line=_on_line if on_progress else None,
            )
        except RuntimeError as exc:
            last_err = exc
            if is_stop_requested():
                raise
            if "403" in str(exc) or "forbidden" in str(exc).lower():
                continue
            if not extra_args:
                raise
            continue

        cand_path = target if target.exists() else next(iter(sorted(target.parent.glob(f"{target.stem}.*"))), None)
        if cand_path is None or not cand_path.exists():
            continue
        h = _probe_video_height(cand_path)
        if h >= 1080:
            if best_backup is not None:
                best_backup.unlink(missing_ok=True)
            return cand_path, fetch_start
        # Keep the best <1080p file as a fallback if remaining clients fail
        if h > best_height:
            best_height = h
            backup = cand_path.with_name(f"{cand_path.stem}_best{cand_path.suffix}")
            cand_path.replace(backup)
            best_backup = backup

    if best_backup is not None and best_backup.exists():
        best_backup.replace(target)
        return target, fetch_start
    if last_err is not None:
        raise last_err
    if not target.exists():
        matches = sorted(target.parent.glob(f"{target.stem}.*"))
        if not matches:
            raise FileNotFoundError(f"Expected video section output at {target}")
        return matches[0], fetch_start
    return target, fetch_start


def resolve_hls_stream_url(url: str) -> tuple[str, str | None]:
    """Resolve muxed HLS URL or (video_hls_url, audio_hls_url) via yt-dlp for online scrubbing."""
    for extra in (
        ["--extractor-args", "youtube:player_client=web_safari,web", "-f", "95/94/96/best[protocol*=m3u8][vcodec!=none][acodec!=none]"],
        ["-f", "bestvideo[height<=720][protocol*=m3u8]+bestaudio[protocol*=m3u8]/best[vcodec!=none][acodec!=none]"],
    ):
        try:
            proc = _run_ytdlp([*extra, "-g", url], timeout_sec=60)
            lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip().startswith("http")]
            if len(lines) >= 2:
                return lines[0], lines[1]
            if len(lines) == 1:
                return lines[0], None
        except RuntimeError:
            if is_stop_requested():
                raise
    raise RuntimeError("Could not resolve online HLS stream URL for this video.")


def fetch_preview_context(
    url: str,
    start_sec: float,
    end_sec: float,
    out_path: str | Path,
) -> tuple[Path, float]:
    """Fast-fetch a wider 720p context window for the Extend Clip scrubber."""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fetch_start = max(0.0, start_sec)
    fetch_end = max(fetch_start + 10.0, end_sec)
    section = f"*{format_timestamp(fetch_start)}-{format_timestamp(fetch_end)}"
    args = [
        "--download-sections",
        section,
        "-S",
        "res:720,vcodec:h264,ext:mp4:m4a",
        "-f",
        "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        "--merge-output-format",
        "mp4",
        "--force-overwrites",
        "-o",
        str(target),
    ]
    for extra in ([], ["--extractor-args", "youtube:player_client=web_safari,web"]):
        try:
            _run_ytdlp([*args, *extra, url], timeout_sec=300)
            if target.exists():
                return target, fetch_start
        except RuntimeError:
            if is_stop_requested() or extra:
                raise
    raise FileNotFoundError(f"Could not fetch preview context at {target}")
