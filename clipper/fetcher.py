"""Component 5: yt-dlp caption fetcher, audio chunk fetcher, and --download-sections video fetcher."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .models import format_timestamp


def _run_ytdlp(args: list[str], timeout_sec: int = 600) -> subprocess.CompletedProcess[str]:
    cmd = ["yt-dlp", "--no-playlist", *args]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
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
    """Download an audio-only section (e.g. a 10-min chunk) when no platform captions exist."""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
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
    wav_path = target.with_suffix(".wav")
    if wav_path.exists():
        return wav_path
    # Fallback if yt-dlp preserved original extension
    matches = sorted(target.parent.glob(f"{target.stem}.*"))
    if not matches:
        raise FileNotFoundError(f"Expected audio section output at {wav_path}")
    return matches[0]


def fetch_video_section(
    url: str,
    start_sec: float,
    end_sec: float,
    out_path: str | Path,
    pad_before_sec: float = 4.0,
    pad_after_sec: float = 4.0,
) -> tuple[Path, float]:
    """Download only a 20-60s video window (+ small padding) via --download-sections.

    Returns (downloaded_path, actual_window_start_sec).
    """
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fetch_start = max(0.0, start_sec - pad_before_sec)
    fetch_end = max(fetch_start + 5.0, end_sec + pad_after_sec)
    section = f"*{format_timestamp(fetch_start)}-{format_timestamp(fetch_end)}"

    _run_ytdlp(
        [
            "--download-sections",
            section,
            "--force-keyframes-at-cuts",
            "-f",
            "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4][height<=1080]/best",
            "--merge-output-format",
            "mp4",
            "--force-overwrites",
            "-o",
            str(target),
            url,
        ],
        timeout_sec=300,
    )
    if not target.exists():
        matches = sorted(target.parent.glob(f"{target.stem}.*"))
        if not matches:
            raise FileNotFoundError(f"Expected video section output at {target}")
        return matches[0], fetch_start
    return target, fetch_start
