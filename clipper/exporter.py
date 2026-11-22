"""Component 9: Multi-ratio ffmpeg compositor (9:16, 1:1, 16:9) + loudnorm + H.264/AAC export."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .captions import generate_ass_subtitles
from .models import (
    FacecamBox,
    RATIO_DIMENSIONS,
    RATIO_SUFFIX,
    WordTimestamp,
    slugify,
)


def _escape_ffmpeg_sub_path(path: Path) -> str:
    """Escape a Windows or POSIX file path for ffmpeg's ass='...' filter."""
    s = path.resolve().as_posix()
    # Escape drive colon (D:/ -> D\:/) and single quotes
    s = s.replace(":", r"\:").replace("'", r"\'")
    return s


def build_video_filtergraph(
    ratio: str,
    facecam: FacecamBox,
    ass_path: Path | None = None,
) -> tuple[str, str]:
    """Build the ffmpeg -filter_complex graph and output video pad name for a ratio & layout."""
    out_w, out_h = RATIO_DIMENSIONS[ratio]
    sub_suffix = f",ass='{_escape_ffmpeg_sub_path(ass_path)}'" if ass_path else ""

    if ratio == "9:16" and facecam.layout == "facecam" and facecam.w > 0 and facecam.h > 0:
        # Split-screen: top 640px facecam, bottom 1280px gameplay
        top_h = 640
        bot_h = out_h - top_h
        fg = (
            f"[0:v]split=2[cam_in][game_in];"
            f"[cam_in]crop={facecam.w}:{facecam.h}:{facecam.x}:{facecam.y},"
            f"scale={out_w}:{top_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{top_h},setsar=1[top];"
            f"[game_in]scale={out_w}:{bot_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{bot_h},setsar=1[bot];"
            f"[top][bot]vstack=inputs=2{sub_suffix}[vout]"
        )
        return fg, "[vout]"

    if ratio == "9:16" and facecam.layout == "cam":
        # Full-frame vertical center crop for fullscreen webcam
        fg = (
            f"[0:v]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{out_h},setsar=1{sub_suffix}[vout]"
        )
        return fg, "[vout]"

    if ratio == "16:9":
        # Native landscape with blur-fill safety if source is non-16:9
        fg = (
            f"[0:v]split=2[bg_in][fg_in];"
            f"[bg_in]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{out_h},boxblur=20:5,setsar=1[bg];"
            f"[fg_in]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,setsar=1[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2{sub_suffix}[vout]"
        )
        return fg, "[vout]"

    # 9:16 gameplay fallback OR 1:1 feed square: blurred zoomed background + centered sharp video
    fg = (
        f"[0:v]split=2[bg_in][fg_in];"
        f"[bg_in]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
        f"crop={out_w}:{out_h},boxblur=20:5,setsar=1[bg];"
        f"[fg_in]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,setsar=1[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2{sub_suffix}[vout]"
    )
    return fg, "[vout]"


def export_clip_ratio(
    source_video: str | Path,
    clip_start_in_source: float,
    clip_duration: float,
    ratio: str,
    facecam: FacecamBox,
    words: list[WordTimestamp],
    hook_text: str,
    out_mp4: str | Path,
    enable_subtitles: bool = True,
    work_dir: str | Path | None = None,
) -> Path:
    """Render a single aspect ratio MP4 with optional burned-in subtitles and loudnorm."""
    out_path = Path(out_mp4)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ass_path: Path | None = None
    if enable_subtitles:
        sub_dir = Path(work_dir) if work_dir else out_path.parent
        ass_path = sub_dir / f"{out_path.stem}.ass"
        generate_ass_subtitles(
            words=words,
            hook_text=hook_text,
            clip_duration=clip_duration,
            ratio=ratio,
            out_path=ass_path,
        )

    filter_complex, v_pad = build_video_filtergraph(ratio=ratio, facecam=facecam, ass_path=ass_path)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{max(0.0, clip_start_in_source):.3f}",
        "-i",
        str(source_video),
        "-t",
        f"{max(1.0, clip_duration):.3f}",
        "-filter_complex",
        filter_complex,
        "-map",
        v_pad,
        "-map",
        "0:a?",
        "-af",
        "loudnorm=I=-14:TP=-1.5:LRA=11",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(out_path),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg export failed for {ratio} ({out_path.name}):\n{proc.stderr[-1500:]}"
        )
    return out_path


def export_multi_ratio(
    source_video: str | Path,
    clip_start_in_source: float,
    clip_duration: float,
    clip_index: int,
    title: str,
    hook_text: str,
    facecam: FacecamBox,
    words: list[WordTimestamp],
    clips_dir: str | Path,
    ratios: tuple[str, ...] | list[str] = ("9:16", "1:1", "16:9"),
    enable_subtitles: bool = True,
    work_dir: str | Path | None = None,
) -> dict[str, str]:
    """Export the same clipped master across all requested aspect ratios."""
    out_dir = Path(clips_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(title)
    exported: dict[str, str] = {}

    for ratio in ratios:
        suffix = RATIO_SUFFIX[ratio]
        mp4_path = out_dir / f"clip_{clip_index:02d}_{slug}_{suffix}.mp4"
        export_clip_ratio(
            source_video=source_video,
            clip_start_in_source=clip_start_in_source,
            clip_duration=clip_duration,
            ratio=ratio,
            facecam=facecam,
            words=words,
            hook_text=hook_text,
            out_mp4=mp4_path,
            enable_subtitles=enable_subtitles,
            work_dir=work_dir,
        )
        exported[ratio] = str(mp4_path)

    return exported


def probe_media(media_path: str | Path) -> dict[str, Any]:
    """Run ffprobe on an output clip to verify resolution, codecs, and duration."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(media_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {media_path}: {proc.stderr}")
    data = json.loads(proc.stdout)
    v_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    a_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), {})
    return {
        "width": int(v_stream.get("width", 0)),
        "height": int(v_stream.get("height", 0)),
        "video_codec": v_stream.get("codec_name", ""),
        "audio_codec": a_stream.get("codec_name", ""),
        "duration": float(data.get("format", {}).get("duration", 0.0)),
    }
