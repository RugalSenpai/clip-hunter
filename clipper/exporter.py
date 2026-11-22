"""Component 9: Multi-ratio ffmpeg compositor (9:16, 1:1, 16:9) + loudnorm + H.264/AAC export."""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
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
    profile: str = "profile1",
    game_x_pct: float = 50.0,
) -> tuple[str, str]:
    """Build the ffmpeg -filter_complex graph and output video pad name for a ratio & layout."""
    out_w, out_h = RATIO_DIMENSIONS[ratio]
    sub_suffix = f",ass='{_escape_ffmpeg_sub_path(ass_path)}'" if ass_path else ""

    x_frac = max(0.0, min(1.0, float(game_x_pct) / 100.0))

    if ratio == "9:16" and facecam.layout == "full_game":
        # Whole-screen 9:16 zoomed gameplay with horizontal pan (no facecam, no blur bars)
        fg = (
            f"[0:v]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{out_h}:(iw-{out_w})*{x_frac:.4f}:(ih-{out_h})/2,setsar=1{sub_suffix}[vout]"
        )
        return fg, "[vout]"

    use_split = ratio == "9:16" and (
        profile == "profile2"
        or (facecam.layout == "facecam" and facecam.w > 0 and facecam.h > 0)
    )
    if use_split:
        # 30% facecam overlaid at the top of 100% full-height horizontally panned gameplay
        fw = facecam.frame_w or 1920
        fh = facecam.frame_h or 1080
        cw = facecam.w if facecam.w > 0 else int(fw * 0.28)
        ch = facecam.h if facecam.h > 0 else int(fh * 0.28)
        cx = facecam.x if facecam.w > 0 else 0
        cy = facecam.y if facecam.h > 0 else int(fh * 0.36)
        top_h = int(round(out_h * 0.30 / 2) * 2)
        fg = (
            f"[0:v]split=2[cam_in][game_in];"
            f"[cam_in]crop={cw}:{ch}:{cx}:{cy},"
            f"scale={out_w}:{top_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{top_h},setsar=1[top];"
            f"[game_in]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{out_h}:(iw-{out_w})*{x_frac:.4f}:(ih-{out_h})/2,setsar=1[game];"
            f"[game][top]overlay=0:0{sub_suffix}[vout]"
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


_NVENC_AVAILABLE: bool | None = None


def has_nvenc() -> bool:
    """Return True if ffmpeg supports h264_nvenc."""
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is None:
        try:
            proc = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            _NVENC_AVAILABLE = "h264_nvenc" in proc.stdout
        except Exception:
            _NVENC_AVAILABLE = False
    return _NVENC_AVAILABLE


def resolve_ratio_encoders(
    ratios: tuple[str, ...] | list[str],
    compute_mode: str = "auto",
) -> dict[str, str]:
    """Choose 'h264_nvenc' or 'libx264' per ratio based on compute_mode ('auto','cpu','gpu','hybrid')."""
    mode = (compute_mode or "auto").lower()
    nvenc_ok = has_nvenc()
    if mode == "cpu" or not nvenc_ok:
        return {r: "libx264" for r in ratios}
    if mode == "hybrid" and len(ratios) >= 2:
        # Run GPU (NVENC) and CPU (libx264) simultaneously across ratios
        return {
            r: ("libx264" if i == len(ratios) - 1 else "h264_nvenc")
            for i, r in enumerate(ratios)
        }
    return {r: "h264_nvenc" for r in ratios}


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
    video_encoder: str = "auto",
    enable_hook: bool = True,
    hook_y_pct: float | None = None,
    sub_y_pct: float | None = None,
    highlight_color: str = "#FFFF00",
    profile: str = "profile1",
    game_x_pct: float = 50.0,
    on_progress: Any = None,
) -> Path:
    """Render a single aspect ratio MP4 with optional burned-in subtitles and loudnorm."""
    out_path = Path(out_mp4)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ass_path: Path | None = None
    if enable_subtitles or enable_hook:
        sub_dir = Path(work_dir) if work_dir else out_path.parent
        ass_path = sub_dir / f"{out_path.stem}.ass"
        generate_ass_subtitles(
            words=words if enable_subtitles else [],
            hook_text=hook_text,
            clip_duration=clip_duration,
            ratio=ratio,
            out_path=ass_path,
            include_hook=enable_hook,
            hook_y_pct=hook_y_pct,
            sub_y_pct=sub_y_pct,
            highlight_color=highlight_color,
        )

    filter_complex, v_pad = build_video_filtergraph(
        ratio=ratio, facecam=facecam, ass_path=ass_path, profile=profile, game_x_pct=game_x_pct
    )
    chosen_enc = (
        ("h264_nvenc" if has_nvenc() else "libx264")
        if video_encoder == "auto"
        else video_encoder
    )

    def _run_encode(enc: str) -> subprocess.CompletedProcess[str]:
        v_flags = (
            ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20", "-b:v", "0"]
            if enc == "h264_nvenc"
            else ["-c:v", "libx264", "-preset", "fast", "-crf", "20"]
        )
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
            *v_flags,
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
        from .fetcher import _run_interruptible, parse_progress_pct

        last_pct = -1

        def _on_line(line: str) -> None:
            nonlocal last_pct
            if not on_progress:
                return
            pct = parse_progress_pct(line, total_dur=max(1.0, clip_duration))
            if pct is not None and pct > last_pct:
                last_pct = pct
                on_progress(pct)

        return _run_interruptible(cmd, timeout_sec=600, on_line=_on_line if on_progress else None)

    proc = _run_encode(chosen_enc)
    if proc.returncode != 0 and chosen_enc == "h264_nvenc":
        # Automatic CPU fallback if NVENC fails at runtime
        proc = _run_encode("libx264")
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
    compute_mode: str = "auto",
    enable_hook: bool = True,
    hook_y_pct: float | None = None,
    sub_y_pct: float | None = None,
    highlight_color: str = "#FFFF00",
    profile: str = "profile1",
    game_x_pct: float = 50.0,
    on_progress: Any = None,
) -> dict[str, str]:
    """Export the same clipped master across all requested aspect ratios."""
    out_dir = Path(clips_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(title)
    if profile == "profile2":
        ratios = ("9:16",)
    encoders = resolve_ratio_encoders(ratios, compute_mode=compute_mode)

    def _job(ratio: str) -> tuple[str, str]:
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
            video_encoder=encoders[ratio],
            enable_hook=enable_hook,
            hook_y_pct=hook_y_pct,
            sub_y_pct=sub_y_pct,
            highlight_color=highlight_color,
            profile=profile,
            game_x_pct=game_x_pct,
            on_progress=on_progress,
        )
        return ratio, str(mp4_path)

    # Parallelize multi-ratio encoding when GPU/hybrid is active (or >1 ratio)
    if len(ratios) > 1 and (compute_mode or "auto").lower() in ("auto", "gpu", "hybrid"):
        with ThreadPoolExecutor(max_workers=len(ratios)) as pool:
            results = list(pool.map(_job, ratios))
        return dict(results)

    return dict(_job(r) for r in ratios)


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
