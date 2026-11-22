"""Component 10: Per-clip .txt metadata writer, summary.json writer, and Section 10 report."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .models import ClipArtifact, format_timestamp


def write_clip_metadata_files(artifact: ClipArtifact) -> dict[str, str]:
    """Write a matching .txt file beside every exported ratio .mp4."""
    meta_files: dict[str, str] = {}
    cand = artifact.candidate
    hashtags_line = " ".join(cand.hashtags)
    content = (
        f"Title: {cand.title}\n"
        f"Hook: {cand.hook}\n"
        f"Category: {cand.category} (score {cand.score:.1f})\n"
        f"Source Timestamp: {format_timestamp(artifact.start_sec)} - {format_timestamp(artifact.end_sec)} "
        f"({artifact.duration:.1f}s)\n"
        f"Layout: {artifact.layout}\n"
        f"\n"
        f"Caption:\n"
        f"{cand.caption}\n"
        f"\n"
        f"Hashtags:\n"
        f"{hashtags_line}\n"
    )
    for ratio, mp4_path in artifact.files.items():
        txt_path = Path(mp4_path).with_suffix(".txt")
        txt_path.write_text(content, encoding="utf-8")
        meta_files[ratio] = str(txt_path)
    artifact.metadata_files = meta_files
    return meta_files


def write_summary_json(
    clips_dir: str | Path,
    stream_info: dict[str, Any],
    moments_found: int,
    artifacts: list[ClipArtifact],
    elapsed_sec: float,
) -> Path:
    """Write summary.json inside clips_dir."""
    out_dir = Path(clips_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "stream": stream_info,
        "moments_found": moments_found,
        "clips_kept": len(artifacts),
        "elapsed_sec": round(elapsed_sec, 2),
        "clips": [a.to_dict() for a in artifacts],
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path


def format_final_report(
    stream_duration_sec: float,
    moments_found: int,
    artifacts: list[ClipArtifact],
    elapsed_sec: float,
    clips_dir: str | Path,
) -> str:
    """Format the Section 10 final report."""
    lines = [
        f"Stream length: {format_timestamp(stream_duration_sec)} -> "
        f"Moments found: {moments_found} -> "
        f"Clips kept: {len(artifacts)} (total time: {elapsed_sec:.1f}s)",
        f"Output folder: {Path(clips_dir).resolve()}",
        "",
    ]
    for a in artifacts:
        ratio_paths = ", ".join(f"{r}: {p}" for r, p in a.files.items())
        lines.append(
            f"- Clip {a.index:02d}: \"{a.candidate.title}\" | "
            f"{format_timestamp(a.start_sec)}-{format_timestamp(a.end_sec)} "
            f"({a.duration:.1f}s) | layout: {a.layout} | {ratio_paths}"
        )
    if not artifacts:
        lines.append("- No clips met the selection threshold.")

    lines.append("")
    lines.append(
        "Reminder: Watch every clip before posting (check facecam framing and cut points), "
        "and only clip streamers who are fine with being clipped."
    )
    return "\n".join(lines)


def open_clips_folder(clips_dir: str | Path) -> None:
    """Open the output clips folder in Windows Explorer (best-effort)."""
    folder = str(Path(clips_dir).resolve())
    if sys.platform.startswith("win") and os.path.isdir(folder):
        try:
            subprocess.Popen(["explorer.exe", folder])
        except OSError:
            pass
