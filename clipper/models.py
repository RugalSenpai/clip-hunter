"""Shared data models and timestamp utilities for the clipping pipeline."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

VALID_CATEGORIES = frozenset(
    {"first_reaction", "discovery", "freakout", "fail", "hot_take", "funny"}
)
VALID_RATIOS = ("9:16", "1:1", "16:9")
RATIO_SUFFIX = {"9:16": "916", "1:1": "11", "16:9": "169"}
RATIO_DIMENSIONS = {
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "16:9": (1920, 1080),
}


def parse_timestamp(ts: str | float | int) -> float:
    """Parse HH:MM:SS(.mmm), MM:SS(.mmm), or numeric seconds into float seconds."""
    if isinstance(ts, (int, float)):
        return float(ts)
    s = str(ts).strip()
    if not s:
        raise ValueError("Empty timestamp")
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)
    parts = s.replace(",", ".").split(":")
    if len(parts) == 3:
        h, m, sec = parts
        return int(h) * 3600 + int(m) * 60 + float(sec)
    if len(parts) == 2:
        m, sec = parts
        return int(m) * 60 + float(sec)
    raise ValueError(f"Invalid timestamp format: {ts!r}")


def format_timestamp(seconds: float, include_ms: bool = False) -> str:
    """Format seconds as HH:MM:SS (or HH:MM:SS.mmm if include_ms=True)."""
    seconds = max(0.0, float(seconds))
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_sec = total_ms // 1000
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    s = total_sec % 60
    if include_ms:
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


def slugify(text: str, max_len: int = 40) -> str:
    """Convert a clip title into a filesystem-safe snake_case slug."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    if not cleaned:
        return "clip"
    return cleaned[:max_len].rstrip("_") or "clip"


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"start": round(self.start, 3), "end": round(self.end, 3), "text": self.text}


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float


@dataclass
class TranscriptBlock:
    index: int
    start: float
    end: float
    segments: list[TranscriptSegment]
    prefilter_score: float = 0.0
    signals: dict[str, Any] = field(default_factory=dict)

    def format_transcript(self) -> str:
        """Render block lines with [HH:MM:SS] timestamps for LLM judging."""
        lines = []
        for seg in self.segments:
            stamp = format_timestamp(seg.start)
            lines.append(f"[{stamp}] {seg.text}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": format_timestamp(self.start),
            "end": format_timestamp(self.end),
            "start_sec": round(self.start, 2),
            "end_sec": round(self.end, 2),
            "prefilter_score": round(self.prefilter_score, 3),
            "signals": self.signals,
            "transcript": self.format_transcript(),
        }


@dataclass
class Candidate:
    start: str
    end: str
    category: str
    score: float
    hook: str
    title: str
    caption: str
    hashtags: list[str]
    why: str
    layout_override: str | None = None

    @property
    def start_sec(self) -> float:
        return parse_timestamp(self.start)

    @property
    def end_sec(self) -> float:
        return parse_timestamp(self.end)

    @property
    def duration(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d.get("layout_override") is None:
            d.pop("layout_override", None)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Candidate":
        hashtags = data.get("hashtags") or []
        if isinstance(hashtags, str):
            hashtags = [h.strip() for h in hashtags.split() if h.strip()]
        return cls(
            start=format_timestamp(parse_timestamp(data["start"])),
            end=format_timestamp(parse_timestamp(data["end"])),
            category=str(data.get("category", "funny")).strip().lower(),
            score=float(data.get("score", 0.0)),
            hook=str(data.get("hook", "")).strip(),
            title=str(data.get("title", "")).strip(),
            caption=str(data.get("caption", "")).strip(),
            hashtags=[h if h.startswith("#") else f"#{h}" for h in hashtags],
            why=str(data.get("why", "")).strip(),
            layout_override=data.get("layout_override"),
        )


@dataclass
class FacecamBox:
    x: int
    y: int
    w: int
    h: int
    frame_w: int
    frame_h: int
    layout: str  # "facecam", "cam", or "gameplay"
    confidence: float = 0.0


@dataclass
class ClipArtifact:
    index: int
    candidate: Candidate
    layout: str
    start_sec: float
    end_sec: float
    duration: float
    files: dict[str, str]  # ratio -> mp4 path
    metadata_files: dict[str, str]  # ratio -> txt path
    preview_frame: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "title": self.candidate.title,
            "hook": self.candidate.hook,
            "category": self.candidate.category,
            "score": self.candidate.score,
            "start": format_timestamp(self.start_sec),
            "end": format_timestamp(self.end_sec),
            "duration_sec": round(self.duration, 2),
            "layout": self.layout,
            "files": self.files,
            "metadata_files": self.metadata_files,
            "preview_frame": self.preview_frame,
            "caption": self.candidate.caption,
            "hashtags": self.candidate.hashtags,
            "why": self.candidate.why,
        }
