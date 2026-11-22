"""Component 1: WebVTT parser tailored for YouTube auto-captions."""

from __future__ import annotations

import html
import re
from pathlib import Path

from .models import TranscriptSegment, parse_timestamp

_CUE_TIME_RE = re.compile(
    r"(?P<start>(?:\d{2}:)?\d{2}:\d{2}\.\d{3})\s*-->\s*(?P<end>(?:\d{2}:)?\d{2}:\d{2}\.\d{3})"
)
_INLINE_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_PURE_NOISE_RE = re.compile(r"^[\[\(]?(?:music|♪|♫|silence)[\]\)]?$", re.IGNORECASE)


def _clean_cue_line(raw: str) -> str:
    text = _INLINE_TAG_RE.sub("", raw)
    text = html.unescape(text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _deduplicate_rolling_text(prev_text: str, curr_text: str) -> str:
    """Remove YouTube's rolling 2-line caption overlap from curr_text."""
    if not prev_text:
        return curr_text
    if curr_text == prev_text:
        return ""
    if curr_text.startswith(prev_text):
        return curr_text[len(prev_text) :].strip()
    # Suffix-prefix overlap check (word-aligned)
    prev_words = prev_text.split()
    curr_words = curr_text.split()
    max_overlap = min(len(prev_words), len(curr_words))
    for k in range(max_overlap, 0, -1):
        if [w.lower() for w in prev_words[-k:]] == [w.lower() for w in curr_words[:k]]:
            remainder = " ".join(curr_words[k:]).strip()
            return remainder
    return curr_text


def parse_vtt_text(vtt_content: str) -> list[TranscriptSegment]:
    """Parse raw WebVTT string into clean, deduplicated TranscriptSegments."""
    lines = vtt_content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    raw_cues: list[TranscriptSegment] = []

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        match = _CUE_TIME_RE.search(line)
        if not match:
            i += 1
            continue

        start = parse_timestamp(match.group("start"))
        end = parse_timestamp(match.group("end"))
        i += 1
        cue_lines: list[str] = []
        while i < n and lines[i].strip() != "":
            cleaned = _clean_cue_line(lines[i])
            if cleaned:
                cue_lines.append(cleaned)
            i += 1

        if not cue_lines:
            continue

        # YouTube auto-captions often repeat the previous line as line 1 of a 2-line cue
        combined = " ".join(cue_lines).strip()
        if not combined or _PURE_NOISE_RE.match(combined):
            continue

        raw_cues.append(TranscriptSegment(start=start, end=end, text=combined))

    # Deduplicate rolling overlaps and merge micro-cues into clean timed segments
    segments: list[TranscriptSegment] = []
    last_full_text = ""

    for cue in raw_cues:
        # Skip 10ms YouTube transition cues that only duplicate the previous cue
        if (cue.end - cue.start) < 0.05 and cue.text == last_full_text:
            continue

        novel = _deduplicate_rolling_text(last_full_text, cue.text)
        last_full_text = cue.text
        if not novel or _PURE_NOISE_RE.match(novel):
            continue

        if segments and novel.lower() == segments[-1].text.lower() and (cue.start - segments[-1].end) < 1.5:
            segments[-1].end = max(segments[-1].end, cue.end)
            continue

        segments.append(TranscriptSegment(start=cue.start, end=max(cue.end, cue.start + 0.1), text=novel))

    return segments


def parse_vtt_file(path: str | Path) -> list[TranscriptSegment]:
    """Read and parse a .vtt file from disk."""
    content = Path(path).read_text(encoding="utf-8", errors="replace")
    return parse_vtt_text(content)
