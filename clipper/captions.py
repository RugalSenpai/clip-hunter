"""Component 8: Ratio-aware ASS subtitle generator for hook title + word-by-word captions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import RATIO_DIMENSIONS, WordTimestamp


@dataclass(frozen=True)
class RatioCaptionStyle:
    play_res_x: int
    play_res_y: int
    hook_font_size: int
    hook_margin_v: int
    word_font_size: int
    word_margin_v: int
    words_per_chunk: int


RATIO_STYLES: dict[str, RatioCaptionStyle] = {
    "9:16": RatioCaptionStyle(
        play_res_x=1080,
        play_res_y=1920,
        hook_font_size=62,
        hook_margin_v=145,  # Top safe zone above Shorts/Reels UI
        word_font_size=76,
        word_margin_v=430,  # Lower-middle safe zone above Reels/Shorts buttons
        words_per_chunk=3,
    ),
    "1:1": RatioCaptionStyle(
        play_res_x=1080,
        play_res_y=1080,
        hook_font_size=50,
        hook_margin_v=55,
        word_font_size=62,
        word_margin_v=95,
        words_per_chunk=4,
    ),
    "16:9": RatioCaptionStyle(
        play_res_x=1920,
        play_res_y=1080,
        hook_font_size=54,
        hook_margin_v=45,
        word_font_size=64,
        word_margin_v=85,
        words_per_chunk=5,
    ),
}


def _ass_time(seconds: float) -> str:
    """Format float seconds as ASS H:MM:SS.cc timestamp."""
    cs_total = max(0, int(round(seconds * 100)))
    cs = cs_total % 100
    sec_total = cs_total // 100
    h = sec_total // 3600
    m = (sec_total % 3600) // 60
    s = sec_total % 60
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _escape_ass(text: str) -> str:
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}").replace("\n", r"\N")


def _chunk_words(words: list[WordTimestamp], chunk_size: int) -> list[list[WordTimestamp]]:
    """Group words into short display chunks, breaking early on sentence punctuation or long pauses."""
    chunks: list[list[WordTimestamp]] = []
    current: list[WordTimestamp] = []
    for w in words:
        if current:
            gap = w.start - current[-1].end
            prev_ends_sentence = current[-1].word.endswith((".", "!", "?"))
            if len(current) >= chunk_size or gap > 0.65 or prev_ends_sentence:
                chunks.append(current)
                current = []
        current.append(w)
    if current:
        chunks.append(current)
    return chunks


def _hex_to_ass_bgr(hex_color: str) -> str:
    """Convert #RRGGBB hex string into ASS &H00BBGGRR& color code."""
    h = hex_color.strip().lstrip("#")
    if len(h) != 6:
        return "&H0000FFFF&"
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H00{b.upper()}{g.upper()}{r.upper()}&"


def generate_ass_subtitles(
    words: list[WordTimestamp],
    hook_text: str,
    clip_duration: float,
    ratio: str,
    out_path: str | Path,
    hook_duration_sec: float = 4.5,
    include_hook: bool = True,
    hook_y_pct: float | None = None,
    sub_y_pct: float | None = None,
    highlight_color: str = "#FFFF00",
) -> Path:
    """Generate a ratio-aware .ass subtitle file with hook banner + word-by-word active highlight."""
    if ratio not in RATIO_STYLES:
        raise ValueError(f"Unsupported ratio: {ratio}")
    style = RATIO_STYLES[ratio]
    res_x, res_y = RATIO_DIMENSIONS[ratio]
    hook_mv = int(max(10, min(res_y - 80, (hook_y_pct / 100.0) * res_y))) if hook_y_pct is not None else style.hook_margin_v
    if sub_y_pct is not None:
        word_align = 8  # Top-center so MarginV matches CSS top: %
        word_mv = int(max(10, min(res_y - 90, (sub_y_pct / 100.0) * res_y)))
    else:
        word_align = 2
        word_mv = style.word_margin_v
    hl_ass = _hex_to_ass_bgr(highlight_color)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {res_x}
PlayResY: {res_y}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Hook,Arial Black,{style.hook_font_size},&H00FFFFFF,&H0000FFFF,&H00000000,&HB0000000,1,0,0,0,100,100,0,0,3,6,0,8,60,60,{hook_mv},1
Style: Caption,Arial Black,{style.word_font_size},&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,7,2,{word_align},60,60,{word_mv},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []

    # Top hook title banner (optional, first ~4.5 seconds to grab viewers)
    clean_hook = _escape_ass(hook_text.strip().upper())
    if include_hook and clean_hook:
        h_end = min(clip_duration, max(2.5, hook_duration_sec))
        events.append(f"Dialogue: 1,{_ass_time(0.0)},{_ass_time(h_end)},Hook,,0,0,0,,{clean_hook}")

    # Word-by-word highlighted chunks (active word in bright yellow &H0000FFFF&)
    chunks = _chunk_words(words, style.words_per_chunk)
    for chunk in chunks:
        for active_idx, active_word in enumerate(chunk):
            w_start = active_word.start
            w_end = (
                chunk[active_idx + 1].start
                if active_idx + 1 < len(chunk)
                else max(active_word.end, active_word.start + 0.12)
            )
            w_end = min(clip_duration, max(w_end, w_start + 0.08))
            parts: list[str] = []
            for idx, w in enumerate(chunk):
                escaped = _escape_ass(w.word.upper())
                if idx == active_idx:
                    parts.append(r"{\c" + hl_ass + r"\fscx108\fscy108}" + escaped + r"{\c&H00FFFFFF&\fscx100\fscy100}")
                else:
                    parts.append(escaped)
            line_text = " ".join(parts)
            events.append(
                f"Dialogue: 0,{_ass_time(w_start)},{_ass_time(w_end)},Caption,,0,0,0,,{line_text}"
            )

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return target
