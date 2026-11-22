"""Component 6: Whisper word-timestamp aligner + clip boundary & overlap constraints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import Candidate, TranscriptSegment, WordTimestamp

_SENTENCE_END_CHARS = (".", "!", "?")

_WHISPER_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}


def get_whisper_model(
    model_size: str = "small.en",
    device: str = "cpu",
    compute_type: str = "int8",
) -> Any:
    """Lazy-load and cache a faster-whisper WhisperModel instance."""
    key = (model_size, device, compute_type)
    if key not in _WHISPER_MODEL_CACHE:
        from faster_whisper import WhisperModel

        _WHISPER_MODEL_CACHE[key] = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
    return _WHISPER_MODEL_CACHE[key]


def transcribe_media(
    media_path: str | Path,
    model_size: str = "small.en",
    device: str = "cpu",
    compute_type: str = "int8",
    time_offset: float = 0.0,
) -> tuple[list[TranscriptSegment], list[WordTimestamp]]:
    """Run faster-whisper with word_timestamps=True on a short clip or audio chunk."""
    model = get_whisper_model(model_size=model_size, device=device, compute_type=compute_type)
    raw_segments, _info = model.transcribe(
        str(media_path),
        beam_size=1,
        word_timestamps=True,
        vad_filter=True,
    )

    segments: list[TranscriptSegment] = []
    words: list[WordTimestamp] = []
    for seg in raw_segments:
        text = seg.text.strip()
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                start=float(seg.start) + time_offset,
                end=float(seg.end) + time_offset,
                text=text,
            )
        )
        if getattr(seg, "words", None):
            for w in seg.words:
                w_text = w.word.strip()
                if w_text:
                    words.append(
                        WordTimestamp(
                            word=w_text,
                            start=float(w.start) + time_offset,
                            end=float(w.end) + time_offset,
                        )
                    )
    return segments, words


def snap_to_sentence_boundaries(
    target_start: float,
    target_end: float,
    segments: list[TranscriptSegment],
    max_snap_sec: float = 3.5,
    min_duration_sec: float = 12.0,
    max_duration_sec: float = 75.0,
) -> tuple[float, float]:
    """Snap cut points to the nearest transcript sentence boundaries within bounds."""
    snapped_start = max(0.0, target_start)
    snapped_end = max(snapped_start + 1.0, target_end)

    if segments:
        # Snap start to nearest segment start within max_snap_sec
        best_start = min(segments, key=lambda s: abs(s.start - target_start))
        if abs(best_start.start - target_start) <= max_snap_sec:
            snapped_start = max(0.0, best_start.start - 0.15)

        # Prefer snapping end to a segment ending with punctuation, else nearest segment end
        sentence_ends = [
            s for s in segments if s.text.rstrip().endswith(_SENTENCE_END_CHARS)
        ]
        pool = sentence_ends or segments
        best_end = min(pool, key=lambda s: abs(s.end - target_end))
        if abs(best_end.end - target_end) <= max_snap_sec:
            snapped_end = best_end.end + 0.25

    duration = snapped_end - snapped_start
    if duration > max_duration_sec:
        snapped_end = snapped_start + max_duration_sec
    elif duration < min_duration_sec and segments:
        # Try extending end slightly if transcript continues
        max_seg_end = max(s.end for s in segments)
        snapped_end = min(max_seg_end, snapped_start + min_duration_sec)

    return round(snapped_start, 3), round(snapped_end, 3)


def filter_and_deduplicate_candidates(
    candidates: list[Candidate],
    min_score: float = 7.0,
    max_clips: int = 3,
    min_duration_sec: float = 12.0,
    max_duration_sec: float = 75.0,
) -> list[Candidate]:
    """Enforce Section 6 rendering constraints: score bar, 12-75s bounds, no overlaps, top K."""
    eligible: list[Candidate] = []
    for c in candidates:
        if c.score < min_score:
            continue
        dur = c.duration
        if dur < min_duration_sec:
            continue
        if dur > max_duration_sec:
            # Cap at 75s
            from .models import format_timestamp

            c = Candidate(
                start=c.start,
                end=format_timestamp(c.start_sec + max_duration_sec),
                category=c.category,
                score=c.score,
                hook=c.hook,
                title=c.title,
                caption=c.caption,
                hashtags=list(c.hashtags),
                why=c.why,
                layout_override=c.layout_override,
            )
        eligible.append(c)

    # Sort highest score first, resolve overlaps greedily
    eligible.sort(key=lambda c: (-c.score, c.start_sec))
    kept: list[Candidate] = []
    for cand in eligible:
        has_overlap = any(
            min(cand.end_sec, k.end_sec) - max(cand.start_sec, k.start_sec) > 2.0
            for k in kept
        )
        if has_overlap:
            continue
        kept.append(cand)
        if len(kept) >= max_clips:
            break

    # Return in chronological order
    kept.sort(key=lambda c: c.start_sec)
    return kept


def rebase_words_for_clip(
    words: list[WordTimestamp],
    clip_start: float,
    clip_end: float,
) -> list[WordTimestamp]:
    """Filter words to [clip_start, clip_end] and shift timestamps so clip_start == 0.0."""
    rebased: list[WordTimestamp] = []
    for w in words:
        if w.end <= clip_start or w.start >= clip_end:
            continue
        rebased.append(
            WordTimestamp(
                word=w.word,
                start=max(0.0, w.start - clip_start),
                end=max(0.05, min(clip_end, w.end) - clip_start),
            )
        )
    return rebased
