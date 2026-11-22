"""Component 2: Block splitter for VOD (10-min blocks) and Live (5-min overlapping windows)."""

from __future__ import annotations

from .models import TranscriptBlock, TranscriptSegment


def split_into_blocks(
    segments: list[TranscriptSegment],
    block_duration_sec: float = 600.0,
    overlap_sec: float = 0.0,
) -> list[TranscriptBlock]:
    """Split timed transcript segments into fixed-duration blocks with optional overlap."""
    if not segments:
        return []

    step = block_duration_sec - overlap_sec
    if step <= 0:
        raise ValueError("block_duration_sec must be greater than overlap_sec")

    min_t = max(0.0, segments[0].start)
    max_t = max(seg.end for seg in segments)
    # Align VOD blocks cleanly to 0, 600, 1200...
    window_start = 0.0 if overlap_sec == 0.0 else min_t

    blocks: list[TranscriptBlock] = []
    idx = 0
    while window_start < max_t or (idx == 0 and max_t == 0.0):
        window_end = window_start + block_duration_sec
        window_segs = [
            seg for seg in segments if seg.end > window_start and seg.start < window_end
        ]
        if window_segs:
            blocks.append(
                TranscriptBlock(
                    index=idx,
                    start=window_start,
                    end=min(window_end, max_t),
                    segments=window_segs,
                )
            )
            idx += 1
        window_start += step
        if window_end >= max_t:
            break

    return blocks


def build_live_window(
    segments: list[TranscriptSegment],
    window_end: float,
    window_duration_sec: float = 300.0,
    block_index: int = 0,
    signals: dict | None = None,
) -> TranscriptBlock | None:
    """Build a single 5-minute rolling live window ending at window_end."""
    window_start = max(0.0, window_end - window_duration_sec)
    window_segs = [
        seg for seg in segments if seg.end > window_start and seg.start <= window_end
    ]
    if not window_segs:
        return None
    return TranscriptBlock(
        index=block_index,
        start=window_start,
        end=window_end,
        segments=window_segs,
        signals=signals or {},
    )


def overlaps_existing(
    start_sec: float,
    end_sec: float,
    picked_intervals: list[tuple[float, float]],
    min_overlap_sec: float = 5.0,
) -> bool:
    """Return True if [start_sec, end_sec] overlaps any previously picked interval."""
    for p_start, p_end in picked_intervals:
        overlap = min(end_sec, p_end) - max(start_sec, p_start)
        if overlap >= min_overlap_sec:
            return True
    return False
