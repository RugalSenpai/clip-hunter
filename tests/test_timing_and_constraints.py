"""Tests for timing constraints, overlap removal, schema validation, and live windowing."""

from clipper.aligner import filter_and_deduplicate_candidates, snap_to_sentence_boundaries
from clipper.blocker import build_live_window, overlaps_existing, split_into_blocks
from clipper.judge import validate_candidates
from clipper.models import Candidate, TranscriptSegment


def test_filter_and_deduplicate_enforces_score_duration_and_overlaps():
    cands = [
        Candidate("00:01:00", "00:01:08", "funny", 9.0, "TOO SHORT", "Too short", "c", ["#a"], "why"),
        Candidate("00:02:00", "00:02:40", "freakout", 8.5, "BIG MOMENT", "Big moment", "c", ["#a"], "why"),
        Candidate("00:02:20", "00:03:00", "freakout", 7.8, "OVERLAP", "Overlap", "c", ["#a"], "why"),
        Candidate("00:05:00", "00:06:40", "discovery", 8.0, "LONG CLIP", "Long clip", "c", ["#a"], "why"),
        Candidate("00:08:00", "00:08:30", "fail", 6.5, "LOW SCORE", "Low score", "c", ["#a"], "why"),
    ]
    kept = filter_and_deduplicate_candidates(cands, min_score=7.0, max_clips=3)
    assert [c.title for c in kept] == ["Big moment", "Long clip"]
    assert kept[1].duration == 75.0  # capped at 75s


def test_snap_to_sentence_boundaries():
    segs = [
        TranscriptSegment(10.2, 14.0, "Wait what is that?!"),
        TranscriptSegment(14.1, 25.4, "No way that actually worked."),
        TranscriptSegment(25.8, 30.0, "Anyway moving on."),
    ]
    start, end = snap_to_sentence_boundaries(11.0, 24.8, segs)
    assert abs(start - 10.05) < 0.2
    assert abs(end - 25.65) < 0.2


def test_validate_candidates_normalizes_fields():
    raw = [
        {
            "start": "00:01:10",
            "end": "00:01:45",
            "category": "first_reaction",
            "score": 8.5,
            "hook": "SPEED SEES THE MAP",
            "title": "Speed reacts to the reveal",
            "caption": "He lost it. Would you play?",
            "hashtags": ["GTA6", "#IShowSpeed"],
            "why": "instant reaction",
        }
    ]
    valid = validate_candidates(raw, min_score=7.0)
    assert len(valid) == 1
    assert valid[0].hashtags == ["#GTA6", "#IShowSpeed"]


def test_live_window_and_overlap_guard():
    segs = [TranscriptSegment(i * 30.0, i * 30.0 + 10.0, f"line {i}") for i in range(20)]
    blocks = split_into_blocks(segs, block_duration_sec=300.0, overlap_sec=45.0)
    assert len(blocks) >= 2
    win = build_live_window(segs, window_end=360.0, window_duration_sec=300.0)
    assert win is not None
    assert win.start == 60.0
    assert overlaps_existing(100.0, 140.0, [(120.0, 160.0)]) is True
    assert overlaps_existing(100.0, 140.0, [(150.0, 190.0)]) is False
