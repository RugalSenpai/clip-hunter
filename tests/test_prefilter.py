"""Tests for the free heuristic arousal pre-filter."""

from clipper.models import TranscriptBlock, TranscriptSegment
from clipper.prefilter import score_block, shortlist_blocks


def test_prefilter_ranks_hype_block_above_filler_block():
    calm = TranscriptBlock(
        index=0,
        start=0.0,
        end=600.0,
        segments=[
            TranscriptSegment(10.0, 20.0, "Thank you for the sub, use code gamer in the description."),
            TranscriptSegment(30.0, 40.0, "Okay so anyway let us walk over here slowly."),
        ],
    )
    hype = TranscriptBlock(
        index=1,
        start=600.0,
        end=1200.0,
        segments=[
            TranscriptSegment(610.0, 612.0, "NO WAY! WHAT THE FUCK JUST HAPPENED?!"),
            TranscriptSegment(612.0, 614.0, "LET'S GO! OH MY GOD I CAN'T! [Laughter]"),
        ],
    )
    assert score_block(hype) > score_block(calm) * 3

    shortlisted = shortlist_blocks([calm, hype], top_fraction=0.2, min_blocks=1)
    assert [b.index for b in shortlisted] == [1]
