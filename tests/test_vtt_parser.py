"""Tests for YouTube WebVTT parsing and rolling-cue deduplication."""

from clipper.vtt_parser import parse_vtt_text

SAMPLE_YOUTUBE_VTT = """WEBVTT
Kind: captions
Language: en

00:00:01.000 --> 00:00:03.500 align:start position:0%
no<00:00:01.200><c> way</c><00:00:01.500><c> bro</c>

00:00:03.500 --> 00:00:03.510 align:start position:0%
no way bro
 

00:00:03.510 --> 00:00:06.000 align:start position:0%
no way bro
what<00:00:04.000><c> just</c><00:00:04.500><c> happened</c>

00:00:06.100 --> 00:00:08.000
[Music]

00:00:08.500 --> 00:00:10.500
LET&apos;S GO! [Laughter]
"""


def test_parse_vtt_strips_tags_and_deduplicates_rolling_lines():
    segs = parse_vtt_text(SAMPLE_YOUTUBE_VTT)
    texts = [s.text for s in segs]
    assert texts == [
        "no way bro",
        "what just happened",
        "LET'S GO! [Laughter]",
    ]
    assert segs[0].start == 1.0
    assert segs[1].start == 3.51
    assert segs[2].start == 8.5
