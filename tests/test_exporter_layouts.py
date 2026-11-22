"""Tests for multi-ratio ffmpeg filtergraph generation and ASS subtitle generation."""

from pathlib import Path

from clipper.captions import generate_ass_subtitles
from clipper.exporter import build_video_filtergraph
from clipper.models import FacecamBox, WordTimestamp


def test_filtergraph_for_all_ratios_and_layouts(tmp_path: Path):
    ass = tmp_path / "test.ass"
    ass.write_text("", encoding="utf-8")

    cam_box = FacecamBox(x=1400, y=50, w=480, h=270, frame_w=1920, frame_h=1080, layout="facecam", confidence=0.9)
    fg_916, pad = build_video_filtergraph("9:16", cam_box, ass_path=ass)
    assert pad == "[vout]"
    assert "vstack=inputs=2" in fg_916
    assert "ass='" in fg_916

    # Subtitles disabled
    fg_no_sub, _ = build_video_filtergraph("9:16", cam_box, ass_path=None)
    assert "ass='" not in fg_no_sub

    # Gameplay blur-fill fallback
    gp_box = FacecamBox(0, 0, 0, 0, 1920, 1080, layout="gameplay", confidence=0.0)
    fg_gp, _ = build_video_filtergraph("9:16", gp_box, ass_path=None)
    assert "boxblur=20:5" in fg_gp

    # 1:1 and 16:9
    fg_11, _ = build_video_filtergraph("1:1", cam_box, ass_path=ass)
    assert "scale=1080:1080" in fg_11
    fg_169, _ = build_video_filtergraph("16:9", cam_box, ass_path=ass)
    assert "scale=1920:1080" in fg_169


def test_generate_ass_subtitles_creates_hook_and_word_events(tmp_path: Path):
    words = [
        WordTimestamp("No", 0.5, 0.8),
        WordTimestamp("way", 0.8, 1.1),
        WordTimestamp("bro!", 1.1, 1.5),
    ]
    out = generate_ass_subtitles(words, "INSANE REACTION", 20.0, "9:16", tmp_path / "clip.ass")
    text = out.read_text(encoding="utf-8")
    assert "PlayResX: 1080" in text
    assert "PlayResY: 1920" in text
    assert "INSANE REACTION" in text
    assert r"{\c&H0000FFFF&\fscx108\fscy108}WAY" in text
