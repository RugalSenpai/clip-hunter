"""CLI entrypoint for Mode A (VOD prepare/render/stream) and Mode B (Live)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .aligner import filter_and_deduplicate_candidates
from .blocker import split_into_blocks
from .fetcher import fetch_captions, get_video_info
from .judge import OpenAICompatibleJudge, SessionLLMJudge
from .live_recorder import run_live_clipper
from .models import VALID_RATIOS, format_timestamp
from .prefilter import shortlist_blocks
from .reporter import format_final_report, open_clips_folder, write_summary_json
from .vod_stream import render_single_candidate, run_streaming_vod
from .vtt_parser import parse_vtt_file


def _parse_ratios(raw: str) -> list[str]:
    ratios = [r.strip() for r in raw.split(",") if r.strip()]
    for r in ratios:
        if r not in VALID_RATIOS:
            raise argparse.ArgumentTypeError(f"Unsupported ratio {r!r}; choose from {VALID_RATIOS}")
    return ratios


def _cmd_vod_prepare(args: argparse.Namespace) -> int:
    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)

    info = get_video_info(args.url)
    vtt_path = Path(args.vtt) if args.vtt else fetch_captions(args.url, run_dir / "captions")
    if not vtt_path or not vtt_path.exists():
        print("Error: No auto-captions (.vtt) available. Use `vod stream` for audio-chunk Whisper mode.", file=sys.stderr)
        return 1

    segments = parse_vtt_file(vtt_path)
    blocks = split_into_blocks(segments, block_duration_sec=args.block_min * 60.0)
    shortlisted = shortlist_blocks(blocks, top_fraction=args.top_fraction, min_blocks=args.min_blocks)

    judge = SessionLLMJudge(run_dir)
    shortlist_file = judge.write_shortlist(shortlisted, metadata=info)
    print(
        f"[prepare] {info['title']} ({format_timestamp(info['duration'])})\n"
        f"  Parsed {len(segments)} segments -> {len(blocks)} blocks -> shortlisted {len(shortlisted)} blocks\n"
        f"  Shortlist written to: {shortlist_file}\n"
        f"  Next: judge shortlisted blocks -> write {run_dir / 'candidates.json'} -> run `vod render`."
    )
    return 0


def _cmd_vod_render(args: argparse.Namespace) -> int:
    t0 = time.monotonic()
    run_dir = Path(args.out)
    clips_dir = run_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    info = get_video_info(args.url)
    judge = SessionLLMJudge(run_dir)
    raw_cands = judge.load_candidates(args.candidates)
    kept = filter_and_deduplicate_candidates(
        raw_cands,
        min_score=args.min_score,
        max_clips=args.max_clips,
    )

    artifacts = []
    for idx, cand in enumerate(kept, start=1):
        print(f"[render] Clip {idx:02d}/{len(kept):02d}: \"{cand.title}\" [{cand.start}-{cand.end}]...", flush=True)
        art = render_single_candidate(
            url=args.url,
            candidate=cand,
            clip_index=idx,
            run_dir=run_dir,
            ratios=args.ratios,
            enable_subtitles=args.subtitles,
            whisper_model=args.whisper_model,
        )
        if art is not None:
            artifacts.append(art)

    elapsed = time.monotonic() - t0
    write_summary_json(clips_dir, info, len(raw_cands), artifacts, elapsed)
    print("\n" + format_final_report(info["duration"], len(raw_cands), artifacts, elapsed, clips_dir))
    if args.open_folder:
        open_clips_folder(clips_dir)
    return 0


def _cmd_vod_stream(args: argparse.Namespace) -> int:
    judge = OpenAICompatibleJudge(
        base_url=args.llm_base_url,
        model=args.llm_model,
        api_key=args.llm_api_key,
    )
    run_streaming_vod(
        url=args.url,
        run_dir=args.out,
        judge=judge,
        ratios=args.ratios,
        enable_subtitles=args.subtitles,
        block_duration_sec=args.block_min * 60.0,
        min_score=args.min_score,
        max_clips=args.max_clips,
        whisper_model=args.whisper_model,
        game_tag=args.game_tag,
        open_folder=args.open_folder,
    )
    return 0


def _cmd_live(args: argparse.Namespace) -> int:
    judge = OpenAICompatibleJudge(
        base_url=args.llm_base_url,
        model=args.llm_model,
        api_key=args.llm_api_key,
    )
    run_live_clipper(
        url=args.url,
        run_dir=args.out,
        judge=judge,
        buffer_min=args.buffer_min,
        ratios=args.ratios,
        enable_subtitles=args.subtitles,
        min_score=args.min_score,
        whisper_model=args.whisper_model,
        twitch_channel=args.twitch_channel,
        streamer=args.streamer,
        game_tag=args.game_tag,
    )
    return 0


def _add_shared_render_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ratios", type=_parse_ratios, default=list(VALID_RATIOS), help="Comma-separated ratios: 9:16,1:1,16:9")
    parser.add_argument("--subtitles", action=argparse.BooleanOptionalAction, default=True, help="Burn in hook + word-by-word subtitles")
    parser.add_argument("--whisper-model", default="small.en", help="faster-whisper model name")
    parser.add_argument("--open-folder", action="store_true", help="Open clips folder in Explorer when done")


def _add_llm_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--llm-base-url", default=None, help="OpenAI-compatible base URL (e.g. http://localhost:11434/v1 for Ollama)")
    parser.add_argument("--llm-model", default=None, help="LLM model name (e.g. qwen2.5:7b, gpt-4o-mini)")
    parser.add_argument("--llm-api-key", default=None, help="API key (optional for local LLMs)")
    parser.add_argument("--game-tag", default=None, help="Required game hashtag (e.g. #GTA6)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clipper", description="Viral-clipping pipeline for long gaming streams (VOD + live)")
    sub = parser.add_subparsers(dest="mode", required=True)

    # Mode A: VOD
    vod = sub.add_parser("vod", help="Mode A — Finished VOD clipping")
    vod_sub = vod.add_subparsers(dest="action", required=True)

    p_prep = vod_sub.add_parser("prepare", help="Fetch VTT, split into 10-min blocks, pre-filter -> shortlist.json")
    p_prep.add_argument("--url", required=True)
    p_prep.add_argument("--out", required=True, help="Run output directory")
    p_prep.add_argument("--vtt", default=None, help="Optional existing .vtt file on disk")
    p_prep.add_argument("--block-min", type=float, default=10.0)
    p_prep.add_argument("--top-fraction", type=float, default=0.20)
    p_prep.add_argument("--min-blocks", type=int, default=2)
    p_prep.set_defaults(func=_cmd_vod_prepare)

    p_ren = vod_sub.add_parser("render", help="Download winning sections from candidates.json and export multi-ratio clips")
    p_ren.add_argument("--url", required=True)
    p_ren.add_argument("--candidates", required=True, help="Path to candidates.json")
    p_ren.add_argument("--out", required=True, help="Run output directory")
    p_ren.add_argument("--min-score", type=float, default=7.0)
    p_ren.add_argument("--max-clips", type=int, default=3)
    _add_shared_render_flags(p_ren)
    p_ren.set_defaults(func=_cmd_vod_render)

    p_stream = vod_sub.add_parser("stream", help="Concurrent 3-worker streaming VOD pipeline (10m blocks -> LLM -> immediate clip render)")
    p_stream.add_argument("--url", required=True)
    p_stream.add_argument("--out", required=True, help="Run output directory")
    p_stream.add_argument("--block-min", type=float, default=10.0)
    p_stream.add_argument("--min-score", type=float, default=7.0)
    p_stream.add_argument("--max-clips", type=int, default=10)
    _add_shared_render_flags(p_stream)
    _add_llm_flags(p_stream)
    p_stream.set_defaults(func=_cmd_vod_stream)

    # Mode B: Live
    live = sub.add_parser("live", help="Mode B — Live stream rolling 15-min buffer clipper")
    live.add_argument("--url", required=True)
    live.add_argument("--out", required=True, help="Run output directory")
    live.add_argument("--buffer-min", type=int, default=15)
    live.add_argument("--min-score", type=float, default=7.5)
    live.add_argument("--twitch-channel", default=None, help="Optional Twitch channel for live chat velocity signal")
    live.add_argument("--streamer", default="Streamer")
    _add_shared_render_flags(live)
    _add_llm_flags(live)
    live.set_defaults(func=_cmd_live)

    # Desktop Studio GUI
    gui_p = sub.add_parser("gui", help="Launch the Clipper Studio desktop GUI")
    gui_p.set_defaults(func=lambda _args: _cmd_gui())

    return parser


def _cmd_gui() -> int:
    from .gui import launch_gui

    return launch_gui()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
