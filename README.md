# Clip Hunter `v1.00`

> **Motive:** Turning 9–24 hour gaming streams into ready-to-post vertical clips shouldn't require downloading 10–30 GB of video or burning CPU transcribing 24 hours of dead air. **Clip Hunter** scans *text*, not media — using an LLM solely to pick viral timestamps & metadata, then downloading and rendering **only** the winning 20–60 second windows across `9:16`, `1:1`, and `16:9` with optional burned-in word-by-word subtitles.

---

## Core Idea

1. **Scan text, not video:** Fetch tiny `.vtt` auto-captions (or stream 10-minute audio-only chunks + local Whisper when captions don't exist).
2. **LLM picks moments only:** A local LLM (LM Studio, Ollama, llama.cpp) or cloud OpenAI-compatible API scores 10-minute blocks and returns strict JSON timestamps + hook, title, caption, and hashtags.
3. **Engine fetches & edits only the winners:** `yt-dlp --download-sections` pulls only the winning 20–60s windows, `faster-whisper` aligns exact word timestamps, OpenCV detects facecams, and `ffmpeg` exports `9:16`, `1:1`, and `16:9` with `loudnorm` and optional word-by-word subtitles.
4. **Concurrent 3-worker pipeline:** Transcript production, LLM judging, and clip rendering run in parallel so your first clip lands in ~1–2 minutes instead of waiting for the entire stream to finish.

---

## Features (`v1.00`)

- **Clipper Studio Desktop GUI (`PySide6`):** Dark-mode studio UI with LLM preset & connection tester, saved API settings, live 3-stage pipeline tracker, built-in multi-ratio video player, metadata copy, and graceful **Stop** (preserves all completed clips).
- **Mode A — VOD (`vod stream` / `vod prepare` + `vod render`):** Supports both concurrent local/cloud LLM streaming and free zero-API-key session-LLM judging.
- **Mode B — Live Streams (`live`):** Rolling 15-minute buffer of 1-minute `.ts` segments, per-minute local Whisper + audio loudness (+ optional Twitch IRC chat velocity), 5-minute overlapping windows, and immediate buffer cutting.
- **Multi-Ratio Export:** Renders `9:16` (`1080×1920`), `1:1` (`1080×1080`), and `16:9` (`1920×1080`) with ratio-aware safe-zone captions, facecam split-screen or blur-fill fallback, `.txt` post metadata, and `summary.json`.
- **Optional Subtitles Toggle:** Enable or disable burned-in hook titles and word-by-word highlighted captions (`--subtitles` / `--no-subtitles`).

---

## Requirements

- Python 3.10+
- `ffmpeg` and `ffprobe` on `PATH`
- Python packages:
  ```powershell
  pip install -r requirements.txt
  ```

---

## Quickstart

### Launch the Desktop GUI
```powershell
python -m clipper gui
```
Or double-click `launch_gui.bat`.

### Mode A — Concurrent Streaming VOD (CLI)
```powershell
python -m clipper vod stream --url "https://www.youtube.com/watch?v=<VIDEO_ID>" --out runs/stream_test --ratios 9:16,1:1,16:9 --subtitles --llm-base-url http://localhost:1234/v1 --llm-model <MODEL>
```

### Mode A — Two-Step Free Session-LLM Workflow (No API Key)
```powershell
# 1. Fetch VTT, split into 10-min blocks, and shortlist top arousal blocks
python -m clipper vod prepare --url "https://www.youtube.com/watch?v=<VIDEO_ID>" --out runs/test_run

# 2. Judge runs/test_run/shortlist.json -> save runs/test_run/candidates.json
# 3. Download only the winning windows and export all ratios
python -m clipper vod render --url "https://www.youtube.com/watch?v=<VIDEO_ID>" --candidates runs/test_run/candidates.json --out runs/test_run --ratios 9:16,1:1,16:9 --subtitles --open-folder
```

### Mode B — Live Stream Clipper (Rolling 15-Min Buffer)
```powershell
python -m clipper live --url "https://www.twitch.tv/<CHANNEL>" --out runs/live_test --buffer-min 15 --ratios 9:16,1:1,16:9 --subtitles --twitch-channel <CHANNEL> --llm-base-url http://localhost:11434/v1 --llm-model qwen2.5:7b
```

---

## Running Tests

```powershell
python -m pytest tests/
```
