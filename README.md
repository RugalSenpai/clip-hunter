# Clip Hunter `v2.1.0`

> **Motive:** Turning 9–24 hour gaming streams into ready-to-post vertical clips shouldn't require downloading 10–30 GB of video or burning CPU transcribing 24 hours of dead air. **Clip Hunter** scans *text*, not media — using an LLM solely to pick viral timestamps & metadata, then downloading and rendering **only** the winning 20–60 second windows across `9:16`, `1:1`, and `16:9` with optional burned-in word-by-word subtitles.

---

## Core Idea

1. **Scan text, not video:** Fetch tiny `.vtt` auto-captions (or stream 10-minute audio-only chunks + local Whisper when captions don't exist).
2. **LLM picks moments only:** A local LLM (LM Studio, Ollama, llama.cpp) or cloud OpenAI-compatible API scores 10-minute blocks and returns strict JSON timestamps + hook, title, caption, and hashtags.
3. **Engine fetches & edits only the winners:** `yt-dlp --download-sections` pulls only the winning 20–60s windows, `faster-whisper` aligns exact word timestamps, OpenCV detects facecams, and `ffmpeg` exports `9:16`, `1:1`, and `16:9` with `loudnorm` and optional word-by-word subtitles.
4. **Concurrent 3-worker pipeline:** Transcript production, LLM judging, and clip rendering run in parallel so your first clip lands in ~1–2 minutes instead of waiting for the entire stream to finish.

---

## Features (`v2.1.0`)

- **Web Studio UI (`web_server.py` + `web_ui.html`):** Browser-based dark studio with byte-range MP4 streaming, LLM preset & connection tester, saved settings, run History (Load / Resume / Remove / Clear), and automatic fresh run folder creation.
- **Interactive Edit Clip Workspace:** Drag hook banner & subtitle Y-position, pick/resize facecam box, toggle facecam off for full-screen gameplay, pan gameplay horizontally, adjust trim sliders, edit transcript words or re-transcribe, and apply/re-render.
- **Extend Clip from Full VOD (`hls.js` + `/api/hls_proxy`):** Open an online HLS stream jumped directly to the clip's VOD timestamp, scrub earlier/later across the full stream with audio, set new Start/End points, and re-download at 1080p+ + re-transcribe + re-render with live per-clip progress.
- **Profile 1 & Profile 2 (`9:16` Split):** Profile 1 exports `9:16`, `1:1`, and `16:9`; Profile 2 overlays a 30% facecam at the top of 100% full-height horizontally panned gameplay (`1080×1920`).
- **Smarter Facecam Detection (`OpenCV` YuNet + Edge Clustering):** Excludes middle-of-screen game/loading-screen character faces, prefers stationary left/right streamer cams, and distinguishes boxed (`16:9`/overlay) cams from transparent/chroma-keyed cams.
- **1080p+ Video & Fast Audio Fetching:** Enforces `>=1080p` clip downloads with automatic client fallback (`web_safari,web`, `tv,web`), single-pass `yt-dlp -g` + `ffmpeg` 16kHz mono WAV extraction (~5s per 10-min chunk), and **YouTube Cookies** support (`Firefox`/`Chrome`/`Edge`/`Brave` or `cookies.txt`).
- **Live Per-Task Progress & Stop / Resume:** Real-time `0–100%` progress bars per stage (`Downloading`, `Transcribing`, `Rendering`) as soon as the LLM picks a winner, auto-play on completion, prompt subprocess cancellation on **Stop**, and **Resume** / **Start Again** controls.
- **GPU / Hybrid Acceleration & Multi-line Subtitles:** NVENC + CUDA acceleration with automatic CPU fallback and 2-line ASS word-highlight captions.
- **Mode A (VOD) & Mode B (Live Streams):** Supports both VOD blocks (`10m` default) and rolling 15-minute live buffers with optional Twitch IRC chat velocity.

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
