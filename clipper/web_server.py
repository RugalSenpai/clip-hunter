"""Local HTTP API + byte-range video streaming server backing the Clip Hunter HTML UI."""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import threading
import time
import urllib.parse
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .aligner import rebase_words_for_clip, transcribe_media
from .exporter import export_multi_ratio, probe_media
from .judge import OpenAICompatibleJudge, list_available_models, test_llm_connection
from .live_recorder import run_live_clipper
from .models import Candidate, ClipArtifact, FacecamBox, WordTimestamp, parse_timestamp
from .reporter import open_clips_folder, write_clip_metadata_files
from .vod_stream import run_streaming_vod

SETTINGS_FILE = Path.home() / ".clipper_studio_settings.json"
LOGO_PATH = Path(__file__).parent / "assets" / "clip_hunter_logo.png"
HTML_PATH = Path(__file__).parent / "web_ui.html"


class _LineBufferStream(io.TextIOBase):
    """Capture stdout/stderr lines into StudioState while echoing to original stream."""

    def __init__(self, on_line, original) -> None:
        super().__init__()
        self._on_line = on_line
        self._original = original
        self._buf = ""

    def write(self, s: str) -> int:
        if self._original:
            try:
                self._original.write(s)
            except Exception:
                pass
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._on_line(line)
        return len(s)

    def flush(self) -> None:
        if self._buf.strip():
            self._on_line(self._buf.strip())
            self._buf = ""
        if self._original:
            try:
                self._original.flush()
            except Exception:
                pass


class StudioState:
    """Thread-safe controller bridging the web UI and the clipping engine."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.status = "idle"  # idle | running | stopping | done | error
        self.stage = 1
        self.blocks_count = 0
        self.winners_count = 0
        self.artifacts: list[dict[str, Any]] = []
        self.logs: list[str] = []
        self.active_run_dir = f"runs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.stop_event: threading.Event | None = None
        self.worker: threading.Thread | None = None
        self.start_time = 0.0
        self._autoload_latest()

    def _autoload_latest(self) -> None:
        default_summary = Path("runs/test_15m/clips/summary.json")
        if default_summary.exists():
            self.load_summary(default_summary, update_run_dir=False)

    def load_settings(self) -> dict[str, Any]:
        defaults = {
            "mode_index": 0,
            "preset": "Ollama (Local)",
            "base_url": "http://localhost:11434/v1",
            "model": "qwen2.5:7b",
            "api_key": "",
            "ratio_916": True,
            "ratio_11": True,
            "ratio_169": True,
            "subtitles": True,
            "enable_hook": True,
            "min_score": 7.0,
            "max_clips": 10,
            "block_min": 10,
            "whisper_model": "small.en",
            "compute_mode": "auto",
            "buffer_min": 15,
            "profile": "profile1",
        }
        if SETTINGS_FILE.exists():
            try:
                loaded = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
                defaults.update(loaded)
            except Exception:
                pass
        return defaults

    def save_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        current = self.load_settings()
        current.update(payload)
        SETTINGS_FILE.write_text(json.dumps(current, indent=2), encoding="utf-8")
        return current

    def _append_log(self, line: str) -> None:
        with self.lock:
            ts = datetime.now().strftime("%H:%M:%S")
            self.logs.append(f"{ts}  {line}")
            if len(self.logs) > 600:
                self.logs = self.logs[-600:]
            if "[producer]" in line:
                self.stage = 1
            elif "[judge] Evaluating" in line:
                self.blocks_count += 1
                self.stage = 2
            elif "[judge] Winner" in line:
                self.winners_count += 1
            elif "[render]" in line or "[clip ready]" in line:
                self.stage = 3

    def _artifact_to_json(self, art: ClipArtifact) -> dict[str, Any]:
        d = art.to_dict()
        d["ratios"] = list(art.files.keys())
        d["status"] = "completed"
        d["progress"] = 100
        d["status_text"] = "Ready"
        return d

    def load_summary(self, summary_path: Path, update_run_dir: bool = True) -> bool:
        if not summary_path.exists():
            return False
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        clips = []
        for item in data.get("clips", []):
            item_copy = dict(item)
            item_copy["ratios"] = list((item.get("files") or {}).keys())
            item_copy["status"] = "completed"
            item_copy["progress"] = 100
            item_copy["status_text"] = "Ready"
            clips.append(item_copy)
        with self.lock:
            self.artifacts = clips
            self.winners_count = int(data.get("moments_found", len(clips)))
            if update_run_dir:
                self.active_run_dir = str(summary_path.parent.parent)
            if self.status == "idle" and clips:
                self.stage = 3
        return True

    def list_history(self) -> list[dict[str, Any]]:
        runs_root = Path("runs")
        if not runs_root.exists():
            return []
        entries: list[tuple[float, dict[str, Any]]] = []
        for sub in runs_root.iterdir():
            if not sub.is_dir():
                continue
            state_file = sub / "state.json"
            summary_file = sub / "clips" / "summary.json"
            cands_file = sub / "candidates.json"
            if not (state_file.exists() or summary_file.exists() or cands_file.exists()):
                continue
            meta: dict[str, Any] = {
                "run_dir": str(sub),
                "name": sub.name,
                "title": sub.name,
                "status": "INCOMPLETE",
                "winners_count": 0,
                "rendered_count": 0,
                "last_judged_block": -1,
                "total_blocks": 0,
                "url": "",
            }
            if state_file.exists():
                try:
                    st = json.loads(state_file.read_text(encoding="utf-8"))
                    meta.update(st)
                    raw_status = str(st.get("status", "running")).lower()
                    meta["status"] = "CRASHED" if raw_status == "running" else raw_status.upper()
                except Exception:
                    pass
            elif summary_file.exists():
                try:
                    sm = json.loads(summary_file.read_text(encoding="utf-8"))
                    stream = sm.get("stream", {})
                    meta["title"] = stream.get("title") or sub.name
                    meta["url"] = stream.get("webpage_url") or stream.get("url") or ""
                    meta["winners_count"] = sm.get("moments_found", len(sm.get("clips", [])))
                    meta["rendered_count"] = len(sm.get("clips", []))
                    meta["status"] = "COMPLETED" if meta["rendered_count"] > 0 else "INCOMPLETE"
                except Exception:
                    pass
            if cands_file.exists() and not meta.get("winners_count"):
                try:
                    meta["winners_count"] = len(json.loads(cands_file.read_text(encoding="utf-8")))
                except Exception:
                    pass
            mtime = sub.stat().st_mtime
            if not meta.get("updated_at"):
                meta["updated_at"] = datetime.fromtimestamp(mtime).strftime("%b %d, %H:%M")
            entries.append((mtime, meta))
        entries.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in entries]

    @staticmethod
    def _fresh_run_dir() -> Path:
        base = f"runs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cand = Path(base)
        idx = 1
        while cand.exists():
            cand = Path(f"{base}_{idx}")
            idx += 1
        return cand

    def reset_for_new_run(self) -> str:
        with self.lock:
            if self.worker and self.worker.is_alive():
                return self.active_run_dir
            self.active_run_dir = str(self._fresh_run_dir())
            self.status = "idle"
            self.stage = 1
            self.blocks_count = 0
            self.winners_count = 0
            self.artifacts = []
            self.logs = []
            return self.active_run_dir

    def start_pipeline(self, cfg: dict[str, Any]) -> None:
        is_resume = bool(cfg.get("resume"))
        force_new = bool(cfg.get("force_new"))
        old_worker = None
        with self.lock:
            if self.worker and self.worker.is_alive():
                if self.status == "running":
                    raise RuntimeError("Pipeline is already running.")
                old_worker = self.worker
        if old_worker is not None:
            old_worker.join(timeout=2.0)

        with self.lock:
            url = str(cfg.get("url") or "").strip()
            if not url:
                raise ValueError("Stream / Video URL is required.")
            ratios = cfg.get("ratios") or ["9:16", "1:1", "16:9"]
            raw_out = str(cfg.get("out_dir") or "").strip()
            run_dir = self._fresh_run_dir() if force_new else (Path(raw_out) if raw_out else self._fresh_run_dir())
            # Never silently reuse a completed/existing run folder when starting a new run
            if not is_resume and run_dir.exists() and (
                (run_dir / "state.json").exists() or (run_dir / "clips" / "summary.json").exists()
            ):
                run_dir = self._fresh_run_dir()

            self.active_run_dir = str(run_dir)
            self.status = "running"
            self.stage = 1
            self.blocks_count = 0
            self.winners_count = 0
            self.artifacts = []
            self.logs = []
            self.stop_event = threading.Event()
            self.start_time = time.monotonic()

        # Preload existing clips only when resuming an existing run_dir
        if is_resume:
            summary_file = run_dir / "clips" / "summary.json"
            if summary_file.exists():
                self.load_summary(summary_file, update_run_dir=False)
            state_file = run_dir / "state.json"
            if state_file.exists():
                try:
                    st = json.loads(state_file.read_text(encoding="utf-8"))
                    with self.lock:
                        self.blocks_count = max(0, int(st.get("last_judged_block", -1)) + 1)
                        self.winners_count = int(st.get("winners_count", len(self.artifacts)))
                except Exception:
                    pass

        settings = self.save_settings(
            {
                "mode_index": 1 if cfg.get("mode") == "live" else 0,
                "ratio_916": "9:16" in ratios,
                "ratio_11": "1:1" in ratios,
                "ratio_169": "16:9" in ratios,
                "subtitles": bool(cfg.get("enable_subtitles", True)),
                "enable_hook": bool(cfg.get("enable_hook", True)),
                "min_score": float(cfg.get("min_score", 7.0)),
                "max_clips": int(cfg.get("max_clips", 10)),
                "block_min": int(cfg.get("block_min", 10)),
                "whisper_model": str(cfg.get("whisper_model", "small.en")),
                "compute_mode": str(cfg.get("compute_mode", "auto")),
                "buffer_min": int(cfg.get("buffer_min", 15)),
                "profile": str(cfg.get("profile", "profile1")),
                "cookies_browser": str(cfg.get("cookies_browser", "")),
                "cookies_file": str(cfg.get("cookies_file", "")),
            }
        )

        from .fetcher import set_active_stop_event, set_ytdlp_cookies

        set_ytdlp_cookies(
            browser=str(settings.get("cookies_browser") or ""),
            cookies_file=str(settings.get("cookies_file") or ""),
        )
        set_active_stop_event(self.stop_event)

        judge = OpenAICompatibleJudge(
            base_url=str(settings.get("base_url") or "http://localhost:11434/v1"),
            model=str(settings.get("model") or "qwen2.5:7b"),
            api_key=str(settings.get("api_key") or "") or "local",
        )

        def _on_clip(art: ClipArtifact) -> None:
            item = self._artifact_to_json(art)
            with self.lock:
                # Replace if same index exists, else append
                self.artifacts = [a for a in self.artifacts if a.get("index") != item["index"]]
                self.artifacts.append(item)
                self.artifacts.sort(key=lambda a: a.get("index", 0))

        def _on_clip_progress(item: dict[str, Any]) -> None:
            idx = int(item.get("index", 0))
            with self.lock:
                self.artifacts = [a for a in self.artifacts if int(a.get("index", 0)) != idx]
                if not item.get("remove"):
                    self.artifacts.append(item)
                    self.artifacts.sort(key=lambda a: int(a.get("index", 0)))

        my_stop = self.stop_event

        def _run() -> None:
            orig_out, orig_err = sys.stdout, sys.stderr
            stream = _LineBufferStream(self._append_log, orig_out)
            sys.stdout = stream
            sys.stderr = stream
            try:
                if cfg.get("mode") == "live":
                    run_live_clipper(
                        url=url,
                        run_dir=run_dir,
                        judge=judge,
                        buffer_min=int(cfg.get("buffer_min", 15)),
                        ratios=ratios,
                        enable_subtitles=bool(cfg.get("enable_subtitles", True)),
                        min_score=float(cfg.get("min_score", 7.5)),
                        whisper_model=str(cfg.get("whisper_model", "small.en")),
                        twitch_channel=(str(cfg.get("twitch_channel") or "").strip() or None),
                        stop_event=my_stop,
                        on_clip_ready=_on_clip,
                        compute_mode=str(cfg.get("compute_mode", "auto")),
                        enable_hook=bool(cfg.get("enable_hook", True)),
                        profile=str(cfg.get("profile", "profile1")),
                    )
                else:
                    run_streaming_vod(
                        url=url,
                        run_dir=run_dir,
                        judge=judge,
                        ratios=ratios,
                        enable_subtitles=bool(cfg.get("enable_subtitles", True)),
                        block_duration_sec=float(cfg.get("block_min", 10)) * 60.0,
                        min_score=float(cfg.get("min_score", 7.0)),
                        max_clips=int(cfg.get("max_clips", 10)),
                        whisper_model=str(cfg.get("whisper_model", "small.en")),
                        stop_event=my_stop,
                        on_clip_ready=_on_clip,
                        on_clip_progress=_on_clip_progress,
                        compute_mode=str(cfg.get("compute_mode", "auto")),
                        enable_hook=bool(cfg.get("enable_hook", True)),
                        profile=str(cfg.get("profile", "profile1")),
                    )
                with self.lock:
                    if my_stop is self.stop_event:
                        self.status = "stopped" if my_stop.is_set() else "done"
                        if not my_stop.is_set():
                            self.stage = 4
            except Exception as exc:
                with self.lock:
                    if my_stop is self.stop_event:
                        if my_stop.is_set():
                            self.status = "stopped"
                        else:
                            self._append_log(f"[ERROR] {exc}")
                            self.status = "error"
            finally:
                if my_stop is self.stop_event:
                    set_active_stop_event(None)
                stream.flush()
                sys.stdout = orig_out
                sys.stderr = orig_err

        self.worker = threading.Thread(target=_run, daemon=True)
        self.worker.start()

    def stop_pipeline(self) -> None:
        with self.lock:
            if self.stop_event:
                self.stop_event.set()
            # Drop any incomplete in-progress placeholder clips, keep completed ones
            self.artifacts = [a for a in self.artifacts if (a.get("files") and len(a.get("files")) > 0)]
            self.status = "stopped"
            self._append_log("[cancel] Stopped — completed clips preserved. Click Resume or Start Again.")

    def get_clip_edit_state(self, run_dir: Path, clip_index: int) -> dict[str, Any]:
        work_dir = run_dir / "work" / f"clip_{clip_index:02d}"
        edit_file = work_dir / "edit_state.json"
        if edit_file.exists():
            return json.loads(edit_file.read_text(encoding="utf-8"))

        # Synthesize edit_state for legacy runs from summary.json + window.mp4
        win_mp4 = work_dir / "window.mp4"
        if not win_mp4.exists():
            raise FileNotFoundError(f"Source window.mp4 not found for clip {clip_index:02d}")
        probe = probe_media(win_mp4)
        clip_info: dict[str, Any] = {}
        summary_file = run_dir / "clips" / "summary.json"
        if summary_file.exists():
            sm = json.loads(summary_file.read_text(encoding="utf-8"))
            clip_info = next((c for c in sm.get("clips", []) if int(c.get("index", 0)) == clip_index), {})

        dur = float(clip_info.get("duration_sec") or probe.get("duration") or 20.0)
        return {
            "window_video": str(win_mp4),
            "window_start": 0.0,
            "local_start": 4.0 if probe.get("duration", 0) > dur + 4.0 else 0.0,
            "duration": min(dur, float(probe.get("duration") or dur)),
            "window_duration": float(probe.get("duration") or dur),
            "enable_subtitles": True,
            "enable_hook": True,
            "hook_text": clip_info.get("hook", ""),
            "hook_y_pct": 8.0,
            "sub_y_pct": 76.0,
            "highlight_color": "#FFFF00",
            "game_x_pct": 50.0,
            "layout": clip_info.get("layout", "gameplay"),
            "facecam": {
                "x": 0,
                "y": 0,
                "w": int(probe.get("width", 1920) * 0.28),
                "h": int(probe.get("height", 1080) * 0.28),
                "frame_w": int(probe.get("width", 1920)),
                "frame_h": int(probe.get("height", 1080)),
            },
            "ratios": list((clip_info.get("files") or {"9:16": ""}).keys()),
            "words": [],
        }

    def rerender_clip(self, payload: dict[str, Any]) -> dict[str, Any]:
        run_dir = Path(str(payload.get("run_dir") or self.active_run_dir))
        clip_index = int(payload["index"])
        work_dir = run_dir / "work" / f"clip_{clip_index:02d}"
        clips_dir = run_dir / "clips"
        base_state = self.get_clip_edit_state(run_dir, clip_index)

        win_mp4 = Path(base_state["window_video"])
        if not win_mp4.exists():
            raise FileNotFoundError(f"Missing {win_mp4}")
        probe = probe_media(win_mp4)
        frame_w = int(probe.get("width") or 1920)
        frame_h = int(probe.get("height") or 1080)
        win_dur = float(probe.get("duration") or 60.0)

        local_start = max(0.0, float(payload.get("local_start", base_state.get("local_start", 0.0))))
        duration = max(3.0, min(win_dur - local_start, float(payload.get("duration", base_state.get("duration", 20.0)))))
        enable_subtitles = bool(payload.get("enable_subtitles", base_state.get("enable_subtitles", True)))
        enable_hook = bool(payload.get("enable_hook", base_state.get("enable_hook", True)))
        hook_text = str(payload.get("hook_text", base_state.get("hook_text", "")))
        hook_y_pct = float(payload.get("hook_y_pct", base_state.get("hook_y_pct", 8.0)))
        sub_y_pct = float(payload.get("sub_y_pct", base_state.get("sub_y_pct", 76.0)))
        highlight_color = str(payload.get("highlight_color", base_state.get("highlight_color", "#FFFF00")))
        layout = str(payload.get("layout", base_state.get("layout", "gameplay")))

        fc_raw = payload.get("facecam") or base_state.get("facecam") or {}
        facecam = FacecamBox(
            x=max(0, int(fc_raw.get("x", 0))),
            y=max(0, int(fc_raw.get("y", 0))),
            w=max(32, int(fc_raw.get("w", int(frame_w * 0.25)))),
            h=max(32, int(fc_raw.get("h", int(frame_h * 0.25)))),
            frame_w=frame_w,
            frame_h=frame_h,
            layout=layout,
            confidence=1.0,
        )

        settings = self.load_settings()
        compute_mode = str(settings.get("compute_mode", "auto"))
        whisper_model = str(settings.get("whisper_model", "small.en"))

        raw_words = payload.get("words") if "words" in payload else base_state.get("words", [])
        trim_changed = (
            abs(local_start - float(base_state.get("local_start", 0.0))) > 0.25
            or abs(duration - float(base_state.get("duration", 0.0))) > 0.25
        )
        if enable_subtitles and (not raw_words or (trim_changed and not payload.get("words_edited"))):
            _segs, trans_words = transcribe_media(
                win_mp4, model_size=whisper_model, device=compute_mode, vad_filter=False
            )
            rebased = rebase_words_for_clip(trans_words, local_start, local_start + duration)
        else:
            rebased = [
                WordTimestamp(word=str(w["word"]), start=float(w["start"]), end=float(w["end"]))
                for w in (raw_words or [])
                if str(w.get("word", "")).strip()
            ]

        summary_file = clips_dir / "summary.json"
        summary_data = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else {"clips": []}
        clip_entry = next((c for c in summary_data.get("clips", []) if int(c.get("index", 0)) == clip_index), {})

        title = str(payload.get("title") or clip_entry.get("title") or f"Clip {clip_index:02d}")
        caption = str(payload.get("caption") if "caption" in payload else clip_entry.get("caption", ""))
        hashtags_in = payload.get("hashtags") if "hashtags" in payload else clip_entry.get("hashtags", [])
        if isinstance(hashtags_in, str):
            hashtags = [h.strip() for h in hashtags_in.split() if h.strip()]
        else:
            hashtags = list(hashtags_in or [])

        profile = str(payload.get("profile", base_state.get("profile", "profile1")))
        game_x_pct = float(payload.get("game_x_pct", base_state.get("game_x_pct", 50.0)))
        ratios = ["9:16"] if profile == "profile2" else list(
            payload.get("ratios") or base_state.get("ratios") or list((clip_entry.get("files") or {"9:16": ""}).keys())
        )
        if profile == "profile2" and layout != "full_game":
            layout = "facecam"
        files = export_multi_ratio(
            source_video=win_mp4,
            clip_start_in_source=local_start,
            clip_duration=duration,
            clip_index=clip_index,
            title=title,
            hook_text=hook_text,
            facecam=facecam,
            words=rebased,
            clips_dir=clips_dir,
            ratios=ratios,
            enable_subtitles=enable_subtitles,
            work_dir=work_dir,
            compute_mode=compute_mode,
            enable_hook=enable_hook,
            hook_y_pct=hook_y_pct,
            sub_y_pct=sub_y_pct,
            highlight_color=highlight_color,
            profile=profile,
            game_x_pct=game_x_pct,
        )

        cand = Candidate(
            start=str(clip_entry.get("start", "00:00:00")),
            end=str(clip_entry.get("end", "00:00:30")),
            category=str(clip_entry.get("category", "funny")),
            score=float(clip_entry.get("score", 8.0)),
            hook=hook_text,
            title=title,
            caption=caption,
            hashtags=hashtags,
            why=str(clip_entry.get("why", "")),
        )
        art = ClipArtifact(
            index=clip_index,
            candidate=cand,
            layout=layout,
            start_sec=parse_timestamp(cand.start),
            end_sec=parse_timestamp(cand.start) + duration,
            duration=duration,
            files=files,
            metadata_files={},
            preview_frame=clip_entry.get("preview_frame"),
        )
        write_clip_metadata_files(art)
        art_dict = self._artifact_to_json(art)

        # Persist updated summary.json & edit_state.json
        updated_clips = [c for c in summary_data.get("clips", []) if int(c.get("index", 0)) != clip_index]
        updated_clips.append(art_dict)
        updated_clips.sort(key=lambda c: int(c.get("index", 0)))
        summary_data["clips"] = updated_clips
        summary_file.write_text(json.dumps(summary_data, indent=2, ensure_ascii=False), encoding="utf-8")

        new_edit_state = {
            "window_video": str(win_mp4),
            "window_start": base_state.get("window_start", 0.0),
            "local_start": round(local_start, 3),
            "duration": round(duration, 3),
            "window_duration": round(win_dur, 3),
            "enable_subtitles": enable_subtitles,
            "enable_hook": enable_hook,
            "hook_text": hook_text,
            "hook_y_pct": hook_y_pct,
            "sub_y_pct": sub_y_pct,
            "highlight_color": highlight_color,
            "profile": profile,
            "game_x_pct": game_x_pct,
            "layout": layout,
            "facecam": {
                "x": facecam.x,
                "y": facecam.y,
                "w": facecam.w,
                "h": facecam.h,
                "frame_w": frame_w,
                "frame_h": frame_h,
            },
            "ratios": ratios,
            "words": [{"word": w.word, "start": w.start, "end": w.end} for w in rebased],
        }
        (work_dir / "edit_state.json").write_text(json.dumps(new_edit_state, indent=2, ensure_ascii=False), encoding="utf-8")

        with self.lock:
            self.artifacts = [a for a in self.artifacts if int(a.get("index", 0)) != clip_index]
            self.artifacts.append(art_dict)
            self.artifacts.sort(key=lambda a: int(a.get("index", 0)))

        return art_dict

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            elapsed = int(time.monotonic() - self.start_time) if self.status in ("running", "stopping") else 0
            return {
                "status": self.status,
                "stage": self.stage,
                "blocks_count": self.blocks_count,
                "winners_count": self.winners_count,
                "clips": list(self.artifacts),
                "logs": list(self.logs),
                "active_run_dir": self.active_run_dir,
                "elapsed_sec": elapsed,
                "settings": self.load_settings(),
                "history": self.list_history(),
            }


def create_http_server(
    state: StudioState | None = None,
    shutdown_event: threading.Event | None = None,
) -> tuple[ThreadingHTTPServer, StudioState, int]:
    """Start the local HTTP server on 127.0.0.1:<ephemeral_port>."""
    studio = state or StudioState()
    stop_signal = shutdown_event or threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *args: Any) -> None:
            return

        def _send_json(self, obj: Any, code: int = 200) -> None:
            raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            body = self.rfile.read(length).decode("utf-8", errors="replace")
            return json.loads(body) if body else {}

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/":
                raw = HTML_PATH.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return

            if parsed.path == "/api/logo":
                if LOGO_PATH.exists():
                    raw = LOGO_PATH.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                else:
                    self.send_error(404)
                return

            if parsed.path == "/api/state":
                self._send_json(studio.snapshot())
                return

            if parsed.path == "/api/clip/edit_state":
                qs = urllib.parse.parse_qs(parsed.query)
                run_dir = Path((qs.get("run_dir") or [studio.active_run_dir])[0])
                idx = int((qs.get("index") or ["1"])[0])
                try:
                    self._send_json({"ok": True, "edit_state": studio.get_clip_edit_state(run_dir, idx)})
                except Exception as exc:
                    self._send_json({"ok": False, "error": str(exc)}, 404)
                return

            if parsed.path == "/api/video":
                qs = urllib.parse.parse_qs(parsed.query)
                rel = (qs.get("path") or [""])[0]
                file_path = Path(rel).resolve()
                if not file_path.exists() or not file_path.is_file():
                    self.send_error(404)
                    return
                size = file_path.stat().st_size
                range_hdr = self.headers.get("Range")
                start, end = 0, size - 1
                status_code = HTTPStatus.OK
                if range_hdr and range_hdr.startswith("bytes="):
                    spec = range_hdr.replace("bytes=", "").split(",")[0].strip()
                    s_part, e_part = spec.split("-", 1)
                    if s_part:
                        start = int(s_part)
                    if e_part:
                        end = min(size - 1, int(e_part))
                    status_code = HTTPStatus.PARTIAL_CONTENT
                length = max(0, end - start + 1)
                self.send_response(status_code)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(length))
                if status_code == HTTPStatus.PARTIAL_CONTENT:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                try:
                    with file_path.open("rb") as fh:
                        fh.seek(start)
                        remaining = length
                        while remaining > 0:
                            chunk = fh.read(min(65536, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
                return

            self.send_error(404)

        def do_POST(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            try:
                body = self._read_json()
                if parsed.path == "/api/settings":
                    saved = studio.save_settings(body)
                    self._send_json({"ok": True, "settings": saved})
                elif parsed.path == "/api/llm/models":
                    models = list_available_models(
                        str(body.get("base_url") or "http://localhost:11434/v1"),
                        api_key=str(body.get("api_key") or "local"),
                    )
                    self._send_json({"ok": True, "models": models})
                elif parsed.path == "/api/llm/test":
                    studio.save_settings(
                        {
                            "preset": body.get("preset", "Custom"),
                            "base_url": body.get("base_url", ""),
                            "model": body.get("model", ""),
                            "api_key": body.get("api_key", ""),
                        }
                    )
                    reply = test_llm_connection(
                        str(body.get("base_url") or "http://localhost:11434/v1"),
                        str(body.get("model") or "qwen2.5:7b"),
                        api_key=str(body.get("api_key") or "local"),
                    )
                    self._send_json({"ok": True, "reply": reply})
                elif parsed.path == "/api/start":
                    studio.start_pipeline(body)
                    self._send_json({"ok": True, "active_run_dir": studio.active_run_dir})
                elif parsed.path == "/api/reset":
                    new_dir = studio.reset_for_new_run()
                    self._send_json({"ok": True, "active_run_dir": new_dir})
                elif parsed.path == "/api/stop":
                    studio.stop_pipeline()
                    self._send_json({"ok": True})
                elif parsed.path == "/api/open_folder":
                    run_dir = Path(str(body.get("run_dir") or studio.active_run_dir or "runs"))
                    clips_dir = run_dir / "clips"
                    open_clips_folder(clips_dir if clips_dir.exists() else run_dir)
                    self._send_json({"ok": True})
                elif parsed.path == "/api/open_external":
                    p = Path(str(body.get("path") or ""))
                    if p.exists():
                        os.startfile(str(p.resolve()))
                        self._send_json({"ok": True})
                    else:
                        self._send_json({"ok": False, "error": "File not found"}, 404)
                elif parsed.path == "/api/history/load":
                    run_dir = Path(str(body.get("run_dir") or ""))
                    ok = studio.load_summary(run_dir / "clips" / "summary.json", update_run_dir=True)
                    self._send_json({"ok": ok})
                elif parsed.path == "/api/history/resume":
                    run_dir = Path(str(body.get("run_dir") or ""))
                    st_file = run_dir / "state.json"
                    st = json.loads(st_file.read_text(encoding="utf-8")) if st_file.exists() else {}
                    resume_url = str(st.get("url") or body.get("url") or "").strip()
                    if not resume_url:
                        raise ValueError("No state.json or URL in this run folder.")
                    cur_settings = studio.load_settings()
                    studio.start_pipeline(
                        {
                            "resume": True,
                            "mode": "vod",
                            "url": resume_url,
                            "out_dir": str(run_dir),
                            "ratios": st.get("ratios", ["9:16", "1:1", "16:9"]),
                            "enable_subtitles": st.get("enable_subtitles", cur_settings.get("subtitles", True)),
                            "enable_hook": st.get("enable_hook", cur_settings.get("enable_hook", True)),
                            "min_score": st.get("min_score", cur_settings.get("min_score", 7.0)),
                            "max_clips": st.get("max_clips", cur_settings.get("max_clips", 10)),
                            "block_min": int(float(st.get("block_duration_sec", cur_settings.get("block_min", 10) * 60)) / 60),
                            "whisper_model": st.get("whisper_model", cur_settings.get("whisper_model", "small.en")),
                            "compute_mode": st.get("compute_mode", cur_settings.get("compute_mode", "auto")),
                            "profile": st.get("profile", cur_settings.get("profile", "profile1")),
                            "cookies_browser": body.get("cookies_browser", cur_settings.get("cookies_browser", "")),
                            "cookies_file": body.get("cookies_file", cur_settings.get("cookies_file", "")),
                        }
                    )
                    self._send_json({"ok": True})
                elif parsed.path == "/api/history/remove":
                    run_dir = Path(str(body.get("run_dir") or ""))
                    if studio.worker and studio.worker.is_alive() and Path(studio.active_run_dir) == run_dir:
                        raise RuntimeError("Cannot remove the currently running job.")
                    shutil.rmtree(run_dir, ignore_errors=True)
                    self._send_json({"ok": True})
                elif parsed.path == "/api/history/clear":
                    if studio.worker and studio.worker.is_alive():
                        raise RuntimeError("Stop the running job before clearing history.")
                    runs_root = Path("runs")
                    if runs_root.exists():
                        for sub in runs_root.iterdir():
                            if sub.is_dir():
                                shutil.rmtree(sub, ignore_errors=True)
                    with studio.lock:
                        studio.artifacts = []
                    self._send_json({"ok": True})
                elif parsed.path == "/api/clip/rerender":
                    updated = studio.rerender_clip(body)
                    self._send_json({"ok": True, "clip": updated})
                elif parsed.path == "/api/clip/retranscribe":
                    run_dir = Path(str(body.get("run_dir") or studio.active_run_dir))
                    idx = int(body["index"])
                    base_state = studio.get_clip_edit_state(run_dir, idx)
                    win_mp4 = Path(base_state["window_video"])
                    ls = float(body.get("local_start", base_state.get("local_start", 0.0)))
                    dur = float(body.get("duration", base_state.get("duration", 20.0)))
                    settings = studio.load_settings()
                    _segs, trans_words = transcribe_media(
                        win_mp4,
                        model_size=str(settings.get("whisper_model", "small.en")),
                        device=str(settings.get("compute_mode", "auto")),
                        vad_filter=False,
                    )
                    rebased = rebase_words_for_clip(trans_words, ls, ls + dur)
                    self._send_json(
                        {
                            "ok": True,
                            "words": [{"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3)} for w in rebased],
                        }
                    )
                elif parsed.path == "/api/shutdown":
                    stop_signal.set()
                    self._send_json({"ok": True})
                else:
                    self.send_error(404)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, 400)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, name="clip-hunter-web", daemon=True)
    thread.start()
    return server, studio, port
