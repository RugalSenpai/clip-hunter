"""Professional PySide6 Desktop Studio GUI for the Viral-Clipping Pipeline."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtGui, QtMultimedia, QtMultimediaWidgets, QtWidgets

from .judge import OpenAICompatibleJudge, list_available_models, test_llm_connection
from .live_recorder import run_live_clipper
from .models import Candidate, ClipArtifact, parse_timestamp
from .reporter import open_clips_folder
from .vod_stream import run_streaming_vod

STUDIO_QSS = """
QMainWindow, QWidget {
    background-color: #0B0F17;
    color: #E2E8F0;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}
QFrame#HeaderBar {
    background-color: #111827;
    border-bottom: 1px solid #1F2937;
}
QLabel#AppTitle {
    font-size: 18px;
    font-weight: 700;
    color: #F8FAFC;
    letter-spacing: 0.4px;
}
QLabel#AppSubtitle {
    font-size: 12px;
    color: #94A3B8;
}
QFrame#Card {
    background-color: #131A26;
    border: 1px solid #1E293B;
    border-radius: 10px;
}
QLabel#CardTitle {
    font-size: 12px;
    font-weight: 700;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
QLabel#Muted {
    color: #94A3B8;
    font-size: 12px;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
    background-color: #0B101B;
    border: 1px solid #263248;
    border-radius: 7px;
    padding: 7px 10px;
    color: #F1F5F9;
    selection-background-color: #4F46E5;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #6366F1;
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QPushButton {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 7px;
    padding: 7px 13px;
    color: #E2E8F0;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #283548;
    border-color: #475569;
}
QPushButton:pressed {
    background-color: #0F172A;
}
QPushButton:disabled {
    background-color: #111827;
    color: #475569;
    border-color: #1E293B;
}
QPushButton#PrimaryBtn {
    background-color: #4F46E5;
    border: 1px solid #6366F1;
    color: #FFFFFF;
    padding: 10px 16px;
    font-size: 13px;
    font-weight: 700;
}
QPushButton#PrimaryBtn:hover {
    background-color: #4338CA;
}
QPushButton#PrimaryBtn:disabled {
    background-color: #1E1B4B;
    border-color: #312E81;
    color: #6366F1;
}
QPushButton#DangerBtn {
    background-color: #7F1D1D;
    border: 1px solid #EF4444;
    color: #FEE2E2;
    padding: 10px 16px;
    font-weight: 700;
}
QPushButton#DangerBtn:hover {
    background-color: #991B1B;
}
QPushButton#DangerBtn:disabled {
    background-color: #1F1215;
    border-color: #3F1D22;
    color: #6B3B42;
}
QPushButton#RatioTab {
    background-color: #0F172A;
    border: 1px solid #263248;
    border-radius: 6px;
    padding: 5px 12px;
    font-weight: 600;
    color: #94A3B8;
}
QPushButton#RatioTab:checked {
    background-color: #4F46E5;
    border-color: #6366F1;
    color: #FFFFFF;
}
QCheckBox {
    spacing: 8px;
    color: #E2E8F0;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid #475569;
    background: #0B101B;
}
QCheckBox::indicator:checked {
    background: #4F46E5;
    border-color: #6366F1;
}
QTabWidget::pane {
    border: 1px solid #1E293B;
    border-radius: 10px;
    background-color: #131A26;
    top: -1px;
}
QTabBar::tab {
    background-color: #0F172A;
    border: 1px solid #1E293B;
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 8px 18px;
    margin-right: 4px;
    color: #94A3B8;
    font-weight: 600;
}
QTabBar::tab:selected {
    background-color: #131A26;
    color: #F8FAFC;
    border-bottom: 2px solid #6366F1;
}
QListWidget {
    background-color: #0D131F;
    border: 1px solid #1E293B;
    border-radius: 8px;
    padding: 6px;
    outline: none;
}
QListWidget::item {
    background-color: #151E2E;
    border: 1px solid #222E44;
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 6px;
}
QListWidget::item:selected {
    background-color: #1E293B;
    border: 1px solid #6366F1;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #1E293B;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #6366F1;
    width: 14px;
    height: 14px;
    margin: -4px 0;
    border-radius: 7px;
}
QSlider::sub-page:horizontal {
    background: #4F46E5;
    border-radius: 3px;
}
QScrollBar:vertical {
    background: #0B0F17;
    width: 8px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #334155;
    border-radius: 4px;
    min-height: 24px;
}
"""

LLM_PRESETS: dict[str, tuple[str, str]] = {
    "Ollama (Local)": ("http://localhost:11434/v1", "qwen2.5:7b"),
    "LM Studio (Local)": ("http://localhost:1234/v1", "local-model"),
    "OpenAI API": ("https://api.openai.com/v1", "gpt-4o-mini"),
    "Groq API": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
    "Custom": ("http://localhost:11434/v1", ""),
}

SETTINGS_FILE = Path.home() / ".clipper_studio_settings.json"


class _SignalStream(io.TextIOBase):
    """Redirect stdout/stderr lines into a Qt signal while preserving original stream."""

    def __init__(self, emit_fn, original) -> None:
        super().__init__()
        self._emit_fn = emit_fn
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
                self._emit_fn(line)
        return len(s)

    def flush(self) -> None:
        if self._buf.strip():
            self._emit_fn(self._buf.strip())
            self._buf = ""
        if self._original:
            try:
                self._original.flush()
            except Exception:
                pass


class PipelineSignals(QtCore.QObject):
    log_line = QtCore.Signal(str)
    clip_ready = QtCore.Signal(object)  # ClipArtifact
    finished = QtCore.Signal(int)  # clip count
    failed = QtCore.Signal(str)
    models_loaded = QtCore.Signal(list, str)  # models, error
    llm_tested = QtCore.Signal(bool, str)


def _badge_label(text: str, bg: str = "#1E293B", fg: str = "#CBD5E1") -> QtWidgets.QLabel:
    lbl = QtWidgets.QLabel(text)
    lbl.setStyleSheet(
        f"background-color: {bg}; color: {fg}; border-radius: 5px; "
        "padding: 3px 8px; font-size: 11px; font-weight: 700;"
    )
    return lbl


class ClipperStudioWindow(QtWidgets.QMainWindow):
    """Main desktop studio window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Clipper Studio — AI Moment Selector & Multi-Ratio Video Engine")
        self.resize(1380, 860)
        self.setMinimumSize(1120, 720)

        self.signals = PipelineSignals()
        self.signals.log_line.connect(self._on_log_line)
        self.signals.clip_ready.connect(self._on_clip_ready)
        self.signals.finished.connect(self._on_pipeline_finished)
        self.signals.failed.connect(self._on_pipeline_failed)
        self.signals.models_loaded.connect(self._on_models_loaded)
        self.signals.llm_tested.connect(self._on_llm_tested)

        self._stop_event: threading.Event | None = None
        self._worker_thread: threading.Thread | None = None
        self._run_start_time: float = 0.0
        self._artifacts: list[ClipArtifact] = []
        self._current_artifact: ClipArtifact | None = None
        self._current_ratio: str = "9:16"
        self._blocks_count = 0
        self._moments_count = 0

        self._elapsed_timer = QtCore.QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._update_elapsed)

        self._setup_media_player()
        self._build_ui()
        self._load_settings()
        self._on_mode_changed()
        self._auto_load_latest_run()

    # ------------------------------------------------------------------ Media Player
    def _setup_media_player(self) -> None:
        self.player = QtMultimedia.QMediaPlayer(self)
        self.audio_output = QtMultimedia.QAudioOutput(self)
        self.audio_output.setVolume(0.85)
        self.player.setAudioOutput(self.audio_output)
        self.player.positionChanged.connect(self._on_player_pos)
        self.player.durationChanged.connect(self._on_player_dur)
        self.player.playbackStateChanged.connect(self._on_player_state)

    # ------------------------------------------------------------------ UI Layout
    def _build_ui(self) -> None:
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        main_vbox = QtWidgets.QVBoxLayout(root)
        main_vbox.setContentsMargins(0, 0, 0, 0)
        main_vbox.setSpacing(0)

        main_vbox.addWidget(self._build_header())

        body = QtWidgets.QWidget()
        body_hbox = QtWidgets.QHBoxLayout(body)
        body_hbox.setContentsMargins(16, 14, 16, 16)
        body_hbox.setSpacing(16)

        body_hbox.addWidget(self._build_left_sidebar(), 0)
        body_hbox.addWidget(self._build_right_workspace(), 1)
        main_vbox.addWidget(body, 1)

    def _build_header(self) -> QtWidgets.QFrame:
        bar = QtWidgets.QFrame()
        bar.setObjectName("HeaderBar")
        bar.setFixedHeight(62)
        lay = QtWidgets.QHBoxLayout(bar)
        lay.setContentsMargins(20, 8, 20, 8)
        lay.setSpacing(12)

        title_col = QtWidgets.QVBoxLayout()
        title_col.setSpacing(1)
        title = QtWidgets.QLabel("CLIPPER STUDIO")
        title.setObjectName("AppTitle")
        sub = QtWidgets.QLabel("LLM selects viral moments  •  Engine downloads, crops, edits & burns subtitles")
        sub.setObjectName("AppSubtitle")
        title_col.addWidget(title)
        title_col.addWidget(sub)
        lay.addLayout(title_col)

        lay.addStretch(1)

        self.status_badge = _badge_label("IDLE", "#1E293B", "#94A3B8")
        self.elapsed_lbl = QtWidgets.QLabel("00:00")
        self.elapsed_lbl.setStyleSheet("font-family: Consolas, monospace; font-size: 13px; color: #94A3B8;")
        lay.addWidget(self.status_badge)
        lay.addWidget(self.elapsed_lbl)

        btn_load = QtWidgets.QPushButton("Load Run...")
        btn_load.clicked.connect(self._browse_and_load_run)
        btn_open = QtWidgets.QPushButton("Open Clips Folder")
        btn_open.clicked.connect(self._open_current_clips_folder)
        lay.addWidget(btn_load)
        lay.addWidget(btn_open)
        return bar

    def _build_left_sidebar(self) -> QtWidgets.QScrollArea:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setFixedWidth(375)

        container = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 6, 0)
        vbox.setSpacing(12)

        # --- Card 1: Source & Mode ---
        c_src = self._make_card("1. Stream / VOD Source")
        src_lay = c_src.layout()

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Mode A — VOD Stream (10m blocks)", "Mode B — Live Stream (15m buffer)"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        src_lay.addWidget(QtWidgets.QLabel("Pipeline Mode"))
        src_lay.addWidget(self.mode_combo)

        src_lay.addWidget(QtWidgets.QLabel("Video / Stream URL"))
        self.url_input = QtWidgets.QLineEdit()
        self.url_input.setPlaceholderText("https://www.youtube.com/watch?v=... or Twitch URL")
        src_lay.addWidget(self.url_input)

        src_lay.addWidget(QtWidgets.QLabel("Output Run Folder"))
        out_row = QtWidgets.QHBoxLayout()
        default_run = f"runs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.out_input = QtWidgets.QLineEdit(default_run)
        btn_dir = QtWidgets.QPushButton("...")
        btn_dir.setFixedWidth(36)
        btn_dir.clicked.connect(self._pick_output_dir)
        out_row.addWidget(self.out_input, 1)
        out_row.addWidget(btn_dir)
        src_lay.addLayout(out_row)

        # Live-only widgets
        self.live_opts_widget = QtWidgets.QWidget()
        live_lay = QtWidgets.QFormLayout(self.live_opts_widget)
        live_lay.setContentsMargins(0, 4, 0, 0)
        self.buffer_spin = QtWidgets.QSpinBox()
        self.buffer_spin.setRange(5, 60)
        self.buffer_spin.setValue(15)
        self.twitch_input = QtWidgets.QLineEdit()
        self.twitch_input.setPlaceholderText("optional (e.g. shroud)")
        live_lay.addRow("Buffer (min):", self.buffer_spin)
        live_lay.addRow("Twitch chat:", self.twitch_input)
        src_lay.addWidget(self.live_opts_widget)
        vbox.addWidget(c_src)

        # --- Card 2: LLM Moment Selector ---
        c_llm = self._make_card("2. LLM Moment Selector (Text Only)")
        llm_lay = c_llm.layout()

        note = QtWidgets.QLabel("Only chooses viral timestamps & metadata. Never touches video.")
        note.setObjectName("Muted")
        note.setWordWrap(True)
        llm_lay.addWidget(note)

        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems(list(LLM_PRESETS.keys()))
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        llm_lay.addWidget(QtWidgets.QLabel("Provider Preset"))
        llm_lay.addWidget(self.preset_combo)

        llm_lay.addWidget(QtWidgets.QLabel("API Base URL"))
        base_row = QtWidgets.QHBoxLayout()
        self.base_url_input = QtWidgets.QLineEdit("http://localhost:11434/v1")
        btn_save_llm = QtWidgets.QPushButton("Save")
        btn_save_llm.setToolTip("Save LLM URL, model, and export settings")
        btn_save_llm.clicked.connect(lambda: self._save_settings(show_feedback=True))
        base_row.addWidget(self.base_url_input, 1)
        base_row.addWidget(btn_save_llm)
        llm_lay.addLayout(base_row)

        llm_lay.addWidget(QtWidgets.QLabel("Model"))
        model_row = QtWidgets.QHBoxLayout()
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setCurrentText("qwen2.5:7b")
        btn_refresh = QtWidgets.QPushButton("Refresh")
        btn_refresh.clicked.connect(self._refresh_models)
        model_row.addWidget(self.model_combo, 1)
        model_row.addWidget(btn_refresh)
        llm_lay.addLayout(model_row)

        llm_lay.addWidget(QtWidgets.QLabel("API Key (optional for local LLMs)"))
        self.api_key_input = QtWidgets.QLineEdit()
        self.api_key_input.setEchoMode(QtWidgets.QLineEdit.Password)
        self.api_key_input.setPlaceholderText("local (not required for Ollama / LM Studio)")
        llm_lay.addWidget(self.api_key_input)

        test_row = QtWidgets.QHBoxLayout()
        self.btn_test_llm = QtWidgets.QPushButton("Test LLM Connection")
        self.btn_test_llm.clicked.connect(self._test_llm)
        self.llm_status_lbl = QtWidgets.QLabel("")
        self.llm_status_lbl.setObjectName("Muted")
        test_row.addWidget(self.btn_test_llm)
        test_row.addWidget(self.llm_status_lbl, 1)
        llm_lay.addLayout(test_row)
        vbox.addWidget(c_llm)

        # --- Card 3: Engine Editing & Export ---
        c_exp = self._make_card("3. Engine Editing & Export")
        exp_lay = c_exp.layout()

        exp_lay.addWidget(QtWidgets.QLabel("Output Aspect Ratios"))
        ratio_row = QtWidgets.QHBoxLayout()
        self.chk_916 = QtWidgets.QCheckBox("9:16")
        self.chk_916.setChecked(True)
        self.chk_11 = QtWidgets.QCheckBox("1:1")
        self.chk_11.setChecked(True)
        self.chk_169 = QtWidgets.QCheckBox("16:9")
        self.chk_169.setChecked(True)
        ratio_row.addWidget(self.chk_916)
        ratio_row.addWidget(self.chk_11)
        ratio_row.addWidget(self.chk_169)
        ratio_row.addStretch(1)
        exp_lay.addLayout(ratio_row)

        self.chk_subtitles = QtWidgets.QCheckBox("Enable burned-in subtitles (Hook + Word-by-Word)")
        self.chk_subtitles.setChecked(True)
        exp_lay.addWidget(self.chk_subtitles)

        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Min Score:"), 0, 0)
        self.score_spin = QtWidgets.QDoubleSpinBox()
        self.score_spin.setRange(5.0, 9.5)
        self.score_spin.setSingleStep(0.5)
        self.score_spin.setValue(7.0)
        grid.addWidget(self.score_spin, 0, 1)

        grid.addWidget(QtWidgets.QLabel("Max Clips:"), 0, 2)
        self.max_clips_spin = QtWidgets.QSpinBox()
        self.max_clips_spin.setRange(1, 50)
        self.max_clips_spin.setValue(10)
        grid.addWidget(self.max_clips_spin, 0, 3)

        grid.addWidget(QtWidgets.QLabel("Block (min):"), 1, 0)
        self.block_min_spin = QtWidgets.QSpinBox()
        self.block_min_spin.setRange(2, 20)
        self.block_min_spin.setValue(10)
        grid.addWidget(self.block_min_spin, 1, 1)

        grid.addWidget(QtWidgets.QLabel("Whisper:"), 1, 2)
        self.whisper_combo = QtWidgets.QComboBox()
        self.whisper_combo.addItems(["small.en", "base.en", "tiny.en", "medium.en"])
        grid.addWidget(self.whisper_combo, 1, 3)
        exp_lay.addLayout(grid)
        vbox.addWidget(c_exp)

        # --- Action Buttons ---
        act_row = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("▶  Start Clipping")
        self.btn_start.setObjectName("PrimaryBtn")
        self.btn_start.clicked.connect(self._start_pipeline)
        self.btn_stop = QtWidgets.QPushButton("■  Stop")
        self.btn_stop.setObjectName("DangerBtn")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_pipeline)
        act_row.addWidget(self.btn_start, 2)
        act_row.addWidget(self.btn_stop, 1)
        vbox.addLayout(act_row)
        vbox.addStretch(1)

        scroll.setWidget(container)
        return scroll

    def _build_right_workspace(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(12)

        # --- Pipeline Stage Strip ---
        stage_card = QtWidgets.QFrame()
        stage_card.setObjectName("Card")
        s_lay = QtWidgets.QHBoxLayout(stage_card)
        s_lay.setContentsMargins(14, 10, 14, 10)
        s_lay.setSpacing(10)

        self.pill_stage1 = _badge_label("1. Transcript / Buffer", "#1E293B", "#94A3B8")
        self.pill_stage2 = _badge_label("2. LLM Moment Judge", "#1E293B", "#94A3B8")
        self.pill_stage3 = _badge_label("3. Download, Edit & Subtitles", "#1E293B", "#94A3B8")
        s_lay.addWidget(self.pill_stage1)
        s_lay.addWidget(QtWidgets.QLabel("→"))
        s_lay.addWidget(self.pill_stage2)
        s_lay.addWidget(QtWidgets.QLabel("→"))
        s_lay.addWidget(self.pill_stage3)
        s_lay.addStretch(1)

        self.stats_lbl = QtWidgets.QLabel("Blocks: 0  •  Winners: 0  •  Clips Ready: 0")
        self.stats_lbl.setStyleSheet("font-weight: 600; color: #CBD5E1;")
        s_lay.addWidget(self.stats_lbl)
        vbox.addWidget(stage_card)

        self.latest_event_lbl = QtWidgets.QLabel("Ready. Configure URL & LLM, then click Start Clipping.")
        self.latest_event_lbl.setStyleSheet(
            "background-color: #0D131F; border: 1px solid #1E293B; border-radius: 7px; "
            "padding: 6px 12px; color: #94A3B8; font-family: Consolas, monospace; font-size: 12px;"
        )
        vbox.addWidget(self.latest_event_lbl)

        # --- Tabs: Clips & Preview | Live Log ---
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_clips_tab(), "Clips & Preview")
        self.tabs.addTab(self._build_log_tab(), "Pipeline Log")
        vbox.addWidget(self.tabs, 1)
        return panel

    def _build_clips_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        hbox = QtWidgets.QHBoxLayout(tab)
        hbox.setContentsMargins(12, 12, 12, 12)
        hbox.setSpacing(14)

        # Left: Clips List
        left = QtWidgets.QVBoxLayout()
        left.setSpacing(8)
        hdr = QtWidgets.QLabel("Generated Clips")
        hdr.setObjectName("CardTitle")
        left.addWidget(hdr)

        self.clips_list = QtWidgets.QListWidget()
        self.clips_list.currentRowChanged.connect(self._on_clip_selected)
        left.addWidget(self.clips_list, 1)
        hbox.addLayout(left, 4)

        # Right: Video Player + Metadata Inspector
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(10)

        # Ratio switcher bar
        ratio_bar = QtWidgets.QHBoxLayout()
        ratio_bar.addWidget(QtWidgets.QLabel("Preview Ratio:"))
        self.ratio_btns: dict[str, QtWidgets.QPushButton] = {}
        self.ratio_group = QtWidgets.QButtonGroup(self)
        for r in ("9:16", "1:1", "16:9"):
            btn = QtWidgets.QPushButton(r)
            btn.setObjectName("RatioTab")
            btn.setCheckable(True)
            if r == "9:16":
                btn.setChecked(True)
            btn.clicked.connect(lambda _chk, ratio=r: self._switch_preview_ratio(ratio))
            self.ratio_group.addButton(btn)
            self.ratio_btns[r] = btn
            ratio_bar.addWidget(btn)
        ratio_bar.addStretch(1)
        self.btn_ext_play = QtWidgets.QPushButton("External Player")
        self.btn_ext_play.clicked.connect(self._open_selected_external)
        ratio_bar.addWidget(self.btn_ext_play)
        right.addLayout(ratio_bar)

        # Video Widget container
        video_frame = QtWidgets.QFrame()
        video_frame.setStyleSheet("background-color: #05080F; border: 1px solid #1E293B; border-radius: 10px;")
        vf_lay = QtWidgets.QVBoxLayout(video_frame)
        vf_lay.setContentsMargins(4, 4, 4, 4)
        self.video_widget = QtMultimediaWidgets.QVideoWidget()
        self.video_widget.setMinimumHeight(310)
        self.player.setVideoOutput(self.video_widget)
        vf_lay.addWidget(self.video_widget, 1)

        # Transport controls
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.setContentsMargins(6, 2, 6, 4)
        self.btn_play = QtWidgets.QPushButton("▶ Play")
        self.btn_play.setFixedWidth(80)
        self.btn_play.clicked.connect(self._toggle_play)
        self.seek_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.seek_slider.sliderMoved.connect(self.player.setPosition)
        self.time_lbl = QtWidgets.QLabel("00:00 / 00:00")
        self.time_lbl.setStyleSheet("font-family: Consolas, monospace; font-size: 11px; color: #94A3B8;")
        ctrl.addWidget(self.btn_play)
        ctrl.addWidget(self.seek_slider, 1)
        ctrl.addWidget(self.time_lbl)
        vf_lay.addLayout(ctrl)
        right.addWidget(video_frame, 3)

        # Metadata Inspector
        meta_card = self._make_card("Post Metadata & Caption")
        m_lay = meta_card.layout()
        self.meta_badges = QtWidgets.QLabel("Select a clip to inspect title, hook, caption & hashtags.")
        self.meta_badges.setObjectName("Muted")
        m_lay.addWidget(self.meta_badges)

        self.meta_text = QtWidgets.QPlainTextEdit()
        self.meta_text.setReadOnly(True)
        self.meta_text.setMaximumHeight(135)
        m_lay.addWidget(self.meta_text)

        copy_row = QtWidgets.QHBoxLayout()
        btn_copy_all = QtWidgets.QPushButton("Copy Title + Caption + Hashtags")
        btn_copy_all.clicked.connect(self._copy_selected_metadata)
        copy_row.addWidget(btn_copy_all)
        copy_row.addStretch(1)
        m_lay.addLayout(copy_row)
        right.addWidget(meta_card, 2)

        hbox.addLayout(right, 6)
        return tab

    def _build_log_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(tab)
        lay.setContentsMargins(12, 12, 12, 12)
        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet(
            "background-color: #070A11; font-family: Consolas, 'Cascadia Code', monospace; "
            "font-size: 12px; color: #CBD5E1; border: 1px solid #1E293B; border-radius: 8px;"
        )
        lay.addWidget(self.log_box)
        return tab

    def _make_card(self, title: str) -> QtWidgets.QFrame:
        card = QtWidgets.QFrame()
        card.setObjectName("Card")
        lay = QtWidgets.QVBoxLayout(card)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(7)
        lbl = QtWidgets.QLabel(title)
        lbl.setObjectName("CardTitle")
        lay.addWidget(lbl)
        return card

    # ------------------------------------------------------------------ Actions & Events
    def _on_mode_changed(self) -> None:
        is_live = self.mode_combo.currentIndex() == 1
        self.live_opts_widget.setVisible(is_live)
        self.score_spin.setValue(7.5 if is_live else 7.0)

    def _on_preset_changed(self, name: str) -> None:
        base_url, model = LLM_PRESETS.get(name, ("http://localhost:11434/v1", ""))
        self.base_url_input.setText(base_url)
        if model:
            self.model_combo.setCurrentText(model)

    def _pick_output_dir(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Run Output Folder", str(Path.cwd()))
        if folder:
            self.out_input.setText(folder)

    def _refresh_models(self) -> None:
        base = self.base_url_input.text().strip()
        key = self.api_key_input.text().strip() or "local"
        self.llm_status_lbl.setText("Fetching models...")

        def _work():
            try:
                models = list_available_models(base, api_key=key)
                self.signals.models_loaded.emit(models, "")
            except Exception as exc:
                self.signals.models_loaded.emit([], str(exc))

        threading.Thread(target=_work, daemon=True).start()

    @QtCore.Slot(list, str)
    def _on_models_loaded(self, models: list, err: str) -> None:
        if err:
            self.llm_status_lbl.setText("Offline / unreachable")
            self.llm_status_lbl.setStyleSheet("color: #F87171; font-size: 12px;")
            return
        if models:
            curr = self.model_combo.currentText()
            self.model_combo.clear()
            self.model_combo.addItems(models)
            if curr in models:
                self.model_combo.setCurrentText(curr)
        self.llm_status_lbl.setText(f"Found {len(models)} model(s)")
        self.llm_status_lbl.setStyleSheet("color: #34D399; font-size: 12px;")

    def _test_llm(self) -> None:
        self._save_settings()
        base = self.base_url_input.text().strip()
        model = self.model_combo.currentText().strip()
        key = self.api_key_input.text().strip() or "local"
        self.btn_test_llm.setEnabled(False)
        self.llm_status_lbl.setText("Testing...")

        def _work():
            try:
                reply = test_llm_connection(base, model, api_key=key)
                self.signals.llm_tested.emit(True, f"Connected ({reply[:18]})")
            except Exception as exc:
                self.signals.llm_tested.emit(False, str(exc)[:60])

        threading.Thread(target=_work, daemon=True).start()

    @QtCore.Slot(bool, str)
    def _on_llm_tested(self, ok: bool, msg: str) -> None:
        self.btn_test_llm.setEnabled(True)
        self.llm_status_lbl.setText(msg)
        color = "#34D399" if ok else "#F87171"
        self.llm_status_lbl.setStyleSheet(f"color: {color}; font-size: 12px;")

    def _selected_ratios(self) -> list[str]:
        ratios = []
        if self.chk_916.isChecked():
            ratios.append("9:16")
        if self.chk_11.isChecked():
            ratios.append("1:1")
        if self.chk_169.isChecked():
            ratios.append("16:9")
        return ratios

    def _start_pipeline(self) -> None:
        url = self.url_input.text().strip()
        if not url:
            QtWidgets.QMessageBox.warning(self, "Missing URL", "Please enter a YouTube / Twitch stream URL.")
            return
        ratios = self._selected_ratios()
        if not ratios:
            QtWidgets.QMessageBox.warning(self, "Missing Ratio", "Select at least one output aspect ratio.")
            return

        run_dir = Path(self.out_input.text().strip() or f"runs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self._save_settings()
        self._artifacts.clear()
        self.clips_list.clear()
        self.log_box.clear()
        self._blocks_count = 0
        self._moments_count = 0
        self._update_stats()

        self._stop_event = threading.Event()
        self._run_start_time = time.monotonic()
        self._elapsed_timer.start()
        self._set_running_ui(True)
        self._set_stage(1)

        judge = OpenAICompatibleJudge(
            base_url=self.base_url_input.text().strip(),
            model=self.model_combo.currentText().strip(),
            api_key=self.api_key_input.text().strip() or "local",
        )
        is_live = self.mode_combo.currentIndex() == 1
        enable_subs = self.chk_subtitles.isChecked()
        min_score = float(self.score_spin.value())
        max_clips = int(self.max_clips_spin.value())
        block_sec = float(self.block_min_spin.value() * 60.0)
        whisper_model = self.whisper_combo.currentText()
        buffer_min = int(self.buffer_spin.value())
        twitch_chan = self.twitch_input.text().strip() or None

        def _run():
            orig_out, orig_err = sys.stdout, sys.stderr
            stream = _SignalStream(self.signals.log_line.emit, orig_out)
            sys.stdout = stream
            sys.stderr = stream
            try:
                if is_live:
                    arts = run_live_clipper(
                        url=url,
                        run_dir=run_dir,
                        judge=judge,
                        buffer_min=buffer_min,
                        ratios=ratios,
                        enable_subtitles=enable_subs,
                        min_score=min_score,
                        whisper_model=whisper_model,
                        twitch_channel=twitch_chan,
                        stop_event=self._stop_event,
                        on_clip_ready=lambda a: self.signals.clip_ready.emit(a),
                    )
                else:
                    arts = run_streaming_vod(
                        url=url,
                        run_dir=run_dir,
                        judge=judge,
                        ratios=ratios,
                        enable_subtitles=enable_subs,
                        block_duration_sec=block_sec,
                        min_score=min_score,
                        max_clips=max_clips,
                        whisper_model=whisper_model,
                        stop_event=self._stop_event,
                        on_clip_ready=lambda a: self.signals.clip_ready.emit(a),
                    )
                self.signals.finished.emit(len(arts))
            except Exception as exc:
                self.signals.failed.emit(str(exc))
            finally:
                stream.flush()
                sys.stdout = orig_out
                sys.stderr = orig_err

        self._worker_thread = threading.Thread(target=_run, daemon=True)
        self._worker_thread.start()

    def _stop_pipeline(self) -> None:
        if self._stop_event:
            self._stop_event.set()
            self.status_badge.setText("STOPPING...")
            self.status_badge.setStyleSheet(
                "background-color: #78350F; color: #FDE68A; border-radius: 5px; padding: 3px 8px; font-weight: 700;"
            )
            self.btn_stop.setEnabled(False)
            self._on_log_line("[cancel] Stop requested — finishing in-flight write & preserving completed clips...")

    def _set_running_ui(self, running: bool) -> None:
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        if running:
            self.status_badge.setText("RUNNING")
            self.status_badge.setStyleSheet(
                "background-color: #312E81; color: #C7D2FE; border-radius: 5px; padding: 3px 8px; font-weight: 700;"
            )

    def _set_stage(self, active: int) -> None:
        pills = [self.pill_stage1, self.pill_stage2, self.pill_stage3]
        for idx, pill in enumerate(pills, start=1):
            if idx == active:
                pill.setStyleSheet(
                    "background-color: #312E81; color: #E0E7FF; border-radius: 5px; padding: 3px 8px; font-weight: 700;"
                )
            else:
                pill.setStyleSheet(
                    "background-color: #1E293B; color: #94A3B8; border-radius: 5px; padding: 3px 8px; font-weight: 700;"
                )

    def _update_elapsed(self) -> None:
        secs = int(time.monotonic() - self._run_start_time)
        self.elapsed_lbl.setText(f"{secs // 60:02d}:{secs % 60:02d}")

    def _update_stats(self) -> None:
        self.stats_lbl.setText(
            f"Blocks: {self._blocks_count}  •  Winners: {self._moments_count}  •  Clips Ready: {len(self._artifacts)}"
        )

    @QtCore.Slot(str)
    def _on_log_line(self, line: str) -> None:
        self.log_box.appendPlainText(line)
        is_err = "ERROR]" in line or "[warnings]" in line
        color = "#FCA5A5" if is_err else "#CBD5E1"
        self.latest_event_lbl.setText(line[:150])
        self.latest_event_lbl.setStyleSheet(
            f"background-color: #0D131F; border: 1px solid {'#7F1D1D' if is_err else '#1E293B'}; "
            f"border-radius: 7px; padding: 6px 12px; color: {color}; "
            "font-family: Consolas, monospace; font-size: 12px;"
        )
        if is_err and len(self._artifacts) == 0:
            self.tabs.setCurrentIndex(1)
        if "[producer]" in line:
            self._set_stage(1)
        elif "[judge] Evaluating" in line:
            self._blocks_count += 1
            self._set_stage(2)
            self._update_stats()
        elif "[judge] Winner" in line:
            self._moments_count += 1
            self._update_stats()
        elif "[render]" in line or "[clip ready]" in line:
            self._set_stage(3)

    @QtCore.Slot(object)
    def _on_clip_ready(self, artifact: ClipArtifact) -> None:
        self._artifacts.append(artifact)
        self._update_stats()
        ratios_str = " ".join(artifact.files.keys())
        item = QtWidgets.QListWidgetItem(
            f"#{artifact.index:02d}  {artifact.candidate.title}\n"
            f"Score {artifact.candidate.score:.1f}  •  {artifact.duration:.1f}s  •  {artifact.layout}  •  [{ratios_str}]"
        )
        self.clips_list.addItem(item)
        if self.clips_list.count() == 1:
            self.clips_list.setCurrentRow(0)

    @QtCore.Slot(int)
    def _on_pipeline_finished(self, count: int) -> None:
        self._elapsed_timer.stop()
        self._set_running_ui(False)
        self.status_badge.setText(f"DONE ({count} CLIPS)")
        self.status_badge.setStyleSheet(
            "background-color: #064E3B; color: #6EE7B7; border-radius: 5px; padding: 3px 8px; font-weight: 700;"
        )

    @QtCore.Slot(str)
    def _on_pipeline_failed(self, err: str) -> None:
        self._elapsed_timer.stop()
        self._set_running_ui(False)
        self.status_badge.setText("ERROR")
        self.status_badge.setStyleSheet(
            "background-color: #7F1D1D; color: #FECACA; border-radius: 5px; padding: 3px 8px; font-weight: 700;"
        )
        self.log_box.appendPlainText(f"[ERROR] {err}")
        QtWidgets.QMessageBox.critical(self, "Pipeline Error", err)

    # ------------------------------------------------------------------ Preview & Metadata
    def _on_clip_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._artifacts):
            return
        art = self._artifacts[row]
        self._current_artifact = art
        c = art.candidate
        self.meta_badges.setText(
            f"Category: {c.category}  •  Score: {c.score:.1f}  •  Layout: {art.layout}  •  {art.duration:.1f}s"
        )
        self.meta_text.setPlainText(
            f"HOOK: {c.hook}\n"
            f"TITLE: {c.title}\n\n"
            f"{c.caption}\n\n"
            f"{' '.join(c.hashtags)}"
        )
        # Enable only available ratio buttons
        for r, btn in self.ratio_btns.items():
            btn.setEnabled(r in art.files)
        if self._current_ratio not in art.files and art.files:
            first_r = next(iter(art.files))
            self._current_ratio = first_r
            self.ratio_btns[first_r].setChecked(True)
        self._load_video_for_current_ratio()

    def _switch_preview_ratio(self, ratio: str) -> None:
        self._current_ratio = ratio
        self._load_video_for_current_ratio()

    def _load_video_for_current_ratio(self) -> None:
        if not self._current_artifact:
            return
        path = self._current_artifact.files.get(self._current_ratio)
        if path and Path(path).exists():
            self.player.setSource(QtCore.QUrl.fromLocalFile(str(Path(path).resolve())))
            self.player.play()

    def _toggle_play(self) -> None:
        if self.player.playbackState() == QtMultimedia.QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _on_player_state(self, state) -> None:
        if state == QtMultimedia.QMediaPlayer.PlayingState:
            self.btn_play.setText("❚❚ Pause")
        else:
            self.btn_play.setText("▶ Play")

    def _on_player_pos(self, pos_ms: int) -> None:
        if not self.seek_slider.isSliderDown():
            self.seek_slider.setValue(pos_ms)
        dur_ms = self.player.duration()
        self.time_lbl.setText(f"{pos_ms // 60000:02d}:{(pos_ms // 1000) % 60:02d} / {dur_ms // 60000:02d}:{(dur_ms // 1000) % 60:02d}")

    def _on_player_dur(self, dur_ms: int) -> None:
        self.seek_slider.setRange(0, dur_ms)

    def _open_selected_external(self) -> None:
        if not self._current_artifact:
            return
        path = self._current_artifact.files.get(self._current_ratio)
        if path and Path(path).exists():
            os.startfile(str(Path(path).resolve()))

    def _copy_selected_metadata(self) -> None:
        if not self._current_artifact:
            return
        QtWidgets.QApplication.clipboard().setText(self.meta_text.toPlainText())
        self.statusBar().showMessage("Copied post metadata to clipboard!", 2500)

    def _open_current_clips_folder(self) -> None:
        run_dir = Path(self.out_input.text().strip() or "runs")
        clips_dir = run_dir / "clips"
        open_clips_folder(clips_dir if clips_dir.exists() else run_dir)

    # ------------------------------------------------------------------ Load Saved Run
    def _browse_and_load_run(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open summary.json", str(Path("runs").resolve()), "Summary JSON (summary.json)"
        )
        if file_path:
            self._load_summary_file(Path(file_path))

    def _auto_load_latest_run(self) -> None:
        default_summary = Path("runs/test_15m/clips/summary.json")
        if default_summary.exists():
            self._load_summary_file(default_summary, update_out_dir=False)

    def _load_summary_file(self, summary_path: Path, update_out_dir: bool = True) -> None:
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            return
        self._artifacts.clear()
        self.clips_list.clear()
        for item in data.get("clips", []):
            cand = Candidate(
                start=item.get("start", "00:00:00"),
                end=item.get("end", "00:00:30"),
                category=item.get("category", "funny"),
                score=float(item.get("score", 7.0)),
                hook=item.get("hook", ""),
                title=item.get("title", "Clip"),
                caption=item.get("caption", ""),
                hashtags=item.get("hashtags", []),
                why=item.get("why", ""),
            )
            art = ClipArtifact(
                index=int(item.get("index", 1)),
                candidate=cand,
                layout=item.get("layout", "gameplay"),
                start_sec=parse_timestamp(cand.start),
                end_sec=parse_timestamp(cand.end),
                duration=float(item.get("duration_sec", 0.0)),
                files=item.get("files", {}),
                metadata_files=item.get("metadata_files", {}),
                preview_frame=item.get("preview_frame"),
            )
            self._on_clip_ready(art)
        if update_out_dir:
            self.out_input.setText(str(summary_path.parent.parent))
        self._moments_count = int(data.get("moments_found", len(self._artifacts)))
        self._update_stats()
        self.status_badge.setText(f"LOADED ({len(self._artifacts)} CLIPS)")

    # ------------------------------------------------------------------ Settings Persistence
    def _save_settings(self, show_feedback: bool = False) -> None:
        data = {
            "mode_index": self.mode_combo.currentIndex(),
            "preset": self.preset_combo.currentText(),
            "base_url": self.base_url_input.text().strip(),
            "model": self.model_combo.currentText().strip(),
            "api_key": self.api_key_input.text(),
            "ratio_916": self.chk_916.isChecked(),
            "ratio_11": self.chk_11.isChecked(),
            "ratio_169": self.chk_169.isChecked(),
            "subtitles": self.chk_subtitles.isChecked(),
            "min_score": float(self.score_spin.value()),
            "max_clips": int(self.max_clips_spin.value()),
            "block_min": int(self.block_min_spin.value()),
            "whisper_model": self.whisper_combo.currentText(),
            "buffer_min": int(self.buffer_spin.value()),
        }
        try:
            SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
            if show_feedback:
                self.llm_status_lbl.setText("Saved!")
                self.llm_status_lbl.setStyleSheet("color: #34D399; font-size: 12px;")
        except OSError:
            pass

    def _load_settings(self) -> None:
        if not SETTINGS_FILE.exists():
            return
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return
        if "mode_index" in data:
            self.mode_combo.setCurrentIndex(int(data["mode_index"]))
        if data.get("preset") in LLM_PRESETS:
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentText(data["preset"])
            self.preset_combo.blockSignals(False)
        if data.get("base_url"):
            self.base_url_input.setText(str(data["base_url"]))
        if data.get("model"):
            self.model_combo.setCurrentText(str(data["model"]))
        if "api_key" in data:
            self.api_key_input.setText(str(data["api_key"]))
        if "ratio_916" in data:
            self.chk_916.setChecked(bool(data["ratio_916"]))
        if "ratio_11" in data:
            self.chk_11.setChecked(bool(data["ratio_11"]))
        if "ratio_169" in data:
            self.chk_169.setChecked(bool(data["ratio_169"]))
        if "subtitles" in data:
            self.chk_subtitles.setChecked(bool(data["subtitles"]))
        if "min_score" in data:
            self.score_spin.setValue(float(data["min_score"]))
        if "max_clips" in data:
            self.max_clips_spin.setValue(int(data["max_clips"]))
        if "block_min" in data:
            self.block_min_spin.setValue(int(data["block_min"]))
        if data.get("whisper_model"):
            self.whisper_combo.setCurrentText(str(data["whisper_model"]))
        if "buffer_min" in data:
            self.buffer_spin.setValue(int(data["buffer_min"]))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._save_settings()
        super().closeEvent(event)


def launch_gui() -> int:
    """Launch the PySide6 Clipper Studio desktop application."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STUDIO_QSS)
    win = ClipperStudioWindow()
    win.show()
    return app.exec()
