"""Clip Hunter — Native Desktop App Window hosting the HTML/Tailwind Studio UI."""

from __future__ import annotations

import shutil
import subprocess
import threading
import webbrowser
from pathlib import Path

from .web_server import create_http_server

_APP_BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]


def _find_app_browser() -> str | None:
    for candidate in _APP_BROWSERS:
        if Path(candidate).exists():
            return candidate
    for exe in ("msedge", "chrome", "chromium"):
        found = shutil.which(exe)
        if found:
            return found
    return None


def launch_gui() -> int:
    """Start the local studio server and open Clip Hunter in a frameless Edge/Chrome app window."""
    shutdown_event = threading.Event()
    server, _studio, port = create_http_server(shutdown_event=shutdown_event)
    url = f"http://127.0.0.1:{port}/"

    browser_exe = _find_app_browser()
    if browser_exe:
        subprocess.Popen(
            [
                browser_exe,
                f"--app={url}",
                "--window-size=1460,880",
            ]
        )
    else:
        webbrowser.open(url)

    try:
        while not shutdown_event.is_set():
            shutdown_event.wait(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            server.shutdown()
            server.server_close()
        except Exception:
            pass
    return 0
