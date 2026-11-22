"""Daemon process control used by the CLI: start detached, stop, health checks.

``drm daemon start`` spawns the daemon with the same interpreter and
environment, redirects its output to the state-dir log, and waits for
``/health``. ``drm daemon stop`` terminates the pid recorded in the state
dir. Both are idempotent.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HEALTH_TIMEOUT = 3.0
START_TIMEOUT = 15.0
POLL_INTERVAL = 0.25


class DaemonControlError(RuntimeError):
    """The daemon could not be stopped or started as requested."""


def is_healthy(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=HEALTH_TIMEOUT) as response:
            if response.status != 200:
                return False
            body = json.load(response)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False
    return body.get("status") == "ok" and "contract" in body


def wait_healthy(base_url: str, timeout: float = START_TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_healthy(base_url):
            return True
        time.sleep(POLL_INTERVAL)
    return False


def read_pid(pid_file: Path) -> int | None:
    try:
        return int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return None


def start_detached(state_dir: Path, log_file: Path, env: dict | None = None) -> int:
    """Spawn ``python -m drm.daemon`` detached; return its pid."""
    state_dir.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_file, "ab")
    try:
        extra: dict = {}
        if os.name == "nt":
            extra["creationflags"] = (
                subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            extra["start_new_session"] = True
        process = subprocess.Popen(
            [sys.executable, "-m", "drm.daemon"],
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env if env is not None else os.environ.copy(),
            **extra,
        )
        return process.pid
    finally:
        log_handle.close()


def stop(pid_file: Path, base_url: str, timeout: float = 8.0) -> dict:
    """Terminate the daemon recorded in pid_file; idempotent."""
    was_running = is_healthy(base_url)
    pid = read_pid(pid_file)
    if not was_running:
        if pid_file.exists():
            pid_file.unlink()  # stale pid from a crashed daemon
        return {"status": "stopped", "was_running": False, "pid": None}
    if pid is None:
        raise DaemonControlError(
            f"daemon is running but {pid_file} is missing; stop that process manually"
        )
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_healthy(base_url):
            if pid_file.exists():
                pid_file.unlink()
            return {"status": "stopped", "was_running": True, "pid": pid}
        time.sleep(POLL_INTERVAL)
    raise DaemonControlError(f"daemon pid {pid} did not stop within {timeout:.0f}s")
