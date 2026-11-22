"""Runtime configuration shared by the daemon and its clients.

Environment overrides:
  DRM_STATE_DIR  state directory (default: <current directory>/.drm)
  DRM_DB         SQLite database path (default: <state dir>/drm.db)
  DRM_HOST       bind host, loopback only (default: 127.0.0.1)
  DRM_PORT       bind port (default: 8737)
  DRM_URL        base URL clients use to reach the daemon
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8737


def state_dir() -> Path:
    root = os.environ.get("DRM_STATE_DIR")
    if root:
        return Path(root)
    return Path.cwd() / ".drm"


def db_path() -> Path:
    override = os.environ.get("DRM_DB")
    if override:
        return Path(override)
    return state_dir() / "drm.db"


def host() -> str:
    return os.environ.get("DRM_HOST", DEFAULT_HOST)


def port() -> int:
    return int(os.environ.get("DRM_PORT", str(DEFAULT_PORT)))


def base_url() -> str:
    override = os.environ.get("DRM_URL")
    if override:
        return override.rstrip("/")
    return f"http://{host()}:{port()}"


def pid_file() -> Path:
    return state_dir() / "daemon.pid"


def log_file() -> Path:
    return state_dir() / "daemon.log"


def ensure_state_dir() -> Path:
    directory = state_dir()
    directory.mkdir(parents=True, exist_ok=True)
    return directory
