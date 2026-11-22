"""Entry point for the daemon process: ``python -m drm.daemon``."""

from __future__ import annotations

import os
import sys

import uvicorn

from drm.config import db_path, ensure_state_dir, host, log_file, pid_file, port
from drm.daemon.app import create_app

_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def main() -> None:
    bind_host = host()
    if bind_host not in _LOOPBACK_HOSTS:
        sys.exit(f"refusing to bind non-loopback host {bind_host!r}: the daemon is loopback-only")
    ensure_state_dir()
    pid_file().write_text(f"{os.getpid()}\n")
    try:
        uvicorn.run(create_app(db_path()), host=bind_host, port=port(), log_level="info")
    finally:
        try:
            pid_file().unlink()
        except OSError:
            pass  # a hard kill leaves a stale file that start/stop clean up


if __name__ == "__main__":
    main()
