"""Scheduling loop stub (spec Sections 3 and 7).

Sprint 0 runs an idle heartbeat so the daemon reports scheduler state.
Placement and dispatch arrive with the planner sprints. Automated scheduling
pauses while the coordinator is offline; ``pause``/``resume`` expose that
switch for the daemon lifecycle.
"""

from __future__ import annotations

import threading

from drm.store.db import utcnow


class Scheduler:
    def __init__(self, interval_seconds: float = 5.0) -> None:
        self.interval_seconds = interval_seconds
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._paused = False
        self._ticks = 0
        self._last_tick_at: str | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="drm-scheduler", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            with self._lock:
                if self._paused:
                    continue
                self._ticks += 1
                self._last_tick_at = utcnow()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def pause(self) -> None:
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        with self._lock:
            self._paused = False

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "state": "paused" if self._paused else "running",
                "ticks": self._ticks,
                "last_tick_at": self._last_tick_at,
                "interval_seconds": self.interval_seconds,
            }
