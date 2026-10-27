"""
src/models/jev_worker.py
────────────────────────
Bounded, throttled background worker so Jev requests never block the camera loop.

  • exactly one worker thread → at most one request in flight
  • at most one pending job; a newer submission replaces an older unsent one
  • ``min_interval_seconds`` between request starts (plus the API's own rate limit)
  • results are tagged with frame id / capture time / pose epoch so the caller can
    discard stale answers
  • frames are held in memory only until sent; nothing is written to disk
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from src.models.jev_client import JevClient, JevResult


@dataclass
class JevWorkerStats:
    submitted: int = 0
    throttled: int = 0
    replaced: int = 0
    completed: int = 0
    failures: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    last_status: Optional[str] = None
    last_error: Optional[str] = None

    @property
    def mean_latency_ms(self) -> Optional[float]:
        return float(np.mean(self.latencies_ms)) if self.latencies_ms else None

    @property
    def p95_latency_ms(self) -> Optional[float]:
        return float(np.percentile(self.latencies_ms, 95)) if self.latencies_ms else None


class JevWorker:
    def __init__(
        self,
        client: JevClient,
        min_interval_seconds: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
        start: bool = True,
    ):
        self._client = client
        self._min_interval = float(min_interval_seconds)
        self._clock = clock
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._pending: Optional[tuple] = None
        self._in_flight = False
        self._last_start: Optional[float] = None
        self._latest: Optional[JevResult] = None
        self._new_result = False
        self._stop = False
        self.stats = JevWorkerStats()
        self._thread: Optional[threading.Thread] = None
        if start:
            self._thread = threading.Thread(target=self._run, name="jev-worker", daemon=True)
            self._thread.start()

    # ── Producer side (UI thread) ─────────────────────────────────────────────
    def seconds_until_allowed(self) -> float:
        with self._lock:
            if self._last_start is None:
                return 0.0
            return max(0.0, self._min_interval - (self._clock() - self._last_start))

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._in_flight or self._pending is not None

    def submit(self, frame_bgr: np.ndarray, candidates: list[str], frame_id: int,
               frame_timestamp: float, epoch: int, force: bool = False,
               focus_pose: Optional[str] = None) -> bool:
        """
        Queue a still frame for analysis.  Returns False when throttled.
        ``force`` (manual request) bypasses the minimum interval but not the
        single-in-flight bound.
        """
        with self._lock:
            if self._stop:
                return False
            now = self._clock()
            if not force and self._last_start is not None and now - self._last_start < self._min_interval:
                self.stats.throttled += 1
                return False
            if not force and (self._in_flight or self._pending is not None):
                self.stats.throttled += 1
                return False
            if self._pending is not None:
                self.stats.replaced += 1
            self._pending = (frame_bgr.copy(), list(candidates), frame_id, frame_timestamp, epoch, focus_pose)
            self._last_start = now  # throttle from acceptance, not from worker pickup
            self.stats.submitted += 1
            self._wake.notify()
            return True

    def poll(self) -> Optional[JevResult]:
        """Return a newly completed result once (None if nothing new)."""
        with self._lock:
            if not self._new_result:
                return None
            self._new_result = False
            return self._latest

    @property
    def latest(self) -> Optional[JevResult]:
        with self._lock:
            return self._latest

    # ── Worker thread ─────────────────────────────────────────────────────────
    def _run(self) -> None:
        while True:
            with self._lock:
                while not self._stop and self._pending is None:
                    self._wake.wait(timeout=0.5)
                if self._stop:
                    return
                job = self._pending
                self._pending = None
                self._in_flight = True
            result = self.process(job)
            with self._lock:
                self._in_flight = False
                self._latest = result
                self._new_result = True

    def process(self, job: tuple) -> JevResult:
        frame, cands, fid, fts, epoch, focus = job
        try:
            result = self._client.classify(frame, cands, frame_id=fid, frame_timestamp=fts, epoch=epoch,
                                           focus_pose=focus)
        except Exception as exc:  # defensive: the UI must never crash because of Jev
            result = JevResult(ok=False, status="network_error", error=type(exc).__name__,
                               frame_id=fid, frame_timestamp=fts, epoch=epoch, candidates=cands)
        del frame
        self.stats.completed += 1
        self.stats.last_status = result.status
        self.stats.last_error = result.error
        if result.ok:
            self.stats.latencies_ms.append(result.latency_ms)
        else:
            self.stats.failures += 1
        return result

    def shutdown(self, timeout: float = 2.0) -> None:
        with self._lock:
            self._stop = True
            self._pending = None
            self._wake.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
