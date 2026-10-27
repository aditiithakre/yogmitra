"""
src/pose/landmark_extraction.py
────────────────────────────────
Runs MediaPipe PoseLandmarker on BGR frames and returns raw landmark dicts.
Mirrors the extraction logic used in the training notebook.

Running modes
  • "video" (default for the webcam app): uses ``detect_for_video`` with
    monotonically increasing millisecond timestamps, so MediaPipe can reuse
    the previous frame's ROI and track the person between frames instead of
    re-running full-frame person detection on every frame.
  • "image": the original per-frame ``detect`` behaviour, kept for still
    images, dataset tools, and tests that call the detector on unrelated frames.

The public interface (``detect(frame) -> list[dict] | None``, ``close()``,
context manager) is unchanged; ``detect`` accepts an optional
``timestamp_ms`` argument in video mode.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── MediaPipe connection indices used for skeleton drawing ────────────────────
SKELETON_CONNECTIONS: list[tuple[int, int]] = [
    (11, 12),  # shoulders
    (11, 13), (13, 15),  # L arm
    (12, 14), (14, 16),  # R arm
    (11, 23), (12, 24),  # torso sides
    (23, 24),  # hips
    (23, 25), (25, 27), (27, 29), (27, 31),  # L leg
    (24, 26), (26, 28), (28, 30), (28, 32),  # R leg
]

RUNNING_MODES = {
    "image": vision.RunningMode.IMAGE,
    "video": vision.RunningMode.VIDEO,
}


def resolve_project_path(path: str | Path) -> Path:
    """Resolve *path* against the project root unless it is absolute or already exists."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    return _PROJECT_ROOT / p


class PoseLandmarkerWrapper:
    """Wraps MediaPipe PoseLandmarker (VIDEO or IMAGE running mode)."""

    def __init__(
        self,
        model_path: str | Path,
        visibility_threshold: float = 0.30,
        running_mode: str = "video",
        min_pose_detection_confidence: float = 0.50,
        min_pose_presence_confidence: float = 0.50,
        min_tracking_confidence: float = 0.50,
    ):
        mode = str(running_mode).lower()
        if mode not in RUNNING_MODES:
            raise ValueError(f"running_mode must be one of {sorted(RUNNING_MODES)}, got '{running_mode}'")

        self.visibility_threshold = visibility_threshold
        self.running_mode = mode
        self.model_path = str(resolve_project_path(model_path))
        self._detection_conf = min_pose_detection_confidence
        self._presence_conf = min_pose_presence_confidence
        self._tracking_conf = min_tracking_confidence

        # Timestamp bookkeeping (video mode)
        self._last_timestamp_ms: Optional[int] = None
        self._clock_origin: float = time.monotonic()
        self.timestamp_corrections: int = 0      # how many non-monotonic/invalid ts were repaired
        self.last_latency_ms: float = 0.0         # wall-clock latency of the last detect call
        self.frames_processed: int = 0

        self._detector = self._create_detector()
        self._closed = False

    # ── Construction ──────────────────────────────────────────────────────────
    def _create_detector(self):
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=RUNNING_MODES[self.running_mode],
            num_poses=1,
            min_pose_detection_confidence=self._detection_conf,
            min_pose_presence_confidence=self._presence_conf,
            min_tracking_confidence=self._tracking_conf,
        )
        return vision.PoseLandmarker.create_from_options(options)

    # ── Timestamps ────────────────────────────────────────────────────────────
    def _next_timestamp(self, timestamp_ms: Optional[float]) -> int:
        """
        Return a strictly increasing integer millisecond timestamp.

        A missing, non-finite, negative, or non-increasing timestamp is
        replaced by (last + 1) or by the internal monotonic clock, so
        ``detect_for_video`` never raises on an out-of-order frame.
        """
        valid = (
            timestamp_ms is not None
            and isinstance(timestamp_ms, (int, float))
            and math.isfinite(float(timestamp_ms))
            and float(timestamp_ms) >= 0
        )
        if valid:
            ts = int(timestamp_ms)
        else:
            if timestamp_ms is not None:
                self.timestamp_corrections += 1
            ts = int((time.monotonic() - self._clock_origin) * 1000.0)

        if self._last_timestamp_ms is not None and ts <= self._last_timestamp_ms:
            if valid:
                self.timestamp_corrections += 1
            ts = self._last_timestamp_ms + 1
        self._last_timestamp_ms = ts
        return ts

    @property
    def last_timestamp_ms(self) -> Optional[int]:
        return self._last_timestamp_ms

    # ── Detection ─────────────────────────────────────────────────────────────
    def detect(self, frame_bgr, timestamp_ms: Optional[float] = None) -> Optional[list[dict]]:
        """
        Run detection on a BGR frame.

        In video mode, *timestamp_ms* should be the capture time of the frame in
        milliseconds; when omitted, an internal monotonic clock is used.
        It is ignored in image mode.

        Returns:
            List of 33 landmark dicts {x, y, z, visibility} in normalised coords,
            or None when no person is detected.
        """
        if self._closed:
            raise RuntimeError("PoseLandmarkerWrapper.detect() called after close()")
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            return None

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        t0 = time.perf_counter()
        if self.running_mode == "video":
            result = self._detector.detect_for_video(mp_image, self._next_timestamp(timestamp_ms))
        else:
            result = self._detector.detect(mp_image)
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        self.frames_processed += 1

        if not result.pose_landmarks:
            return None

        detected = result.pose_landmarks[0]
        return [
            {
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(lm.z),
                "visibility": float(getattr(lm, "visibility", 0.0) or 0.0),
            }
            for lm in detected
        ]

    def reset_tracking(self) -> None:
        """
        Drop MediaPipe's internal tracking state (new session / scene change).
        Video-mode graphs cannot rewind timestamps, so the detector is rebuilt.
        """
        if self._closed or self.running_mode != "video":
            return
        self._detector.close()
        self._detector = self._create_detector()
        self._last_timestamp_ms = None
        self._clock_origin = time.monotonic()

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def close(self) -> None:
        """Release the MediaPipe graph. Safe to call more than once."""
        if getattr(self, "_closed", True):
            return
        self._closed = True
        try:
            self._detector.close()
        except Exception:
            pass

    @property
    def closed(self) -> bool:
        return self._closed

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
