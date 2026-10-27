"""
src/pose/missing_landmarks.py
──────────────────────────────
Handles low-visibility landmarks for live video inference.

Policy:
  • Temporal fallback only: a joint that is not reliably observed in the
    current frame may reuse its *last reliably observed* position, but only
    while that observation is younger than ``max_age_ms``.
  • Every estimated joint is flagged (``imputed_flags[i] == 1`` and
    ``landmark["imputed"] is True``) so downstream reliability logic can never
    mistake it for a real observation.  Estimated joints are capped at the
    "medium" reliability tier by the evaluator and the reliability tracker.
  • Expired / never-observed joints are marked unavailable: they keep their raw
    (low) visibility, so feature extraction produces NaN for any angle using them.
  • No symmetry mirroring across the body centerline – yoga poses are
    asymmetric and mirroring distorts genuine posture geometry.
  • MediaPipe VIDEO mode smooths visibility over time, so a joint that has just
    become occluded keeps its old position with a visibility that decays over
    roughly 8–10 frames.  A joint whose visibility is falling steeply while it
    is still above threshold is treated as *suspect* (estimated, not observed).
  • History is reset when the person disappears, when the torso jumps (new
    person / camera moved), or when a new session starts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

VISIBILITY_THRESHOLD = 0.30
IMPUTED_VISIBILITY = 0.35  # conservative value assigned to estimated joints

# Bilateral landmark pairs (left_id, right_id) – kept for reference only.
OPPOSITE_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]

TORSO_IDS = (11, 12, 23, 24)

STATUS_OBSERVED = "observed"
STATUS_IMPUTED = "imputed"
STATUS_UNAVAILABLE = "unavailable"


def _is_reliable(point: dict, threshold: float = VISIBILITY_THRESHOLD) -> bool:
    return point is not None and float(point.get("visibility", 0.0)) >= threshold


def apply_temporal_fallback(
    raw_landmarks: list[dict],
    previous_raw: Optional[list[dict]],
    threshold: float = VISIBILITY_THRESHOLD,
) -> tuple[list[dict], list[int], list[dict]]:
    """
    Legacy single-frame fallback (kept for tools/tests that use it directly).

    For each low-visibility joint, fall back to the previous frame's position
    when that position was directly observed.  The live app uses
    :class:`LandmarkHistory`, which adds a maximum age and session resets.

    Returns:
        effective  – 33-element list of landmark dicts used for feature extraction
        flags      – 1 where a joint was estimated, 0 where directly observed
        raw        – original detections (for the caller to store as next prev_raw)
    """
    effective = [dict(p) for p in raw_landmarks]
    flags = [0] * len(raw_landmarks)

    for joint_id, point in enumerate(raw_landmarks):
        if _is_reliable(point, threshold):
            continue

        if previous_raw is not None and joint_id < len(previous_raw):
            prev = previous_raw[joint_id]
            if _is_reliable(prev, threshold) and not prev.get("imputed", False):
                effective[joint_id] = {
                    "x": prev["x"],
                    "y": prev["y"],
                    "z": prev["z"],
                    "visibility": IMPUTED_VISIBILITY,
                    "imputed": True,
                }
                flags[joint_id] = 1

    return effective, flags, raw_landmarks


@dataclass
class LandmarkFrame:
    """Output of :meth:`LandmarkHistory.update` for one camera frame."""
    effective: list[dict]                 # 33 dicts used for posture evaluation / display
    imputed_flags: list[int]              # 1 = estimated from history, 0 = observed or unavailable
    status: list[str]                     # per-landmark: observed | imputed | unavailable
    age_ms: list[Optional[float]]         # age of the observation each effective point comes from
    observed_only: list[dict]             # raw detections with estimated joints forced below threshold
    suspect_ids: list[int] = field(default_factory=list)  # visibility-decay ghosts (subset of imputed)
    history_reset: bool = False           # True if this frame triggered a reset

    @property
    def n_imputed(self) -> int:
        return int(sum(self.imputed_flags))

    @property
    def n_unavailable(self) -> int:
        return sum(1 for s in self.status if s == STATUS_UNAVAILABLE)


class LandmarkHistory:
    """
    Per-landmark memory of the last reliable observation (coordinates + time).

    Args:
        threshold:            visibility needed to count as a direct observation
        max_age_ms:           how long a past observation may stand in for a missing joint
        scene_change_norm:    torso jump (normalised image units) that triggers a reset
        decay_drop:           visibility drop over ``decay_window`` frames that marks a
                              still-above-threshold joint as suspect (0 disables)
        decay_window:         frames over which the drop is measured
        decay_ceiling:        only joints currently below this visibility can be suspect
    """

    def __init__(
        self,
        threshold: float = VISIBILITY_THRESHOLD,
        max_age_ms: float = 300.0,
        scene_change_norm: float = 0.25,
        decay_drop: float = 0.15,
        decay_window: int = 3,
        decay_ceiling: float = 0.75,
    ):
        self.threshold = float(threshold)
        self.max_age_ms = float(max_age_ms)
        self.scene_change_norm = float(scene_change_norm)
        self.decay_drop = float(decay_drop)
        self.decay_window = max(2, int(decay_window))
        self.decay_ceiling = float(decay_ceiling)
        self.reset_count = 0
        self.reset()

    # ── State ─────────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Forget all history (person left, scene changed, or new session)."""
        self._last_point: list[Optional[dict]] = [None] * 33
        self._last_ts: list[Optional[float]] = [None] * 33
        self._vis_hist: list[list[float]] = [[] for _ in range(33)]
        self._last_torso: Optional[np.ndarray] = None
        self._last_frame_ts: Optional[float] = None

    def _torso_center(self, lm: list[dict]) -> Optional[np.ndarray]:
        pts = [lm[i] for i in TORSO_IDS if _is_reliable(lm[i], self.threshold)]
        if len(pts) < 3:
            return None
        return np.array([np.mean([p["x"] for p in pts]), np.mean([p["y"] for p in pts])])

    # ── Update ────────────────────────────────────────────────────────────────
    def update(self, raw_landmarks: Optional[list[dict]], timestamp_ms: Optional[float] = None) -> Optional[LandmarkFrame]:
        """
        Process one frame.  Returns None (and resets) when no person is detected.
        """
        if raw_landmarks is None or len(raw_landmarks) < 33:
            if self._last_frame_ts is not None:
                self.reset_count += 1
            self.reset()
            return None

        now = float(timestamp_ms) if timestamp_ms is not None else time.monotonic() * 1000.0
        did_reset = False

        # Scene change: torso centre jumped (different person / camera moved)
        torso = self._torso_center(raw_landmarks)
        if torso is not None and self._last_torso is not None:
            if float(np.linalg.norm(torso - self._last_torso)) > self.scene_change_norm:
                self.reset()
                self.reset_count += 1
                did_reset = True
        # Out-of-order timestamps: start over rather than mixing time bases
        if self._last_frame_ts is not None and now < self._last_frame_ts:
            self.reset()
            self.reset_count += 1
            did_reset = True

        effective: list[dict] = []
        observed_only: list[dict] = []
        flags = [0] * 33
        status = [STATUS_UNAVAILABLE] * 33
        ages: list[Optional[float]] = [None] * 33
        suspects: list[int] = []

        for i in range(33):
            p = raw_landmarks[i]
            vis = float(p.get("visibility", 0.0))
            hist = self._vis_hist[i]
            hist.append(vis)
            if len(hist) > self.decay_window:
                del hist[0]

            decaying = (
                self.decay_drop > 0
                and len(hist) == self.decay_window
                and all(b < a for a, b in zip(hist, hist[1:]))
                and (hist[0] - hist[-1]) >= self.decay_drop
                and vis < self.decay_ceiling
            )

            if vis >= self.threshold and not decaying:
                pt = {"x": float(p["x"]), "y": float(p["y"]), "z": float(p.get("z", 0.0)), "visibility": vis}
                effective.append(pt)
                observed_only.append(pt)
                status[i] = STATUS_OBSERVED
                ages[i] = 0.0
                self._last_point[i] = pt
                self._last_ts[i] = now
                continue

            # Not directly observed in this frame
            hidden = {"x": float(p["x"]), "y": float(p["y"]), "z": float(p.get("z", 0.0)),
                      "visibility": min(vis, self.threshold * 0.5)}
            observed_only.append(hidden)

            if decaying:
                suspects.append(i)
            last, last_ts = self._last_point[i], self._last_ts[i]
            age = (now - last_ts) if last_ts is not None else None
            if last is not None and age is not None and age <= self.max_age_ms:
                effective.append({
                    "x": last["x"], "y": last["y"], "z": last["z"],
                    "visibility": IMPUTED_VISIBILITY,
                    "imputed": True,
                    "age_ms": round(age, 1),
                })
                flags[i] = 1
                status[i] = STATUS_IMPUTED
                ages[i] = age
            else:
                # Expired or never seen: unavailable (raw low visibility → NaN features)
                effective.append({"x": float(p["x"]), "y": float(p["y"]), "z": float(p.get("z", 0.0)),
                                  "visibility": min(vis, self.threshold * 0.5), "unavailable": True})
                if age is not None and age > self.max_age_ms:
                    self._last_point[i] = None
                    self._last_ts[i] = None

        self._last_torso = torso if torso is not None else self._last_torso
        self._last_frame_ts = now
        return LandmarkFrame(
            effective=effective,
            imputed_flags=flags,
            status=status,
            age_ms=ages,
            observed_only=observed_only,
            suspect_ids=suspects,
            history_reset=did_reset,
        )
