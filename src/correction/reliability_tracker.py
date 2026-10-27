"""
src/correction/reliability_tracker.py
─────────────────────────────────────
Stateful temporal reliability and jitter tracker for confidence-aware feedback.

Slide 7 of Yoga Mitra PPT specifies combining:
  1. Landmark Confidence (MediaPipe visibility scores)
  2. Joint Visibility (presence in camera viewport)
  3. Temporal Stability (consistency across consecutive frames, distinguishing
     natural smooth movement from high-frequency detector jitter)

Reliability Tiers & Coaching Policy:
  • High: Actionable, authoritative joint-specific angle corrections.
  • Medium: Qualified, gentle suggestions ("Hold steady...", "Gently adjust...").
  • Low: Suppress unsupported angle corrections; instruct user to reposition,
    step back, improve lighting, or bring the affected body part into view.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from src.features.geometry import ANGLE_TRIPLETS, VISIBILITY_THRESHOLD


@dataclass
class JointReliability:
    joint_name: str
    tier: str                           # "high" | "medium" | "low"
    is_reliable: bool                   # True if tier != "low"
    visibility: float                   # Minimum visibility score across constituent landmarks [0, 1]
    is_imputed: bool                    # Whether any constituent landmark was imputed
    jitter_norm: float                  # Normalized high-frequency jitter score [0, 1]
    suppress_correction: bool           # True if corrections should not be uttered
    reason: str                         # Diagnostic explanation


@dataclass
class FrameReliabilityAssessment:
    overall_tier: str                   # "high" | "medium" | "low"
    is_overall_reliable: bool
    mean_visibility: float
    imputed_fraction: float
    mean_jitter: float
    joint_assessments: dict[str, JointReliability] = field(default_factory=dict)
    global_guidance: Optional[str] = None


class TemporalReliabilityTracker:
    """
    Maintains a rolling window of recent landmark frames to calculate
    velocity, acceleration, and high-frequency jitter, isolating true
    tracking instability from deliberate user movement.
    """

    def __init__(
        self,
        history_window: int = 5,
        jitter_threshold: float = 0.30,   # Normalized torso fraction (allows natural yoga movement)
        min_visibility: float = VISIBILITY_THRESHOLD,
    ):
        self.history_window = history_window
        self.jitter_threshold = jitter_threshold
        self.min_visibility = min_visibility

        # Rolling deque of landmarks: each element is list of 33 dicts
        self._history: deque[list[dict[str, float]]] = deque(maxlen=history_window)
        self._imputed_history: deque[list[int]] = deque(maxlen=history_window)

    def reset(self) -> None:
        """Reset historical buffers upon scene or pose transition."""
        self._history.clear()
        self._imputed_history.clear()

    def update(
        self,
        landmarks: list[dict[str, float]],
        imputed_flags: list[int],
    ) -> FrameReliabilityAssessment:
        """
        Evaluate frame landmarks against historical trajectory to compute
        joint-by-joint and overall reliability.
        """
        if not landmarks or len(landmarks) < 33:
            return FrameReliabilityAssessment(
                overall_tier="low",
                is_overall_reliable=False,
                mean_visibility=0.0,
                imputed_fraction=1.0,
                mean_jitter=1.0,
                global_guidance="No person detected in camera view.",
            )

        self._history.append(landmarks)
        self._imputed_history.append(imputed_flags if imputed_flags else [0] * len(landmarks))

        visibilities = [float(lm.get("visibility", 0.0)) for lm in landmarks]
        mean_vis = float(np.mean(visibilities))
        imp_frac = float(np.mean(imputed_flags)) if imputed_flags else 0.0

        # Calculate high-frequency landmark jitter
        jitter_scores = self._calculate_jitter()

        joint_assessments: dict[str, JointReliability] = {}
        low_count = 0
        med_count = 0

        for angle_name, triplet in ANGLE_TRIPLETS.items():
            joint_vis = [visibilities[idx] for idx in triplet if idx < len(visibilities)]
            min_j_vis = min(joint_vis) if joint_vis else 0.0

            # Check if any triplet joint is imputed in recent frames
            is_imp = False
            for h_flags in self._imputed_history:
                if any(idx < len(h_flags) and h_flags[idx] for idx in triplet):
                    is_imp = True
                    break

            # Joint jitter is max jitter among constituent triplet landmarks
            j_jitters = [jitter_scores.get(idx, 0.0) for idx in triplet]
            max_j_jitter = max(j_jitters) if j_jitters else 0.0

            # Determine Tier
            if min_j_vis < self.min_visibility:
                tier = "low"
                reason = "Joint occluded or partially outside camera view"
                suppress = True
                low_count += 1
            elif max_j_jitter > self.jitter_threshold:
                # Distinguish from legitimate large motion: rapid oscillation = jitter
                tier = "low"
                reason = "Severe tracking jitter / camera blur"
                suppress = True
                low_count += 1
            elif is_imp or min_j_vis < 0.55 or max_j_jitter > (self.jitter_threshold * 0.5):
                tier = "medium"
                reason = "Marginal visibility or mild landmark jitter"
                suppress = False
                med_count += 1
            else:
                tier = "high"
                reason = "Clear visibility and stable tracking"
                suppress = False

            joint_assessments[angle_name] = JointReliability(
                joint_name=angle_name,
                tier=tier,
                is_reliable=(tier != "low"),
                visibility=round(min_j_vis, 3),
                is_imputed=is_imp,
                jitter_norm=round(max_j_jitter, 3),
                suppress_correction=suppress,
                reason=reason,
            )

        mean_jitter = float(np.mean(list(jitter_scores.values()))) if jitter_scores else 0.0

        # Overall frame tier
        if low_count >= 3 or imp_frac > 0.35:
            overall_tier = "low"
            global_msg = "Step back into clear lighting to improve pose tracking."
        elif low_count > 0 or med_count >= 3 or mean_vis < 0.60:
            overall_tier = "medium"
            global_msg = "Hold steady to confirm posture alignment."
        else:
            overall_tier = "high"
            global_msg = None

        return FrameReliabilityAssessment(
            overall_tier=overall_tier,
            is_overall_reliable=(overall_tier != "low"),
            mean_visibility=round(mean_vis, 3),
            imputed_fraction=round(imp_frac, 3),
            mean_jitter=round(mean_jitter, 3),
            joint_assessments=joint_assessments,
            global_guidance=global_msg,
        )

    def _calculate_jitter(self) -> dict[int, float]:
        """
        Computes high-frequency tracking jitter per landmark using the
        second difference (discrete acceleration / direction reversal).
        Natural movements exhibit smooth velocity profiles, whereas tracking
        jitter shows rapid directional reversals with high second difference.
        """
        if len(self._history) < 3:
            return {i: 0.0 for i in range(33)}

        frames = list(self._history)
        n_lms = len(frames[0])
        jitter_map: dict[int, float] = {}

        # Reference scale: torso distance (left shoulder 11 to left hip 23)
        ls = np.array([frames[-1][11]["x"], frames[-1][11]["y"]])
        lh = np.array([frames[-1][23]["x"], frames[-1][23]["y"]])
        torso_scale = float(np.linalg.norm(ls - lh))
        if torso_scale < 0.05:
            torso_scale = 0.30  # Safe floor

        for idx in range(n_lms):
            pts = [np.array([f[idx]["x"], f[idx]["y"]]) for f in frames[-3:]]
            # Velocity vectors: v1 = p1 - p0, v2 = p2 - p1
            v1 = pts[1] - pts[0]
            v2 = pts[2] - pts[1]
            # Second difference (acceleration / direction change)
            accel = v2 - v1
            accel_norm = float(np.linalg.norm(accel)) / torso_scale

            # Directional oscillation check: negative dot product indicates sharp reversal
            dot = float(np.dot(v1, v2))
            if dot < 0:
                oscillation_penalty = 1.5
            else:
                oscillation_penalty = 0.8

            jitter_score = accel_norm * oscillation_penalty
            jitter_map[idx] = float(jitter_score)

        return jitter_map
