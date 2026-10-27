"""
src/correction/calibration.py
──────────────────────────────
User-specific biomechanical calibration module (v2.0).

Measures baseline body proportions (arm span to torso, leg to torso, shoulder width),
quantifies landmark stability, records pose-specific neutral baselines in Mountain Pose (Tadasana),
and persists the validated profile to data/user_profile.json.

Biomechanical Personalization Policy:
- Leg-to-torso ratio adapts knee and hip flexion ranges for long/short femurs.
- Arm-span ratio adapts shoulder reach and tabletop placement tolerances.
- Shoulder-width ratio adapts bilateral width alignments.
- Measurement stability modulates overall tolerance multiplier (avoids false alarms for jittery setups).
- Validates profile schema and handles corrupted or legacy v1 profiles safely.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.features.geometry import LANDMARK_IDS, VISIBILITY_THRESHOLD

logger = logging.getLogger(__name__)

DEFAULT_PROFILE_PATH = Path("data/user_profile.json")
PROFILE_SCHEMA_VERSION = "2.0"


@dataclass
class UserProfile:
    version: str = PROFILE_SCHEMA_VERSION
    calibrated: bool = False
    user_name: str = "Default Yogi"
    tolerance_multiplier: float = 1.0       # Range tolerance modifier [0.85, 1.30]
    arm_span_ratio: float = 1.80            # Arm span relative to torso length
    leg_to_torso_ratio: float = 1.65        # Leg length relative to torso length
    thigh_to_torso_ratio: float = 0.85      # Standing vertical thigh relative to torso length
    shoulder_width_ratio: float = 0.65      # Shoulder width relative to torso
    baseline_torso_px: float = 240.0        # Measured reference pixel scale
    measurement_stability: float = 0.95     # 0.0 to 1.0 stability quality during calibration
    calibration_date: Optional[str] = None
    pose_baselines: dict[str, dict[str, float]] = field(default_factory=dict)
    flexibility_tier: str = "standard"      # "tight", "standard", "flexible"


def validate_profile_dict(data: dict[str, Any]) -> tuple[bool, str]:
    """
    Validates a loaded user profile dictionary against anatomical sanity bounds.
    Returns (is_valid, reason).
    """
    if not isinstance(data, dict):
        return False, "Profile data is not a dictionary"

    # Anatomical plausibility checks
    arm = float(data.get("arm_span_ratio", 1.80))
    leg = float(data.get("leg_to_torso_ratio", 1.65))
    thigh = float(data.get("thigh_to_torso_ratio", 0.85))
    sh = float(data.get("shoulder_width_ratio", 0.65))
    tol = float(data.get("tolerance_multiplier", 1.0))

    if not (0.8 <= arm <= 2.6):
        return False, f"arm_span_ratio ({arm}) outside anatomical bounds [0.8, 2.6]"
    if not (0.8 <= leg <= 2.6):
        return False, f"leg_to_torso_ratio ({leg}) outside anatomical bounds [0.8, 2.6]"
    if not (0.25 <= thigh <= 1.8):
        return False, f"thigh_to_torso_ratio ({thigh}) outside anatomical bounds [0.25, 1.8]"
    if not (0.3 <= sh <= 1.3):
        return False, f"shoulder_width_ratio ({sh}) outside anatomical bounds [0.3, 1.3]"
    if not (0.7 <= tol <= 1.5):
        return False, f"tolerance_multiplier ({tol}) outside safe range [0.7, 1.5]"

    return True, "Valid"


def load_profile(path: Path | str = DEFAULT_PROFILE_PATH) -> UserProfile:
    """
    Load user profile from disk, validating against v2.0 schema.
    Returns safe default if missing, unparseable, or anatomically invalid.
    """
    p = Path(path)
    if not p.exists():
        return UserProfile()

    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        is_valid, reason = validate_profile_dict(data)
        if not is_valid:
            logger.warning("[Calibration] Corrupt/invalid profile at %s (%s). Using safe default.", p, reason)
            return UserProfile()

        # Migrate legacy v1.0 profiles gracefully
        pose_baselines = data.get("pose_baselines", {})
        if not isinstance(pose_baselines, dict):
            pose_baselines = {}

        return UserProfile(
            version=str(data.get("version", PROFILE_SCHEMA_VERSION)),
            calibrated=bool(data.get("calibrated", False)),
            user_name=str(data.get("user_name", "Default Yogi")),
            tolerance_multiplier=float(data.get("tolerance_multiplier", 1.0)),
            arm_span_ratio=float(data.get("arm_span_ratio", 1.80)),
            leg_to_torso_ratio=float(data.get("leg_to_torso_ratio", 1.65)),
            thigh_to_torso_ratio=float(data.get("thigh_to_torso_ratio", 0.85)),
            shoulder_width_ratio=float(data.get("shoulder_width_ratio", 0.65)),
            baseline_torso_px=float(data.get("baseline_torso_px", 240.0)),
            measurement_stability=float(data.get("measurement_stability", 0.95)),
            calibration_date=data.get("calibration_date"),
            pose_baselines=pose_baselines,
            flexibility_tier=str(data.get("flexibility_tier", "standard")),
        )
    except Exception as exc:
        logger.warning("[Calibration] Could not parse profile: %s. Using default.", exc)
        return UserProfile()


def save_profile(profile: UserProfile, path: Path | str = DEFAULT_PROFILE_PATH) -> bool:
    """Save user profile to disk with schema validation."""
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(profile)
        is_valid, reason = validate_profile_dict(data)
        if not is_valid:
            logger.error("[Calibration] Attempted to save invalid profile: %s", reason)
            return False

        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("[Calibration] Profile v%s saved to %s (calibrated=%s)", profile.version, p, profile.calibrated)
        return True
    except Exception as exc:
        logger.error("[Calibration] Failed to save profile: %s", exc)
        return False


def apply_biomechanical_personalization(
    base_ranges: dict[str, tuple[float, float]],
    profile: UserProfile | dict[str, Any],
    pose_name: str,
) -> dict[str, tuple[float, float]]:
    """
    Applies measurable biomechanical adaptations to target angle ranges based on
    the user's individual limb ratios and calibrated baselines.

    Adaptation Rules:
    1. Leg-to-Torso Ratio (> 1.68 indicates longer femur lever arms):
       - In lunges and flexion poses (Virabhadrasana One/Two, Utkatasana, Vajrasana),
         extends knee bend target range downward by up to 6°-8° to accommodate
         biomechanical reach without forcing unnatural lumbar shear.
    2. Arm-Span Ratio (< 1.75 indicates shorter reach):
       - In overhead reach and tabletop poses (Parvatasana, Ardhakati Chakrasana, Marjariasana),
         provides +4° allowance on shoulder extension.
    3. Shoulder-Width Ratio (> 0.70 indicates broad shoulders):
       - Tabletop & plank poses adjust shoulder alignment targets.
    4. Tolerance Multiplier (derived from user calibration stability):
       - Widens or narrows allowable tolerance band around target midpoint.
    """
    if isinstance(profile, UserProfile):
        prof_dict = asdict(profile)
    else:
        prof_dict = profile or {}

    if not prof_dict.get("calibrated", False):
        return dict(base_ranges)

    adapted: dict[str, tuple[float, float]] = {}
    leg_ratio = float(prof_dict.get("leg_to_torso_ratio", 1.65))
    arm_ratio = float(prof_dict.get("arm_span_ratio", 1.80))
    sh_ratio = float(prof_dict.get("shoulder_width_ratio", 0.65))
    mult = float(prof_dict.get("tolerance_multiplier", 1.0))

    # 1. Base copy
    for joint, (low, high) in base_ranges.items():
        adj_low, adj_high = low, high

        # Adapt knee flexion for longer or shorter legs
        if "knee" in joint:
            if leg_ratio > 1.68 and adj_low < 120.0:  # bent knee target (e.g. 85-115)
                # Expand lower bound to allow deeper safe flexion
                delta = min(8.0, (leg_ratio - 1.65) * 25.0)
                adj_low = max(65.0, adj_low - delta)
            elif leg_ratio < 1.55 and adj_low < 120.0:  # shorter legs
                delta = min(5.0, (1.65 - leg_ratio) * 20.0)
                adj_low = adj_low + delta

        # Adapt shoulder angles for shorter/longer arm reach
        if "shoulder" in joint:
            if arm_ratio < 1.72:
                # Shorter arms: allow slightly more relaxed shoulder angle
                adj_low = max(25.0, adj_low - 4.0)
                adj_high = min(190.0, adj_high + 4.0)

        # Adapt for broad shoulders in tabletop/overhead poses
        if sh_ratio > 0.70 and "shoulder" in joint:
            adj_low = max(30.0, adj_low - 3.0)

        # Apply overall flexibility tolerance multiplier
        if 0.80 <= mult <= 1.35 and mult != 1.0:
            mid = (adj_high + adj_low) / 2.0
            span = (adj_high - adj_low) / 2.0
            adj_low = round(mid - span * mult, 1)
            adj_high = round(mid + span * mult, 1)

        adapted[joint] = (round(adj_low, 1), round(adj_high, 1))

    return adapted


class CalibrationSession:
    """
    Guides the user through a 5-second calibration in Mountain Pose (Tadasana).
    Measures body proportions (thigh-to-torso, leg-to-torso, arm-span),
    quantifies stability variance, and records anthropometric baselines.
    """

    def __init__(
        self,
        duration_seconds: float = 5.0,
        target_frames: int = 35,
        save_path: Path | str = DEFAULT_PROFILE_PATH,
    ):
        self.duration_seconds = duration_seconds
        self.target_frames = target_frames
        self.save_path = Path(save_path)
        self.is_active = False
        self.frame_count = 0
        self.start_time = 0.0
        self._arm_spans: list[float] = []
        self._leg_ratios: list[float] = []
        self._thigh_ratios: list[float] = []
        self._vert_thigh_ratios: list[float] = []
        self._torso_lengths: list[float] = []
        self._shoulder_widths: list[float] = []
        self._knee_angles: list[float] = []
        self.status_message = "Press C to start 5-second calibration."

    def start(self) -> None:
        """Start a new 5-second calibration session."""
        self.is_active = True
        self.frame_count = 0
        self.start_time = time.time()
        self._arm_spans.clear()
        self._leg_ratios.clear()
        self._thigh_ratios.clear()
        self._vert_thigh_ratios.clear()
        self._torso_lengths.clear()
        self._shoulder_widths.clear()
        self._knee_angles.clear()
        self.status_message = f"Calibrating Body Baseline: {self.duration_seconds:.0f}s — Stand tall in Mountain Pose..."

    def cancel(self) -> None:
        self.is_active = False
        self.status_message = "Calibration cancelled."

    def update(
        self,
        landmarks: list[dict],
        width: int,
        height: int,
        feature_dict: Optional[dict] = None,
    ) -> tuple[bool, str, Optional[UserProfile]]:
        """
        Process a single frame during calibration.
        Returns (is_complete, status_message, Optional[UserProfile]).
        """
        if not self.is_active:
            return False, self.status_message, None

        if not landmarks or len(landmarks) < 33:
            return False, "Searching for full body in camera...", None

        # Verify key landmarks visibility
        required_ids = [11, 12, 13, 14, 23, 24, 25, 26, 27, 28]
        if any(float(landmarks[i].get("visibility", 0.0)) < VISIBILITY_THRESHOLD for i in required_ids):
            return False, "Step back: Ensure head, shoulders, hips, and feet are visible.", None

        if self.start_time <= 0.0:
            self.start_time = time.time()

        # Extract 2D pixel coordinates
        pts = {}
        for name, idx in LANDMARK_IDS.items():
            lm = landmarks[idx]
            pts[name] = np.array([float(lm["x"]) * width, float(lm["y"]) * height], dtype=float)

        shoulder_mid = (pts["left_shoulder"] + pts["right_shoulder"]) / 2.0
        hip_mid = (pts["left_hip"] + pts["right_hip"]) / 2.0
        torso_len = float(np.linalg.norm(shoulder_mid - hip_mid))
        if torso_len < 20.0:
            return False, "Torso scale too small. Stand 2 to 3 meters from camera.", None

        # Measurements
        arm_span = float(np.linalg.norm(pts["left_wrist"] - pts["right_wrist"]))
        left_thigh = float(np.linalg.norm(pts["left_hip"] - pts["left_knee"]))
        right_thigh = float(np.linalg.norm(pts["right_hip"] - pts["right_knee"]))
        avg_thigh = (left_thigh + right_thigh) / 2.0

        left_leg = float(
            np.linalg.norm(pts["left_hip"] - pts["left_knee"]) +
            np.linalg.norm(pts["left_knee"] - pts["left_ankle"])
        )
        right_leg = float(
            np.linalg.norm(pts["right_hip"] - pts["right_knee"]) +
            np.linalg.norm(pts["right_knee"] - pts["right_ankle"])
        )
        avg_leg = (left_leg + right_leg) / 2.0
        shoulder_w = float(np.linalg.norm(pts["left_shoulder"] - pts["right_shoulder"]))

        # Vertical displacements in normalized coordinates
        sy = (float(landmarks[11]["y"]) + float(landmarks[12]["y"])) / 2.0
        hy = (float(landmarks[23]["y"]) + float(landmarks[24]["y"])) / 2.0
        ky = (float(landmarks[25]["y"]) + float(landmarks[26]["y"])) / 2.0
        torso_dy = max(0.01, hy - sy)
        thigh_dy = max(0.005, ky - hy)
        vert_thigh_ratio = thigh_dy / torso_dy

        self._arm_spans.append(arm_span / torso_len)
        self._leg_ratios.append(avg_leg / torso_len)
        self._thigh_ratios.append(avg_thigh / torso_len)
        self._vert_thigh_ratios.append(vert_thigh_ratio)
        self._shoulder_widths.append(shoulder_w / torso_len)
        self._torso_lengths.append(torso_len)

        if feature_dict:
            lk = feature_dict.get("left_knee_angle", np.nan)
            if np.isfinite(lk):
                self._knee_angles.append(float(lk))

        self.frame_count += 1
        elapsed = time.time() - self.start_time
        remaining = max(0.0, self.duration_seconds - elapsed)
        pct = min(100, int((elapsed / self.duration_seconds) * 100))
        self.status_message = f"Calibrating: {remaining:.1f}s remaining ({pct}%) — Stand tall in Mountain Pose..."

        if (elapsed >= self.duration_seconds and self.frame_count >= 15) or (self.frame_count >= self.target_frames):
            # Quantify measurement stability across frames
            arm_std = float(np.std(self._arm_spans))
            leg_std = float(np.std(self._leg_ratios))
            sh_std = float(np.std(self._shoulder_widths))
            avg_std = (arm_std + leg_std + sh_std) / 3.0

            # Stability quality score: lower std -> higher stability
            stability = round(max(0.70, min(0.99, 1.0 - avg_std * 2.5)), 2)

            # Derive tolerance multiplier from stability
            if stability >= 0.93:
                tolerance_mult = 0.98   # High stability -> sharp coaching
            elif stability >= 0.85:
                tolerance_mult = 1.05   # Moderate stability -> balanced
            else:
                tolerance_mult = 1.15   # Low stability / jitter -> forgiving

            # Store baseline angles for Tadasana
            pose_baselines: dict[str, dict[str, float]] = {}
            if self._knee_angles:
                pose_baselines["Tadasana"] = {
                    "measured_knee_baseline": round(float(np.median(self._knee_angles)), 1),
                }

            median_thigh_ratio = float(np.median(self._vert_thigh_ratios)) if self._vert_thigh_ratios else float(np.median(self._thigh_ratios))

            profile = UserProfile(
                version=PROFILE_SCHEMA_VERSION,
                calibrated=True,
                user_name="Calibrated Yogi",
                tolerance_multiplier=round(tolerance_mult, 2),
                arm_span_ratio=round(float(np.median(self._arm_spans)), 3),
                leg_to_torso_ratio=round(float(np.median(self._leg_ratios)), 3),
                thigh_to_torso_ratio=round(median_thigh_ratio, 3),
                shoulder_width_ratio=round(float(np.median(self._shoulder_widths)), 3),
                baseline_torso_px=round(float(np.median(self._torso_lengths)), 1),
                measurement_stability=stability,
                calibration_date=time.strftime("%Y-%m-%d %H:%M:%S"),
                pose_baselines=pose_baselines,
                flexibility_tier="standard" if tolerance_mult <= 1.05 else "flexible",
            )
            save_profile(profile, self.save_path)
            self.is_active = False
            self.status_message = "Calibration complete! Body baseline locked."
            return True, self.status_message, profile

        return False, self.status_message, None
