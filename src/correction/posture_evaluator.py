"""
src/correction/posture_evaluator.py
─────────────────────────────────────
Evaluates posture alignment, calculates multi-tier reliability,
determines deviation severity and natural directional action hints,
and computes a composite 0–100% alignment score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from src.correction.reference_ranges import REFERENCE_RANGES, JOINT_DISPLAY_NAMES
from src.features import ANGLE_TRIPLETS
from src.pose.missing_landmarks import VISIBILITY_THRESHOLD

TIER_RANK = {"low": 0, "medium": 1, "high": 2}


def min_tier(*tiers: Optional[str]) -> str:
    """Most conservative of the given reliability tiers (None entries ignored)."""
    valid = [t for t in tiers if t in TIER_RANK]
    return min(valid, key=TIER_RANK.get) if valid else "low"


@dataclass
class AngleCheck:
    angle_name: str
    display_name: str
    value: Optional[float]              # Measured angle in degrees (None if unavailable)
    target_range: tuple[float, float]   # Expected (min_deg, max_deg)
    status: str                         # "ok" | "too_low" | "too_high" | "unavailable"
    deviation: float                    # Absolute degrees outside range (0.0 if ok)
    severity: str                       # "none" | "mild" | "moderate" | "major"
    direction: str                      # "ok" | "bend" | "straighten" | "raise" | "lower" | "align"
    action_hint: str                    # Short actionable natural English correction
    reliable: bool                      # True if landmarks meet visibility requirements
    reliability_tier: str               # "high" | "medium" | "low"

    @property
    def joint_name(self) -> str:
        return self.display_name

    @property
    def deviation_magnitude(self) -> float:
        return self.deviation


PostureDeviation = AngleCheck


@dataclass
class PostureEvaluation:
    pose_name: str
    checks: list[AngleCheck]
    imputed_fraction: float
    has_reference: bool
    overall_status: str                 # "correct" | "needs_work" | "no_reference" | "low_visibility"
    alignment_score: float              # 0.0 to 100.0 percentage score
    primary_check: Optional[AngleCheck] # The most critical deviation requiring correction


# ── Directional action generator ──────────────────────────────────────────────

def _derive_action_hint(
    angle_name: str,
    display_name: str,
    status: str,
    deviation: float,
    severity: str,
    is_front_knee: bool = False,
    is_rear_knee: bool = False,
) -> tuple[str, str]:
    """
    Returns (direction, action_hint) with human-coach natural phrasing.
    """
    if status == "ok":
        return "ok", f"{display_name.capitalize()} is well aligned."

    if status == "unavailable":
        return "align", f"Bring your {display_name} clearly into camera view."

    # Specific limb heuristics
    if is_front_knee:
        if status == "too_high":  # angle is open (> 115°), knee needs more bend
            direction = "bend"
            if severity == "mild":
                hint = "Bend your front knee a little more."
            else:
                hint = "Bend your front knee deeper toward 90 degrees."
            return direction, hint
        else:  # angle is acute (< 85°), knee bent too much forward
            direction = "straighten"
            return direction, "Ease back your front knee so it stays over your ankle."

    if is_rear_knee:
        direction = "straighten"
        return direction, "Keep your rear knee and leg straight."

    if "elbow" in angle_name or "arm" in display_name:
        if status == "too_low":
            return "straighten", "Extend and straighten your arms fully."
        else:
            return "bend", "Avoid locking your elbows completely."

    if "shoulder" in angle_name:
        if status == "too_low":
            return "raise", "Raise your arms slightly higher."
        else:
            return "lower", "Relax your shoulders away from your ears."

    if "hip" in angle_name:
        if status == "too_low":
            return "straighten", "Open and lengthen through your hips."
        else:
            return "bend", "Hinge comfortably at the hips."

    # General fallback
    if status == "too_low":
        return "straighten", f"Straighten your {display_name} slightly."
    else:
        return "bend", f"Bend your {display_name} a little more."


# ── Bilateral Asymmetric Lunge Adaptation ────────────────────────────────────

def get_adapted_ranges(
    pose_name: str,
    feature_dict: dict,
    user_profile: Optional[dict] = None,
) -> tuple[dict[str, tuple[float, float]], dict[str, str]]:
    """
    Return reference angle ranges adapted to the user's active side and
    optional personalized calibration tolerances.
    """
    base = REFERENCE_RANGES.get(pose_name)
    if not base:
        return {}, {}

    ranges = dict(base)
    role_map: dict[str, str] = {}  # angle_name -> role description (e.g. "front knee")

    # Dynamic leading leg detection for asymmetric lunges
    lunge_poses = (
        "Virabhadrasana One",
        "Virabhadrasana Two",
        "Utthita Parsvakonasana",
        "Anjaneyasana",
        "Parsva Virabhadrasana",
    )
    if pose_name in lunge_poses:
        lk = feature_dict.get("left_knee_angle", np.nan)
        rk = feature_dict.get("right_knee_angle", np.nan)
        if np.isfinite(lk) and np.isfinite(rk) and rk < lk:
            ranges["right_knee_angle"] = (85, 115)
            ranges["left_knee_angle"] = (160, 185)
            role_map["right_knee_angle"] = "front knee"
            role_map["left_knee_angle"] = "rear leg"
        else:
            ranges["left_knee_angle"] = (85, 115)
            ranges["right_knee_angle"] = (160, 185)
            role_map["left_knee_angle"] = "front knee"
            role_map["right_knee_angle"] = "rear leg"

    elif pose_name == "Vrksasana":  # Tree Pose
        lk = feature_dict.get("left_knee_angle", np.nan)
        rk = feature_dict.get("right_knee_angle", np.nan)
        if np.isfinite(lk) and np.isfinite(rk) and rk < lk:
            ranges["right_knee_angle"] = (35, 75)
            ranges["left_knee_angle"] = (160, 185)
            role_map["right_knee_angle"] = "raised foot knee"
            role_map["left_knee_angle"] = "standing leg"
        else:
            ranges["left_knee_angle"] = (35, 75)
            ranges["right_knee_angle"] = (160, 185)
            role_map["left_knee_angle"] = "raised foot knee"
            role_map["right_knee_angle"] = "standing leg"

    # Zenodo bilateral poses:
    elif pose_name == "Ardhakati Chakrasana":  # Lateral Side Bend
        ls = feature_dict.get("left_shoulder_angle", np.nan)
        rs = feature_dict.get("right_shoulder_angle", np.nan)
        if np.isfinite(ls) and np.isfinite(rs) and rs > ls:
            role_map["right_shoulder_angle"] = "raised arm"
            role_map["left_shoulder_angle"] = "lowered arm"
            ranges["right_shoulder_angle"] = (150, 185)
            ranges["left_shoulder_angle"] = (20, 60)
        elif np.isfinite(ls) and np.isfinite(rs) and ls > rs:
            role_map["left_shoulder_angle"] = "raised arm"
            role_map["right_shoulder_angle"] = "lowered arm"
            ranges["left_shoulder_angle"] = (150, 185)
            ranges["right_shoulder_angle"] = (20, 60)

    elif pose_name == "Anantasana":  # Side-Reclining Sleeping Vishnu
        lk = feature_dict.get("left_knee_angle", np.nan)
        rk = feature_dict.get("right_knee_angle", np.nan)
        lh = feature_dict.get("left_hip_angle", np.nan)
        rh = feature_dict.get("right_hip_angle", np.nan)
        if np.isfinite(lh) and np.isfinite(rh) and lh < rh:
            role_map["left_knee_angle"] = "raised leg knee"
            role_map["right_knee_angle"] = "bottom leg knee"
        else:
            role_map["right_knee_angle"] = "raised leg knee"
            role_map["left_knee_angle"] = "bottom leg knee"

    # Apply measurable biomechanical personalization from user profile
    if user_profile:
        from src.correction.calibration import apply_biomechanical_personalization
        ranges = apply_biomechanical_personalization(ranges, user_profile, pose_name)

    return ranges, role_map


# ── Main Evaluator ────────────────────────────────────────────────────────────

def evaluate(
    pose_name: str,
    feature_dict: dict,
    landmarks: list[dict],
    imputed_flags: list[int],
    imputed_fraction_limit: float = 0.35,
    user_profile: Optional[dict] = None,
    reliability_assessment: Optional[Any] = None,
    decision_tier: str = "high",
    max_unavailable_fraction: float = 0.5,
) -> PostureEvaluation:
    """
    Evaluates posture against biomechanical targets with multi-tier reliability,
    severity categorization, temporal jitter suppression, and composite 0-100% alignment scoring.

    Each check's reliability tier is the *minimum* of:
      • landmark visibility of the angle's three joints,
      • imputation (an estimated joint can never be "high"),
      • the temporal reliability tracker (jitter / recent imputation), and
      • ``decision_tier`` – how certain the pose being assessed is
        (e.g. "medium" when the auto-detected label is only moderately confident).
    A "low" tier makes the check unavailable: no angle-specific correction is issued.
    """
    imputed_fraction = float(np.mean(imputed_flags)) if imputed_flags else 0.0

    if imputed_fraction > imputed_fraction_limit:
        return PostureEvaluation(
            pose_name=pose_name,
            checks=[],
            imputed_fraction=imputed_fraction,
            has_reference=pose_name in REFERENCE_RANGES,
            overall_status="low_visibility",
            alignment_score=0.0,
            primary_check=None,
        )

    ranges, role_map = get_adapted_ranges(pose_name, feature_dict, user_profile)
    if not ranges:
        return PostureEvaluation(
            pose_name=pose_name,
            checks=[],
            imputed_fraction=imputed_fraction,
            has_reference=False,
            overall_status="no_reference",
            alignment_score=0.0,
            primary_check=None,
        )

    checks: list[AngleCheck] = []
    any_issues = False
    valid_scores: list[float] = []

    for angle_name, (low, high) in ranges.items():
        role = role_map.get(angle_name)
        display = role if role else JOINT_DISPLAY_NAMES.get(angle_name, angle_name.replace("_", " "))

        # ── Landmark Reliability Tier ─────────────────────────────────────────
        triplet_ids = ANGLE_TRIPLETS.get(angle_name)
        tier = "high"
        reliable = True

        if triplet_ids:
            vis_scores = []
            for lm_id in triplet_ids:
                if lm_id >= len(landmarks):
                    vis_scores.append(0.0)
                else:
                    vis_scores.append(float(landmarks[lm_id].get("visibility", 0.0)))

            min_vis = min(vis_scores) if vis_scores else 0.0
            if min_vis < VISIBILITY_THRESHOLD:
                tier = "low"
                reliable = False
            elif min_vis < 0.50:
                tier = "medium"
                reliable = True
            else:
                tier = "high"
                reliable = True

        # Check imputation for this angle's joints
        if triplet_ids and imputed_flags:
            if any(lm_id < len(imputed_flags) and imputed_flags[lm_id] for lm_id in triplet_ids):
                if tier == "high":
                    tier = "medium"

        # Combine with the stateful temporal reliability tracker (never upgrade a tier)
        if reliability_assessment and hasattr(reliability_assessment, "joint_assessments"):
            j_rel = reliability_assessment.joint_assessments.get(angle_name)
            if j_rel:
                tier = min_tier(tier, j_rel.tier)

        tier = min_tier(tier, decision_tier)
        reliable = tier != "low"

        # ── Angle Value ───────────────────────────────────────────────────────
        value_raw = feature_dict.get(angle_name, np.nan)
        try:
            value = float(value_raw)
        except (TypeError, ValueError):
            value = np.nan

        if not reliable or not np.isfinite(value):
            action_hint = f"Keep your {display} clearly visible and steady."

            checks.append(AngleCheck(
                angle_name=angle_name,
                display_name=display,
                value=None,
                target_range=(low, high),
                status="unavailable",
                deviation=0.0,
                severity="none",
                direction="align",
                action_hint=action_hint,
                reliable=False,
                reliability_tier="low",
            ))
            continue

        # ── Compare Against Target Range ──────────────────────────────────────
        is_front = (role == "front knee")
        is_rear = (role == "rear leg")

        if value < low:
            status = "too_low"
            deviation = low - value
            any_issues = True
        elif value > high:
            status = "too_high"
            deviation = value - high
            any_issues = True
        else:
            status = "ok"
            deviation = 0.0

        # Severity
        if deviation == 0.0:
            severity = "none"
        elif deviation <= 12.0:
            severity = "mild"
        elif deviation <= 25.0:
            severity = "moderate"
        else:
            severity = "major"

        # Action hint
        direction, action_hint = _derive_action_hint(
            angle_name=angle_name,
            display_name=display,
            status=status,
            deviation=deviation,
            severity=severity,
            is_front_knee=is_front,
            is_rear_knee=is_rear,
        )

        # Joint alignment score contribution (100 down to 0)
        deduction = min(100.0, deviation * 2.0)
        check_score = max(0.0, 100.0 - deduction)
        valid_scores.append(check_score)

        checks.append(AngleCheck(
            angle_name=angle_name,
            display_name=display,
            value=round(value, 1),
            target_range=(low, high),
            status=status,
            deviation=round(deviation, 1),
            severity=severity,
            direction=direction,
            action_hint=action_hint,
            reliable=True,
            reliability_tier=tier,
        ))

    # Overall alignment score
    if valid_scores:
        overall_score = round(float(np.mean(valid_scores)), 1)
    else:
        overall_score = 0.0

    # Pick primary check: highest deviation among reliable checks
    deviant_checks = [c for c in checks if c.status != "ok" and c.reliable]
    deviant_checks.sort(key=lambda c: -c.deviation)
    primary_check = deviant_checks[0] if deviant_checks else None

    overall = "needs_work" if any_issues else "correct"

    # Most required joints unmeasurable → do not present a partial assessment as a verdict
    n_unavail = sum(1 for c in checks if c.status == "unavailable")
    if checks and n_unavail / len(checks) >= max_unavailable_fraction:
        overall = "low_visibility"
        primary_check = None

    return PostureEvaluation(
        pose_name=pose_name,
        checks=checks,
        imputed_fraction=imputed_fraction,
        has_reference=True,
        overall_status=overall,
        alignment_score=overall_score,
        primary_check=primary_check,
    )
