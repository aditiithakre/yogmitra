"""
src/models/sequence_tracker.py
───────────────────────────────
Hierarchical Pose Sequence State Machine & Dynamic Candidate Elimination.

Organizes yoga asanas into base kinematic orientation families and multi-step
temporal state progressions. Dynamically prunes incompatible pose branches
in real time:
  1. Base Kinematic Gating:
     Identifies base support posture (Standing, Seated/Kneeling, Prone, Inverted, Quadruped).
     Immediately prunes 80% of invalid poses from the candidate pool.
  2. Multi-Step State Traversal:
     Tracks progression through steps (Step 1 -> Step 2 -> Step 3 -> Step 4).
     e.g., in Seated/Kneeling:
       • Step 1 & 2: Candidates are [Vajrasana, Parvatasana]
       • Step 3 & 4 (Arms Overhead): Prunes Vajrasana -> Locks on Parvatasana!
  3. Dynamic Backtracking / Auto-Reset:
     When user changes base posture (e.g., stands up from kneeling),
     the candidate tree seamlessly resets to the new family without app restart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np


# ── Canonical Asana Base Orientation Mapping ──────────────────────────────────
BASE_ORIENTATION_MAP: dict[str, str] = {
    # Standing Family
    "Tadasana": "STANDING",
    "Kati Chakrasana": "STANDING",
    "Ardhakati Chakrasana": "STANDING",
    "Virabhadrasana One": "STANDING",
    "Virabhadrasana Two": "STANDING",
    "Virabhadrasana Three": "STANDING",
    "Utkatasana": "STANDING",
    "Trikonasana": "STANDING",
    "Vrksasana": "STANDING",
    "Uttanasana": "STANDING",
    "Utthita Parsvakonasana": "STANDING",
    "Parsva Virabhadrasana": "STANDING",

    # Seated / Kneeling Family
    "Vajrasana": "SEATED_KNEELING",
    "Parvatasana": "SEATED_KNEELING",
    "Padmasana": "SEATED_KNEELING",
    "Sukhasana": "SEATED_KNEELING",
    "Dandasana": "SEATED_KNEELING",

    # Prone Family (Stomach on floor)
    "Bhujangasana": "PRONE",
    "Phalakasana": "PRONE",

    # Inverted / Supine Family
    "Viparita Karani": "INVERTED",
    "Sarvangasana": "INVERTED",
    "Anantasana": "INVERTED",

    # Quadruped Family (All-fours tabletop)
    "Marjariasana": "QUADRUPED",
    "Adho Mukha Svanasana": "QUADRUPED",
}

# Total documented progression steps per Zenodo asana
ASANA_TOTAL_STEPS: dict[str, int] = {
    "Tadasana": 5,
    "Kati Chakrasana": 4,
    "Ardhakati Chakrasana": 5,
    "Vajrasana": 2,
    "Parvatasana": 4,
    "Bhujangasana": 3,
    "Viparita Karani": 2,
    "Sarvangasana": 2,
    "Anantasana": 4,
    "Marjariasana": 4,
}


@dataclass
class SequenceState:
    """Active temporal state output by the sequence tracker."""
    base_posture: str                           # "STANDING" | "SEATED_KNEELING" | "PRONE" | "INVERTED" | "QUADRUPED" | "UNKNOWN"
    active_candidates: list[str] = field(default_factory=list)
    pruned_poses: list[str] = field(default_factory=list)
    step: int = 1                               # 1-indexed current sequence step
    step_name: str = "Base Setup"
    step_total: int = 1
    transition_confidence: float = 1.0


class YogaSequenceTracker:
    """
    Finite State Machine tracking posture progression and eliminating
    incompatible poses step-by-step.
    """

    def __init__(self, debounce_frames: int = 2):
        self._debounce_frames = debounce_frames
        self._pending_base: Optional[str] = None
        self._pending_count: int = 0
        self._current_base: str = "UNKNOWN"
        self._current_step: int = 1
        self._current_step_name: str = "Base Setup"
        self._history_steps: list[int] = []

    def detect_base_posture(
        self,
        landmarks: Optional[list[dict]],
        user_profile: Optional[Any] = None,
    ) -> str:
        """
        Classifies the primary physical support orientation relative to gravity:
          • STANDING: Vertical torso, full vertical leg drop matching calibrated baseline
          • SEATED_KNEELING: Vertical torso, but thighs/legs foreshortened/compressed on floor
          • PRONE: Horizontal torso, stomach toward floor
          • INVERTED: Ankles/knees above hips or shoulders below hips
          • QUADRUPED: Horizontal torso with hands and knees supporting on floor
        """
        if not landmarks or len(landmarks) < 33:
            return "UNKNOWN"

        try:
            sy = (float(landmarks[11].get("y", 0.5)) + float(landmarks[12].get("y", 0.5))) / 2.0
            hy = (float(landmarks[23].get("y", 0.5)) + float(landmarks[24].get("y", 0.5))) / 2.0
            ky = (float(landmarks[25].get("y", 0.5)) + float(landmarks[26].get("y", 0.5))) / 2.0
            ay = (float(landmarks[27].get("y", 0.5)) + float(landmarks[28].get("y", 0.5))) / 2.0
            sx = (float(landmarks[11].get("x", 0.5)) + float(landmarks[12].get("x", 0.5))) / 2.0
            hx = (float(landmarks[23].get("x", 0.5)) + float(landmarks[24].get("x", 0.5))) / 2.0
            wy = (float(landmarks[15].get("y", 0.5)) + float(landmarks[16].get("y", 0.5))) / 2.0

            ankles_vis = max(float(landmarks[27].get("visibility", 0.0)), float(landmarks[28].get("visibility", 0.0)))

            dy_torso = hy - sy
            dx_torso = abs(hx - sx)
            leg_dy = ay - hy
            shin_dy = ay - ky
            ratio_leg = leg_dy / max(0.01, dy_torso)

            # Check Calibrated Profile
            is_calibrated = False
            baseline_thigh_ratio = 0.85
            baseline_leg_ratio = 1.25

            if user_profile is not None:
                if hasattr(user_profile, "calibrated") and user_profile.calibrated:
                    is_calibrated = True
                    baseline_thigh_ratio = float(getattr(user_profile, "thigh_to_torso_ratio", 0.85))
                    baseline_leg_ratio = float(getattr(user_profile, "leg_to_torso_ratio", 1.25))
                elif isinstance(user_profile, dict) and user_profile.get("calibrated", False):
                    is_calibrated = True
                    baseline_thigh_ratio = float(user_profile.get("thigh_to_torso_ratio", 0.85))
                    baseline_leg_ratio = float(user_profile.get("leg_to_torso_ratio", 1.25))

            # 1. INVERTED / SUPINE
            if (ay < hy - 0.08 and ky < hy - 0.04) or (sy > hy + 0.06):
                return "INVERTED"

            # 2. PRONE vs QUADRUPED (Horizontal spine)
            is_horizontal_torso = abs(dy_torso) < 0.10 and dx_torso > 0.08
            if is_horizontal_torso:
                # Quadruped tabletop has hands down (wy > sy + 0.12) and knees down (ky > hy + 0.08)
                if wy > sy + 0.10 and ky > hy + 0.06:
                    return "QUADRUPED"
                if ay > sy - 0.15 and ky > sy - 0.15:
                    return "PRONE"

            # 3. UPRIGHT TORSO
            if dy_torso > 0.07 and (dx_torso / max(0.01, dy_torso) < 0.95):
                live_thigh_dy = max(0.0, ky - hy)
                live_thigh_ratio = live_thigh_dy / max(0.01, dy_torso)
                live_leg_ratio = ratio_leg

                if is_calibrated:
                    is_seated = (
                        (live_thigh_ratio < 0.55 * baseline_thigh_ratio)
                        or (live_leg_ratio < 0.55 * baseline_leg_ratio)
                        or (shin_dy < 0.045)
                        or (ay <= ky + 0.04)
                        or (abs(ky - hy) < 0.08 and abs(ay - hy) < 0.12)
                        or (ankles_vis < 0.30 and ky >= 0.60 and hy >= 0.48)
                    )
                else:
                    is_seated = (
                        (ratio_leg < 0.60)
                        or (shin_dy < 0.045)
                        or (ay <= ky + 0.04)
                        or (abs(ky - hy) < 0.08 and abs(ay - hy) < 0.12)
                        or (ankles_vis < 0.30 and ky >= 0.60 and hy >= 0.48)
                    )

                return "SEATED_KNEELING" if is_seated else "STANDING"

            return "UNKNOWN"
        except Exception:
            return "UNKNOWN"

    def update(
        self,
        landmarks: Optional[list[dict]],
        feature_dict: Optional[dict] = None,
        user_profile: Optional[Any] = None,
        available_classes: Optional[list[str]] = None,
    ) -> SequenceState:
        """
        Evaluate live landmarks and compute the active sequence state,
        pruning incompatible asanas dynamically.
        """
        all_classes = available_classes or list(BASE_ORIENTATION_MAP.keys())
        raw_base = self.detect_base_posture(landmarks, user_profile)

        # ── Debounce base orientation to avoid transit flicker ───────────────
        if raw_base != self._pending_base:
            self._pending_base = raw_base
            self._pending_count = 1
        else:
            self._pending_count += 1

        if self._pending_count >= self._debounce_frames and self._pending_base != "UNKNOWN":
            if self._current_base != self._pending_base:
                self._current_base = self._pending_base
                # Reset step progression on base family switch
                self._current_step = 1
                self._current_step_name = "Base Posture"

        active_base = self._current_base if self._current_base != "UNKNOWN" else raw_base

        # ── Step 1: Base Kinematic Gating ─────────────────────────────────────
        if active_base != "UNKNOWN":
            base_candidates = [
                c for c in all_classes
                if BASE_ORIENTATION_MAP.get(c, "STANDING") == active_base
            ]
            pruned = [c for c in all_classes if c not in base_candidates]
        else:
            base_candidates = list(all_classes)
            pruned = []

        # ── Step 2: Multi-Step Sequence Elimination Within Family ─────────────
        step = 1
        step_name = "Base Posture"
        step_total = 1
        active_candidates = list(base_candidates)

        if landmarks and len(landmarks) >= 33:
            try:
                sy = (float(landmarks[11].get("y", 0.5)) + float(landmarks[12].get("y", 0.5))) / 2.0
                hy = (float(landmarks[23].get("y", 0.5)) + float(landmarks[24].get("y", 0.5))) / 2.0
                ky = (float(landmarks[25].get("y", 0.5)) + float(landmarks[26].get("y", 0.5))) / 2.0
                ay = (float(landmarks[27].get("y", 0.5)) + float(landmarks[28].get("y", 0.5))) / 2.0
                sx = (float(landmarks[11].get("x", 0.5)) + float(landmarks[12].get("x", 0.5))) / 2.0
                hx = (float(landmarks[23].get("x", 0.5)) + float(landmarks[24].get("x", 0.5))) / 2.0
                wy = (float(landmarks[15].get("y", 0.5)) + float(landmarks[16].get("y", 0.5))) / 2.0

                dy_torso = hy - sy
                dx_torso = abs(hx - sx)
                l_sh_angle = feature_dict.get("left_shoulder_angle", 0.0) if feature_dict else 0.0
                r_sh_angle = feature_dict.get("right_shoulder_angle", 0.0) if feature_dict else 0.0
                max_sh_angle = max(l_sh_angle or 0.0, r_sh_angle or 0.0)
                torso_twist = abs(float(landmarks[11].get("x", 0.5)) - float(landmarks[23].get("x", 0.5)))

                # Family A: SEATED / KNEELING ([Vajrasana, Parvatasana])
                if active_base == "SEATED_KNEELING":
                    step_total = 4
                    arms_overhead = (wy < sy - 0.02)
                    arms_shoulder_level = (wy < sy + 0.08) and not arms_overhead

                    if arms_overhead:
                        # Step 3 or 4: Parvatasana ONLY (Vajrasana has no overhead state)
                        step = 3 if wy > sy - 0.10 else 4
                        step_name = "Arms Raised Overhead" if step == 3 else "Palms Interlocked Overhead"
                        if "Vajrasana" in active_candidates:
                            active_candidates.remove("Vajrasana")
                            pruned.append("Vajrasana")
                    elif arms_shoulder_level:
                        # Step 2: Arms preparing / outstretched at shoulder level
                        step = 2
                        step_name = "Arms at Shoulder Height"
                    else:
                        # Step 1: Hands resting on knees/thighs
                        step = 1
                        step_name = "Hands on Knees / Neutral"

                # Family B: STANDING ([Tadasana, Kati Chakrasana, Ardhakati Chakrasana])
                elif active_base == "STANDING":
                    step_total = 4
                    torso_lean = feature_dict.get("torso_lean_from_vertical_deg", 0.0) if feature_dict else 0.0
                    is_lateral_bend = torso_lean is not None and torso_lean > 14.0
                    is_twist = torso_twist > 0.06 or (abs((l_sh_angle or 0.0) - (r_sh_angle or 0.0)) > 45.0)

                    one_arm_overhead = (
                        (float(landmarks[15].get("y", 0.5)) < sy - 0.02 and float(landmarks[16].get("y", 0.5)) > sy) or
                        (float(landmarks[16].get("y", 0.5)) < sy - 0.02 and float(landmarks[15].get("y", 0.5)) > sy)
                    )

                    if is_lateral_bend or one_arm_overhead:
                        # Branch: Ardhakati Chakrasana
                        step = 3 if is_lateral_bend else 2
                        step_name = "Lateral Side Bend" if is_lateral_bend else "One Arm Raised Overhead"
                        active_candidates = [c for c in active_candidates if c == "Ardhakati Chakrasana"]
                        pruned.extend([c for c in ("Tadasana", "Kati Chakrasana") if c not in pruned])
                    elif is_twist:
                        # Branch: Kati Chakrasana
                        step = 3
                        step_name = "Spinal Torso Twist"
                        active_candidates = [c for c in active_candidates if c == "Kati Chakrasana"]
                        pruned.extend([c for c in ("Tadasana", "Ardhakati Chakrasana") if c not in pruned])
                    elif wy < sy + 0.08:
                        # Step 2: Arms raised forward / horizontal
                        step = 2
                        step_name = "Arms at Shoulder Level"
                    else:
                        # Step 1: Standing tall, arms relaxed by sides
                        step = 1
                        step_name = "Standing Neutral"

                # Family C: PRONE ([Bhujangasana])
                elif active_base == "PRONE":
                    step_total = 3
                    elbow_angle = feature_dict.get("left_elbow_angle", 180.0) if feature_dict else 180.0
                    if dy_torso < -0.06:
                        step = 3
                        step_name = "Full Chest Lift (Cobra)"
                    elif wy < sy + 0.15:
                        step = 2
                        step_name = "Palms Grounded by Ribs"
                    else:
                        step = 1
                        step_name = "Prone Alignment on Mat"

                # Family D: INVERTED ([Viparita Karani, Sarvangasana, Anantasana])
                elif active_base == "INVERTED":
                    step_total = 2
                    step = 2 if ay < hy - 0.12 else 1
                    step_name = "Vertical Leg Extension" if step == 2 else "Inverted Pelvis Lift"

                # Family E: QUADRUPED ([Marjariasana])
                elif active_base == "QUADRUPED":
                    step_total = 4
                    step = 2
                    step_name = "Tabletop Position"
            except Exception:
                step = 1
                step_name = "Base Posture"

        self._current_step = step
        self._current_step_name = step_name

        return SequenceState(
            base_posture=active_base,
            active_candidates=active_candidates,
            pruned_poses=list(set(pruned)),
            step=step,
            step_name=step_name,
            step_total=step_total,
            transition_confidence=1.0,
        )

    def filter_probabilities(
        self,
        probabilities: np.ndarray,
        class_names: list[str],
        state: SequenceState,
        suppression: float = 0.001,
    ) -> np.ndarray:
        """
        Applies dynamic candidate elimination by zeroing out pruned classes
        and renormalizing the active probability vector.
        """
        filtered = probabilities.copy()

        # Zero out completely pruned poses
        for idx, name in enumerate(class_names):
            if name in state.pruned_poses:
                filtered[idx] *= suppression

        p_sum = float(np.sum(filtered))
        if p_sum > 1e-6:
            filtered /= p_sum

        return filtered

    def reset(self) -> None:
        """Reset internal state tracker."""
        self._pending_base = None
        self._pending_count = 0
        self._current_base = "UNKNOWN"
        self._current_step = 1
        self._current_step_name = "Base Setup"
        self._history_steps.clear()
