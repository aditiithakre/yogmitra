"""
src/correction/feedback_engine.py
───────────────────────────────────
Intelligent feedback engine delivering natural coach-style audio cues,
multi-tier confidence gating, per-joint cooldowns, and jitter dampening.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from src.correction.posture_evaluator import PostureEvaluation, AngleCheck
from src.models.inference import PosePrediction


@dataclass
class FeedbackItem:
    pose_name: str
    body_part: str                # e.g. "front knee", "rear leg", "arms"
    issue: str                    # e.g. "too_low", "too_high", "low_confidence"
    instruction: str              # Concise natural English sentence for display & speech
    severity: str                 # "info" | "warning" | "correction"
    reliable: bool                # Based on visible landmarks
    confidence_ok: bool           # Prediction confidence above threshold
    reliability_tier: str = "high"# "high" | "medium" | "low"


class FeedbackEngine:
    """
    Translates PostureEvaluation into prioritized feedback items and audio cues.

    Features:
      • Per-joint cooldowns (prevents repeating the same joint cue within 7s)
      • Instant announcement when a different joint deviates
      • Minimum error persistence (dampens single-frame tracking noise)
      • Obsolete queue flushing when posture updates
      • Multi-tier reliability suppression of false corrections
    """

    def __init__(
        self,
        cooldown_seconds: float = 7.0,
        max_display: int = 4,
        min_persistence_frames: int = 2,
    ):
        self._cooldown = cooldown_seconds
        self._max_display = max_display
        self._min_persistence = min_persistence_frames

        # Tracking state
        self._last_spoken_by_joint: dict[str, float] = {}  # joint_key -> timestamp
        self._last_spoken_time: float = 0.0                # global pacing
        self._consecutive_error_counts: dict[str, int] = {}# joint_key -> frames seen
        self._speech_queue: list[str] = []
        self._current_items: list[FeedbackItem] = []

    # ── Main Update ───────────────────────────────────────────────────────────

    def update(
        self,
        prediction: PosePrediction,
        evaluation: Optional[PostureEvaluation],
        target_pose: Optional[str] = None,
        decision: Optional[Any] = None,
    ) -> list[FeedbackItem]:
        """
        Processes prediction and posture evaluation.
        Returns a list of prioritized FeedbackItems for HUD and enqueues audio.

        *decision* (a ``PoseDecision`` from ``src.models.jev_fusion``) states whether a
        pose-specific assessment is allowed.  When it is not, only a neutral
        "uncertain" message with the reason is produced – never a joint correction.
        """
        items: list[FeedbackItem] = []
        now = time.monotonic()

        # ── 1. No Person Detected ─────────────────────────────────────────────
        if prediction.status == "no_person":
            self._consecutive_error_counts.clear()
            msg = "No person detected. Please stand in front of the camera."
            items.append(FeedbackItem(
                pose_name="",
                body_part="",
                issue="no_person",
                instruction=msg,
                severity="info",
                reliable=False,
                confidence_ok=False,
                reliability_tier="low",
            ))
            self._enqueue_speech("system", msg, min_interval=6.0)
            self._current_items = items
            return items

        # ── 1b. Pose assessment not available (uncertain label / poor tracking) ─
        if decision is not None and not getattr(decision, "assessment_allowed", True):
            self._consecutive_error_counts.clear()
            reason = getattr(decision, "user_message", None) or "Pose not confirmed yet. Hold the pose with your full body in view."
            items.append(FeedbackItem(
                pose_name=target_pose or "",
                body_part="",
                issue="uncertain",
                instruction=reason,
                severity="warning",
                reliable=False,
                confidence_ok=False,
                reliability_tier="low",
            ))
            self._enqueue_speech("uncertain", reason, min_interval=max(self._cooldown, 8.0))
            self._current_items = items
            return items

        # ── 2. Severe Low Visibility / Occlusion ──────────────────────────────
        if evaluation and evaluation.overall_status == "low_visibility":
            self._consecutive_error_counts.clear()
            msg = "Full body not clearly visible. Step back so your legs and feet are in view."
            items.append(FeedbackItem(
                pose_name=target_pose or prediction.pose_name,
                body_part="body",
                issue="low_visibility",
                instruction=msg,
                severity="warning",
                reliable=False,
                confidence_ok=True,
                reliability_tier="low",
            ))
            self._enqueue_speech("visibility", msg, min_interval=8.0)
            self._current_items = items
            return items

        # ── 3. Target Pose Guided Mode ────────────────────────────────────────
        if target_pose and evaluation and evaluation.has_reference:
            active_pose = target_pose
            corrections = [c for c in evaluation.checks if c.status != "ok"]

            if not corrections:
                self._consecutive_error_counts.clear()
                congrats = f"Good form! {active_pose} posture looks correct."
                items.append(FeedbackItem(
                    pose_name=active_pose,
                    body_part="",
                    issue="perfect",
                    instruction=congrats,
                    severity="info",
                    reliable=True,
                    confidence_ok=True,
                    reliability_tier="high",
                ))
                self._enqueue_speech(f"{active_pose}_ok", congrats, min_interval=12.0)
            else:
                # Sort: major severity first, then deviation magnitude
                severity_rank = {"major": 3, "moderate": 2, "mild": 1, "none": 0}
                corrections.sort(key=lambda c: (-severity_rank.get(c.severity, 0), -c.deviation))

                for check in corrections[: self._max_display]:
                    items.append(self._correction_item(active_pose, check))

                # Primary audio guidance from top deviant check
                self._speak_top_correction(active_pose, corrections[0])

            self._current_items = items
            return items

        # ── 4. Free Auto-Detect Mode: Low Confidence ──────────────────────────
        if prediction.status == "low_confidence":
            self._consecutive_error_counts.clear()
            msg = f"Detecting pose ({prediction.confidence*100:.0f}%). Hold your posture clearly."
            items.append(FeedbackItem(
                pose_name=prediction.pose_name,
                body_part="",
                issue="low_confidence",
                instruction=msg,
                severity="warning",
                reliable=False,
                confidence_ok=False,
                reliability_tier="medium",
            ))
            self._enqueue_speech("low_confidence", msg, min_interval=self._cooldown)
            self._current_items = items
            return items

        # ── 5. Free Auto-Detect Mode: Pose Recognized Without Rules ───────────
        if evaluation is None or not evaluation.has_reference:
            self._consecutive_error_counts.clear()
            msg = f"Pose detected: {prediction.pose_name}."
            items.append(FeedbackItem(
                pose_name=prediction.pose_name,
                body_part="",
                issue="no_reference",
                instruction=msg,
                severity="info",
                reliable=True,
                confidence_ok=True,
                reliability_tier="high",
            ))
            self._current_items = items
            return items

        # ── 6. Free Auto-Detect Mode: Evaluated Corrections ───────────────────
        corrections = [c for c in evaluation.checks if c.status != "ok"]
        if not corrections:
            self._consecutive_error_counts.clear()
            congrats = f"Good form! {prediction.pose_name} alignment looks correct."
            items.append(FeedbackItem(
                pose_name=prediction.pose_name,
                body_part="",
                issue="perfect",
                instruction=congrats,
                severity="info",
                reliable=True,
                confidence_ok=True,
                reliability_tier="high",
            ))
            self._enqueue_speech(f"{prediction.pose_name}_ok", congrats, min_interval=12.0)
        else:
            severity_rank = {"major": 3, "moderate": 2, "mild": 1, "none": 0}
            corrections.sort(key=lambda c: (-severity_rank.get(c.severity, 0), -c.deviation))

            for check in corrections[: self._max_display]:
                items.append(self._correction_item(prediction.pose_name, check))

            self._speak_top_correction(prediction.pose_name, corrections[0])

        self._current_items = items
        return items

    # ── Correction wording & speech policy by reliability tier ───────────────
    #   high   → specific joint-angle guidance, spoken after min_persistence inferences
    #   medium → qualified wording, spoken only after one extra confirming inference
    #   low    → never spoken; shown as a "make this joint visible" hint only

    @staticmethod
    def _qualified(check: AngleCheck) -> str:
        if check.status == "unavailable" or not check.reliable:
            return check.action_hint
        if check.reliability_tier == "medium":
            return f"Possibly: {check.action_hint[0].lower()}{check.action_hint[1:]}"
        return check.action_hint

    def _correction_item(self, pose_name: str, check: AngleCheck) -> FeedbackItem:
        return FeedbackItem(
            pose_name=pose_name,
            body_part=check.display_name,
            issue=check.status,
            instruction=self._qualified(check),
            severity="correction" if (check.reliable and check.reliability_tier == "high") else "warning",
            reliable=check.reliable,
            confidence_ok=True,
            reliability_tier=check.reliability_tier,
        )

    def _speak_top_correction(self, pose_name: str, top_check: AngleCheck) -> None:
        joint_key = f"{pose_name}_{top_check.angle_name}"
        self._consecutive_error_counts[joint_key] = self._consecutive_error_counts.get(joint_key, 0) + 1
        if not top_check.reliable or top_check.status == "unavailable":
            return
        needed = self._min_persistence + (1 if top_check.reliability_tier == "medium" else 0)
        if top_check.reliability_tier in ("high", "medium") and self._consecutive_error_counts[joint_key] >= needed:
            phrase = top_check.action_hint
            if top_check.reliability_tier == "medium":
                phrase = f"If you can, {phrase[0].lower()}{phrase[1:]}"
            self._enqueue_speech(top_check.angle_name, phrase, min_interval=self._cooldown)

    # ── Jev-primary coaching ──────────────────────────────────────────────────

    def update_from_jev(
        self,
        decision: Any,
        coaching: Any,
        fresh: bool,
        min_confidence: float = 0.6,
    ) -> list[FeedbackItem]:
        """
        Build feedback from Jev's checkpoint answers (``JevCoaching``).

        ``fresh`` is True only for a newly arrived Jev answer: persistence counts and
        speech advance once per answer, not once per display refresh.  A correction is
        spoken after it appears in ``min_persistence`` consecutive answers.
        """
        pose = getattr(decision, "final_pose", None)
        if decision is None or not decision.assessment_allowed or not pose:
            self._current_items = []
            return self._current_items
        if coaching is None or getattr(coaching, "pose", None) != pose:
            self._current_items = [FeedbackItem(pose, "pose", "checking", f"{pose}: checking your form…",
                                                "info", True, True, decision.tier)]
            return self._current_items

        needed = sorted((c for c in coaching.corrections if c.needs_change and c.confidence >= min_confidence),
                        key=lambda c: -c.confidence)
        items = [FeedbackItem(pose, c.body_part, c.choice, c.instruction, "correction", True, True, decision.tier)
                 for c in needed]
        if not items:
            items = [FeedbackItem(pose, "pose", "ok", "Good form! Hold the pose and breathe steadily.",
                                  "info", True, True, decision.tier)]
        self._current_items = items[: self._max_display]

        if fresh:
            keys = {f"jev_{pose}_{c.checkpoint}_{c.choice}" for c in needed}
            for k in [k for k in self._consecutive_error_counts if k.startswith("jev_") and k not in keys]:
                del self._consecutive_error_counts[k]
            for k in keys:
                self._consecutive_error_counts[k] = self._consecutive_error_counts.get(k, 0) + 1
            if needed:
                top = needed[0]
                k = f"jev_{pose}_{top.checkpoint}_{top.choice}"
                if self._consecutive_error_counts[k] >= self._min_persistence:
                    self._enqueue_speech(f"jev_{top.checkpoint}", top.instruction, min_interval=self._cooldown)
            elif coaching.corrections:
                self._enqueue_speech("jev_good_form", "Good. Hold the pose.", min_interval=self._cooldown * 3)
        return self._current_items

    def announce(self, key: str, phrase: str, min_interval: float) -> None:
        """Queue a one-off phrase (e.g. a newly detected pose) under its own cooldown."""
        self._enqueue_speech(key, phrase, min_interval)

    # ── Speech Queue Access ───────────────────────────────────────────────────

    def pop_speech(self) -> Optional[str]:
        """Pop and return the next message to speak, or None."""
        return self._speech_queue.pop(0) if self._speech_queue else None

    @property
    def current_items(self) -> list[FeedbackItem]:
        return self._current_items

    # ── Internal Speech Dispatch & Pacing ─────────────────────────────────────

    def _enqueue_speech(self, key: str, phrase: str, min_interval: float) -> None:
        now = time.monotonic()
        # Per-key cooldown check
        last_key_time = self._last_spoken_by_joint.get(key, 0.0)
        if now - last_key_time < min_interval:
            return

        # Global utterance spacing (avoid rapid machine-gun sentences)
        if now - self._last_spoken_time < 2.2:
            return

        # Keep speech queue fresh (replace older unuttered instruction)
        self._speech_queue.clear()
        self._speech_queue.append(phrase)
        self._last_spoken_by_joint[key] = now
        self._last_spoken_time = now

    def clear(self) -> None:
        """Clear all active feedback and cooldown history."""
        self._current_items.clear()
        self._speech_queue.clear()
        self._last_spoken_by_joint.clear()
        self._consecutive_error_counts.clear()
        self._last_spoken_time = 0.0
