"""
src/models/jev_fusion.py
────────────────────────
Final pose-decision policy: combines the local classifier, an optional Jev
image assessment, MediaPipe landmark reliability, and the practice mode.

This is a *gated decision rule*, not a probability fusion: Jev's numbers are
never mixed arithmetically with the local probabilities.  The local-side
thresholds come from validation data (see scripts/evaluate_runtime_pipeline.py
and data/zenodo/confidence_calibration.json); the Jev-side thresholds could not
be validated without live API access and are deliberately conservative:

  1. Both agree (and evidence is reliable)       → keep the agreed pose.
  2. Local uncertain, Jev valid/recent/confident → accept Jev only if its label is
     also among the local top-k with non-trivial local support; tier capped "medium".
  3. Disagreement                                 → never trust Jev over a confident
     local prediction; if the local prediction is not strong, report uncertainty.
  4. Orientation veto: when torso and legs are reliably observed, a label from the
     other coarse group (upright vs floor) is not assessed (catches confident
     out-of-distribution errors; veto rate on correct labels is measured in
     data/zenodo/confidence_calibration.json).
  5. Landmarks unreliable                         → no pose-specific assessment,
     whatever Jev says (Jev cannot supply joint angles).
  6. Jev's visual form opinion is informational only; it is never converted
     into a joint-angle measurement.

Guided mode keeps the user's target pose.  Classification can only *lower*
certainty there (strong evidence of a different pose), never raise it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

TIER_RANK = {"low": 0, "medium": 1, "high": 2}


def _min_tier(*tiers: str) -> str:
    return min((t for t in tiers if t in TIER_RANK), key=TIER_RANK.get, default="low")


@dataclass
class FusionConfig:
    local_threshold: float = 0.28            # calibrated "detected" threshold
    local_high_confidence: float = 0.60      # above this a local label can be "high" tier
    class_min_precision: float = 0.0         # validation precision a class needs at the threshold
    class_precision: dict[str, float] = field(default_factory=dict)  # from validation calibration
    jev_min_confidence: float = 0.70
    jev_max_age_seconds: float = 4.0
    jev_topk: int = 3
    jev_min_local_support: float = 0.10
    orientation_veto: bool = True            # reject labels contradicting reliable body orientation
    min_label_streak: int = 2                # auto mode: consecutive inferences a new local label needs

    def class_trusted(self, label: Optional[str]) -> bool:
        if not label or not self.class_precision or self.class_min_precision <= 0:
            return True
        return self.class_precision.get(label, 0.0) >= self.class_min_precision


@dataclass
class PoseDecision:
    final_pose: Optional[str]
    assessment_allowed: bool
    tier: str                                # high | medium | low
    source: str                              # local | local+jev | jev_resolved | guided_target | none
    agreement: str                           # agree | disagree | jev_unclear | jev_unavailable | jev_stale |
                                             # jev_low_confidence | jev_not_requested | n/a
    reason: str
    user_message: Optional[str] = None
    local_pose: Optional[str] = None
    local_confidence: float = 0.0
    jev_pose: Optional[str] = None
    jev_confidence: Optional[float] = None
    jev_age_seconds: Optional[float] = None
    jev_form: Optional[str] = None

    @property
    def reliable(self) -> bool:
        return self.assessment_allowed and self.tier != "low"


CAMERA_MSG = "Make sure your whole body is in view, step back or improve the lighting."


def jev_usability(jev: Any, candidates: list[str], now: float, cfg: FusionConfig,
                  current_epoch: Optional[int] = None) -> tuple[bool, str, Optional[float]]:
    """(usable, agreement-status-if-not, age_seconds)."""
    if jev is None:
        return False, "jev_not_requested", None
    age = None
    if getattr(jev, "frame_timestamp", None) is not None:
        age = max(0.0, now - float(jev.frame_timestamp))
    if not getattr(jev, "ok", False):
        return False, "jev_unavailable", age
    if age is None or age > cfg.jev_max_age_seconds:
        return False, "jev_stale", age
    if current_epoch is not None and getattr(jev, "epoch", None) is not None and jev.epoch != current_epoch:
        return False, "jev_stale", age
    if getattr(jev, "unclear", False) or not jev.label:
        return False, "jev_unclear", age
    if jev.label not in candidates:
        return False, "jev_unavailable", age
    if jev.confidence is None or jev.confidence < cfg.jev_min_confidence:
        return False, "jev_low_confidence", age
    return True, "", age


def decide(
    local_pred: Any,
    landmark_tier: str,
    mode: str,
    candidates: list[str],
    now: float,
    cfg: Optional[FusionConfig] = None,
    target_pose: Optional[str] = None,
    jev: Any = None,
    current_epoch: Optional[int] = None,
    label_streak: Optional[int] = None,
) -> PoseDecision:
    """
    Args:
        local_pred:     PosePrediction (pose_name, confidence, status, top3)
        landmark_tier:  overall MediaPipe reliability for this frame ("high"/"medium"/"low")
        mode:           "guided" or "auto"
        candidates:     classes the active profile supports
        now:            monotonic seconds (for Jev result age)
        jev:            latest JevResult or None
        current_epoch:  pose epoch; a Jev result from another epoch is stale
        label_streak:   consecutive inferences with the same local label (None = not tracked)
    """
    cfg = cfg or FusionConfig()
    lp = getattr(local_pred, "pose_name", None)
    lc = float(getattr(local_pred, "confidence", 0.0) or 0.0)
    status = getattr(local_pred, "status", "no_person")
    use, why_not, age = jev_usability(jev, candidates, now, cfg, current_epoch)
    jp = getattr(jev, "label", None) if jev is not None else None
    jc = getattr(jev, "confidence", None) if jev is not None else None
    base = dict(local_pose=lp, local_confidence=lc, jev_pose=jp, jev_confidence=jc,
                jev_age_seconds=age, jev_form=getattr(jev, "form_assessment", None) if jev is not None else None)

    if status == "no_person":
        return PoseDecision(None, False, "low", "none", "n/a", "no person detected",
                            "No person detected. Please stand in front of the camera.", **base)

    if landmark_tier == "low":
        return PoseDecision(target_pose if mode == "guided" else None, False, "low", "none",
                            "n/a" if not use else ("agree" if jp == lp else "disagree"),
                            "landmark tracking unreliable", CAMERA_MSG, **base)

    # Orientation consistency: a reliably upright body cannot be a floor asana (and vice versa).
    fd = getattr(local_pred, "filter_decision", None)
    observed_group = getattr(fd, "coarse_group", None) if cfg.orientation_veto else None

    def _orientation_ok(label: Optional[str]) -> bool:
        if observed_group is None or not label:
            return True
        from src.models.inference import class_coarse_group
        g = class_coarse_group(label)
        return g is None or g == observed_group

    local_ok = (status == "detected" and lc >= cfg.local_threshold and cfg.class_trusted(lp)
                and _orientation_ok(lp))
    local_tier = "high" if lc >= cfg.local_high_confidence else "medium"

    # ── Guided mode: the target is never replaced ─────────────────────────────
    if mode == "guided" and target_pose:
        tier = landmark_tier
        reason = "guided target; assessment limited by landmark reliability"
        msg = None
        agreement = why_not or ("agree" if jp == target_pose else "disagree")
        local_disagrees = local_ok and lc >= cfg.local_high_confidence and lp != target_pose
        jev_disagrees = use and jp != target_pose
        if local_disagrees and (jev_disagrees or not use):
            tier = _min_tier(tier, "medium")
            reason = f"classifier sees {lp} ({lc:.0%}), not the target"
            msg = f"This looks like {lp}. Move into {target_pose} for detailed guidance."
        elif jev_disagrees and lp != target_pose:
            tier = _min_tier(tier, "medium")
            reason = f"Jev sees {jp}, not the target"
        return PoseDecision(target_pose, True, tier, "guided_target", agreement, reason, msg, **base)

    # ── Auto-detect mode ──────────────────────────────────────────────────────
    if local_ok and use:
        if jp == lp:
            return PoseDecision(lp, True, _min_tier(landmark_tier, "high"), "local+jev", "agree",
                                "local classifier and Jev agree", **base)
        if lc >= cfg.local_high_confidence:
            return PoseDecision(lp, True, _min_tier(landmark_tier, "medium"), "local", "disagree",
                                f"Jev suggests {jp}; keeping confident local prediction", **base)
        return PoseDecision(None, False, "low", "none", "disagree",
                            f"local ({lp}) and Jev ({jp}) disagree",
                            "Pose unclear. Hold the pose steadily with your full body in view.", **base)

    if local_ok:
        if label_streak is not None and label_streak < cfg.min_label_streak:
            return PoseDecision(None, False, "low", "none", why_not or "n/a",
                                f"confirming {lp} ({label_streak}/{cfg.min_label_streak} consistent inferences)",
                                "Hold the pose steady…", **base)
        return PoseDecision(lp, True, _min_tier(landmark_tier, local_tier), "local", why_not or "n/a",
                            "local classifier above calibrated threshold", **base)

    # Local uncertain (low confidence or untrusted class)
    if use:
        top = list(getattr(local_pred, "top3", []) or [])[: cfg.jev_topk]
        support = dict(top).get(jp, 0.0)
        if jp in dict(top) and support >= cfg.jev_min_local_support and _orientation_ok(jp):
            return PoseDecision(jp, True, _min_tier(landmark_tier, "medium"), "jev_resolved", "disagree" if jp != lp else "agree",
                                f"Jev resolved ambiguity among local candidates ({jp}, local {support:.0%})", **base)
        return PoseDecision(None, False, "low", "none", "disagree",
                            f"Jev's {jp} is not supported by the local classifier",
                            "Pose unclear. Hold the pose steadily with your full body in view.", **base)

    if lc < cfg.local_threshold or status != "detected":
        reason = "local confidence below calibrated threshold"
    elif not _orientation_ok(lp):
        reason = f"{lp} contradicts the observed body orientation ({observed_group.lower()})"
    else:
        reason = f"{lp} is not reliably recognised on validation data"
    return PoseDecision(None, False, "low", "none", why_not or "n/a", reason,
                        f"Detecting pose ({lc:.0%}). Hold your posture steadily with your full body visible.", **base)


def decide_jev_primary(
    jev: Any,
    local_pred: Any,
    mode: str,
    candidates: list[str],
    now: float,
    cfg: Optional[FusionConfig] = None,
    target_pose: Optional[str] = None,
    current_epoch: Optional[int] = None,
    high_confidence: float = 0.85,
) -> PoseDecision:
    """
    Jev-primary mode: Jev's answer alone decides the pose; MediaPipe landmarks are not
    required.  The local classifier is reported for comparison only.
    """
    cfg = cfg or FusionConfig()
    lp = getattr(local_pred, "pose_name", None) if local_pred is not None else None
    lc = float(getattr(local_pred, "confidence", 0.0) or 0.0) if local_pred is not None else 0.0
    use, why_not, age = jev_usability(jev, candidates, now, cfg, current_epoch)
    jp = getattr(jev, "label", None) if jev is not None else None
    jc = getattr(jev, "confidence", None) if jev is not None else None
    base = dict(local_pose=lp, local_confidence=lc, jev_pose=jp, jev_confidence=jc,
                jev_age_seconds=age, jev_form=getattr(jev, "form_assessment", None) if jev is not None else None)
    guided = mode == "guided" and target_pose is not None

    if not use:
        reason, msg = {
            "jev_not_requested": ("waiting for Jev's first answer", "Analysing your pose…"),
            "jev_unavailable": ("Jev request failed", "Pose check unavailable; retrying…"),
            "jev_stale": ("Jev answer is out of date", "Hold the pose steady…"),
            "jev_unclear": ("Jev sees no clear yoga pose", CAMERA_MSG),
            "jev_low_confidence": (f"Jev unsure ({(jc or 0.0):.0%})", "Hold the pose steadily with your full body in view."),
        }.get(why_not, ("Jev answer not usable", None))
        return PoseDecision(target_pose if guided else None, False, "low", "none", why_not, reason, msg, **base)

    agreement = "agree" if jp == lp else "disagree"
    tier = "high" if jc >= high_confidence else "medium"
    if guided and jp != target_pose:
        return PoseDecision(target_pose, False, "low", "none", agreement,
                            f"Jev sees {jp}, not the target {target_pose}",
                            f"Move into {target_pose}.", **base)
    return PoseDecision(jp, True, tier, "jev", agreement, f"Jev: {jp} ({jc:.0%})", **base)
