"""
src/models/inference.py
────────────────────────
Frame-level pose inference pipeline:
  raw features  →  impute  →  scale (RF only needs impute)  →  predict  →  decode

Also implements a lightweight rolling majority-vote smoothing buffer.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.models.model_loader import YogaArtifacts, YogaArtifactsZenodo
from src.models.sequence_tracker import YogaSequenceTracker, SequenceState, BASE_ORIENTATION_MAP


@dataclass
class PosePrediction:
    pose_name: str               # decoded class label
    confidence: float            # probability of the top class  [0, 1]
    probabilities: np.ndarray    # full class probability vector
    raw_prediction: str          # top class before smoothing
    smoothed: bool               # whether smoothing changed the result
    status: str                  # "detected" | "low_confidence" | "no_person"
    top3: list[tuple[str, float]] = field(default_factory=list)
    sequence_state: Optional[SequenceState] = None
    raw_probabilities: Optional[np.ndarray] = None   # classifier output before filters/smoothing
    filter_decision: Optional[Any] = None             # FilterDecision diagnostics


@dataclass
class DualTaskPrediction(PosePrediction):
    """
    Extends PosePrediction with binary form correctness verification
    (Right vs Wrong form prediction from Model B).
    """
    is_form_correct: bool = True
    correctness_label: str = "Right"       # "Right" | "Wrong" | "Unknown"
    correctness_confidence: float = 1.0     # Probability of predicted correctness [0, 1]
    correctness_probabilities: Optional[np.ndarray] = None


def compute_gravitational_posture_prior(
    landmarks: Optional[list[dict]],
    class_names: list[str],
    user_profile: Optional[Any] = None,
    suppression: float = 0.005,
) -> np.ndarray:
    """
    Computes an anatomical gravitational likelihood mask based on physical
    keypoint spatial hierarchy and calibrated user anthropometric baselines:
      • INVERTED/SUPINE: Feet/knees above hips or shoulders below hips
        -> Activates Viparita Karani, Sarvangasana, Anantasana
      • PRONE: Shoulders and hips horizontal on the floor
        -> Activates Bhujangasana
      • UPRIGHT SEATED/KNEELING: Vertical torso, but thighs/legs foreshortened/compressed
        -> Activates Vajrasana, Parvatasana (distinguishing arms overhead in Step 3/4)
        -> Strongly suppresses standing poses (Tadasana, Kati Chakrasana, Ardhakati Chakrasana)
      • UPRIGHT STANDING: Vertical torso with full leg extension matching calibrated baseline
        -> Activates Tadasana, Kati Chakrasana, Ardhakati Chakrasana
        -> Suppresses ground, prone, seated, and inverted poses
    """
    n_classes = len(class_names)
    if not landmarks or len(landmarks) < 33:
        w_vec = np.ones(n_classes, dtype=float)
        return w_vec / np.sum(w_vec)

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

        # Calibrated user baseline integration
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

        weights = {c: 1.0 for c in class_names}

        # 1. INVERTED / SUPINE (Feet above hips or shoulders below hips)
        is_inverted = (ay < hy - 0.08 and ky < hy - 0.04) or (sy > hy + 0.06)
        if is_inverted:
            for pose in ("Tadasana", "Kati Chakrasana", "Ardhakati Chakrasana", "Bhujangasana", "Marjariasana", "Vajrasana", "Parvatasana"):
                if pose in weights:
                    weights[pose] = suppression
            for pose in ("Viparita Karani", "Sarvangasana", "Anantasana"):
                if pose in weights:
                    weights[pose] = 1.0
            w_vec = np.array([weights.get(c, 1.0) for c in class_names], dtype=float)
            return w_vec / np.sum(w_vec)

        # 2. PRONE (Horizontal on stomach, e.g. Cobra Pose Bhujangasana)
        is_horizontal_torso = abs(dy_torso) < 0.10 and dx_torso > 0.08
        if is_horizontal_torso and (ay > sy - 0.15) and (ky > sy - 0.15):
            for pose in ("Tadasana", "Kati Chakrasana", "Ardhakati Chakrasana", "Vajrasana", "Parvatasana", "Sarvangasana", "Viparita Karani"):
                if pose in weights:
                    weights[pose] = suppression
            if "Bhujangasana" in weights:
                weights["Bhujangasana"] = 1.0
            if "Marjariasana" in weights:
                weights["Marjariasana"] = 0.50
            w_vec = np.array([weights.get(c, 1.0) for c in class_names], dtype=float)
            return w_vec / np.sum(w_vec)

        # 3. UPRIGHT TORSO (Head/shoulders at top, hips below shoulders, vertical spine)
        is_upright = (dy_torso > 0.07) and (dx_torso / max(0.01, dy_torso) < 0.95)
        if is_upright:
            # Live vertical body ratios relative to torso
            live_thigh_dy = max(0.0, ky - hy)
            live_thigh_ratio = live_thigh_dy / max(0.01, dy_torso)
            live_leg_ratio = ratio_leg

            if is_calibrated:
                # Calibrated Anthropometric Comparison:
                # When seated/kneeling, 2D vertical thigh drops to < 55% of standing baseline
                is_seated_or_kneeling = (
                    (live_thigh_ratio < 0.55 * baseline_thigh_ratio)
                    or (live_leg_ratio < 0.55 * baseline_leg_ratio)
                    or (shin_dy < 0.045)
                    or (ay <= ky + 0.04)
                    or (abs(ky - hy) < 0.08 and abs(ay - hy) < 0.12)
                    or (ankles_vis < 0.30 and ky >= 0.60 and hy >= 0.48)
                )
            else:
                # Fallback geometric check when uncalibrated
                is_seated_or_kneeling = (
                    (ratio_leg < 0.60)
                    or (shin_dy < 0.045)
                    or (ay <= ky + 0.04)
                    or (abs(ky - hy) < 0.08 and abs(ay - hy) < 0.12)
                    or (ankles_vis < 0.30 and ky >= 0.60 and hy >= 0.48)
                )

            # Suppress prone and inverted poses whenever torso is upright
            for pose in ("Bhujangasana", "Viparita Karani", "Sarvangasana", "Anantasana", "Marjariasana"):
                if pose in weights:
                    weights[pose] = suppression

            if is_seated_or_kneeling:
                # SEATED / KNEELING: Suppress standing postures
                for pose in ("Tadasana", "Kati Chakrasana", "Ardhakati Chakrasana"):
                    if pose in weights:
                        weights[pose] = suppression

                # Distinguish Step 1 & 2 vs Step 3 & 4 (Parvatasana arms overhead)
                arms_overhead = (wy < sy - 0.02)
                if arms_overhead:
                    if "Parvatasana" in weights:
                        weights["Parvatasana"] = 1.0
                    if "Vajrasana" in weights:
                        weights["Vajrasana"] = suppression
                else:
                    if "Vajrasana" in weights:
                        weights["Vajrasana"] = 1.0
                    if "Parvatasana" in weights:
                        weights["Parvatasana"] = 1.0
            else:
                # STANDING: Suppress seated postures
                for pose in ("Vajrasana", "Parvatasana"):
                    if pose in weights:
                        weights[pose] = suppression
                for pose in ("Tadasana", "Kati Chakrasana", "Ardhakati Chakrasana"):
                    if pose in weights:
                        weights[pose] = 1.0

            w_vec = np.array([weights.get(c, 1.0) for c in class_names], dtype=float)
            return w_vec / np.sum(w_vec)

        w_vec = np.array([weights.get(c, 1.0) for c in class_names], dtype=float)
        return w_vec / np.sum(w_vec)
    except Exception:
        w_vec = np.ones(n_classes, dtype=float)
        return w_vec / np.sum(w_vec)


# ── Candidate filtering policy (gravitational prior + sequence pruning) ──────

ORIENTATION_CORE_IDS = (11, 12, 23, 24)     # shoulders + hips: torso orientation
ORIENTATION_LEG_IDS = (25, 26, 27, 28)      # knees + ankles: standing vs seated / inverted
COARSE_GROUP = {
    "STANDING": "UPRIGHT", "SEATED_KNEELING": "UPRIGHT",
    "PRONE": "FLOOR", "INVERTED": "FLOOR", "QUADRUPED": "FLOOR",
}


@dataclass
class FilterPolicy:
    """
    Controls how geometric rules may restrict the classifier's candidates.

    mode="legacy": original behaviour (sequence pruning x0.001, then prior x0.005,
                   applied on every frame regardless of landmark quality).
    mode="safe":   rules apply only when the landmarks they read are reliably
                   observed and the base orientation has been stable for
                   ``stable_frames`` inferences; suppression is a single soft
                   multiplier (``soft_floor``) instead of a near-zero product, and a
                   class the classifier itself strongly prefers
                   (>= ``override_confidence``) is never suppressed.
    """
    use_gravity_prior: bool = True
    use_sequence_pruning: bool = True
    mode: str = "legacy"
    prior_suppression: float = 0.005
    sequence_suppression: float = 0.001
    soft_floor: float = 0.15
    min_landmark_visibility: float = 0.50
    stable_frames: int = 2
    override_confidence: float = 0.60
    granularity: str = "coarse"   # safe mode: "coarse" (upright vs floor) | "family" (5 base families)

    @classmethod
    def from_config(cls, cfg: Optional[dict]) -> "FilterPolicy":
        if not cfg:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in cfg.items() if k in known})


@dataclass
class FilterDecision:
    """Diagnostics describing what the candidate filters did on one inference."""
    mode: str
    base_posture: str = "UNKNOWN"
    applied_prior: bool = False
    applied_sequence: bool = False
    skipped_reason: Optional[str] = None
    suppressed: list[str] = field(default_factory=list)
    overridden: list[str] = field(default_factory=list)
    stable_count: int = 0
    coarse_group: Optional[str] = None   # "UPRIGHT"/"FLOOR" when orientation is reliable & stable, else None


def class_coarse_group(label: Optional[str]) -> Optional[str]:
    """UPRIGHT / FLOOR group of a class, or None for classes without a known family."""
    return COARSE_GROUP.get(BASE_ORIENTATION_MAP.get(label or "", ""))


def _landmarks_reliable(landmarks: list[dict], ids: tuple[int, ...], min_vis: float) -> bool:
    for i in ids:
        if i >= len(landmarks):
            return False
        lm = landmarks[i]
        if lm.get("imputed") or lm.get("unavailable") or float(lm.get("visibility", 0.0)) < min_vis:
            return False
    return True


class CandidateFilter:
    """Applies the configured geometric filters and keeps their temporal state."""

    def __init__(self, policy: Optional[FilterPolicy] = None, debounce_frames: int = 2):
        self.policy = policy or FilterPolicy()
        self.seq_tracker = YogaSequenceTracker(debounce_frames=debounce_frames)
        self._last_raw_base: Optional[str] = None
        self._stable_count = 0

    def reset(self) -> None:
        self.seq_tracker.reset()
        self._last_raw_base = None
        self._stable_count = 0

    def apply(
        self,
        proba: np.ndarray,
        class_names: list[str],
        landmarks: Optional[list[dict]],
        feature_dict: Optional[dict],
        user_profile: Optional[Any],
    ) -> tuple[np.ndarray, Optional[SequenceState], FilterDecision]:
        pol = self.policy
        decision = FilterDecision(mode=pol.mode)
        if not landmarks:
            decision.skipped_reason = "no landmarks"
            return proba, None, decision

        seq_state = self.seq_tracker.update(
            landmarks, feature_dict=feature_dict, user_profile=user_profile, available_classes=class_names,
        )
        decision.base_posture = seq_state.base_posture

        # Orientation evidence (used by the decision layer as a consistency check in every mode)
        raw_base = self.seq_tracker.detect_base_posture(landmarks, user_profile)
        if raw_base == self._last_raw_base:
            self._stable_count += 1
        else:
            self._last_raw_base = raw_base
            self._stable_count = 1
        decision.stable_count = self._stable_count
        orientation_reliable = (
            raw_base != "UNKNOWN"
            and self._stable_count >= max(1, pol.stable_frames)
            and _landmarks_reliable(landmarks, ORIENTATION_CORE_IDS, pol.min_landmark_visibility)
            and _landmarks_reliable(landmarks, ORIENTATION_LEG_IDS, pol.min_landmark_visibility)
        )
        decision.coarse_group = COARSE_GROUP.get(raw_base) if orientation_reliable else None

        if pol.mode == "legacy":
            out = proba
            if pol.use_sequence_pruning:
                out = self.seq_tracker.filter_probabilities(out, class_names, seq_state, suppression=pol.sequence_suppression)
                decision.applied_sequence = True
            if pol.use_gravity_prior:
                prior = compute_gravitational_posture_prior(
                    landmarks, class_names, user_profile=user_profile, suppression=pol.prior_suppression)
                out = out * prior
                s = float(np.sum(out))
                if s > 1e-6:
                    out = out / s
                decision.applied_prior = True
            decision.suppressed = [c for c, p, q in zip(class_names, proba, out) if q < p * 0.1]
            return out, seq_state, decision

        # ── safe mode ─────────────────────────────────────────────────────────
        if not (pol.use_gravity_prior or pol.use_sequence_pruning):
            decision.skipped_reason = "filters disabled"
            return proba, seq_state, decision
        if not _landmarks_reliable(landmarks, ORIENTATION_CORE_IDS, pol.min_landmark_visibility):
            decision.skipped_reason = "torso landmarks not reliably observed"
            return proba, seq_state, decision
        if not _landmarks_reliable(landmarks, ORIENTATION_LEG_IDS, pol.min_landmark_visibility):
            decision.skipped_reason = "leg landmarks not reliably observed"
            return proba, seq_state, decision
        if raw_base == "UNKNOWN":
            decision.skipped_reason = "orientation ambiguous"
            return proba, seq_state, decision
        if self._stable_count < pol.stable_frames:
            decision.skipped_reason = f"orientation not yet stable ({self._stable_count}/{pol.stable_frames})"
            return proba, seq_state, decision

        mult = np.ones(len(class_names), dtype=float)
        if pol.granularity == "coarse":
            # Measured: the geometric rules separate upright from floor poses far more
            # reliably than they separate prone / inverted / quadruped from each other.
            group = COARSE_GROUP.get(raw_base)
            mult = np.array([
                pol.soft_floor if COARSE_GROUP.get(BASE_ORIENTATION_MAP.get(c, ""), group) != group else 1.0
                for c in class_names
            ])
            decision.applied_prior = pol.use_gravity_prior
            decision.applied_sequence = pol.use_sequence_pruning
        elif pol.use_sequence_pruning:
            seq_m = np.array([pol.soft_floor if c in seq_state.pruned_poses else 1.0 for c in class_names])
            mult = np.minimum(mult, seq_m)
            decision.applied_sequence = True
        if pol.use_gravity_prior and pol.granularity != "coarse":
            prior = compute_gravitational_posture_prior(landmarks, class_names, user_profile=user_profile, suppression=0.0)
            prior_m = np.where(prior > 0, 1.0, pol.soft_floor)
            mult = np.minimum(mult, prior_m)
            decision.applied_prior = True

        for idx, c in enumerate(class_names):
            if mult[idx] < 1.0 and proba[idx] >= pol.override_confidence:
                mult[idx] = 1.0
                decision.overridden.append(c)
        decision.suppressed = [c for c, m in zip(class_names, mult) if m < 1.0]

        out = proba * mult
        s = float(np.sum(out))
        out = out / s if s > 1e-9 else proba
        return out, seq_state, decision


class PoseInference:
    """
    Wraps preprocessing + classifier into a single predict() call.
    Maintains a rolling prediction history for majority-vote smoothing.
    """

    def __init__(
        self,
        artifacts: YogaArtifacts,
        active_classifier: str = "rf",
        confidence_threshold: float = 0.45,
        smoothing_window: int = 5,
        filter_policy: Optional[FilterPolicy] = None,
    ):
        self._art = artifacts
        self._clf = active_classifier.lower()
        self._threshold = confidence_threshold
        self._history: deque[str] = deque(maxlen=smoothing_window)
        self._ema_proba: Optional[np.ndarray] = None
        self._alpha = 0.55
        self._filter = CandidateFilter(filter_policy)
        self._seq_tracker = self._filter.seq_tracker

        if self._clf not in ("rf", "mlp"):
            raise ValueError(f"active_classifier must be 'rf' or 'mlp', got '{self._clf}'")
        if self._clf == "mlp" and artifacts.mlp_model is None:
            print("[Inference] MLP not available, falling back to RF.")
            self._clf = "rf"

    def predict(
        self,
        feature_series: pd.Series,
        landmarks: Optional[list[dict]] = None,
        user_profile: Optional[Any] = None,
    ) -> PosePrediction:
        """
        Run the full inference pipeline on a single feature Series.

        feature_series must have the same index/column names as feature_cols.
        Returns a PosePrediction even when confidence is low.
        """
        art = self._art

        if isinstance(feature_series, dict):
            feature_series = pd.Series(feature_series)
        series_aligned = feature_series.reindex(art.feature_cols)
        # Replace ±inf with NaN before passing to imputer
        values = series_aligned.values.astype(float)
        values[~np.isfinite(values)] = np.nan
        X_df = pd.DataFrame([values], columns=art.feature_cols)

        # ── Impute (always required) ──────────────────────────────────────────
        X_imp = art.imputer.transform(X_df)

        # ── Predict ───────────────────────────────────────────────────────────
        if self._clf == "rf":
            proba = art.rf_model.predict_proba(X_imp)[0]
        else:
            X_scaled = art.scaler.transform(X_imp)
            proba = art.mlp_model.predict(X_scaled, verbose=0)[0]

        # ── Gravitational Orientation Prior Filter & Sequence Pruning ─────────
        raw_proba = np.asarray(proba, dtype=float).copy()
        feat_dict = feature_series.to_dict() if hasattr(feature_series, "to_dict") else None
        proba, seq_state, decision = self._filter.apply(
            raw_proba, art.class_names, landmarks, feat_dict, user_profile,
        )

        raw_idx = int(np.argmax(proba))
        raw_name = art.class_names[raw_idx]

        # ── Responsive Probability Smoothing (EMA) ───────────────────────────
        if self._ema_proba is None:
            self._ema_proba = proba.copy()
        else:
            self._ema_proba = self._alpha * proba + (1.0 - self._alpha) * self._ema_proba

        smoothed_idx = int(np.argmax(self._ema_proba))
        smoothed_conf = float(self._ema_proba[smoothed_idx])
        smoothed_name = art.class_names[smoothed_idx]
        was_smoothed = smoothed_name != raw_name

        self._history.append(smoothed_name)

        # ── Status ────────────────────────────────────────────────────────────
        if smoothed_conf < self._threshold:
            status = "low_confidence"
        else:
            status = "detected"

        top3_indices = np.argsort(self._ema_proba)[::-1][:3]
        top3 = [(art.class_names[i], float(self._ema_proba[i])) for i in top3_indices]

        return PosePrediction(
            pose_name=smoothed_name,
            confidence=smoothed_conf,
            probabilities=self._ema_proba,
            raw_prediction=raw_name,
            smoothed=was_smoothed,
            status=status,
            top3=top3,
            sequence_state=seq_state,
            raw_probabilities=raw_proba,
            filter_decision=decision,
        )

    def reset_history(self) -> None:
        self._history.clear()
        self._ema_proba = None
        self._filter.reset()


class DualTaskInference:
    """
    Inference engine for Zenodo dual-task models:
      • Model A: 10-Class Asana Classifier
      • Model B: Binary Form Verifier (Right vs Wrong)

    Pipeline per feature vector:
      features → align columns → impute → [scale if MLP] → predict asana & correctness
    """

    def __init__(
        self,
        artifacts: YogaArtifactsZenodo,
        confidence_threshold: float = 0.30,
        verifier_threshold: float = 0.50,
        smoothing_window: int = 3,
        alpha: float = 0.55,
        filter_policy: Optional[FilterPolicy] = None,
    ):
        self._art = artifacts
        self._threshold = confidence_threshold
        self._verifier_threshold = verifier_threshold
        self._alpha = alpha
        self._ema_proba: Optional[np.ndarray] = None
        self._history: deque[str] = deque(maxlen=smoothing_window)
        self._filter = CandidateFilter(filter_policy)
        self._seq_tracker = self._filter.seq_tracker

    def predict(
        self,
        feature_series: pd.Series,
        landmarks: Optional[list[dict]] = None,
        user_profile: Optional[Any] = None,
    ) -> DualTaskPrediction:
        """
        Run inference on feature vector, producing both Asana classification
        and Form verification predictions with responsive probability smoothing.
        """
        art = self._art

        if isinstance(feature_series, dict):
            feature_series = pd.Series(feature_series)
        series_aligned = feature_series.reindex(art.feature_cols)
        values = series_aligned.values.astype(float)
        values[~np.isfinite(values)] = np.nan
        X_df = pd.DataFrame([values], columns=art.feature_cols)

        # ── Impute ────────────────────────────────────────────────────────────
        X_imp = art.imputer.transform(X_df)
        X_eval = art.scaler.transform(X_imp) if getattr(art, "model_type", "rf") == "mlp" else X_imp

        # ── Model A: Asana Prediction ─────────────────────────────────────────
        proba_asana = art.asana_model.predict_proba(X_eval)[0]

        # ── Gravitational Orientation Prior Filter & Sequence Pruning ─────────
        raw_proba = np.asarray(proba_asana, dtype=float).copy()
        feat_dict = feature_series.to_dict() if hasattr(feature_series, "to_dict") else None
        proba_asana, seq_state, decision = self._filter.apply(
            raw_proba, art.class_names, landmarks, feat_dict, user_profile,
        )

        raw_idx = int(np.argmax(proba_asana))
        raw_name = art.class_names[raw_idx]

        # ── Responsive Probability Smoothing (EMA) ───────────────────────────
        if self._ema_proba is None:
            self._ema_proba = proba_asana.copy()
        else:
            self._ema_proba = self._alpha * proba_asana + (1.0 - self._alpha) * self._ema_proba

        smoothed_idx = int(np.argmax(self._ema_proba))
        smoothed_conf = float(self._ema_proba[smoothed_idx])
        smoothed_name = art.class_names[smoothed_idx]
        was_smoothed = smoothed_name != raw_name

        self._history.append(smoothed_name)

        status = "detected" if smoothed_conf >= self._threshold else "low_confidence"
        top3_indices = np.argsort(self._ema_proba)[::-1][:3]
        top3 = [(art.class_names[i], float(self._ema_proba[i])) for i in top3_indices]

        # ── Model B: Form Verifier Prediction ─────────────────────────────────
        proba_corr = art.verifier_model.predict_proba(X_eval)[0]
        corr_classes = list(art.correctness_label_encoder.classes_)
        right_idx = corr_classes.index("Right") if "Right" in corr_classes else 1
        wrong_idx = corr_classes.index("Wrong") if "Wrong" in corr_classes else 0

        p_right = float(proba_corr[right_idx])
        p_wrong = float(proba_corr[wrong_idx])
        is_right = p_right >= self._verifier_threshold
        corr_label = "Right" if is_right else "Wrong"
        corr_conf = p_right if is_right else p_wrong

        return DualTaskPrediction(
            pose_name=smoothed_name,
            confidence=smoothed_conf,
            probabilities=self._ema_proba,
            raw_prediction=raw_name,
            smoothed=was_smoothed,
            status=status,
            top3=top3,
            is_form_correct=is_right,
            correctness_label=corr_label,
            correctness_confidence=corr_conf,
            correctness_probabilities=proba_corr,
            sequence_state=seq_state,
            raw_probabilities=raw_proba,
            filter_decision=decision,
        )

    def reset_history(self) -> None:
        self._history.clear()
        self._ema_proba = None
        self._filter.reset()

