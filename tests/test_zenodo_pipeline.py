"""
tests/test_zenodo_pipeline.py
──────────────────────────────
Comprehensive automated test suite for Zenodo migration, dual-task inference,
biomechanical personalization, temporal reliability, and longitudinal progress tracking.
All tests run headless without physical webcam or audio hardware.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent

from src.features import extract_features, features_to_series
from src.models.model_loader import (
    YogaArtifactsZenodo,
    load_profile_artifacts,
    load_zenodo_artifacts,
)
from src.models.inference import (
    DualTaskInference,
    DualTaskPrediction,
    PoseInference,
    PosePrediction,
)
from src.models.zenodo_annotations import (
    ZENODO_ASANAS,
    parse_combined_label,
    parse_group_id,
)
from src.correction.reference_ranges import (
    REFERENCE_RANGES,
    POSE_METADATA,
    ZENODO_GUIDED_POSES,
)
from src.app.reference_display import get_target_poses_for_profile
from src.correction.posture_evaluator import evaluate as evaluate_posture, PostureEvaluation

from src.correction.calibration import (
    UserProfile,
    load_profile,
    save_profile,
    validate_profile_dict,
    apply_biomechanical_personalization,
)
from src.correction.reliability_tracker import (
    TemporalReliabilityTracker,
    FrameReliabilityAssessment,
)
from src.correction.progress_tracker import ProgressTracker
from scripts.prepare_zenodo_dataset import (
    landmark_vector_to_mediapipe_dicts,
    stratified_group_split,
)


@pytest.fixture
def dummy_visible_landmarks():
    """33 standard anatomical landmarks."""
    base = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.95} for _ in range(33)]
    # Place major joints in a plausible standing posture (Tadasana)
    base[11] = {"x": 0.42, "y": 0.25, "z": 0.0, "visibility": 0.95}  # L shoulder
    base[12] = {"x": 0.58, "y": 0.25, "z": 0.0, "visibility": 0.95}  # R shoulder
    base[13] = {"x": 0.38, "y": 0.45, "z": 0.0, "visibility": 0.95}  # L elbow
    base[14] = {"x": 0.62, "y": 0.45, "z": 0.0, "visibility": 0.95}  # R elbow
    base[15] = {"x": 0.35, "y": 0.65, "z": 0.0, "visibility": 0.95}  # L wrist
    base[16] = {"x": 0.65, "y": 0.65, "z": 0.0, "visibility": 0.95}  # R wrist
    base[23] = {"x": 0.44, "y": 0.55, "z": 0.0, "visibility": 0.95}  # L hip
    base[24] = {"x": 0.56, "y": 0.55, "z": 0.0, "visibility": 0.95}  # R hip
    base[25] = {"x": 0.44, "y": 0.75, "z": 0.0, "visibility": 0.95}  # L knee
    base[26] = {"x": 0.56, "y": 0.75, "z": 0.0, "visibility": 0.95}  # R knee
    base[27] = {"x": 0.44, "y": 0.95, "z": 0.0, "visibility": 0.95}  # L ankle
    base[28] = {"x": 0.56, "y": 0.95, "z": 0.0, "visibility": 0.95}  # R ankle
    return base


@pytest.fixture
def mock_zenodo_artifacts(dummy_visible_landmarks):
    """Construct mock YogaArtifactsZenodo for unit testing without full model fitting."""
    from src.features.geometry import LANDMARK_IDS, ANGLE_TRIPLETS, DISTANCE_PAIRS

    flags = [0] * 33
    f_dict = extract_features(dummy_visible_landmarks, flags, 1280, 720)
    feature_cols = list(f_dict.keys())

    # Preprocessing
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    X_dummy = pd.DataFrame([f_dict])
    imputer.fit(X_dummy)


    scaler = StandardScaler()
    X_imp = imputer.transform(X_dummy)
    scaler.fit(X_imp)

    asana_le = LabelEncoder().fit(sorted(ZENODO_ASANAS))
    corr_le = LabelEncoder().fit(["Wrong", "Right"])

    # Mock Models
    model_a = MagicMock()
    # Predict Tadasana (index 7 in sorted ZENODO_ASANAS) with 0.88 probability
    p_a = np.zeros(10)
    tadasana_idx = list(asana_le.classes_).index("Tadasana")
    p_a[tadasana_idx] = 0.88
    p_a[(tadasana_idx + 1) % 10] = 0.12
    model_a.predict_proba.return_value = np.array([p_a])

    model_b = MagicMock()
    corr_classes = list(corr_le.classes_)
    p_b = np.zeros(2)
    p_b[corr_classes.index("Right")] = 0.92
    p_b[corr_classes.index("Wrong")] = 0.08
    model_b.predict_proba.return_value = np.array([p_b])


    return YogaArtifactsZenodo(
        imputer=imputer,
        scaler=scaler,
        asana_label_encoder=asana_le,
        correctness_label_encoder=corr_le,
        feature_cols=feature_cols,
        asana_model=model_a,
        verifier_model=model_b,
        class_names=asana_le.classes_.tolist(),
        correctness_names=["Wrong", "Right"],
        model_type="rf",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Zenodo Raw Landmark Vector to MediaPipe Dictionaries Conversion
# ─────────────────────────────────────────────────────────────────────────────

def test_landmark_vector_to_mediapipe_dicts():
    # 132 raw values: 33 landmarks * 4 (x, y, z, vis)
    raw_vec = np.zeros(132, dtype=float)
    # Set landmark 11 (left shoulder): x=0.42, y=0.25, z=-0.05, vis=0.98
    raw_vec[11 * 4] = 0.42
    raw_vec[11 * 4 + 1] = 0.25
    raw_vec[11 * 4 + 2] = -0.05
    raw_vec[11 * 4 + 3] = 0.98

    lm_dicts = landmark_vector_to_mediapipe_dicts(raw_vec)
    assert len(lm_dicts) == 33
    assert abs(lm_dicts[11]["x"] - 0.42) < 1e-5
    assert abs(lm_dicts[11]["y"] - 0.25) < 1e-5
    assert abs(lm_dicts[11]["z"] - (-0.05)) < 1e-5
    assert abs(lm_dicts[11]["visibility"] - 0.98) < 1e-5

    # Invalid length raises ValueError
    with pytest.raises(ValueError):
        landmark_vector_to_mediapipe_dicts(np.zeros(130))


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: 39 Biomechanical Feature Extraction Parity
# ─────────────────────────────────────────────────────────────────────────────

def test_feature_extraction_parity(dummy_visible_landmarks):
    flags = [0] * 33
    feat_dict = extract_features(dummy_visible_landmarks, flags, 1280, 720)
    assert len(feat_dict) == 39

    # Check key expected feature names
    expected_keys = [
        "left_knee_angle", "right_knee_angle",
        "left_elbow_angle", "right_elbow_angle",
        "shoulder_line_angle_deg", "torso_lean_from_vertical_deg",
        "wrist_to_wrist_norm", "imputed_landmark_fraction",
    ]
    for k in expected_keys:
        assert k in feat_dict, f"Missing expected feature: {k}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Group-Aware Stratified Split Prevents Burst Leakage & Preserves Vajrasana
# ─────────────────────────────────────────────────────────────────────────────

def test_group_aware_leakage_prevention():
    # Synthetic dataset mimicking burst sessions and rare class Vajrasana
    records = []
    # 5 classes with 10 sessions each
    for c in ["Tadasana_Right", "Bhujangasana_Right", "Marjariasana_Wrong", "Parvatasana_Right"]:
        for s in range(10):
            for frame in range(5):
                records.append({
                    "session_id": f"sess_{c}_{s}",
                    "combined_label": c,
                    "val": frame,
                })
    # Vajrasana: rare class with 4 sessions
    for s in range(4):
        for frame in range(5):
            records.append({
                "session_id": f"sess_Vajrasana_{s}",
                "combined_label": "Vajrasana_Right",
                "val": frame,
            })

    df = pd.DataFrame(records)
    train_df, val_df, test_df = stratified_group_split(
        df,
        group_col="session_id",
        strat_col="combined_label",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    # Check ZERO group overlap across splits
    train_groups = set(train_df["session_id"].unique())
    val_groups = set(val_df["session_id"].unique())
    test_groups = set(test_df["session_id"].unique())

    assert len(train_groups & val_groups) == 0, "Train and Val share groups!"
    assert len(train_groups & test_groups) == 0, "Train and Test share groups!"
    assert len(val_groups & test_groups) == 0, "Val and Test share groups!"

    # Rare class Vajrasana must be present in Train
    assert "Vajrasana_Right" in train_df["combined_label"].values


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Zenodo Metadata and Annotation Parsing
# ─────────────────────────────────────────────────────────────────────────────

def test_zenodo_annotation_parsing():
    c_asana, c_corr = parse_combined_label("Parvatasana_Wrong")
    assert c_asana == "Parvatasana"
    assert c_corr == "Wrong"

    # Step 3 with leg and hand fault
    ann1 = parse_group_id("parvatasana::step 3 leg and hand")
    assert ann1.step == 3
    assert ann1.correctness == "Wrong"
    assert "leg" in ann1.affected_body_parts
    assert "hand" in ann1.affected_body_parts
    assert "left_knee_angle" in ann1.candidate_angles

    # Step 5 standing (verifies steps exceed 1-4)
    ann2 = parse_group_id("tadasana::step 5 standing")
    assert ann2.step == 5
    assert "standing" in ann2.raw_fault


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Reference Ranges & Metadata for All 10 Zenodo Asanas
# ─────────────────────────────────────────────────────────────────────────────

def test_zenodo_reference_ranges_and_metadata():
    for asana in ZENODO_ASANAS:
        assert asana in REFERENCE_RANGES, f"Missing reference ranges for {asana}"
        assert asana in POSE_METADATA, f"Missing POSE_METADATA for {asana}"

        meta = POSE_METADATA[asana]
        assert "english_name" in meta
        assert "sanskrit_name" in meta
        assert "cues" in meta and len(meta["cues"]) > 0
        assert meta.get("provisional") is True, f"{asana} should be marked provisional"


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Dual-Task Inference Interface (Model A + Model B)
# ─────────────────────────────────────────────────────────────────────────────

def test_dual_task_inference_predictions(mock_zenodo_artifacts, dummy_visible_landmarks):
    inferrer = DualTaskInference(
        artifacts=mock_zenodo_artifacts,
        confidence_threshold=0.40,
        verifier_threshold=0.50,
    )

    flags = [0] * 33
    feat_dict = extract_features(dummy_visible_landmarks, flags, 1280, 720)
    feat_series = features_to_series(feat_dict, mock_zenodo_artifacts.feature_cols)

    pred = inferrer.predict(feat_series)

    # Validates DualTaskPrediction fields
    assert isinstance(pred, DualTaskPrediction)
    assert isinstance(pred, PosePrediction)  # Subtype compatibility!
    assert pred.pose_name == "Tadasana"
    assert pred.confidence == 0.88
    assert pred.is_form_correct is True
    assert pred.correctness_label == "Right"
    assert pred.correctness_confidence == 0.92
    assert pred.status == "detected"


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Biomechanical Personalization v2.0 Adapts Target Ranges
# ─────────────────────────────────────────────────────────────────────────────

def test_biomechanical_personalization_adapts_targets():
    base_ranges = {
        "left_knee_angle": (85.0, 115.0),
        "right_knee_angle": (160.0, 185.0),
        "left_shoulder_angle": (150.0, 185.0),
    }

    # Case A: Standard uncalibrated profile -> ranges unchanged
    uncalibrated = UserProfile(calibrated=False)
    unchanged = apply_biomechanical_personalization(base_ranges, uncalibrated, "Virabhadrasana Two")
    assert unchanged["left_knee_angle"] == (85.0, 115.0)

    # Case B: Long-femur user (leg_to_torso_ratio = 1.82 > 1.68)
    long_femur = UserProfile(
        calibrated=True,
        leg_to_torso_ratio=1.82,
        arm_span_ratio=1.80,
        tolerance_multiplier=1.0,
    )
    adapted = apply_biomechanical_personalization(base_ranges, long_femur, "Virabhadrasana Two")
    # Lower bound on front knee should expand downwards to accommodate longer lever arm
    assert adapted["left_knee_angle"][0] < 85.0

    # Case C: Flexibility tolerance multiplier scales window
    beginner_profile = UserProfile(
        calibrated=True,
        leg_to_torso_ratio=1.65,
        tolerance_multiplier=1.15,
    )
    beg_adapted = apply_biomechanical_personalization(base_ranges, beginner_profile, "Tadasana")
    orig_span = base_ranges["right_knee_angle"][1] - base_ranges["right_knee_angle"][0]
    new_span = beg_adapted["right_knee_angle"][1] - beg_adapted["right_knee_angle"][0]
    assert new_span > orig_span, "Tolerance multiplier should expand the range span"


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: User Profile Validation & Graceful Fallback
# ─────────────────────────────────────────────────────────────────────────────

def test_user_profile_validation():
    # Valid profile
    valid_data = {
        "version": "2.0",
        "calibrated": True,
        "arm_span_ratio": 1.85,
        "leg_to_torso_ratio": 1.68,
        "shoulder_width_ratio": 0.65,
        "tolerance_multiplier": 1.05,
    }
    is_valid, _ = validate_profile_dict(valid_data)
    assert is_valid is True

    # Anatomically impossible arm span ratio (e.g. 5.5) -> fails validation
    corrupt_data = dict(valid_data, arm_span_ratio=5.5)
    is_valid, reason = validate_profile_dict(corrupt_data)
    assert is_valid is False
    assert "arm_span_ratio" in reason


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: Temporal Reliability Tracker & High-Frequency Jitter Detection
# ─────────────────────────────────────────────────────────────────────────────

def test_temporal_reliability_jitter_detection():
    tracker = TemporalReliabilityTracker(history_window=4, jitter_threshold=0.08)

    # Frame 1 to 3: Stable stationary landmarks
    base_lm = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.95} for _ in range(33)]
    base_lm[11] = {"x": 0.40, "y": 0.25, "z": 0.0, "visibility": 0.95}
    base_lm[23] = {"x": 0.40, "y": 0.55, "z": 0.0, "visibility": 0.95}  # Torso scale = 0.30

    ass1 = tracker.update(base_lm, [0] * 33)
    ass2 = tracker.update(base_lm, [0] * 33)
    ass3 = tracker.update(base_lm, [0] * 33)

    assert ass3.overall_tier == "high"
    assert ass3.is_overall_reliable is True

    # Frame 4: Artificial rapid jitter spike on left elbow (landmark 13)
    jittery_lm = [dict(lm) for lm in base_lm]
    jittery_lm[13]["x"] = 0.75  # Sharp unnatural displacement
    ass4 = tracker.update(jittery_lm, [0] * 33)

    # Left elbow angle should be flagged with high jitter / low or medium tier
    elbow_rel = ass4.joint_assessments.get("left_elbow_angle")
    assert elbow_rel is not None
    assert elbow_rel.jitter_norm > 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: Longitudinal Progress Analytics & Recommendations
# ─────────────────────────────────────────────────────────────────────────────

def test_longitudinal_progress_analytics(tmp_path):
    db_path = tmp_path / "longitudinal_test.db"
    tracker = ProgressTracker(db_path=db_path)

    # Session 1: Baseline session with lower score and recurring knee error
    tracker.start_session(calibrated=False)
    dummy_dev = {
        "joint_name": "left knee",
        "status": "too_high",
        "deviation_magnitude": 25.0,
    }
    tracker.log_frame("Tadasana", 72.0, [dummy_dev], is_reliable=True)
    tracker.log_frame("Tadasana", 74.0, [dummy_dev], is_reliable=True)
    tracker.end_session()

    # Session 2: Improved session with calibration
    tracker.start_session(calibrated=True)
    tracker.log_frame("Tadasana", 88.0, [], is_reliable=True)
    tracker.log_frame("Tadasana", 90.0, [], is_reliable=True)
    tracker.end_session()

    summary = tracker.get_longitudinal_summary(limit=10)
    assert summary["total_sessions"] == 2
    assert summary["improvement_delta"] > 0, "Second session had higher scores"
    assert len(summary["personalized_recommendations"]) > 0

    # Format report string contains expected sections
    report_text = tracker.format_progress_report(limit=10)
    assert "LONGITUDINAL PROGRESS" in report_text
    assert "RECOMMENDATIONS" in report_text


# ─────────────────────────────────────────────────────────────────────────────
# Test 11: Missing Zenodo Model Artifacts Fails Gracefully
# ─────────────────────────────────────────────────────────────────────────────

def test_missing_zenodo_artifacts_raises_informative_error():
    with pytest.raises(RuntimeError, match="Zenodo model directory not found"):
        load_zenodo_artifacts("/nonexistent/zenodo/models/dir")


# ─────────────────────────────────────────────────────────────────────────────
# Test 12: Backward Compatibility with Kaggle 47-Class Profile
# ─────────────────────────────────────────────────────────────────────────────

def test_profile_loader_backward_compatibility():
    # Calling with 'kaggle' loads existing 47-class model
    artifacts = load_profile_artifacts("kaggle")
    assert artifacts.n_classes == 47
    assert len(artifacts.feature_cols) == 39

    # Calling with invalid profile raises ValueError
    with pytest.raises(ValueError, match="Unknown model profile"):
        load_profile_artifacts("unsupported_profile_123")


# ─────────────────────────────────────────────────────────────────────────────
# Test 13: Gravitational Posture Prior
# ─────────────────────────────────────────────────────────────────────────────

def test_gravitational_posture_prior():
    from src.models.inference import compute_gravitational_posture_prior

    classes = [
        "Anantasana", "Ardhakati Chakrasana", "Bhujangasana", "Kati Chakrasana",
        "Marjariasana", "Parvatasana", "Sarvangasana", "Tadasana", "Vajrasana", "Viparita Karani"
    ]

    # 1. Standing landmarks: shoulders at 0.25, hips at 0.50, knees at 0.72, ankles at 0.92
    lm_standing = [{"x": 0.5, "y": 0.1, "visibility": 0.9} for _ in range(33)]
    lm_standing[11] = {"x": 0.40, "y": 0.25, "visibility": 0.9}
    lm_standing[12] = {"x": 0.60, "y": 0.25, "visibility": 0.9}
    lm_standing[23] = {"x": 0.42, "y": 0.50, "visibility": 0.9}
    lm_standing[24] = {"x": 0.58, "y": 0.50, "visibility": 0.9}
    lm_standing[25] = {"x": 0.43, "y": 0.72, "visibility": 0.9}
    lm_standing[26] = {"x": 0.57, "y": 0.72, "visibility": 0.9}
    lm_standing[27] = {"x": 0.43, "y": 0.92, "visibility": 0.9}
    lm_standing[28] = {"x": 0.57, "y": 0.92, "visibility": 0.9}

    prior_standing = compute_gravitational_posture_prior(lm_standing, classes)
    c_map_standing = dict(zip(classes, prior_standing))

    # Standing poses must have high prior weight
    assert c_map_standing["Tadasana"] > c_map_standing["Viparita Karani"] * 10
    assert c_map_standing["Kati Chakrasana"] > c_map_standing["Bhujangasana"] * 10
    # Prone, Inverted, and Seated poses must be suppressed
    assert c_map_standing["Bhujangasana"] < 0.05
    assert c_map_standing["Viparita Karani"] < 0.05
    assert c_map_standing["Sarvangasana"] < 0.05
    assert c_map_standing["Vajrasana"] < 0.05

    # 2. Kneeling upright in Vajrasana Step 1/2: shoulders at 0.28, hips at 0.55, knees at 0.70, ankles at 0.72
    # Wrists resting on knees (y=0.68)
    lm_kneeling = [{"x": 0.5, "y": 0.1, "visibility": 0.9} for _ in range(33)]
    lm_kneeling[11] = {"x": 0.40, "y": 0.28, "visibility": 0.9}
    lm_kneeling[12] = {"x": 0.60, "y": 0.28, "visibility": 0.9}
    lm_kneeling[15] = {"x": 0.43, "y": 0.68, "visibility": 0.9}
    lm_kneeling[16] = {"x": 0.57, "y": 0.68, "visibility": 0.9}
    lm_kneeling[23] = {"x": 0.42, "y": 0.55, "visibility": 0.9}
    lm_kneeling[24] = {"x": 0.58, "y": 0.55, "visibility": 0.9}
    lm_kneeling[25] = {"x": 0.43, "y": 0.70, "visibility": 0.9}
    lm_kneeling[26] = {"x": 0.57, "y": 0.70, "visibility": 0.9}
    lm_kneeling[27] = {"x": 0.43, "y": 0.72, "visibility": 0.2}  # Feet tucked under / low visibility
    lm_kneeling[28] = {"x": 0.57, "y": 0.72, "visibility": 0.2}

    prior_kneeling = compute_gravitational_posture_prior(lm_kneeling, classes)
    c_map_kneeling = dict(zip(classes, prior_kneeling))

    # Tadasana and standing poses must be strongly suppressed!
    assert c_map_kneeling["Tadasana"] < 0.05
    assert c_map_kneeling["Kati Chakrasana"] < 0.05
    # Vajrasana and Parvatasana must both be candidate poses for Step 1/2
    assert c_map_kneeling["Vajrasana"] > 0.30
    assert c_map_kneeling["Parvatasana"] > 0.30

    # 3. Kneeling in Parvatasana Step 3/4: arms raised overhead (wrists y=0.15, well above shoulders y=0.28)
    lm_parvatasana_step3 = [{"x": 0.5, "y": 0.1, "visibility": 0.9} for _ in range(33)]
    for i in range(33):
        lm_parvatasana_step3[i] = dict(lm_kneeling[i])
    lm_parvatasana_step3[15] = {"x": 0.48, "y": 0.15, "visibility": 0.9}  # wrists raised high overhead
    lm_parvatasana_step3[16] = {"x": 0.52, "y": 0.15, "visibility": 0.9}

    prior_parvatasana = compute_gravitational_posture_prior(lm_parvatasana_step3, classes)
    c_map_parvatasana = dict(zip(classes, prior_parvatasana))

    # Parvatasana MUST dominate Step 3/4! Vajrasana and Tadasana must be suppressed!
    assert c_map_parvatasana["Parvatasana"] > 0.80
    assert c_map_parvatasana["Vajrasana"] < 0.05
    assert c_map_parvatasana["Tadasana"] < 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Test 14: Calibrated Anthropometric Baseline Discrimination
# ─────────────────────────────────────────────────────────────────────────────

def test_calibrated_standing_vs_sitting_discrimination(tmp_path):
    from src.correction.calibration import CalibrationSession, UserProfile
    from src.models.inference import compute_gravitational_posture_prior

    classes = [
        "Anantasana", "Ardhakati Chakrasana", "Bhujangasana", "Kati Chakrasana",
        "Marjariasana", "Parvatasana", "Sarvangasana", "Tadasana", "Vajrasana", "Viparita Karani"
    ]

    profile_path = tmp_path / "test_user_profile.json"
    session = CalibrationSession(duration_seconds=0.05, target_frames=5, save_path=profile_path)
    session.start()

    # Standing frame for calibration
    lm_standing = [{"x": 0.5, "y": 0.1, "visibility": 0.9} for _ in range(33)]
    lm_standing[11] = {"x": 0.40, "y": 0.25, "visibility": 0.9}
    lm_standing[12] = {"x": 0.60, "y": 0.25, "visibility": 0.9}
    lm_standing[23] = {"x": 0.42, "y": 0.50, "visibility": 0.9}
    lm_standing[24] = {"x": 0.58, "y": 0.50, "visibility": 0.9}
    lm_standing[25] = {"x": 0.43, "y": 0.72, "visibility": 0.9}
    lm_standing[26] = {"x": 0.57, "y": 0.72, "visibility": 0.9}
    lm_standing[27] = {"x": 0.43, "y": 0.92, "visibility": 0.9}
    lm_standing[28] = {"x": 0.57, "y": 0.92, "visibility": 0.9}

    calib_done = False
    profile = None
    for _ in range(6):
        calib_done, msg, prof = session.update(lm_standing, 1280, 720)
        if calib_done:
            profile = prof
            break

    assert calib_done is True
    assert profile is not None
    assert profile.calibrated is True
    assert profile.thigh_to_torso_ratio > 0.5
    assert profile.leg_to_torso_ratio > 1.0

    # Test 1: With calibrated profile, standing landmarks activate Tadasana
    prior_standing = compute_gravitational_posture_prior(lm_standing, classes, user_profile=profile)
    c_map_standing = dict(zip(classes, prior_standing))
    assert c_map_standing["Tadasana"] > 0.25
    assert c_map_standing["Vajrasana"] < 0.05

    # Test 2: With calibrated profile, kneeling landmarks (shortened vertical thigh) activate Vajrasana/Parvatasana
    lm_kneeling = [{"x": 0.5, "y": 0.1, "visibility": 0.9} for _ in range(33)]
    lm_kneeling[11] = {"x": 0.40, "y": 0.28, "visibility": 0.9}
    lm_kneeling[12] = {"x": 0.60, "y": 0.28, "visibility": 0.9}
    lm_kneeling[15] = {"x": 0.43, "y": 0.65, "visibility": 0.9}
    lm_kneeling[16] = {"x": 0.57, "y": 0.65, "visibility": 0.9}
    lm_kneeling[23] = {"x": 0.42, "y": 0.55, "visibility": 0.9}
    lm_kneeling[24] = {"x": 0.58, "y": 0.55, "visibility": 0.9}
    lm_kneeling[25] = {"x": 0.43, "y": 0.65, "visibility": 0.9}  # compressed thigh
    lm_kneeling[26] = {"x": 0.57, "y": 0.65, "visibility": 0.9}
    lm_kneeling[27] = {"x": 0.43, "y": 0.68, "visibility": 0.2}
    lm_kneeling[28] = {"x": 0.57, "y": 0.68, "visibility": 0.2}

    prior_kneeling = compute_gravitational_posture_prior(lm_kneeling, classes, user_profile=profile)
    c_map_kneeling = dict(zip(classes, prior_kneeling))
    assert c_map_kneeling["Tadasana"] < 0.05
    assert c_map_kneeling["Vajrasana"] > 0.30
    assert c_map_kneeling["Parvatasana"] > 0.30


# ─────────────────────────────────────────────────────────────────────────────
# Test 15: Hierarchical Sequence Tracker & Dynamic Candidate Elimination
# ─────────────────────────────────────────────────────────────────────────────

def test_yoga_sequence_tracker_hierarchical_elimination():
    from src.models.sequence_tracker import YogaSequenceTracker, SequenceState

    classes = [
        "Anantasana", "Ardhakati Chakrasana", "Bhujangasana", "Kati Chakrasana",
        "Marjariasana", "Parvatasana", "Sarvangasana", "Tadasana", "Vajrasana", "Viparita Karani"
    ]

    tracker = YogaSequenceTracker(debounce_frames=1)

    # 1. STANDING Setup: Should prune seated, floor, prone, and inverted poses
    lm_standing = [{"x": 0.5, "y": 0.1, "visibility": 0.9} for _ in range(33)]
    lm_standing[11] = {"x": 0.40, "y": 0.25, "visibility": 0.9}
    lm_standing[12] = {"x": 0.60, "y": 0.25, "visibility": 0.9}
    lm_standing[23] = {"x": 0.42, "y": 0.50, "visibility": 0.9}
    lm_standing[24] = {"x": 0.58, "y": 0.50, "visibility": 0.9}
    lm_standing[25] = {"x": 0.43, "y": 0.72, "visibility": 0.9}
    lm_standing[26] = {"x": 0.57, "y": 0.72, "visibility": 0.9}
    lm_standing[27] = {"x": 0.43, "y": 0.92, "visibility": 0.9}
    lm_standing[28] = {"x": 0.57, "y": 0.92, "visibility": 0.9}

    state_standing = tracker.update(lm_standing, available_classes=classes)
    assert state_standing.base_posture == "STANDING"
    assert "Tadasana" in state_standing.active_candidates
    assert "Kati Chakrasana" in state_standing.active_candidates
    assert "Ardhakati Chakrasana" in state_standing.active_candidates
    assert "Vajrasana" in state_standing.pruned_poses
    assert "Bhujangasana" in state_standing.pruned_poses
    assert "Viparita Karani" in state_standing.pruned_poses

    # 2. SEATED / KNEELING Step 1: Retains both Vajrasana and Parvatasana, prunes standing
    lm_kneeling_step1 = [{"x": 0.5, "y": 0.1, "visibility": 0.9} for _ in range(33)]
    lm_kneeling_step1[11] = {"x": 0.40, "y": 0.28, "visibility": 0.9}
    lm_kneeling_step1[12] = {"x": 0.60, "y": 0.28, "visibility": 0.9}
    lm_kneeling_step1[15] = {"x": 0.43, "y": 0.65, "visibility": 0.9}  # wrists low on thighs
    lm_kneeling_step1[16] = {"x": 0.57, "y": 0.65, "visibility": 0.9}
    lm_kneeling_step1[23] = {"x": 0.42, "y": 0.55, "visibility": 0.9}
    lm_kneeling_step1[24] = {"x": 0.58, "y": 0.55, "visibility": 0.9}
    lm_kneeling_step1[25] = {"x": 0.43, "y": 0.65, "visibility": 0.9}  # compressed thigh
    lm_kneeling_step1[26] = {"x": 0.57, "y": 0.65, "visibility": 0.9}
    lm_kneeling_step1[27] = {"x": 0.43, "y": 0.68, "visibility": 0.2}
    lm_kneeling_step1[28] = {"x": 0.57, "y": 0.68, "visibility": 0.2}

    tracker.reset()
    state_kneeling1 = tracker.update(lm_kneeling_step1, available_classes=classes)
    assert state_kneeling1.base_posture == "SEATED_KNEELING"
    assert state_kneeling1.step == 1
    assert "Vajrasana" in state_kneeling1.active_candidates
    assert "Parvatasana" in state_kneeling1.active_candidates
    assert "Tadasana" in state_kneeling1.pruned_poses
    assert "Kati Chakrasana" in state_kneeling1.pruned_poses

    # 3. SEATED / KNEELING Step 3 (Arms Overhead): Prunes Vajrasana, locks onto Parvatasana
    lm_kneeling_step3 = [dict(lm) for lm in lm_kneeling_step1]
    lm_kneeling_step3[15] = {"x": 0.48, "y": 0.15, "visibility": 0.9}  # wrists high overhead
    lm_kneeling_step3[16] = {"x": 0.52, "y": 0.15, "visibility": 0.9}

    feat_overhead = {"left_shoulder_angle": 165.0, "right_shoulder_angle": 165.0}
    state_kneeling3 = tracker.update(lm_kneeling_step3, feature_dict=feat_overhead, available_classes=classes)
    assert state_kneeling3.base_posture == "SEATED_KNEELING"
    assert state_kneeling3.step >= 3
    assert "Parvatasana" in state_kneeling3.active_candidates
    assert "Vajrasana" not in state_kneeling3.active_candidates
    assert "Vajrasana" in state_kneeling3.pruned_poses

    # 4. Filter Probabilities: Zeroes out pruned poses and renormalizes
    uniform_proba = np.ones(len(classes)) / len(classes)
    filtered = tracker.filter_probabilities(uniform_proba, classes, state_kneeling3)
    c_filtered = dict(zip(classes, filtered))
    assert c_filtered["Parvatasana"] > 0.95
    assert c_filtered["Vajrasana"] < 0.01
    assert c_filtered["Tadasana"] < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Test 16: Dual-Task Inference Integration with Sequence State
# ─────────────────────────────────────────────────────────────────────────────

def test_dual_task_inference_with_sequence_tracker(mock_zenodo_artifacts):
    inferrer = DualTaskInference(
        artifacts=mock_zenodo_artifacts,
        confidence_threshold=0.30,
        verifier_threshold=0.50,
    )

    # Kneeling frame
    lm_kneeling = [{"x": 0.5, "y": 0.1, "visibility": 0.9} for _ in range(33)]
    lm_kneeling[11] = {"x": 0.40, "y": 0.28, "visibility": 0.9}
    lm_kneeling[12] = {"x": 0.60, "y": 0.28, "visibility": 0.9}
    lm_kneeling[13] = {"x": 0.41, "y": 0.45, "visibility": 0.9}
    lm_kneeling[14] = {"x": 0.59, "y": 0.45, "visibility": 0.9}
    lm_kneeling[15] = {"x": 0.43, "y": 0.65, "visibility": 0.9}
    lm_kneeling[16] = {"x": 0.57, "y": 0.65, "visibility": 0.9}
    lm_kneeling[23] = {"x": 0.42, "y": 0.55, "visibility": 0.9}
    lm_kneeling[24] = {"x": 0.58, "y": 0.55, "visibility": 0.9}
    lm_kneeling[25] = {"x": 0.43, "y": 0.65, "visibility": 0.9}
    lm_kneeling[26] = {"x": 0.57, "y": 0.65, "visibility": 0.9}
    lm_kneeling[27] = {"x": 0.43, "y": 0.68, "visibility": 0.2}
    lm_kneeling[28] = {"x": 0.57, "y": 0.68, "visibility": 0.2}

    flags = [0] * 33
    feat_dict = extract_features(lm_kneeling, flags, 1280, 720)
    feat_series = features_to_series(feat_dict, mock_zenodo_artifacts.feature_cols)

    pred = inferrer.predict(feat_series, landmarks=lm_kneeling)

    assert pred.sequence_state is not None
    assert pred.sequence_state.base_posture == "SEATED_KNEELING"
    assert "Tadasana" in pred.sequence_state.pruned_poses
    assert pred.sequence_state.step == 1
    # Tadasana probability must be virtually zero due to candidate elimination
    tad_idx = mock_zenodo_artifacts.class_names.index("Tadasana")
    assert pred.probabilities[tad_idx] < 0.05


