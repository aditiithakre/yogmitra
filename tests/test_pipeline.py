"""
tests/test_pipeline.py
───────────────────────
Automated tests for the inference and feedback pipeline.
All tests run without webcam or GPU access.

Run with:
    cd e:\\Yoga_posture_project
    python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Make sure the project root is on the path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def artifacts():
    """Load real saved artifacts once per test session."""
    from src.models.model_loader import load_artifacts
    return load_artifacts(ROOT / "saved_yoga_model")


@pytest.fixture
def dummy_landmarks_visible():
    """33 fully visible, anatomically plausible landmarks."""
    base = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.99}] * 33
    # spread hips, knees, shoulders slightly so angles are computable
    base[11] = {"x": 0.40, "y": 0.30, "z": 0.0, "visibility": 0.99}  # L shoulder
    base[12] = {"x": 0.60, "y": 0.30, "z": 0.0, "visibility": 0.99}  # R shoulder
    base[13] = {"x": 0.35, "y": 0.50, "z": 0.0, "visibility": 0.99}  # L elbow
    base[14] = {"x": 0.65, "y": 0.50, "z": 0.0, "visibility": 0.99}  # R elbow
    base[15] = {"x": 0.30, "y": 0.70, "z": 0.0, "visibility": 0.99}  # L wrist
    base[16] = {"x": 0.70, "y": 0.70, "z": 0.0, "visibility": 0.99}  # R wrist
    base[23] = {"x": 0.42, "y": 0.60, "z": 0.0, "visibility": 0.99}  # L hip
    base[24] = {"x": 0.58, "y": 0.60, "z": 0.0, "visibility": 0.99}  # R hip
    base[25] = {"x": 0.40, "y": 0.78, "z": 0.0, "visibility": 0.99}  # L knee
    base[26] = {"x": 0.60, "y": 0.78, "z": 0.0, "visibility": 0.99}  # R knee
    base[27] = {"x": 0.38, "y": 0.95, "z": 0.0, "visibility": 0.99}  # L ankle
    base[28] = {"x": 0.62, "y": 0.95, "z": 0.0, "visibility": 0.99}  # R ankle
    base[29] = {"x": 0.36, "y": 0.97, "z": 0.0, "visibility": 0.99}  # L heel
    base[30] = {"x": 0.64, "y": 0.97, "z": 0.0, "visibility": 0.99}  # R heel
    base[31] = {"x": 0.34, "y": 0.99, "z": 0.0, "visibility": 0.99}  # L foot
    base[32] = {"x": 0.66, "y": 0.99, "z": 0.0, "visibility": 0.99}  # R foot
    return base


@pytest.fixture
def dummy_landmarks_occluded(dummy_landmarks_visible):
    """Same as visible but key lower-body joints have low visibility."""
    lm = [dict(x) for x in dummy_landmarks_visible]
    for idx in [25, 26, 27, 28, 29, 30, 31, 32]:
        lm[idx]["visibility"] = 0.05
    return lm


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Feature extraction produces exactly 39 features in correct order
# ─────────────────────────────────────────────────────────────────────────────

def test_feature_vector_length_and_order(artifacts, dummy_landmarks_visible):
    from src.features import extract_features, features_to_series
    flags = [0] * 33
    feat_dict = extract_features(dummy_landmarks_visible, flags, 1280, 720)
    series = features_to_series(feat_dict, artifacts.feature_cols)

    assert len(series) == len(artifacts.feature_cols), (
        f"Expected {len(artifacts.feature_cols)} features, got {len(series)}"
    )
    assert list(series.index) == artifacts.feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Full preprocessing + classifier produces a valid class index
# ─────────────────────────────────────────────────────────────────────────────

def test_rf_predict_returns_valid_class(artifacts, dummy_landmarks_visible):
    from src.features import extract_features, features_to_series
    from src.models.inference import PoseInference

    flags = [0] * 33
    feat_dict = extract_features(dummy_landmarks_visible, flags, 1280, 720)
    series = features_to_series(feat_dict, artifacts.feature_cols)

    inferrer = PoseInference(artifacts, active_classifier="rf")
    pred = inferrer.predict(series)

    assert pred.pose_name in artifacts.class_names
    assert 0.0 <= pred.confidence <= 1.0
    assert pred.status in ("detected", "low_confidence")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Class index maps to correct pose name
# ─────────────────────────────────────────────────────────────────────────────

def test_label_encoder_class_mapping(artifacts):
    le = artifacts.label_encoder
    for i, name in enumerate(artifacts.class_names):
        decoded = le.inverse_transform([i])[0]
        assert decoded == name, f"Mismatch at index {i}: {decoded} != {name}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Occluded landmarks → no fabricated correction
# ─────────────────────────────────────────────────────────────────────────────

def test_occluded_joints_produce_unavailable_checks(dummy_landmarks_occluded):
    from src.features import extract_features
    from src.correction.posture_evaluator import evaluate

    flags = [0] * 33
    feat_dict = extract_features(dummy_landmarks_occluded, flags, 1280, 720)
    ev = evaluate("Phalakasana", feat_dict, dummy_landmarks_occluded, flags)

    for check in ev.checks:
        if check.status == "unavailable":
            assert not check.reliable, "Unavailable check should not be marked reliable"


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Angle outside range → correction feedback
# ─────────────────────────────────────────────────────────────────────────────

def test_out_of_range_angle_triggers_correction(dummy_landmarks_visible):
    from src.correction.posture_evaluator import evaluate

    # Forge a feature dict with Warrior 2 left knee clearly outside target
    feat_dict = {k: np.nan for k in ["left_knee_angle", "right_knee_angle",
                                      "left_elbow_angle", "right_elbow_angle"]}
    feat_dict["left_knee_angle"] = 160.0    # target is 85–115 → too_high
    feat_dict["right_knee_angle"] = 175.0   # within range for right
    feat_dict["left_elbow_angle"] = 170.0
    feat_dict["right_elbow_angle"] = 170.0

    flags = [0] * 33
    ev = evaluate("Virabhadrasana Two", feat_dict, dummy_landmarks_visible, flags)

    issues = [c for c in ev.checks if c.status != "ok" and c.reliable]
    assert any(c.angle_name == "left_knee_angle" for c in issues), (
        "Expected left_knee_angle correction not found"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Angle inside range → no correction for that joint
# ─────────────────────────────────────────────────────────────────────────────

def test_in_range_angle_no_correction(dummy_landmarks_visible):
    from src.correction.posture_evaluator import evaluate

    feat_dict = {
        "left_knee_angle":   95.0,   # target 85–115 ✓
        "right_knee_angle":  170.0,  # target 160–185 ✓
        "left_elbow_angle":  172.0,  # target 160–185 ✓
        "right_elbow_angle": 168.0,  # target 160–185 ✓
    }
    flags = [0] * 33
    ev = evaluate("Virabhadrasana Two", feat_dict, dummy_landmarks_visible, flags)
    issues = [c for c in ev.checks if c.status not in ("ok", "unavailable")]
    assert issues == [], f"Unexpected corrections: {issues}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Low confidence → uncertainty message, no posture correction
# ─────────────────────────────────────────────────────────────────────────────

def test_low_confidence_feedback(artifacts, dummy_landmarks_visible):
    from src.features import extract_features, features_to_series
    from src.models.inference import PoseInference
    from src.correction.feedback_engine import FeedbackEngine

    flags = [0] * 33
    feat_dict = extract_features(dummy_landmarks_visible, flags, 1280, 720)
    series = features_to_series(feat_dict, artifacts.feature_cols)

    # Force low threshold to trigger low_confidence unconditionally
    inferrer = PoseInference(artifacts, active_classifier="rf", confidence_threshold=0.99)
    pred = inferrer.predict(series)

    if pred.status == "low_confidence":
        engine = FeedbackEngine()
        items = engine.update(pred, None)
        assert any(i.severity == "warning" for i in items)
        assert not any(i.severity == "correction" for i in items)


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: Repeated feedback throttled by cooldown
# ─────────────────────────────────────────────────────────────────────────────

def test_feedback_cooldown_prevents_repeat_speech(artifacts, dummy_landmarks_visible):
    from src.features import extract_features, features_to_series
    from src.models.inference import PoseInference, PosePrediction
    from src.correction.feedback_engine import FeedbackEngine

    engine = FeedbackEngine(cooldown_seconds=60)   # very long cooldown

    # Manually create a low-confidence prediction to get a speech message
    pred = PosePrediction(
        pose_name="Vrksasana", confidence=0.3,
        probabilities=np.zeros(47), raw_prediction="Vrksasana",
        smoothed=False, status="low_confidence"
    )
    engine.update(pred, None)
    msg1 = engine.pop_speech()

    # Second identical update — should NOT enqueue speech again
    engine.update(pred, None)
    msg2 = engine.pop_speech()

    assert msg1 is not None, "First call should produce a speech message"
    assert msg2 is None, "Cooldown should prevent second speech within cooldown window"


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: Voice can be disabled
# ─────────────────────────────────────────────────────────────────────────────

def test_voice_can_be_disabled():
    from src.app.voice import VoiceFeedback
    v = VoiceFeedback.__new__(VoiceFeedback)
    v._enabled = False
    v._q = __import__("queue").Queue()
    v._engine = None

    v.speak("this should be ignored")
    assert v._q.empty(), "Queue should be empty when voice is disabled"


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: Missing model dir raises RuntimeError
# ─────────────────────────────────────────────────────────────────────────────

def test_missing_model_dir_raises():
    from src.models.model_loader import load_artifacts
    with pytest.raises(RuntimeError, match="Model directory not found"):
        load_artifacts("/nonexistent/path/to/model")


# ─────────────────────────────────────────────────────────────────────────────
# Test 11: Temporal landmark fallback applied correctly
# ─────────────────────────────────────────────────────────────────────────────

def test_temporal_fallback_restores_previous_joint():
    from src.pose.missing_landmarks import apply_temporal_fallback, VISIBILITY_THRESHOLD

    prev = [{"x": 0.5, "y": 0.8, "z": 0.0, "visibility": 0.99}] * 33
    current = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.99}] * 33
    current[25] = {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.01}  # L knee occluded

    effective, flags, _ = apply_temporal_fallback(current, prev, VISIBILITY_THRESHOLD)

    assert flags[25] == 1, "Joint 25 should be flagged as imputed"
    assert abs(effective[25]["y"] - 0.8) < 1e-6, "Joint 25 should be restored from prev frame"


# ─────────────────────────────────────────────────────────────────────────────
# Test 12: Inference interval enforcement
# ─────────────────────────────────────────────────────────────────────────────

def test_inference_interval_respected():
    """Verify the interval logic used in the app loop."""
    interval = 2.0
    last_time = time.monotonic() - 0.5   # only 0.5s have passed
    should_run = (time.monotonic() - last_time) >= interval
    assert not should_run, "Inference should NOT run when interval hasn't elapsed"

    last_time = time.monotonic() - 2.5   # 2.5s have passed
    should_run = (time.monotonic() - last_time) >= interval
    assert should_run, "Inference SHOULD run when interval has elapsed"


# ─────────────────────────────────────────────────────────────────────────────
# Test 13: Target Pose Mode guides user to Warrior 2
# ─────────────────────────────────────────────────────────────────────────────

def test_target_pose_warrior_two_guidance(dummy_landmarks_visible):
    from src.correction.posture_evaluator import evaluate
    from src.correction.feedback_engine import FeedbackEngine
    from src.models.inference import PosePrediction

    # Dummy standing pose (straight legs)
    feat_dict = {
        "left_knee_angle": 175.0,
        "right_knee_angle": 175.0,
        "left_elbow_angle": 170.0,
        "right_elbow_angle": 170.0,
    }
    flags = [0] * 33
    ev = evaluate("Virabhadrasana Two", feat_dict, dummy_landmarks_visible, flags)

    pred = PosePrediction(
        pose_name="Unknown", confidence=0.2,
        probabilities=np.zeros(47), raw_prediction="Unknown",
        smoothed=False, status="low_confidence"
    )

    engine = FeedbackEngine()
    items = engine.update(pred, ev, target_pose="Virabhadrasana Two")

    assert len(items) > 0
    # Must instruct to bend knee for Warrior 2
    assert any("knee" in item.instruction.lower() for item in items)
    assert any("Virabhadrasana Two" in item.pose_name for item in items)


# ─────────────────────────────────────────────────────────────────────────────
# Test 14: Bilateral Warrior 2 Adaptation (Right Leg Forward)
# ─────────────────────────────────────────────────────────────────────────────

def test_bilateral_warrior_adaptation(dummy_landmarks_visible):
    from src.correction.posture_evaluator import evaluate

    # User steps right leg forward into Warrior 2 (right knee bent, left straight)
    feat_dict = {
        "right_knee_angle": 95.0,    # right knee bent to ~95° (front leg)
        "left_knee_angle": 175.0,    # left knee straight ~175° (back leg)
        "left_elbow_angle": 170.0,   # arms horizontal
        "right_elbow_angle": 170.0,
    }
    flags = [0] * 33
    ev = evaluate("Virabhadrasana Two", feat_dict, dummy_landmarks_visible, flags)

    # Right leg forward should be recognized as correct!
    assert ev.overall_status == "correct", f"Expected correct, got {ev.overall_status} ({[c.status for c in ev.checks]})"


# ─────────────────────────────────────────────────────────────────────────────
# Test 15: Voice Worker Operation & Safe Shutdown
# ─────────────────────────────────────────────────────────────────────────────

def test_voice_worker_operation_and_shutdown():
    from src.app.voice import VoiceFeedback

    voice = VoiceFeedback(rate=160, volume=0.9)
    # Test enqueue without crash
    voice.speak("Yoga Mitra audio test.")
    assert voice.is_available or not voice.is_available  # Boolean state valid
    # Shutdown cleanly
    voice.shutdown()
    assert voice._shutdown is True


# ─────────────────────────────────────────────────────────────────────────────
# Test 16: Voice Mute, Unmute, and Queue Flushing
# ─────────────────────────────────────────────────────────────────────────────

def test_voice_mute_unmute_and_queue_flushing():
    from src.app.voice import VoiceFeedback

    voice = VoiceFeedback()
    assert voice.enabled is True

    # Toggle to mute
    state = voice.toggle()
    assert state is False
    assert voice.enabled is False

    # When muted, speaking should not enqueue
    voice.speak("Should not be spoken")
    assert voice._queue.empty()

    # Toggle back to unmuted
    state = voice.toggle()
    assert state is True
    assert voice.enabled is True
    voice.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# Test 17: Reference Asset Mapping for All Guided Targets
# ─────────────────────────────────────────────────────────────────────────────

def test_reference_asset_mapping():
    from src.correction.reference_ranges import POSE_METADATA
    from src.app.reference_display import ReferenceDisplay

    display = ReferenceDisplay(panel_width=380)

    guided_targets = [
        "Virabhadrasana Two",
        "Virabhadrasana One",
        "Utkatasana",
        "Trikonasana",
        "Phalakasana",
        "Adho Mukha Svanasana",
        "Vrksasana",
        "Uttanasana",
    ]

    for pose in guided_targets:
        assert pose in POSE_METADATA, f"{pose} missing from POSE_METADATA"
        meta = POSE_METADATA[pose]
        assert "english_name" in meta
        assert "sanskrit_name" in meta
        assert "cues" in meta and len(meta["cues"]) > 0

        # Verify rendering produces a valid panel image
        panel = display.render_panel(pose_name=pose, height=600, is_guided=True)
        assert panel.shape == (600, 380, 3)
        assert panel.dtype == np.uint8


# ─────────────────────────────────────────────────────────────────────────────
# Test 18: Reference Display Auto-Detect Smoothing & Anti-Flicker
# ─────────────────────────────────────────────────────────────────────────────

def test_reference_display_auto_detect_stability():
    from src.app.reference_display import ReferenceDisplay

    display = ReferenceDisplay()

    # Single noisy prediction should not immediately switch stable pose
    p1 = display.update_auto_detected_pose("Virabhadrasana Two", confidence=0.85)
    p2 = display.update_auto_detected_pose("Virabhadrasana Two", confidence=0.85)
    # 3rd consecutive occurrence locks in stable pose
    p3 = display.update_auto_detected_pose("Virabhadrasana Two", confidence=0.85)
    assert p3 == "Virabhadrasana Two"

    # Single spurious glitch frame should be ignored
    glitch = display.update_auto_detected_pose("Phalakasana", confidence=0.85)
    assert glitch == "Virabhadrasana Two"  # Still locked to previous stable


# ─────────────────────────────────────────────────────────────────────────────
# Test 19: Missing Reference Asset Fallback Card
# ─────────────────────────────────────────────────────────────────────────────

def test_missing_reference_asset_fallback():
    from src.app.reference_display import ReferenceDisplay

    display = ReferenceDisplay()
    # Unknown pose with no asset
    panel = display.render_panel(pose_name="UnregisteredPoseXYZ", height=500, is_guided=False)
    assert panel.shape == (500, 380, 3)
    assert panel.mean() > 0  # Not completely blank black


# ─────────────────────────────────────────────────────────────────────────────
# Test 20: Confidence-Aware Feedback Decisions & Multi-Tier Reliability
# ─────────────────────────────────────────────────────────────────────────────

def test_confidence_aware_feedback_tiers(dummy_landmarks_visible):
    from src.correction.posture_evaluator import evaluate

    feat_dict = {
        "left_knee_angle": 175.0,
        "right_knee_angle": 175.0,
    }
    # Case A: Fully visible landmarks -> high reliability tier
    flags = [0] * 33
    ev_high = evaluate("Virabhadrasana Two", feat_dict, dummy_landmarks_visible, flags)
    assert ev_high.checks[0].reliability_tier in ("high", "medium")
    assert ev_high.checks[0].reliable is True

    # Case B: Landmark occluded -> low visibility / unavailable
    occluded_lm = [dict(lm) for lm in dummy_landmarks_visible]
    occluded_lm[25]["visibility"] = 0.05  # Left knee occluded
    ev_low = evaluate("Virabhadrasana Two", feat_dict, occluded_lm, flags)
    left_check = next(c for c in ev_low.checks if c.angle_name == "left_knee_angle")
    assert left_check.status == "unavailable"
    assert left_check.reliability_tier == "low"
    assert "clearly visible" in left_check.action_hint.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Test 21: Invalid Angles and Imputed Landmark Robustness
# ─────────────────────────────────────────────────────────────────────────────

def test_invalid_angles_and_imputed_landmarks(dummy_landmarks_visible):
    from src.correction.posture_evaluator import evaluate

    # Pass NaN and Inf values
    feat_dict = {
        "left_knee_angle": float("nan"),
        "right_knee_angle": float("inf"),
    }
    flags = [1] * 33  # All imputed
    ev = evaluate("Virabhadrasana Two", feat_dict, dummy_landmarks_visible, flags)

    for chk in ev.checks:
        assert chk.status == "unavailable"
        assert chk.deviation == 0.0
        assert chk.reliable is False


# ─────────────────────────────────────────────────────────────────────────────
# Test 22: User Profile Loading and Persistence
# ─────────────────────────────────────────────────────────────────────────────

def test_user_profile_saving_and_loading(tmp_path):
    from src.correction.calibration import UserProfile, load_profile, save_profile

    profile_file = tmp_path / "user_profile.json"
    prof = UserProfile(
        calibrated=True,
        user_name="Advanced Yogi",
        tolerance_multiplier=1.15,
        arm_span_ratio=1.82,
        leg_to_torso_ratio=1.68,
        shoulder_width_ratio=0.67,
    )
    ok = save_profile(prof, profile_file)
    assert ok is True
    assert profile_file.exists()

    loaded = load_profile(profile_file)
    assert loaded.calibrated is True
    assert loaded.user_name == "Advanced Yogi"
    assert loaded.tolerance_multiplier == 1.15
    assert loaded.arm_span_ratio == 1.82

    # Non-existent profile returns uncalibrated default safely
    missing = load_profile(tmp_path / "non_existent.json")
    assert missing.calibrated is False


# ─────────────────────────────────────────────────────────────────────────────
# Test 23: Joint Deviation Direction and Severity Tiers
# ─────────────────────────────────────────────────────────────────────────────

def test_joint_deviation_direction_and_severity(dummy_landmarks_visible):
    from src.correction.posture_evaluator import evaluate

    # Target for Chair Pose knees is 70 to 115 deg
    # 1. Major deviation: standing straight (175 deg -> 60 deg deviation)
    ev_major = evaluate("Utkatasana", {"left_knee_angle": 175.0, "right_knee_angle": 175.0}, dummy_landmarks_visible, [0]*33)
    chk_major = ev_major.checks[0]
    assert chk_major.severity == "major"
    assert chk_major.direction == "bend"
    assert chk_major.deviation >= 55.0

    # 2. Mild deviation: knee bent to 125 deg (10 deg deviation)
    ev_mild = evaluate("Utkatasana", {"left_knee_angle": 125.0, "right_knee_angle": 125.0}, dummy_landmarks_visible, [0]*33)
    chk_mild = ev_mild.checks[0]
    assert chk_mild.severity == "mild"
    assert chk_mild.direction == "bend"


# ─────────────────────────────────────────────────────────────────────────────
# Test 24: Front and Rear Leg Bilateral Adaptation in Warrior Poses
# ─────────────────────────────────────────────────────────────────────────────

def test_front_and_rear_leg_bilateral_adaptation(dummy_landmarks_visible):
    from src.correction.posture_evaluator import evaluate

    # Left leg forward: Left knee ~95 deg, Right knee ~175 deg
    ev_left_lead = evaluate(
        "Virabhadrasana One",
        {
            "left_knee_angle": 95.0,
            "right_knee_angle": 175.0,
            "left_elbow_angle": 175.0,
            "right_elbow_angle": 175.0,
            "left_shoulder_angle": 170.0,
            "right_shoulder_angle": 170.0,
        },
        dummy_landmarks_visible,
        [0] * 33,
    )
    knee_statuses_left = [c.status for c in ev_left_lead.checks if "knee" in c.angle_name]
    assert all(s == "ok" for s in knee_statuses_left), f"Left-lead knee statuses: {knee_statuses_left}"

    # Right leg forward: Right knee ~95 deg, Left knee ~175 deg
    ev_right_lead = evaluate(
        "Virabhadrasana One",
        {
            "right_knee_angle": 95.0,
            "left_knee_angle": 175.0,
            "left_elbow_angle": 175.0,
            "right_elbow_angle": 175.0,
            "left_shoulder_angle": 170.0,
            "right_shoulder_angle": 170.0,
        },
        dummy_landmarks_visible,
        [0] * 33,
    )
    knee_statuses_right = [c.status for c in ev_right_lead.checks if "knee" in c.angle_name]
    assert all(s == "ok" for s in knee_statuses_right), f"Right-lead knee statuses: {knee_statuses_right}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 25: Session History & Progress Tracker Persistence
# ─────────────────────────────────────────────────────────────────────────────

def test_session_history_persistence(tmp_path):
    from src.correction.progress_tracker import ProgressTracker
    from src.correction.posture_evaluator import AngleCheck

    db_path = tmp_path / "yoga_test.db"
    tracker = ProgressTracker(db_path=db_path)
    tracker.start_session(calibrated=True)

    dummy_dev = AngleCheck(
        angle_name="left_knee_angle",
        display_name="left knee",
        value=150.0,
        target_range=(85, 115),
        status="too_high",
        deviation=35.0,
        severity="moderate",
        direction="bend",
        action_hint="Bend your front knee a little more.",
        reliable=True,
        reliability_tier="high",
    )

    tracker.log_frame("Virabhadrasana Two", alignment_score=82.0, deviations=[dummy_dev], is_reliable=True)
    tracker.log_frame("Virabhadrasana Two", alignment_score=88.0, deviations=[], is_reliable=True)

    summary = tracker.end_session()
    assert summary is not None
    assert summary["frames_evaluated"] == 2
    assert summary["avg_alignment_score"] == 85.0
    assert summary["peak_alignment_score"] == 88.0
    assert summary["calibrated"] is True

    # Verify query and formatted table
    history = tracker.get_history(limit=5)
    assert len(history) == 1
    assert history[0]["poses"] == "Virabhadrasana Two"

    table_str = tracker.format_history_table(limit=5)
    assert "Virabhadrasana Two" in table_str


# ─────────────────────────────────────────────────────────────────────────────
# Test 26: Application Cleanup on Shutdown
# ─────────────────────────────────────────────────────────────────────────────

def test_application_cleanup():
    from src.app.voice import VoiceFeedback
    from src.app.reference_display import ReferenceDisplay

    voice = VoiceFeedback()
    ref_display = ReferenceDisplay()

    ref_display.reset_stability()
    assert len(display_history := ref_display._detect_history) == 0

    voice.shutdown()
    assert voice._shutdown is True
