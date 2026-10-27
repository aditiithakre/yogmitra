"""
tests/test_reliability_fixes.py
───────────────────────────────
Regression tests for confidence gating (Fix 3), candidate filters (Fix 4),
artifact / schema compatibility (Fix 5) and tiered corrections (Fix 8).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.correction.feedback_engine import FeedbackEngine
from src.correction.posture_evaluator import evaluate as evaluate_posture
from src.correction.reliability_tracker import TemporalReliabilityTracker
from src.features import extract_features, features_to_series
from src.models.inference import DualTaskInference, FilterPolicy, PosePrediction
from src.models.jev_fusion import FusionConfig, decide
from src.models.model_loader import YogaArtifactsZenodo
from src.models.zenodo_annotations import ZENODO_ASANAS

ROOT = Path(__file__).resolve().parent.parent
CLASSES = sorted(ZENODO_ASANAS)


# ── Synthetic body orientations (all major joints clearly visible) ────────────

def _body(points: dict[int, tuple[float, float]], vis: float = 0.95) -> list[dict]:
    lm = [{"x": 0.5, "y": 0.1, "z": 0.0, "visibility": vis} for _ in range(33)]
    for i, (x, y) in points.items():
        lm[i] = {"x": x, "y": y, "z": 0.0, "visibility": vis}
    return lm


STANDING = _body({11: (.40, .25), 12: (.60, .25), 13: (.38, .40), 14: (.62, .40), 15: (.37, .55), 16: (.63, .55),
                  23: (.42, .50), 24: (.58, .50), 25: (.43, .72), 26: (.57, .72), 27: (.43, .92), 28: (.57, .92),
                  31: (.45, .95), 32: (.55, .95)})
KNEELING = _body({11: (.40, .28), 12: (.60, .28), 13: (.41, .45), 14: (.59, .45), 15: (.43, .65), 16: (.57, .65),
                  23: (.42, .55), 24: (.58, .55), 25: (.43, .65), 26: (.57, .65), 27: (.43, .68), 28: (.57, .68),
                  31: (.40, .69), 32: (.60, .69)})
PRONE = _body({11: (.30, .60), 12: (.32, .61), 13: (.30, .66), 14: (.32, .66), 15: (.31, .68), 16: (.33, .68),
               23: (.55, .62), 24: (.57, .62), 25: (.72, .64), 26: (.74, .64), 27: (.90, .65), 28: (.92, .65),
               31: (.95, .66), 32: (.96, .66)})
QUADRUPED = _body({11: (.30, .45), 12: (.32, .46), 13: (.30, .58), 14: (.32, .58), 15: (.30, .70), 16: (.32, .70),
                   23: (.60, .47), 24: (.62, .48), 25: (.60, .70), 26: (.62, .70), 27: (.85, .70), 28: (.87, .70),
                   31: (.90, .71), 32: (.91, .71)})
INVERTED = _body({11: (.40, .80), 12: (.60, .80), 13: (.38, .85), 14: (.62, .85), 15: (.36, .88), 16: (.64, .88),
                  23: (.42, .60), 24: (.58, .60), 25: (.43, .40), 26: (.57, .40), 27: (.43, .20), 28: (.57, .20),
                  31: (.45, .15), 32: (.55, .15)})


class ControlledModel:
    """Classifier stub whose top class is set by the test."""

    def __init__(self, classes, p_top=0.7):
        self.classes = classes
        self.truth = classes[0]
        self.p_top = p_top
        self.classes_ = np.arange(len(classes))

    def predict_proba(self, X):
        p = np.full(len(self.classes), (1 - self.p_top) / (len(self.classes) - 1))
        p[self.classes.index(self.truth)] = self.p_top
        return np.array([p] * len(X))


@pytest.fixture
def controlled_artifacts():
    f = extract_features(STANDING, [0] * 33)
    cols = list(f.keys())
    imputer = SimpleImputer(strategy="median", add_indicator=True).fit(pd.DataFrame([f]))
    scaler = StandardScaler().fit(imputer.transform(pd.DataFrame([f])))
    asana_le = LabelEncoder().fit(CLASSES)
    corr_le = LabelEncoder().fit(["Right", "Wrong"])
    verifier = ControlledModel(["Right", "Wrong"], p_top=0.9)
    return YogaArtifactsZenodo(imputer=imputer, scaler=scaler, asana_label_encoder=asana_le,
                               correctness_label_encoder=corr_le, feature_cols=cols,
                               asana_model=ControlledModel(CLASSES), verifier_model=verifier,
                               class_names=CLASSES, correctness_names=["Right", "Wrong"], model_type="rf")


def _run(inf, art, lm, truth, n=1):
    art.asana_model.truth = truth
    pred = None
    for _ in range(n):
        feats = features_to_series(extract_features(lm, [0] * 33), art.feature_cols)
        pred = inf.predict(feats, landmarks=lm)
    return pred


# ── Fix 4: candidate filters ──────────────────────────────────────────────────

TRANSITIONS = [(STANDING, "Tadasana"), (KNEELING, "Vajrasana"), (PRONE, "Bhujangasana"),
               (QUADRUPED, "Marjariasana"), (INVERTED, "Viparita Karani"), (STANDING, "Kati Chakrasana")]


@pytest.mark.parametrize("policy", [
    FilterPolicy(mode="safe", granularity="coarse", soft_floor=0.5, stable_frames=3),
    FilterPolicy(mode="safe", granularity="family", soft_floor=0.05, stable_frames=1, use_sequence_pruning=False),
    FilterPolicy(use_gravity_prior=False, use_sequence_pruning=False),
])
def test_transitions_between_orientations_recover(controlled_artifacts, policy):
    """After each orientation change the correct pose wins within 2 inferences."""
    inf = DualTaskInference(controlled_artifacts, confidence_threshold=0.3, filter_policy=policy)
    for lm, truth in TRANSITIONS:
        preds = [_run(inf, controlled_artifacts, lm, truth).pose_name for _ in range(4)]
        assert truth in preds[:2], (truth, preds)
        assert preds[-1] == truth


def test_detected_orientations_match_fixtures():
    from src.models.sequence_tracker import YogaSequenceTracker
    t = YogaSequenceTracker()
    assert [t.detect_base_posture(lm) for lm in (STANDING, KNEELING, PRONE, QUADRUPED, INVERTED)] == \
        ["STANDING", "SEATED_KNEELING", "PRONE", "QUADRUPED", "INVERTED"]


def test_legacy_filters_suppress_correct_class_when_geometry_is_wrong(controlled_artifacts):
    """Measured failure mode: Bhujangasana frames whose landmarks look INVERTED (43% on val+test)."""
    legacy = DualTaskInference(controlled_artifacts, confidence_threshold=0.3, filter_policy=FilterPolicy())
    pred = _run(legacy, controlled_artifacts, INVERTED, "Bhujangasana", n=3)
    assert pred.pose_name != "Bhujangasana"
    assert "Bhujangasana" in pred.filter_decision.suppressed


def test_safe_coarse_keeps_correct_class_within_same_coarse_group(controlled_artifacts):
    safe = DualTaskInference(controlled_artifacts, confidence_threshold=0.3,
                             filter_policy=FilterPolicy(mode="safe", granularity="coarse"))
    pred = _run(safe, controlled_artifacts, INVERTED, "Bhujangasana", n=3)
    assert pred.pose_name == "Bhujangasana"
    assert "Bhujangasana" not in pred.filter_decision.suppressed


def test_safe_filters_skip_when_leg_landmarks_unreliable(controlled_artifacts):
    lm = [dict(p) for p in STANDING]
    for i in (25, 26, 27, 28):
        lm[i]["visibility"] = 0.2
    safe = DualTaskInference(controlled_artifacts, filter_policy=FilterPolicy(mode="safe"))
    pred = _run(safe, controlled_artifacts, lm, "Bhujangasana", n=4)
    assert pred.filter_decision.skipped_reason == "leg landmarks not reliably observed"
    assert pred.filter_decision.suppressed == []
    assert pred.filter_decision.coarse_group is None


def test_safe_filters_skip_imputed_torso(controlled_artifacts):
    lm = [dict(p) for p in STANDING]
    lm[23] = dict(lm[23], imputed=True, visibility=0.35)
    safe = DualTaskInference(controlled_artifacts, filter_policy=FilterPolicy(mode="safe"))
    pred = _run(safe, controlled_artifacts, lm, "Tadasana", n=4)
    assert pred.filter_decision.skipped_reason == "torso landmarks not reliably observed"


def test_safe_filters_wait_for_stability(controlled_artifacts):
    safe = DualTaskInference(controlled_artifacts,
                             filter_policy=FilterPolicy(mode="safe", stable_frames=3, granularity="coarse",
                                                          override_confidence=0.95))
    p1 = _run(safe, controlled_artifacts, STANDING, "Bhujangasana")
    assert "not yet stable" in (p1.filter_decision.skipped_reason or "")
    p3 = _run(safe, controlled_artifacts, STANDING, "Bhujangasana", n=2)
    assert p3.filter_decision.skipped_reason is None and "Bhujangasana" in p3.filter_decision.suppressed


def test_override_confidence_preserves_strong_classifier_belief(controlled_artifacts):
    controlled_artifacts.asana_model.p_top = 0.9
    safe = DualTaskInference(controlled_artifacts, filter_policy=FilterPolicy(
        mode="safe", stable_frames=1, override_confidence=0.6, soft_floor=0.01))
    pred = _run(safe, controlled_artifacts, STANDING, "Bhujangasana", n=2)
    assert "Bhujangasana" in pred.filter_decision.overridden
    assert pred.pose_name == "Bhujangasana"


def test_filters_disabled_returns_classifier_probabilities(controlled_artifacts):
    inf = DualTaskInference(controlled_artifacts, filter_policy=FilterPolicy(
        mode="safe", use_gravity_prior=False, use_sequence_pruning=False))
    pred = _run(inf, controlled_artifacts, INVERTED, "Tadasana")
    assert np.allclose(pred.probabilities, pred.raw_probabilities)


def test_filter_policy_from_config_ignores_unknown_keys():
    p = FilterPolicy.from_config({"mode": "safe", "soft_floor": 0.3, "bogus": 1})
    assert p.mode == "safe" and p.soft_floor == 0.3


# ── Fix 3: auto-detect gating / guided preservation ───────────────────────────

def _pred(name, conf, coarse=None, status=None):
    return SimpleNamespace(pose_name=name, confidence=conf, top3=[(name, conf)],
                           status=status or ("detected" if conf >= 0.5 else "low_confidence"),
                           filter_decision=SimpleNamespace(coarse_group=coarse))


def test_orientation_veto_blocks_contradicting_label():
    cfg = FusionConfig(local_threshold=0.5)
    d = decide(_pred("Anantasana", 0.96, coarse="UPRIGHT"), "high", "auto", CLASSES, 0, cfg)
    assert not d.assessment_allowed and "orientation" in d.reason
    d = decide(_pred("Tadasana", 0.96, coarse="UPRIGHT"), "high", "auto", CLASSES, 0, cfg)
    assert d.assessment_allowed
    d = decide(_pred("Anantasana", 0.96, coarse=None), "high", "auto", CLASSES, 0, cfg)
    assert d.assessment_allowed  # no reliable orientation evidence → no veto


def test_new_label_needs_consecutive_confirmation():
    cfg = FusionConfig(local_threshold=0.5, min_label_streak=2)
    assert not decide(_pred("Tadasana", 0.9), "high", "auto", CLASSES, 0, cfg, label_streak=1).assessment_allowed
    assert decide(_pred("Tadasana", 0.9), "high", "auto", CLASSES, 0, cfg, label_streak=2).assessment_allowed


def test_calibration_file_matches_artifacts_and_is_loaded():
    cal_path = ROOT / "data" / "zenodo" / "confidence_calibration.json"
    if not cal_path.exists():
        pytest.skip("calibration not generated")
    from src.app.webcam_app import load_config, load_fusion_config
    cal = json.loads(cal_path.read_text())
    cfg = load_config("config.yaml")
    for clf in ("rf", "mlp"):
        fc = load_fusion_config(cfg, "zenodo", clf)
        assert fc.local_threshold == cal[clf]["threshold"]
        assert set(fc.class_precision) == set(CLASSES)


# ── Fix 8: tiered corrections ─────────────────────────────────────────────────

def _tadasana_feats(bad_knee=True):
    lm = [dict(p) for p in STANDING]
    if bad_knee:
        lm[25] = dict(lm[25], x=0.30)  # bend left knee strongly
    return lm, extract_features(lm, [0] * 33)


def test_decision_tier_caps_check_tiers():
    lm, f = _tadasana_feats()
    ev = evaluate_posture("Tadasana", f, lm, [0] * 33, decision_tier="medium")
    assert ev.checks and all(c.reliability_tier in ("medium", "low") for c in ev.checks)


def test_imputed_joint_never_high_even_if_tracker_says_high():
    lm, f = _tadasana_feats()
    flags = [0] * 33
    flags[25] = 1
    tracker_high = SimpleNamespace(joint_assessments={
        "left_knee_angle": SimpleNamespace(tier="high", is_reliable=True)})
    ev = evaluate_posture("Tadasana", f, lm, flags, reliability_assessment=tracker_high)
    knee = next(c for c in ev.checks if c.angle_name == "left_knee_angle")
    assert knee.reliability_tier == "medium"


def test_low_tier_check_is_unavailable_not_corrected():
    lm, f = _tadasana_feats()
    tracker_low = SimpleNamespace(joint_assessments={
        "left_knee_angle": SimpleNamespace(tier="low", is_reliable=False)})
    ev = evaluate_posture("Tadasana", f, lm, [0] * 33, reliability_assessment=tracker_low)
    knee = next(c for c in ev.checks if c.angle_name == "left_knee_angle")
    assert knee.status == "unavailable" and not knee.reliable


def test_feedback_uncertain_decision_issues_no_correction():
    fe = FeedbackEngine(cooldown_seconds=0)
    lm, f = _tadasana_feats()
    ev = evaluate_posture("Tadasana", f, lm, [0] * 33)
    pred = PosePrediction("Tadasana", 0.3, np.zeros(10), "Tadasana", False, "low_confidence", [("Tadasana", 0.3)])
    d = decide(pred, "high", "auto", CLASSES, 0, FusionConfig(local_threshold=0.5))
    items = fe.update(pred, ev, decision=d)
    assert [i.issue for i in items] == ["uncertain"]
    assert all(i.severity != "correction" for i in items)


def test_medium_tier_wording_is_qualified_and_spoken_later():
    fe = FeedbackEngine(cooldown_seconds=0, min_persistence_frames=2)
    lm, f = _tadasana_feats()
    pred = PosePrediction("Tadasana", 0.9, np.zeros(10), "Tadasana", False, "detected", [("Tadasana", 0.9)])
    spoken = []
    for k in range(3):
        fe._last_spoken_time = 0.0
        ev = evaluate_posture("Tadasana", f, lm, [0] * 33, decision_tier="medium")
        items = fe.update(pred, ev, target_pose="Tadasana")
        spoken.append(fe.pop_speech())
    corr = [i for i in items if i.issue in ("too_low", "too_high")]
    assert corr and all(i.instruction.startswith("Possibly") for i in corr)
    assert all(i.severity == "warning" for i in corr)
    assert spoken[0] is None and spoken[1] is None and spoken[2] and spoken[2].startswith("If you can")


def test_high_tier_correction_specific():
    fe = FeedbackEngine(cooldown_seconds=0, min_persistence_frames=1)
    lm, f = _tadasana_feats()
    pred = PosePrediction("Tadasana", 0.9, np.zeros(10), "Tadasana", False, "detected", [("Tadasana", 0.9)])
    ev = evaluate_posture("Tadasana", f, lm, [0] * 33)
    items = fe.update(pred, ev, target_pose="Tadasana")
    assert any(i.severity == "correction" and not i.instruction.startswith("Possibly") for i in items)


def test_reference_panel_drops_stale_pose_when_uncertain():
    from src.app.reference_display import ReferenceDisplay
    rd = ReferenceDisplay()
    rd.update_auto_detected_pose("Tadasana", 0.9)
    assert rd.update_auto_detected_pose("Tadasana", 0.9) == "Tadasana"
    rd.update_auto_detected_pose(None, 0.0)
    assert rd.update_auto_detected_pose(None, 0.0) is None


# ── Webcam integration helpers (no camera needed) ─────────────────────────────

class StubWorker:
    def __init__(self):
        self.submitted = []
        self.result = None

    def submit(self, frame, cands, fid, ts, epoch, force=False, focus_pose=None):
        self.submitted.append((fid, ts, epoch, force))
        self.focus = focus_pose
        return True

    def poll(self):
        r, self.result = self.result, None
        return r


def _jev_result(fid, ts, epoch, label="Vajrasana"):
    from src.models.jev_client import JevResult
    return JevResult(ok=True, status="ok", label=label, confidence=0.9, frame_id=fid,
                     frame_timestamp=ts, epoch=epoch)


def test_jev_coordinator_requests_only_on_persistent_ambiguity():
    from src.app.webcam_app import JevCoordinator
    jc = JevCoordinator(StubWorker(), {"ambiguity_inferences": 3}, CLASSES, clock=lambda: 0.0)
    amb = _pred("Vajrasana", 0.3)
    assert [jc.observe_inference(amb, "high", "auto") for _ in range(3)] == [False, False, True]
    assert not jc.observe_inference(amb, "high", "guided")  # never automatic in guided mode
    jc.ambiguous_streak = 0
    assert not any(jc.observe_inference(amb, "low", "auto") for _ in range(5))  # bad tracking: fix camera first


def test_jev_coordinator_discards_stale_answers():
    from src.app.webcam_app import JevCoordinator
    t = [10.0]
    w = StubWorker()
    jc = JevCoordinator(w, {"max_result_age_seconds": 4.0, "max_pose_change": 0.08}, CLASSES, clock=lambda: t[0])
    jc.submit(np.zeros((8, 8, 3), np.uint8), STANDING, frame_id=1)
    w.result = _jev_result(1, 10.0, jc.epoch)
    assert jc.poll(STANDING) is not None and jc.latest is not None          # fresh & same pose
    jc.submit(np.zeros((8, 8, 3), np.uint8), STANDING, frame_id=2)
    w.result = _jev_result(2, 10.0, jc.epoch)
    assert jc.poll(KNEELING) is None and jc.status == "stale"               # person moved
    jc.submit(np.zeros((8, 8, 3), np.uint8), STANDING, frame_id=3)
    w.result = _jev_result(3, 10.0, jc.epoch)
    t[0] = 20.0
    assert jc.poll(STANDING) is None                                        # too old
    jc.submit(np.zeros((8, 8, 3), np.uint8), STANDING, frame_id=4)
    w.result = _jev_result(4, 20.0, jc.epoch)
    jc.new_epoch()
    assert jc.poll(STANDING) is None and jc.latest is None                  # scene reset
    assert jc.discarded_stale == 3


def test_jev_periodic_mode_offers_every_frame_without_landmarks():
    from src.app.webcam_app import JevCoordinator
    frame = np.zeros((8, 8, 3), np.uint8)
    off = StubWorker()
    assert not JevCoordinator(off, {}, CLASSES, clock=lambda: 0.0).tick(frame, None, 1)
    assert off.submitted == []                                              # default: no periodic requests
    w = StubWorker()
    jc = JevCoordinator(w, {"periodic_interval_seconds": 1.0}, CLASSES, clock=lambda: 0.0)
    assert jc.tick(frame, None, 1) and jc.tick(frame, STANDING, 2)          # no landmarks needed
    assert [s[0] for s in w.submitted] == [1, 2] and not any(s[3] for s in w.submitted)  # throttled by worker


def test_jev_coordinator_coaches_target_or_last_confident_pose():
    from src.app.webcam_app import JevCoordinator
    frame = np.zeros((8, 8, 3), np.uint8)
    w = StubWorker()
    jc = JevCoordinator(w, {"min_confidence": 0.7}, CLASSES, clock=lambda: 10.0)
    jc.submit(frame, None, 1)
    assert w.focus is None                                                  # nothing known yet
    w.result = _jev_result(1, 10.0, jc.epoch, label="Vajrasana")
    jc.poll(None)
    jc.submit(frame, None, 2)
    assert w.focus == "Vajrasana"                                           # coach the pose Jev just saw
    jc.target = "Tadasana"
    jc.submit(frame, None, 3)
    assert w.focus == "Tadasana"                                            # guided target wins
    jc.target = None
    jc.new_epoch()
    jc.submit(frame, None, 4)
    assert w.focus is None


def test_jev_periodic_interval_sets_worker_throttle(monkeypatch):
    from src.app.webcam_app import build_jev
    monkeypatch.setenv("JEVMODEL_API_KEY", "sk-test")
    jc = build_jev({"jev": {"enabled": True, "periodic_interval_seconds": 1.0,
                            "min_request_interval_seconds": 6.0}}, CLASSES)
    try:
        assert jc.periodic and jc.worker._min_interval == 1.0
    finally:
        jc.worker.shutdown()


def test_jev_disabled_means_local_only():
    from src.app.webcam_app import build_jev
    jc = build_jev({"jev": {"enabled": False}}, CLASSES)
    assert not jc.enabled and jc.poll(STANDING) is None
    assert not jc.submit(np.zeros((8, 8, 3), np.uint8), STANDING, 1)


def test_jev_enabled_without_key_falls_back(monkeypatch):
    from src.app.webcam_app import build_jev
    monkeypatch.delenv("JEVMODEL_API_KEY", raising=False)
    assert not build_jev({"jev": {"enabled": True}}, CLASSES).enabled


def test_classifier_flag_policy():
    from src.app.webcam_app import classifier_flags
    fr = SimpleNamespace(imputed_flags=[1] + [0] * 32)
    assert classifier_flags(fr, "zenodo", "auto") == [0] * 33
    assert classifier_flags(fr, "kaggle", "auto")[0] == 1
    assert classifier_flags(fr, "kaggle", "zero") == [0] * 33


# ── Fix 5: artifact / schema compatibility with the real saved models ─────────

@pytest.mark.parametrize("clf", ["rf", "mlp"])
def test_zenodo_artifacts_schema_and_class_mapping(clf):
    from src.models.model_loader import load_zenodo_artifacts
    from src.correction.reference_ranges import REFERENCE_RANGES
    from src.app.reference_display import get_target_poses_for_profile
    art = load_zenodo_artifacts(ROOT / "saved_yoga_model_zenodo", active_classifier=clf)
    assert art.model_type == clf                                          # right artifact for the request
    assert list(art.feature_cols) == list(extract_features(STANDING, [0] * 33).keys())
    assert art.asana_model.n_features_in_ == len(art.feature_cols) + len(art.imputer.indicator_.features_)
    assert list(art.asana_label_encoder.classes_) == art.class_names     # index → name mapping
    assert list(art.asana_model.classes_) == list(range(len(art.class_names)))
    assert set(art.class_names) == set(get_target_poses_for_profile("zenodo"))
    assert all(c in REFERENCE_RANGES for c in art.class_names)
    assert list(art.correctness_label_encoder.classes_) == ["Right", "Wrong"]


def test_root_zenodo_artifact_is_mlp_champion_not_rf():
    """The un-suffixed directory holds the MLP champion; RF must be loaded from rf/."""
    import joblib
    prep = joblib.load(ROOT / "saved_yoga_model_zenodo" / "zenodo_preprocessing.joblib")
    assert prep["model_type"] == "mlp"


def test_mlp_inference_uses_scaler_rf_does_not():
    from src.models.model_loader import load_zenodo_artifacts
    for clf in ("rf", "mlp"):
        art = load_zenodo_artifacts(ROOT / "saved_yoga_model_zenodo", active_classifier=clf)
        f = features_to_series(extract_features(STANDING, [0] * 33), art.feature_cols)
        X = art.imputer.transform(pd.DataFrame([f.values], columns=art.feature_cols))
        Xe = art.scaler.transform(X) if clf == "mlp" else X
        expect = art.asana_model.predict_proba(Xe)[0]
        pred = DualTaskInference(art, filter_policy=FilterPolicy(use_gravity_prior=False,
                                                                 use_sequence_pruning=False)).predict(f)
        assert np.allclose(pred.raw_probabilities, expect)


def test_kaggle_artifacts_schema():
    from src.models.model_loader import load_artifacts
    art = load_artifacts(ROOT / "saved_yoga_model")
    assert list(art.feature_cols) == list(extract_features(STANDING, [0] * 33).keys())
    assert art.n_classes == 47 and len(art.rf_model.classes_) == 47


def test_config_paths_resolve_from_any_cwd(tmp_path):
    code = ("from src.app.webcam_app import load_config, project_path;"
            "c = load_config('config.yaml');"
            "assert project_path(c['model']['mediapipe_task']).exists();"
            "assert project_path(c['model']['kaggle_dir']).is_dir();"
            "assert project_path(c['inference']['calibration_file']).parent.is_dir();"
            "from src.pose import resolve_project_path;"
            "assert resolve_project_path(c['model']['mediapipe_task']).exists();"
            "from src.models.model_loader import load_profile_artifacts, load_zenodo_artifacts;"
            "assert load_profile_artifacts('kaggle').n_classes == 47;"
            "assert load_zenodo_artifacts('saved_yoga_model_zenodo', 'rf').model_type == 'rf'")
    env = dict(os.environ, PYTHONPATH=str(ROOT), TF_CPP_MIN_LOG_LEVEL="3")
    res = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-2000:]


def test_calibration_profile_round_trips_thigh_ratio(tmp_path):
    from src.correction.calibration import UserProfile, load_profile, save_profile
    p = tmp_path / "profile.json"
    save_profile(UserProfile(calibrated=True, thigh_to_torso_ratio=0.77), p)
    assert load_profile(p).thigh_to_torso_ratio == 0.77
