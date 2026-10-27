"""
src/app/webcam_app.py
──────────────────────
Main application entry-point for Yoga Mitra: AI-Powered Personalized
Yoga Posture Detection and Correction System.

Pipeline per frame:
  camera frame
    → MediaPipe detection, VIDEO mode with monotonic timestamps
      (unmirrored raw frame for anatomical accuracy)
    → per-landmark history (last reliable observation, max age, reset on scene change)
    → temporal reliability tracker (visibility / imputation / jitter tiers)
    → feature extraction (geometry.py — exact 39-feature training schema)
    → impute + [scale] + RF/MLP predict + reliability-gated candidate filters (inference.py)
    → optional Jev still-image assessment (background worker, throttled, stale-checked)
    → pose decision (jev_fusion.py — is a pose-specific assessment allowed, and how certain?)
    → posture evaluate (posture_evaluator.py — only when the decision allows it)
    → feedback engine build (feedback_engine.py — tiered, confidence-aware cues)
    → biomechanical calibration (calibration.py — Mountain Pose baseline)
    → session progress tracker (progress_tracker.py — SQLite logging)
    → two-panel layout:
        Left panel: Live webcam feed + skeleton HUD + deviant joint highlight + status strip
        Right panel: Persistent Reference Asana card + alignment cues
    → async TTS voice engine (voice.py — non-blocking thread-local SAPI5)
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml

# ── Locate project root so relative imports work when launched from any cwd ───
_ROOT = Path(__file__).resolve().parent.parent.parent   # …/Yoga_posture_project
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.pose import PoseLandmarkerWrapper, SKELETON_CONNECTIONS
from src.pose.missing_landmarks import LandmarkHistory
from src.features import extract_features, features_to_series, ANGLE_TRIPLETS
from src.models.model_loader import load_artifacts
from src.models.inference import PoseInference, PosePrediction, FilterPolicy
from src.models.jev_fusion import FusionConfig, PoseDecision, decide, decide_jev_primary
from src.correction.posture_evaluator import evaluate as evaluate_posture, PostureEvaluation
from src.correction.feedback_engine import FeedbackEngine, FeedbackItem
from src.correction.calibration import UserProfile, load_profile, save_profile, CalibrationSession
from src.correction.progress_tracker import ProgressTracker
from src.app.voice import VoiceFeedback
from src.app.reference_display import ReferenceDisplay
from src.app.overlay import (
    draw_skeleton, draw_hud, draw_key_hints, draw_angle_values, draw_calibration_overlay,
    draw_decision_status, draw_diagnostics,
)

# Landmark ID mapping for visual joint error highlighting
JOINT_LM_MAP: dict[str, int] = {
    "left_knee_angle": 25,
    "right_knee_angle": 26,
    "left_elbow_angle": 13,
    "right_elbow_angle": 14,
    "left_shoulder_angle": 11,
    "right_shoulder_angle": 12,
    "left_hip_angle": 23,
    "right_hip_angle": 24,
    "left_ankle_angle": 27,
    "right_ankle_angle": 28,
}

POSE_CHANGE_IDS = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)


# ─────────────────────────────────────────────────────────────────────────────
def load_config(path: str | Path = "config.yaml") -> dict:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = _ROOT / p
    with open(p, "r") as f:
        return yaml.safe_load(f)


def project_path(path: str | Path) -> Path:
    """Resolve config paths against the project root (independent of the shell cwd)."""
    p = Path(path)
    return p if p.is_absolute() else _ROOT / p


def load_fusion_config(cfg: dict, profile: str, classifier: str) -> FusionConfig:
    """Confidence thresholds: config defaults, replaced by validation calibration if available."""
    infer_cfg = cfg.get("inference", {})
    jev_cfg = cfg.get("jev", {}) or {}
    fc = FusionConfig(
        local_threshold=float(infer_cfg.get("confidence_threshold", 0.28)),
        local_high_confidence=float(infer_cfg.get("high_confidence", 0.60)),
        jev_min_confidence=float(jev_cfg.get("min_confidence", 0.70)),
        jev_max_age_seconds=float(jev_cfg.get("max_result_age_seconds", 4.0)),
        orientation_veto=bool(infer_cfg.get("orientation_veto", True)),
        min_label_streak=int(infer_cfg.get("min_label_streak", 2)),
    )
    calib_path = infer_cfg.get("calibration_file")
    if profile == "zenodo" and calib_path and infer_cfg.get("use_calibration_file", True):
        p = project_path(calib_path)
        if p.exists():
            try:
                cal = json.loads(p.read_text()).get(classifier, {})
                if "threshold" in cal:
                    fc.local_threshold = float(cal["threshold"])
                fc.class_precision = {k: float(v) for k, v in cal.get("per_class_precision", {}).items()}
                fc.class_min_precision = float(cal.get("class_min_precision", 0.0))
            except Exception as exc:
                print(f"[App] Could not read calibration file {p}: {exc}")
    return fc


def classifier_flags(lm_frame: Any, profile: str, policy: str = "auto") -> list[int]:
    """
    Imputed flags passed to the *classifier* (posture evaluation always keeps the real flags).
    Zenodo models were trained with every *_imputed flag equal to 0, so non-zero flags are
    out-of-distribution for them; Kaggle models were trained with flags.
    """
    if policy == "zero" or (policy == "auto" and profile == "zenodo"):
        return [0] * len(lm_frame.imputed_flags)
    return list(lm_frame.imputed_flags)


def pose_change_distance(a: Optional[list[dict]], b: Optional[list[dict]]) -> Optional[float]:
    """Mean displacement of directly observed major joints, in normalised image units."""
    if not a or not b:
        return None
    d = []
    for i in POSE_CHANGE_IDS:
        pa, pb = a[i], b[i]
        if pa.get("imputed") or pb.get("imputed") or pa.get("unavailable") or pb.get("unavailable"):
            continue
        if float(pa.get("visibility", 0)) < 0.5 or float(pb.get("visibility", 0)) < 0.5:
            continue
        d.append(float(np.hypot(pa["x"] - pb["x"], pa["y"] - pb["y"])))
    return float(np.mean(d)) if len(d) >= 4 else None


class JevCoordinator:
    """
    Decides *when* to ask Jev, associates answers with the frame they describe and
    discards answers that no longer match the person's current pose.
    """

    def __init__(self, worker: Any, jev_cfg: dict, candidates: list[str], clock=time.monotonic):
        self.worker = worker
        self.cfg = jev_cfg or {}
        self.candidates = list(candidates)
        self.clock = clock
        self.epoch = 0
        self.latest: Optional[Any] = None          # last *valid* (non-stale) result
        self.status = "idle"
        self.ambiguous_streak = 0
        self.requests = 0
        self.discarded_stale = 0
        self._pending: dict[int, list[dict]] = {}   # frame_id → landmarks at capture
        self.periodic_seconds = float(self.cfg.get("periodic_interval_seconds", 0) or 0)
        self.target: Optional[str] = None           # guided target: always coached
        self.confirmed_pose: Optional[str] = None   # auto mode: last confident Jev pose, coached next

    @property
    def focus_pose(self) -> Optional[str]:
        """Pose whose coaching checkpoints the next request carries."""
        return self.target or self.confirmed_pose

    @property
    def enabled(self) -> bool:
        return self.worker is not None

    @property
    def periodic(self) -> bool:
        return self.enabled and self.periodic_seconds > 0

    def tick(self, frame_bgr: np.ndarray, landmarks: Optional[list[dict]], frame_id: int) -> bool:
        """Periodic mode: offer the current frame every loop; the worker's interval and
        single-in-flight bound decide whether it is actually sent."""
        return self.periodic and self.submit(frame_bgr, landmarks, frame_id)

    def new_epoch(self) -> None:
        """Person left / scene changed / mode changed: older answers no longer apply."""
        self.epoch += 1
        self.latest = None
        self.ambiguous_streak = 0
        self.confirmed_pose = None

    def observe_inference(self, prediction: Any, landmark_tier: str, mode: str) -> bool:
        """Update ambiguity streak; True when an automatic request is warranted."""
        top = list(getattr(prediction, "top3", []) or [])
        margin = (top[0][1] - top[1][1]) if len(top) >= 2 else 1.0
        ambiguous = (getattr(prediction, "status", "") == "low_confidence"
                     or margin < float(self.cfg.get("ambiguity_margin", 0.10)))
        if ambiguous and landmark_tier in ("high", "medium"):
            self.ambiguous_streak += 1
        else:
            self.ambiguous_streak = 0
        return (self.enabled and mode == "auto" and bool(self.cfg.get("auto_on_ambiguity", True))
                and self.ambiguous_streak >= int(self.cfg.get("ambiguity_inferences", 3)))

    def submit(self, frame_bgr: np.ndarray, landmarks: Optional[list[dict]], frame_id: int,
               manual: bool = False) -> bool:
        if not self.enabled or frame_bgr is None:
            return False
        ok = self.worker.submit(frame_bgr, self.candidates, frame_id, self.clock(), self.epoch, force=manual,
                                focus_pose=self.focus_pose)
        if ok:
            self.requests += 1
            self.status = "pending"
            self._pending = {frame_id: [dict(p) for p in landmarks] if landmarks else []}
            if not manual:
                self.ambiguous_streak = 0
        return ok

    def poll(self, current_landmarks: Optional[list[dict]]) -> Optional[Any]:
        """Collect a finished answer; returns it only if it still applies."""
        if not self.enabled:
            return None
        res = self.worker.poll()
        if res is None:
            return None
        captured = self._pending.pop(res.frame_id, None)
        self.status = res.status
        if not res.ok:
            return None
        if res.epoch == self.epoch:
            if res.unclear:
                self.confirmed_pose = None
            elif res.label and (res.confidence or 0.0) >= float(self.cfg.get("min_confidence", 0.70)):
                self.confirmed_pose = res.label
        moved = pose_change_distance(captured, current_landmarks)
        too_old = res.frame_timestamp is None or self.clock() - res.frame_timestamp > float(
            self.cfg.get("max_result_age_seconds", 4.0))
        if res.epoch != self.epoch or too_old or (moved is not None and moved > float(self.cfg.get("max_pose_change", 0.08))):
            self.discarded_stale += 1
            self.status = "stale"
            return None
        self.latest = res
        return res


def build_jev(cfg: dict, candidates: list[str]) -> JevCoordinator:
    jev_cfg = cfg.get("jev", {}) or {}
    if not jev_cfg.get("enabled", False):
        return JevCoordinator(None, jev_cfg, candidates)
    from src.models.jev_client import JevClient, JevConfig
    from src.models.jev_worker import JevWorker
    client = JevClient(JevConfig.from_config(jev_cfg))
    if not client.has_api_key:
        print(f"[App] Jev enabled in config but {JevConfig.from_config(jev_cfg).api_key_env} is not set "
              "— running local-only.")
        return JevCoordinator(None, jev_cfg, candidates)
    print("[App] Jev ENABLED: selected still frames (not every frame) will be sent to "
          f"{client.config.endpoint} for image assessment. Images are not saved locally.")
    min_interval = float(jev_cfg.get("min_request_interval_seconds", 6.0))
    periodic = float(jev_cfg.get("periodic_interval_seconds", 0) or 0)
    if periodic > 0:
        min_interval = periodic
        print(f"[App] Jev periodic mode: one frame every {periodic:g} s (each request uses TypeSafe credits).")
    worker = JevWorker(client, min_interval_seconds=min_interval)
    return JevCoordinator(worker, jev_cfg, candidates)


def correction_state(decision: Optional[PoseDecision], items: list[FeedbackItem]) -> str:
    if decision is None:
        return "-"
    if not decision.assessment_allowed:
        return "suppressed"
    if any(i.severity == "correction" for i in items):
        return "issued"
    if any(i.issue in ("too_low", "too_high") for i in items):
        return "qualified"
    if any(i.issue == "unavailable" for i in items):
        return "partial (joints hidden)"
    return "none needed"


# ─────────────────────────────────────────────────────────────────────────────
def run(config_input: str | Path | dict = "config.yaml") -> None:
    if isinstance(config_input, dict):
        cfg = config_input
    else:
        cfg = load_config(config_input)

    model_cfg     = cfg["model"]
    infer_cfg     = cfg["inference"]
    camera_cfg    = cfg["camera"]
    display_cfg   = cfg["display"]
    voice_cfg     = cfg["voice"]
    mp_cfg        = cfg.get("mediapipe", {}) or {}
    hist_cfg      = cfg.get("landmark_history", {}) or {}

    # ── Model Profile & Artifacts ─────────────────────────────────────────────
    requested_profile = model_cfg.get("profile", "kaggle").lower()
    clf_choice = model_cfg.get("active_classifier", "rf").lower()
    task_path = project_path(model_cfg["mediapipe_task"])
    filter_cfg = dict(infer_cfg.get("filters") or {})
    filter_cfg.update((infer_cfg.get("filters_by_classifier") or {}).get(clf_choice) or {})
    filter_policy = FilterPolicy.from_config(filter_cfg)
    active_profile = requested_profile

    zenodo_dir = project_path(model_cfg.get("dir", "saved_yoga_model_zenodo"))
    if not (zenodo_dir / "zenodo_preprocessing.joblib").exists():
        zenodo_dir = _ROOT / "saved_yoga_model_zenodo"
    zenodo_ready = (
        (zenodo_dir / "zenodo_asana_classifier.joblib").exists() and
        (zenodo_dir / "zenodo_form_verifier.joblib").exists() and
        (zenodo_dir / "zenodo_preprocessing.joblib").exists()
    )

    if requested_profile == "zenodo":
        if zenodo_ready:
            try:
                from src.models.model_loader import load_zenodo_artifacts
                from src.models.inference import DualTaskInference
                artifacts = load_zenodo_artifacts(zenodo_dir, active_classifier=clf_choice)
                if artifacts.model_type != clf_choice:
                    print(f"[App] WARNING: requested classifier '{clf_choice}' but loaded '{artifacts.model_type}'.")
                inferrer = DualTaskInference(
                    artifacts=artifacts,
                    confidence_threshold=infer_cfg["confidence_threshold"],
                    smoothing_window=infer_cfg["smoothing_window"],
                    filter_policy=filter_policy,
                )
                sub = zenodo_dir / artifacts.model_type
                print(f"[App] Zenodo Profile Active: 10 Asanas + Right/Wrong Form Verifier "
                      f"({artifacts.model_type.upper()} from {sub if sub.is_dir() else zenodo_dir})")
            except Exception as exc:
                print(f"[App] Zenodo load error: {exc}. Falling back to Kaggle profile.")
                active_profile = "kaggle"
        else:
            print("[App] Zenodo profile requested, but trained Zenodo artifacts are pending training run approval.")
            print("[App] Falling back safely to verified Kaggle 47-class model.")
            active_profile = "kaggle"

    if active_profile == "kaggle":
        kaggle_dir = project_path(model_cfg.get("kaggle_dir", "saved_yoga_model"))
        print(f"[App] Loading Kaggle model artifacts from: {kaggle_dir}")
        try:
            artifacts = load_artifacts(kaggle_dir)
        except RuntimeError as exc:
            sys.exit(f"[App] FATAL: {exc}")

        inferrer = PoseInference(
            artifacts=artifacts,
            active_classifier=clf_choice,
            confidence_threshold=infer_cfg["confidence_threshold"],
            smoothing_window=infer_cfg["smoothing_window"],
            filter_policy=filter_policy,
        )
        print(f"[App] Kaggle Profile Active: {clf_choice.upper()}, "
              f"classes: {artifacts.n_classes}, features: {artifacts.n_features}")

    fusion_cfg = load_fusion_config(cfg, active_profile, clf_choice)
    inferrer._threshold = fusion_cfg.local_threshold
    print(f"[App] Confidence threshold: {fusion_cfg.local_threshold:.2f} | filters: {filter_policy.mode}"
          f"/{filter_policy.granularity} | MediaPipe mode: {mp_cfg.get('running_mode', 'video')}")

    # ── MediaPipe ─────────────────────────────────────────────────────────────
    if not task_path.exists():
        sys.exit(f"[App] MediaPipe task file not found: {task_path}")

    landmarker = PoseLandmarkerWrapper(
        model_path=task_path,
        visibility_threshold=infer_cfg["visibility_threshold"],
        running_mode=mp_cfg.get("running_mode", "video"),
        min_pose_detection_confidence=float(mp_cfg.get("min_pose_detection_confidence", 0.5)),
        min_pose_presence_confidence=float(mp_cfg.get("min_pose_presence_confidence", 0.5)),
        min_tracking_confidence=float(mp_cfg.get("min_tracking_confidence", 0.5)),
    )
    lm_history = LandmarkHistory(
        threshold=infer_cfg["visibility_threshold"],
        max_age_ms=float(hist_cfg.get("max_age_ms", 300)),
        scene_change_norm=float(hist_cfg.get("scene_change_norm", 0.25)),
        decay_drop=float(hist_cfg.get("decay_drop", 0.15)),
    )

    # Everything below owns resources: release them deterministically.
    voice = None
    cap = None
    jev = JevCoordinator(None, {}, [])
    progress_tracker = None
    try:
        # ── Voice ─────────────────────────────────────────────────────────────
        voice = VoiceFeedback(
            rate=voice_cfg["rate"],
            volume=voice_cfg["volume"],
        )
        if not voice_cfg["enabled"]:
            voice._enabled = False

        # ── Feedback engine & Reliability Tracker ─────────────────────────────
        feedback_engine = FeedbackEngine(
            cooldown_seconds=voice_cfg["cooldown_seconds"],
            max_display=display_cfg["max_feedback_lines"],
        )
        from src.correction.reliability_tracker import TemporalReliabilityTracker
        reliability_tracker = TemporalReliabilityTracker(min_visibility=infer_cfg["visibility_threshold"])

        # ── Reference Display Panel (Right side) ──────────────────────────────
        ref_display = ReferenceDisplay(
            assets_dir=_ROOT / "assets" / "pose_references",
            panel_width=380,
        )

        # ── User Biomechanical Calibration ───────────────────────────────────
        user_profile = load_profile(_ROOT / "data" / "user_profile.json")
        calib_cfg = cfg.get("calibration", {})
        calib_duration = float(calib_cfg.get("duration_seconds", 5.0))
        calib_session = CalibrationSession(duration_seconds=calib_duration, save_path=_ROOT / "data" / "user_profile.json")

        force_calib = calib_cfg.get("force_startup", False)
        auto_calib = calib_cfg.get("auto_on_startup", True)
        if force_calib or (not user_profile.calibrated and auto_calib):
            calib_session.start()
            voice.speak("Welcome to Yoga Mitra. Please stand tall in front of the camera for 5 seconds to calibrate your body baseline.")
            print("[App] Starting 5-second body baseline calibration. Stand tall in Mountain Pose (Tadasana)...")
        elif user_profile.calibrated:
            print(f"[App] Loaded user profile: {user_profile.user_name} (Calibrated: {user_profile.calibration_date})")
            print("[App] Press 'C' anytime to re-calibrate your 5-second body baseline.")

        # ── Session Progress Tracker ──────────────────────────────────────────
        progress_tracker = ProgressTracker(db_path=_ROOT / "data" / "yoga_sessions.db")
        progress_tracker.start_session(calibrated=user_profile.calibrated)

        # ── Camera ────────────────────────────────────────────────────────────
        source = camera_cfg.get("source") or camera_cfg["device_index"]
        from_file = isinstance(source, str) and not str(source).isdigit()
        cap = cv2.VideoCapture(str(project_path(source)) if from_file else int(source))
        if not cap.isOpened():
            sys.exit("[App] Could not open webcam/video. Check camera.device_index / --video.")
        if not from_file:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  camera_cfg["width"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_cfg["height"])
        headless = bool(display_cfg.get("headless", False))
        max_frames = int(camera_cfg.get("max_frames") or 0)
        decision_log_path = (cfg.get("logging", {}) or {}).get("decision_log")
        decision_log = open(project_path(decision_log_path), "w", encoding="utf-8") if decision_log_path else None

        # ── Real-Time Single Mode ─────────────────────────────────────────────
        target_mode: bool = False
        active_target: Optional[str] = None

        # ── Jev (optional, external) ──────────────────────────────────────────
        jev_cfg = cfg.get("jev", {}) or {}
        jev = build_jev(cfg, artifacts.class_names)
        jev_primary = jev.enabled and bool(jev_cfg.get("primary", False))
        jev_coach_min_conf = float(jev_cfg.get("coach_min_confidence", 0.60))
        if jev_primary:
            print("[App] Jev-primary mode: Jev decides the pose and gives the form corrections; "
                  "the local classifier is shown for comparison only.")

        # ── State ─────────────────────────────────────────────────────────────
        interval: float = infer_cfg["interval_seconds"]
        last_inference_time: float = 0.0
        clock_origin = time.monotonic()
        frame_id = 0
        clf_flag_policy = str(infer_cfg.get("classifier_imputed_flags", "auto"))
        last_label: Optional[str] = None
        label_streak = 0

        # Persistent results
        last_prediction: Optional[PosePrediction] = None
        last_decision: Optional[PoseDecision] = None
        last_features: dict = {}
        last_feedback: list[FeedbackItem] = []
        last_lm_effective: Optional[list[dict]] = None
        last_eval_result: Optional[PostureEvaluation] = None
        last_rel = None
        last_decision_tier = "low"
        highlight_joint_ids: set[int] = set()
        display_prediction: Optional[PosePrediction] = None   # Jev-primary: what the HUD shows
        jev_form_score: Optional[float] = None
        timings = {"mediapipe_ms": 0.0, "inference_ms": 0.0}
        fps_hist: list[float] = []

        show_angles: bool = display_cfg["show_angles"]
        show_diag: bool = bool(display_cfg.get("diagnostics", False))
        screenshot_dir = _ROOT / "screenshots"
        screenshot_dir.mkdir(parents=True, exist_ok=True)

        print("[App] Real-Time Pose Detection & Form Feedback Active.")
        print("[App] Controls: Q=Quit, C=Calibrate, V=Voice, A=Angles, D=Diagnostics, S=Screenshot")
        win_title = "Yoga Mitra: AI-Powered Yoga Posture Detection & Correction"
        if not headless:
            cv2.namedWindow(win_title, cv2.WINDOW_AUTOSIZE)
            try:
                cv2.setWindowProperty(win_title, cv2.WND_PROP_TOPMOST, 1)
            except Exception:
                pass

        # ── Main loop ─────────────────────────────────────────────────────────
        while True:
            ret, frame_raw = cap.read()
            if not ret:
                if from_file:
                    print("[App] End of video.")
                    break
                print("[App] Camera read failed. Retrying…")
                time.sleep(0.05)
                continue

            frame_id += 1
            # Video files are replayed on their own timeline (deterministic, faster than real time)
            now = clock_origin + cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if from_file else time.monotonic()
            fps_hist = (fps_hist + [now])[-30:]
            run_inference = (now - last_inference_time) >= interval
            active_target = None
            mode = "realtime"
            ts_ms = int((now - clock_origin) * 1000.0)

            # ── Landmark detection (unmirrored frame, video timestamps) ────────
            raw_lm = landmarker.detect(frame_raw, timestamp_ms=ts_ms)
            timings["mediapipe_ms"] = landmarker.last_latency_ms

            # ── Biomechanical Calibration step (if triggered by user) ─────────
            if calib_session.is_active and raw_lm is not None:
                h_raw, w_raw = frame_raw.shape[:2]
                calib_done, calib_msg, new_prof = calib_session.update(raw_lm, w_raw, h_raw)
                if calib_done and new_prof is not None:
                    user_profile = new_prof
                    progress_tracker.calibrated = True
                    voice.speak("Calibration complete! Your body baseline is locked. You may begin your practice.")
                    print(f"[App] {calib_msg}")

            lm_frame = lm_history.update(raw_lm, ts_ms)
            if lm_frame is None or lm_frame.history_reset:
                reliability_tracker.reset()
                if not jev_primary:   # Jev-primary does not depend on MediaPipe keeping the body
                    jev.new_epoch()

            # ── Collect any finished Jev answer (never blocks) ─────────────────
            new_jev = jev.poll(lm_frame.effective if lm_frame else None)
            jev.target = active_target
            jev.tick(frame_raw, lm_frame.effective if lm_frame else None, frame_id)

            if lm_frame is None:
                last_label, label_streak = None, 0
                last_lm_effective = None
                last_features = {}
                highlight_joint_ids.clear()
                if run_inference:
                    last_inference_time = now
                    dummy_pred = PosePrediction(
                        pose_name="—",
                        confidence=0.0,
                        probabilities=np.zeros(artifacts.n_classes),
                        raw_prediction="—",
                        smoothed=False,
                        status="no_person",
                        top3=[],
                    )
                    last_prediction = dummy_pred
                    last_eval_result = None
                    last_decision_tier = "low"
                    last_decision = decide(dummy_pred, "low", mode, artifacts.class_names, now,
                                           fusion_cfg, target_pose=active_target)
                    if not jev_primary:
                        last_feedback = feedback_engine.update(dummy_pred, None, target_pose=active_target)
            else:
                effective_lm = lm_frame.effective
                flags = lm_frame.imputed_flags
                last_lm_effective = effective_lm

                # ── Temporal Reliability & Jitter Tracking ────────────────────
                rel_assessment = reliability_tracker.update(effective_lm, flags)
                last_rel = rel_assessment

                h, w = frame_raw.shape[:2]
                feat_dict = extract_features(effective_lm, flags, w, h)
                last_features = feat_dict

                # ── Periodic classifier, decision & posture evaluation ─────────
                if run_inference:
                    last_inference_time = now
                    t_inf = time.perf_counter()
                    cls_flags = classifier_flags(lm_frame, active_profile, clf_flag_policy)
                    cls_feats = feat_dict if cls_flags == flags else extract_features(effective_lm, cls_flags, w, h)
                    feat_series = features_to_series(cls_feats, artifacts.feature_cols)

                    prediction = inferrer.predict(feat_series, landmarks=effective_lm, user_profile=user_profile)
                    last_prediction = prediction
                    landmark_tier = rel_assessment.overall_tier
                    last_decision_tier = landmark_tier

                    # Jev: automatic request on persistent ambiguity (bounded & throttled)
                    if jev.observe_inference(prediction, landmark_tier, mode):
                        jev.submit(frame_raw, effective_lm, frame_id)

                    label_streak = label_streak + 1 if prediction.pose_name == last_label else 1
                    last_label = prediction.pose_name
                    decision = decide(
                        prediction, landmark_tier, mode, artifacts.class_names, now, fusion_cfg,
                        target_pose=active_target, jev=jev.latest, current_epoch=jev.epoch,
                        label_streak=label_streak,
                    )
                    last_decision = decision

                    eval_result = None
                    if not jev_primary and decision.assessment_allowed and decision.final_pose:
                        eval_result = evaluate_posture(
                            pose_name=decision.final_pose,
                            feature_dict=feat_dict,
                            landmarks=effective_lm,
                            imputed_flags=flags,
                            user_profile=asdict(user_profile) if user_profile.calibrated else None,
                            reliability_assessment=rel_assessment,
                            decision_tier=decision.tier,
                        )
                    last_eval_result = eval_result

                    # Update joints to highlight on skeleton (reliable, assessed checks only)
                    highlight_joint_ids.clear()
                    if eval_result and eval_result.checks:
                        for chk in eval_result.checks:
                            if chk.status != "ok" and chk.reliable and chk.severity in ("moderate", "major"):
                                lm_id = JOINT_LM_MAP.get(chk.angle_name)
                                if lm_id is not None:
                                    highlight_joint_ids.add(lm_id)

                    # Log to session progress tracker (only assessed frames)
                    if eval_result:
                        is_rel = (eval_result.overall_status != "low_visibility")
                        progress_tracker.log_frame(
                            decision.final_pose,
                            eval_result.alignment_score,
                            [c for c in eval_result.checks if c.reliable],
                            is_reliable=is_rel,
                        )

                    # Generate feedback
                    if not jev_primary:
                        last_feedback = feedback_engine.update(
                            prediction,
                            eval_result,
                            target_pose=active_target,
                            decision=decision,
                        )
                    timings["inference_ms"] = (time.perf_counter() - t_inf) * 1000.0
                    if decision_log is not None and not jev_primary:
                        fd = prediction.filter_decision
                        decision_log.write(json.dumps({
                            "frame_id": frame_id, "t": round(now - clock_origin, 3), "mode": mode,
                            "target": active_target, "local_pose": prediction.pose_name,
                            "local_conf": round(prediction.confidence, 4), "status": prediction.status,
                            "top3": [(n, round(p, 4)) for n, p in prediction.top3],
                            "landmark_tier": landmark_tier, "imputed": lm_frame.n_imputed,
                            "unavailable": lm_frame.n_unavailable, "suspect": len(lm_frame.suspect_ids),
                            "filter": {"mode": fd.mode, "base": fd.base_posture, "skipped": fd.skipped_reason,
                                       "suppressed": fd.suppressed} if fd else None,
                            "decision": {k: getattr(decision, k) for k in (
                                "final_pose", "assessment_allowed", "tier", "source", "agreement", "reason",
                                "jev_pose", "jev_confidence")},
                            "correction_state": correction_state(decision, last_feedback),
                            "feedback": [(i.issue, i.severity, i.reliability_tier, i.instruction) for i in last_feedback],
                            "mediapipe_ms": round(timings["mediapipe_ms"], 2),
                            "inference_ms": round(timings["inference_ms"], 2),
                        }) + "\n")

                    # Dispatch voice
                    msg = feedback_engine.pop_speech()
                    if msg:
                        voice.speak(msg)

            # ── Jev-primary: Jev decides the pose and coaches the form ────────
            if jev_primary and (run_inference or new_jev is not None):
                prev_pose = last_decision.final_pose if (last_decision and last_decision.assessment_allowed) else None
                jd = decide_jev_primary(jev.latest, last_prediction, mode, artifacts.class_names, now, fusion_cfg,
                                        target_pose=active_target, current_epoch=jev.epoch,
                                        high_confidence=float(jev_cfg.get("high_confidence", 0.85)))
                last_decision = jd
                last_decision_tier = jd.tier
                coaching = getattr(jev.latest, "coaching", None) if jd.assessment_allowed else None
                last_feedback = feedback_engine.update_from_jev(jd, coaching, fresh=new_jev is not None,
                                                                min_confidence=jev_coach_min_conf)
                if jd.assessment_allowed and mode == "auto" and jd.final_pose != prev_pose:
                    feedback_engine.announce("jev_pose", f"{jd.final_pose}.", min_interval=4.0)
                highlight_joint_ids = {j for c in (coaching.corrections if coaching else [])
                                       if c.needs_change and c.confidence >= jev_coach_min_conf for j in c.joints}
                jev_form_score = coaching.form_score if coaching else None
                best = jev.latest.label if (jev.latest is not None and jev.latest.ok) else None
                probs = dict(jev.latest.probabilities) if (jev.latest is not None and jev.latest.ok) else {}
                display_prediction = PosePrediction(
                    pose_name=jd.final_pose or best or "No clear pose",
                    confidence=float(jd.jev_confidence or 0.0),
                    probabilities=np.zeros(1),
                    raw_prediction=best or "—",
                    smoothed=False,
                    status="detected" if jd.assessment_allowed else "low_confidence",
                    top3=sorted(probs.items(), key=lambda kv: -kv[1])[:3],
                )
                if new_jev is not None:
                    if jd.assessment_allowed:
                        progress_tracker.log_frame(jd.final_pose, jev_form_score or 0.0, [], is_reliable=True)
                    if decision_log is not None:
                        decision_log.write(json.dumps({
                            "frame_id": frame_id, "t": round(now - clock_origin, 3), "mode": mode,
                            "target": active_target, "source": "jev_primary",
                            "local_pose": getattr(last_prediction, "pose_name", None),
                            "local_conf": round(float(getattr(last_prediction, "confidence", 0.0) or 0.0), 4),
                            "jev_pose": new_jev.label, "jev_conf": new_jev.confidence, "jev_unclear": new_jev.unclear,
                            "jev_status": new_jev.status, "jev_latency_ms": new_jev.latency_ms,
                            "focus_pose": new_jev.focus_pose,
                            "checkpoints": [(c.checkpoint, c.choice, c.confidence)
                                            for c in (new_jev.coaching.corrections if new_jev.coaching else [])],
                            "form_score": new_jev.coaching.form_score if new_jev.coaching else None,
                            "decision": {k: getattr(jd, k) for k in (
                                "final_pose", "assessment_allowed", "tier", "agreement", "reason")},
                            "feedback": [(i.issue, i.severity, i.instruction) for i in last_feedback],
                        }) + "\n")
                msg = feedback_engine.pop_speech()
                if msg:
                    voice.speak(msg)
            hud_prediction = display_prediction if (jev_primary and display_prediction is not None) else last_prediction

            # ── Prepare display frame (mirrored horizontally for natural view) ──────
            if camera_cfg["mirror"]:
                frame_display = cv2.flip(frame_raw, 1)
                display_landmarks = [
                    {**lm, "x": 1.0 - lm["x"]}
                    for lm in last_lm_effective
                ] if last_lm_effective else None
            else:
                frame_display = frame_raw.copy()
                display_landmarks = last_lm_effective

            # ── Draw skeleton with deviant joint highlights ───────────────────
            if display_landmarks and display_cfg["show_landmarks"]:
                draw_skeleton(
                    frame_display,
                    display_landmarks,
                    show_landmarks=display_cfg["show_landmarks"],
                    show_skeleton=display_cfg["show_skeleton"],
                    highlight_joint_ids=highlight_joint_ids,
                )

            # ── Draw angle values ─────────────────────────────────────────────
            if show_angles and display_landmarks and last_features:
                draw_angle_values(
                    frame_display, display_landmarks, last_features,
                    list(ANGLE_TRIPLETS.keys())
                )

            # ── HUD with alignment score ──────────────────────────────────────
            if hud_prediction:
                draw_hud(
                    frame_display,
                    prediction=hud_prediction,
                    feedback_items=last_feedback,
                    voice_enabled=voice.enabled,
                    inference_interval=interval,
                    last_inference_age=now - last_inference_time,
                    show_angles=show_angles,
                    font_scale=display_cfg["font_scale"],
                    max_lines=display_cfg["max_feedback_lines"],
                    target_pose=active_target,
                    landmarks=last_lm_effective,
                    alignment_score=(jev_form_score if jev_primary
                                     else last_eval_result.alignment_score if last_eval_result else None),
                    decision=last_decision,
                    jev_status=(jev.status if jev.enabled else "off"),
                )

            if display_cfg.get("show_decision_status", True) and last_decision is not None:
                draw_decision_status(
                    frame_display, last_decision,
                    landmark_tier=last_decision_tier if last_lm_effective else "low",
                    correction_state=correction_state(last_decision, last_feedback),
                    jev_status=jev.status if jev.enabled else "disabled",
                )

            if show_diag:
                fd = getattr(last_prediction, "filter_decision", None) if last_prediction else None
                span = fps_hist[-1] - fps_hist[0] if len(fps_hist) > 1 else 0.0
                lines = [
                    f"FPS: {(len(fps_hist) - 1) / span:.1f}" if span > 0 else "FPS: -",
                    f"MediaPipe ({landmarker.running_mode}): {timings['mediapipe_ms']:.1f} ms",
                    f"Classifier+decision: {timings['inference_ms']:.1f} ms",
                    f"Mode: {mode}  target: {active_target or '-'}",
                ]
                if lm_frame is not None:
                    lines.append(f"Landmarks imputed: {lm_frame.n_imputed}  unavailable: {lm_frame.n_unavailable}"
                                 f"  suspect: {len(lm_frame.suspect_ids)}")
                if last_prediction and last_prediction.top3:
                    lines.append("Top3: " + ", ".join(f"{n[:10]} {p:.2f}" for n, p in last_prediction.top3))
                if fd is not None:
                    lines.append(f"Filter[{fd.mode}] base={fd.base_posture} "
                                 f"{'skip: ' + fd.skipped_reason if fd.skipped_reason else 'suppr: ' + ','.join(s[:6] for s in fd.suppressed)}")
                if last_decision is not None:
                    lines.append(f"Decision: {last_decision.final_pose or '-'} [{last_decision.tier}/{last_decision.source}]")
                tiers = {k: v.tier for k, v in last_rel.joint_assessments.items()} if (last_rel and last_lm_effective) else {}
                draw_diagnostics(frame_display, lines, tiers)

            # ── Biomechanical Calibration Overlay ─────────────────────────────
            if calib_session.is_active:
                pct = int((calib_session.frame_count / calib_session.target_frames) * 100)
                draw_calibration_overlay(frame_display, True, pct, calib_session.status_message)

            draw_key_hints(frame_display)

            # ── Render Right Reference Asana Panel (Dynamic Real-Time Reference) ────
            assessed = last_decision is not None and last_decision.assessment_allowed
            candidate_pose = last_decision.final_pose if assessed else (
                last_prediction.pose_name if (last_prediction and last_prediction.status == "detected") else None
            )
            candidate_conf = hud_prediction.confidence if hud_prediction else (
                last_prediction.confidence if last_prediction else 0.0
            )
            current_ref_pose = ref_display.update_auto_detected_pose(
                candidate_pose,
                candidate_conf,
            )

            primary_joint = (
                last_eval_result.primary_check.joint_name
                if (last_eval_result and last_eval_result.primary_check and last_eval_result.primary_check.status != "ok")
                else None
            )

            ref_panel = ref_display.render_panel(
                pose_name=current_ref_pose,
                height=frame_display.shape[0],
                focus_joint=primary_joint,
                is_guided=False,
                confidence=hud_prediction.confidence if hud_prediction else None,
            )

            # ── Combine Left Live Panel and Right Reference Panel ─────────────
            combined_display = cv2.hconcat([frame_display, ref_panel])

            if max_frames and frame_id >= max_frames:
                break
            if headless:
                continue
            if not headless and frame_id == 20:
                try:
                    cv2.setWindowProperty(win_title, cv2.WND_PROP_TOPMOST, 0)
                except Exception:
                    pass
            cv2.imshow(win_title, combined_display)

            # ── Key controls ──────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:                # Q / Esc → quit
                break
            elif key == ord("m") or key == ord("t") or key == ord("n"):
                print("[App] Real-Time Auto-Detect & Form Feedback is active.")
            elif key == ord("c"):                            # C → start/cancel calibration
                if not calib_session.is_active:
                    calib_session.start()
                    voice.speak("Stand tall in Mountain Pose for 5 seconds to calibrate your body baseline.")
                    print("[App] Calibration started. Stand tall in Mountain Pose (Tadasana)...")
                else:
                    calib_session.cancel()
                    print("[App] Calibration cancelled.")
            elif key == ord("v"):                            # V → toggle voice
                state = voice.toggle()
                print(f"[App] Voice {'ON' if state else 'OFF'}")
            elif key == ord("a"):                            # A → toggle angles
                show_angles = not show_angles
                print(f"[App] Angle overlay {'ON' if show_angles else 'OFF'}")
            elif key == ord("d"):                            # D → diagnostics panel
                show_diag = not show_diag
                print(f"[App] Diagnostics {'ON' if show_diag else 'OFF'}")
            elif key == ord("j"):                            # J → manual Jev assessment
                if not jev.enabled:
                    print("[App] Jev is disabled (set jev.enabled: true and JEVMODEL_API_KEY).")
                elif not jev_cfg.get("manual_key", True):
                    print("[App] Manual Jev requests are disabled in config.")
                elif jev.submit(frame_raw, last_lm_effective, frame_id, manual=True):
                    print("[App] Jev assessment requested for the current frame.")
                else:
                    print("[App] Jev request not sent (one already in progress).")
            elif key == ord("r"):                            # R → reset history
                inferrer.reset_history()
                feedback_engine.clear()
                ref_display.reset_stability()
                lm_history.reset()
                reliability_tracker.reset()
                jev.new_epoch()
                print("[App] Prediction history reset.")
            elif key == ord("s"):                            # S → screenshot
                fname = screenshot_dir / f"yoga_{int(time.time())}.jpg"
                cv2.imwrite(str(fname), combined_display)
                print(f"[App] Screenshot saved: {fname}")
            elif key == ord("+") or key == ord("="):         # + → increase interval
                interval = min(interval + 0.5, 10.0)
                print(f"[App] Inference interval: {interval:.1f}s")
            elif key == ord("-"):                            # - → decrease interval
                interval = max(interval - 0.5, 0.5)
                print(f"[App] Inference interval: {interval:.1f}s")

    # ── Cleanup & Session Summary (always runs) ───────────────────────────────
    finally:
        if cap is not None:
            cap.release()
        if "decision_log" in locals() and decision_log is not None:
            decision_log.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # headless OpenCV build
        landmarker.close()
        if jev.enabled:
            jev.worker.shutdown()
        if voice is not None:
            voice.shutdown()

    summary = progress_tracker.end_session() if progress_tracker else None
    if summary and summary.get("frames_evaluated", 0) > 0:
        print("\n" + "=" * 65)
        print(f"  YOGA MITRA — PRACTICE SESSION SUMMARY")
        print("=" * 65)
        print(f"  Session Duration:  {summary['duration_sec']:.1f}s")
        print(f"  Frames Evaluated:  {summary['frames_evaluated']}")
        print(f"  Poses Practiced:   {', '.join(summary['poses_practiced']) or 'None'}")
        print(f"  Avg Alignment:     {summary['avg_alignment_score']:.1f}%")
        print(f"  Peak Alignment:    {summary['peak_alignment_score']:.1f}%")
        print(f"  User Calibrated:   {'Yes' if summary['calibrated'] else 'No'}")
        if summary.get("top_joint_faults"):
            print("  Top Recurring Alignment Issues:")
            for fault in summary["top_joint_faults"][:3]:
                print(f"    - {fault['pose']} {fault['joint']}: {fault['count']}x (avg dev: {fault['avg_dev_deg']}°)")
        print("=" * 65 + "\n")

    print("[App] Exited cleanly.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Resolve config path relative to project root
    cfg_path = _ROOT / "config.yaml"
    run(str(cfg_path))
