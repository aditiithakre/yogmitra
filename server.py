"""
server.py  –  Yoga Mitra Web Server & Live Browser Interface
─────────────────────────────────────────────────────────────
Serves a modern, interactive web dashboard on http://localhost:5000 with:
- Live MJPEG video stream with skeletal overlays, angle pills, and reference cards
- Real-time exact biomechanical joint telemetry and Model B Form Verifier status
- REST API for pose selection, guided/free mode switching, and session management
- Native SAPI5 / Web audio voice feedback
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

# UTF-8 console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np
import yaml
from flask import Flask, Response, jsonify, render_template_string, request, send_file

from src.app.overlay import (
    draw_angle_values,
    draw_hud,
    draw_skeleton,
)
from src.app.reference_display import ReferenceDisplay, get_target_poses_for_profile
from src.correction.reference_ranges import POSE_METADATA
from src.app.voice import VoiceFeedback
from src.correction.calibration import UserProfile, load_profile
from src.correction.feedback_engine import FeedbackEngine
from src.correction.posture_evaluator import evaluate as evaluate_posture
from src.correction.progress_tracker import ProgressTracker
from src.correction.reliability_tracker import TemporalReliabilityTracker
from src.features import ANGLE_TRIPLETS, extract_features, features_to_series
from src.models.inference import DualTaskInference, PoseInference, PosePrediction
from src.models.model_loader import load_artifacts, load_zenodo_artifacts
from src.pose import PoseLandmarkerWrapper
from src.pose.missing_landmarks import apply_temporal_fallback

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("YogaMitraServer")

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


class YogaMitraEngine:
    """Core vision and inference pipeline running continuously in a worker thread."""

    def __init__(self, config_path: str = "config.yaml", profile: Optional[str] = None):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        if profile:
            self.cfg["model"]["profile"] = profile

        model_cfg = self.cfg["model"]
        infer_cfg = self.cfg["inference"]
        cam_cfg = self.cfg["camera"]
        voice_cfg = self.cfg["voice"]

        self.requested_profile = model_cfg.get("profile", "zenodo").lower()
        self.camera_idx = cam_cfg.get("device_index", 0)
        self.mirror = cam_cfg.get("mirror", True)
        self.interval = infer_cfg.get("interval_seconds", 0.5)

        # Initialize Models
        zenodo_dir = _ROOT / "saved_yoga_model_zenodo"
        zenodo_ready = (
            (zenodo_dir / "zenodo_asana_classifier.joblib").exists() and
            (zenodo_dir / "zenodo_form_verifier.joblib").exists() and
            (zenodo_dir / "zenodo_preprocessing.joblib").exists()
        )

        if self.requested_profile == "zenodo" and zenodo_ready:
            logger.info("Loading Zenodo Dual-Task Model...")
            clf_choice = model_cfg.get("active_classifier", "rf")
            self.artifacts = load_zenodo_artifacts(zenodo_dir, active_classifier=clf_choice)
            self.inferrer = DualTaskInference(
                artifacts=self.artifacts,
                confidence_threshold=infer_cfg["confidence_threshold"],
                smoothing_window=infer_cfg["smoothing_window"],
            )
            self.active_profile = "zenodo"
        else:
            logger.info("Loading Kaggle Model Profile...")
            model_dir = Path(model_cfg["dir"])
            self.artifacts = load_artifacts(model_dir)
            self.inferrer = PoseInference(
                artifacts=self.artifacts,
                active_classifier=model_cfg["active_classifier"],
                confidence_threshold=infer_cfg["confidence_threshold"],
                smoothing_window=infer_cfg["smoothing_window"],
            )
            self.active_profile = "kaggle"

        # MediaPipe Landmarker
        task_path = str(model_cfg["mediapipe_task"])
        self.landmarker = PoseLandmarkerWrapper(
            model_path=task_path,
            visibility_threshold=infer_cfg["visibility_threshold"],
        )

        # Correction and Display
        self.feedback_engine = FeedbackEngine(
            cooldown_seconds=voice_cfg.get("cooldown_seconds", 4.5),
            min_persistence_frames=1,
            max_display=4,
        )
        self.ref_display = ReferenceDisplay(assets_dir=_ROOT / "assets" / "pose_references")
        self.progress_tracker = ProgressTracker()
        self.reliability_tracker = TemporalReliabilityTracker()
        self.user_profile = load_profile()

        # Voice
        self.voice = VoiceFeedback(
            rate=voice_cfg.get("rate", 175),
            volume=voice_cfg.get("volume", 1.0),
        )
        if not voice_cfg.get("enabled", True):
            self.voice.enabled = False

        # Session State (Real-Time Mode)
        self.practice_poses = get_target_poses_for_profile(self.active_profile)
        self.target_idx = 0
        self.target_mode = False
        self.show_angles = True

        # Latest frame and telemetry cache
        self._lock = threading.Lock()
        self.latest_jpeg: Optional[bytes] = None
        self.latest_status: dict = {
            "pose": "Initializing...",
            "confidence": 0.0,
            "form_label": "N/A",
            "form_correct": False,
            "form_confidence": 0.0,
            "alignment_score": 0.0,
            "target_pose": "Auto-Detect",
            "target_mode": False,
            "exact_angles": [],
            "feedback": [],
            "fps": 0.0,
        }

        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        logger.info("Yoga Mitra worker loop started.")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self.voice.shutdown()

    def set_target_pose(self, pose_name: str) -> bool:
        return False

    def toggle_mode(self) -> bool:
        return False

    def toggle_voice(self) -> bool:
        self.voice._enabled = not self.voice._enabled
        return self.voice._enabled

    def _worker(self):
        cap = cv2.VideoCapture(self.camera_idx)
        if not cap.isOpened():
            logger.warning(f"Cannot open camera index {self.camera_idx}. Trying index 1...")
            cap = cv2.VideoCapture(1)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        prev_raw_landmarks = None
        last_lm_effective = None
        last_features = None
        last_prediction: Optional[PosePrediction] = None
        last_eval_result = None
        last_feedback = []
        highlight_joint_ids = set()

        last_inference_time = 0.0
        fps_t0 = time.monotonic()
        frame_counter = 0
        current_fps = 0.0

        while self.running:
            ret, frame_raw = cap.read()
            if not ret:
                time.sleep(0.04)
                continue

            frame_counter += 1
            now = time.monotonic()
            if now - fps_t0 >= 1.0:
                current_fps = frame_counter / (now - fps_t0)
                frame_counter = 0
                fps_t0 = now

            active_target = None

            # Detect landmarks
            raw_lm = self.landmarker.detect(frame_raw)

            run_inference = (now - last_inference_time) >= self.interval

            if raw_lm is None:
                last_lm_effective = None
                prev_raw_landmarks = None
                highlight_joint_ids.clear()
                if run_inference:
                    last_inference_time = now
                    dummy_pred = PosePrediction(
                        pose_name="—",
                        confidence=0.0,
                        probabilities=np.zeros(getattr(self.artifacts, "n_classes", 10)),
                        raw_prediction="—",
                        smoothed=False,
                        status="no_person",
                        top3=[],
                    )
                    last_prediction = dummy_pred
                    last_eval_result = None
                    last_feedback = self.feedback_engine.update(dummy_pred, None, target_pose=None)
            else:
                effective_lm, flags, raw_for_history = apply_temporal_fallback(
                    raw_lm, prev_raw_landmarks,
                    threshold=self.cfg["inference"]["visibility_threshold"],
                )
                prev_raw_landmarks = raw_for_history
                last_lm_effective = effective_lm

                rel_assessment = self.reliability_tracker.update(effective_lm, flags)
                h, w = frame_raw.shape[:2]
                feat_dict = extract_features(effective_lm, flags, w, h)
                last_features = feat_dict

                if run_inference:
                    last_inference_time = now
                    feat_series = features_to_series(feat_dict, self.artifacts.feature_cols)

                    prediction = self.inferrer.predict(feat_series, landmarks=effective_lm, user_profile=self.user_profile)
                    last_prediction = prediction

                    eval_pose = prediction.pose_name
                    eval_result = evaluate_posture(
                        pose_name=eval_pose,
                        feature_dict=feat_dict,
                        landmarks=effective_lm,
                        imputed_flags=flags,
                        user_profile=asdict(self.user_profile) if self.user_profile.calibrated else None,
                        reliability_assessment=rel_assessment,
                    )
                    last_eval_result = eval_result

                    highlight_joint_ids.clear()
                    if eval_result and eval_result.checks:
                        for chk in eval_result.checks:
                            if chk.status != "ok" and chk.reliable and chk.severity in ("moderate", "major"):
                                lm_id = JOINT_LM_MAP.get(chk.angle_name)
                                if lm_id is not None:
                                    highlight_joint_ids.add(lm_id)

                    last_feedback = self.feedback_engine.update(
                        prediction, eval_result, target_pose=None
                    )

                    # Speak cue
                    msg = self.feedback_engine.pop_speech()
                    if msg:
                        self.voice.speak(msg)

            # Mirror frame
            if self.mirror:
                frame_display = cv2.flip(frame_raw, 1)
                display_landmarks = [
                    {"x": 1.0 - lm["x"], "y": lm["y"], "z": lm["z"], "visibility": lm.get("visibility", 0.0)}
                    for lm in last_lm_effective
                ] if last_lm_effective else None
            else:
                frame_display = frame_raw.copy()
                display_landmarks = last_lm_effective

            # Skeleton
            if display_landmarks:
                draw_skeleton(
                    frame_display,
                    display_landmarks,
                    show_landmarks=True,
                    show_skeleton=True,
                    highlight_joint_ids=highlight_joint_ids,
                )

            # Angle pills
            if self.show_angles and display_landmarks and last_features:
                draw_angle_values(
                    frame_display,
                    display_landmarks,
                    last_features,
                    list(ANGLE_TRIPLETS.keys()),
                )

            # HUD
            if last_prediction:
                draw_hud(
                    frame_display,
                    prediction=last_prediction,
                    feedback_items=last_feedback,
                    voice_enabled=self.voice.enabled,
                    inference_interval=self.interval,
                    last_inference_age=now - last_inference_time,
                    show_angles=self.show_angles,
                    target_pose=active_target,
                    landmarks=last_lm_effective,
                    alignment_score=last_eval_result.alignment_score if last_eval_result else None,
                )

            # Reference Panel (Dynamic Real-Time Reference)
            current_ref = self.ref_display.update_auto_detected_pose(
                last_prediction.pose_name if last_prediction else None,
                last_prediction.confidence if last_prediction else 0.0,
            )

            primary_joint = (
                last_eval_result.primary_check.joint_name
                if (last_eval_result and last_eval_result.primary_check and last_eval_result.primary_check.status != "ok")
                else None
            )

            ref_panel = self.ref_display.render_panel(
                pose_name=current_ref,
                height=frame_display.shape[0],
                focus_joint=primary_joint,
                is_guided=False,
                confidence=last_prediction.confidence if last_prediction else None,
            )

            # Combine Left & Right
            combined = cv2.hconcat([frame_display, ref_panel])

            # Encode JPEG for Web streaming
            ret_enc, jpeg = cv2.imencode(".jpg", combined, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret_enc:
                jpeg_bytes = jpeg.tobytes()

                # Build telemetry for JSON API
                exact_list = []
                if last_eval_result and last_eval_result.checks:
                    for chk in last_eval_result.checks:
                        if chk.value is not None:
                            low, high = chk.target_range
                            exact_list.append({
                                "joint": chk.display_name.title(),
                                "value": round(chk.value, 1),
                                "target": f"{low:.0f}°–{high:.0f}°",
                                "status": chk.status,
                                "deviation": round(chk.deviation, 1),
                            })

                corr_lbl = getattr(last_prediction, "correctness_label", "N/A") if last_prediction else "N/A"
                is_corr = getattr(last_prediction, "is_form_correct", False) if last_prediction else False
                corr_conf = getattr(last_prediction, "correctness_confidence", 0.0) if last_prediction else 0.0

                display_pose = current_ref or active_target or (last_prediction.pose_name if last_prediction else "Tadasana")
                if not display_pose or display_pose == "—" or display_pose not in POSE_METADATA:
                    display_pose = "Tadasana"
                ref_meta = POSE_METADATA.get(display_pose, {})

                with self._lock:
                    self.latest_jpeg = jpeg_bytes
                    self.latest_status = {
                        "pose": last_prediction.pose_name if last_prediction else "Detecting...",
                        "confidence": round(float(last_prediction.confidence) * 100, 1) if last_prediction else 0.0,
                        "form_label": corr_lbl.upper(),
                        "form_correct": bool(is_corr),
                        "form_confidence": round(float(corr_conf) * 100, 1),
                        "alignment_score": round(float(last_eval_result.alignment_score), 1) if last_eval_result else 0.0,
                        "target_pose": active_target or "Auto-Detect",
                        "display_pose": display_pose,
                        "english_name": ref_meta.get("english_name", display_pose),
                        "sanskrit_name": ref_meta.get("sanskrit_name", display_pose),
                        "ref_cues": ref_meta.get("cues", [])[:3],
                        "target_mode": bool(self.target_mode),
                        "exact_angles": exact_list,
                        "feedback": [f.instruction for f in last_feedback[:3]],
                        "fps": round(current_fps, 1),
                        "voice_enabled": bool(self.voice.enabled),
                    }

        cap.release()
        logger.info("Yoga Mitra worker loop terminated.")


# ── Web Application Factory ──────────────────────────────────────────────────
app = Flask(__name__)
engine: Optional[YogaMitraEngine] = None


@app.route("/")
def index():
    global engine
    poses = engine.practice_poses if engine else []
    return render_template_string(HTML_TEMPLATE, poses=poses)


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            if engine and engine.latest_jpeg:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + engine.latest_jpeg + b"\r\n")
            time.sleep(0.033)  # ~30 fps cap
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    if not engine:
        return jsonify({"error": "Engine not running"}), 503
    with engine._lock:
        return jsonify(engine.latest_status)


@app.route("/api/control", methods=["POST"])
def api_control():
    if not engine:
        return jsonify({"error": "Engine not running"}), 503
    data = request.json or {}
    action = data.get("action")
    if action == "set_target":
        pose = data.get("pose")
        success = engine.set_target_pose(pose)
        return jsonify({"success": success, "target": pose})
    elif action == "toggle_mode":
        mode = engine.toggle_mode()
        return jsonify({"success": True, "target_mode": mode})
    elif action == "toggle_voice":
        v = engine.toggle_voice()
        return jsonify({"success": True, "voice_enabled": v})
    return jsonify({"error": "Invalid action"}), 400


@app.route("/api/reference_image/<path:pose_name>")
def api_reference_image(pose_name: str):
    meta = POSE_METADATA.get(pose_name)
    asset_file = meta.get("asset_name") if meta else None
    if not asset_file:
        asset_file = f"{pose_name.lower().replace(' ', '_')}.png"

    img_path = _ROOT / "assets" / "pose_references" / asset_file
    if not img_path.exists():
        img_path = _ROOT / "assets" / "pose_references" / asset_file.lower()
    if not img_path.exists():
        img_path = _ROOT / "assets" / "pose_references" / "tadasana.png"

    if img_path.exists():
        return send_file(str(img_path), mimetype="image/png")
    return Response(status=404)



# ── UI Dashboard HTML Template ───────────────────────────────────────────────
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Yoga Mitra — AI Posture Detection & Biomechanical Correction Studio</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #0c0f14;
            --bg-card: rgba(20, 26, 35, 0.75);
            --bg-card-hover: rgba(28, 36, 48, 0.85);
            --border-card: rgba(255, 255, 255, 0.08);
            --accent: #2dd4bf;
            --accent-glow: rgba(45, 212, 191, 0.25);
            --accent-green: #22c55e;
            --accent-amber: #f59e0b;
            --accent-red: #ef4444;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --font-ui: 'Outfit', -apple-system, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg-base);
            background-image: 
                radial-gradient(at 0% 0%, rgba(45, 212, 191, 0.07) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(59, 130, 246, 0.06) 0px, transparent 50%);
            color: var(--text-main);
            font-family: var(--font-ui);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
        }

        header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 16px 32px;
            background: rgba(12, 15, 20, 0.8);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border-card);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .brand-icon {
            width: 36px;
            height: 36px;
            background: linear-gradient(135deg, var(--accent), #3b82f6);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            box-shadow: 0 4px 12px var(--accent-glow);
        }

        .brand-title {
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            background: linear-gradient(90deg, #fff, #94a3b8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .badge-live {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            background: rgba(34, 197, 94, 0.12);
            border: 1px solid rgba(34, 197, 94, 0.3);
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--accent-green);
        }

        .badge-live::before {
            content: "";
            width: 7px;
            height: 7px;
            background: var(--accent-green);
            border-radius: 50%;
            box-shadow: 0 0 8px var(--accent-green);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.4; transform: scale(0.85); }
        }

        .main-container {
            flex: 1;
            padding: 24px 32px;
            display: grid;
            grid-template-columns: 1fr 380px;
            gap: 24px;
            max-width: 1700px;
            width: 100%;
            margin: 0 auto;
        }

        .stream-card {
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: 18px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            backdrop-filter: blur(12px);
            box-shadow: 0 16px 32px rgba(0, 0, 0, 0.4);
        }

        .stream-header {
            padding: 14px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--border-card);
            background: rgba(255, 255, 255, 0.02);
        }

        .stream-title {
            font-size: 0.90rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .video-wrapper {
            position: relative;
            width: 100%;
            aspect-ratio: 16 / 9;
            background: #000;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .video-wrapper img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        /* Controls Toolbar */
        .controls-toolbar {
            padding: 14px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(255, 255, 255, 0.02);
            border-top: 1px solid var(--border-card);
            flex-wrap: wrap;
            gap: 12px;
        }

        .btn-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        button, select {
            font-family: var(--font-ui);
            font-size: 0.85rem;
            font-weight: 600;
            padding: 8px 14px;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.2s ease;
            outline: none;
        }

        select {
            background: #19202a;
            color: var(--text-main);
            border: 1px solid rgba(255, 255, 255, 0.15);
        }

        button.btn-primary {
            background: linear-gradient(135deg, var(--accent), #0d9488);
            color: #042f2e;
            border: none;
        }

        button.btn-primary:hover {
            box-shadow: 0 4px 14px var(--accent-glow);
            transform: translateY(-1px);
        }

        button.btn-ghost {
            background: rgba(255, 255, 255, 0.06);
            color: var(--text-main);
            border: 1px solid var(--border-card);
        }

        button.btn-ghost:hover {
            background: rgba(255, 255, 255, 0.12);
        }

        /* Sidebar Panels */
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .panel-card {
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: 16px;
            padding: 18px;
            backdrop-filter: blur(12px);
        }

        .card-header {
            font-size: 0.75rem;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .metric-value-huge {
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--text-main);
            letter-spacing: -0.02em;
        }

        .form-banner {
            padding: 10px 14px;
            border-radius: 10px;
            font-size: 0.85rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-top: 10px;
        }

        .form-correct {
            background: rgba(34, 197, 94, 0.15);
            border: 1px solid rgba(34, 197, 94, 0.4);
            color: #4ade80;
        }

        .form-wrong {
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.4);
            color: #f87171;
        }

        /* Exact Table */
        .exact-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.82rem;
            font-family: var(--font-mono);
            margin-top: 8px;
        }

        .exact-table th {
            text-align: left;
            padding: 6px 8px;
            color: var(--text-muted);
            border-bottom: 1px solid var(--border-card);
            font-weight: 500;
            font-size: 0.70rem;
        }

        .exact-table td {
            padding: 8px 8px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
        }

        .stat-ok {
            color: var(--accent-green);
            font-weight: 600;
        }

        .stat-dev {
            color: var(--accent-amber);
            font-weight: 600;
        }

        .cue-item {
            display: flex;
            align-items: flex-start;
            gap: 8px;
            font-size: 0.82rem;
            color: #cbd5e1;
            padding: 6px 0;
        }

        .cue-dot {
            width: 6px;
            height: 6px;
            background: var(--accent);
            border-radius: 50%;
            margin-top: 6px;
        }

        @media (max-width: 1100px) {
            .main-container {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>

    <header>
        <div class="brand">
            <div class="brand-icon">🧘</div>
            <div>
                <div class="brand-title">Yoga Mitra Studio</div>
                <div style="font-size: 0.75rem; color: var(--text-muted);">AI Posture Evaluation & Dual-Task Verification</div>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 14px;">
            <div class="badge-live">SERVER ONLINE</div>
            <div id="fps-badge" style="font-size: 0.75rem; color: var(--text-muted); font-family: var(--font-mono);">FPS: --</div>
        </div>
    </header>

    <div class="main-container">
        
        <!-- Stream Column -->
        <div class="stream-card">
            <div class="stream-header">
                <span class="stream-title">Dual-Panel Live Vision Feed</span>
                <span id="header-target-label" style="font-size: 0.80rem; color: var(--accent); font-weight: 600;">GUIDED MODE</span>
            </div>

            <div class="video-wrapper">
                <img src="/video_feed" alt="Yoga Mitra Live Video Feed">
            </div>

            <div class="controls-toolbar">
                <div class="btn-group">
                    <span style="font-size: 0.85rem; color: var(--accent); font-weight: 600;">⚡ Real-Time Posture AI</span>
                </div>

                <div class="btn-group">
                    <button class="btn-ghost" id="voice-btn" onclick="toggleVoice()">Voice: ON</button>
                </div>
            </div>
        </div>

        <!-- Right Telemetry Column -->
        <div class="sidebar">

            <!-- Reference Asana Card -->
            <div class="panel-card" id="ref-card">
                <div class="card-header">
                    <span>Reference Asana</span>
                    <span id="ref-mode-badge" style="color: var(--accent); font-weight: 600;">REAL-TIME</span>
                </div>
                <div style="display: flex; gap: 14px; align-items: center; margin-bottom: 12px;">
                    <div style="width: 100px; height: 100px; flex-shrink: 0; background: rgba(0,0,0,0.35); border-radius: 12px; border: 1px solid var(--border-card); display: flex; align-items: center; justify-content: center; overflow: hidden; padding: 4px;">
                        <img id="ref-img" src="/api/reference_image/Tadasana" alt="Reference Illustration" style="max-width: 100%; max-height: 100%; object-fit: contain;">
                    </div>
                    <div style="flex: 1; min-width: 0;">
                        <div id="ref-english" style="font-size: 1.15rem; font-weight: 700; color: #fff; line-height: 1.25;">Mountain Pose</div>
                        <div id="ref-sanskrit" style="font-size: 0.85rem; color: var(--accent); margin-top: 4px; font-weight: 500;">Tadasana</div>
                        <div id="ref-status-text" style="font-size: 0.75rem; color: var(--text-muted); margin-top: 6px;">Target posture</div>
                    </div>
                </div>
                <div id="ref-cues-list" style="border-top: 1px solid var(--border-card); padding-top: 8px;">
                    <div class="cue-item"><div class="cue-dot"></div><span>Stand grounded through all four foot corners</span></div>
                    <div class="cue-item"><div class="cue-dot"></div><span>Lengthen spine and roll shoulders down</span></div>
                </div>
            </div>

            <!-- Alignment Score Card -->
            <div class="panel-card">
                <div class="card-header">
                    <span>Alignment Score</span>
                    <span id="score-status" style="color: var(--accent-green); font-weight: 600;">ACTIVE</span>
                </div>
                <div class="metric-value-huge" id="score-value">--%</div>
                
                <!-- Model B Form Verifier Banner -->
                <div id="form-banner" class="form-banner form-correct">
                    <span id="form-label">FORM: VERIFYING...</span>
                    <span id="form-conf" style="font-family: var(--font-mono); font-size: 0.75rem;">--%</span>
                </div>
            </div>

            <!-- Exact Biomechanical Angles Table -->
            <div class="panel-card">
                <div class="card-header">
                    <span>Exact Joint Biomechanics</span>
                    <span style="font-family: var(--font-mono); font-size: 0.70rem; color: var(--accent);">DEGREES (°)</span>
                </div>

                <table class="exact-table">
                    <thead>
                        <tr>
                            <th>JOINT</th>
                            <th>LIVE</th>
                            <th>TARGET</th>
                            <th>STATUS</th>
                        </tr>
                    </thead>
                    <tbody id="exact-table-body">
                        <tr><td colspan="4" style="color: var(--text-muted); text-align: center;">Aligning body in camera...</td></tr>
                    </tbody>
                </table>
            </div>

            <!-- Live Speech Cues -->
            <div class="panel-card">
                <div class="card-header">
                    <span>Active Corrections & Cues</span>
                </div>
                <div id="feedback-list">
                    <div class="cue-item"><div class="cue-dot"></div><span>Maintain steady breathing throughout pose</span></div>
                    <div class="cue-item"><div class="cue-dot"></div><span>Keep spine long and core engaged</span></div>
                </div>
            </div>

        </div>

    </div>

    <script>
        // Poll status every 250ms for live telemetry
        async function fetchStatus() {
            try {
                const res = await fetch('/api/status');
                if (!res.ok) return;
                const data = await res.json();

                // Score
                document.getElementById('score-value').innerText = data.alignment_score + '%';
                document.getElementById('fps-badge').innerText = 'FPS: ' + data.fps;

                // Form Verifier
                const formBanner = document.getElementById('form-banner');
                const formLabel = document.getElementById('form-label');
                const formConf = document.getElementById('form-conf');

                if (data.form_correct) {
                    formBanner.className = 'form-banner form-correct';
                    formLabel.innerText = 'FORM VERIFIER: RIGHT ✓';
                } else {
                    formBanner.className = 'form-banner form-wrong';
                    formLabel.innerText = 'FORM VERIFIER: WRONG ✗';
                }
                formConf.innerText = data.form_confidence + '%';

                // Reference Pose Card
                if (data.display_pose) {
                    const refImg = document.getElementById('ref-img');
                    const targetSrc = '/api/reference_image/' + encodeURIComponent(data.display_pose);
                    if (!refImg.src.endsWith(targetSrc)) {
                        refImg.src = targetSrc;
                    }
                    if (data.english_name) document.getElementById('ref-english').innerText = data.english_name;
                    if (data.sanskrit_name) document.getElementById('ref-sanskrit').innerText = data.sanskrit_name;
                    document.getElementById('ref-mode-badge').innerText = 'REAL-TIME';
                    document.getElementById('ref-status-text').innerText = 'Detected Live Reference Posture';

                    const cuesContainer = document.getElementById('ref-cues-list');
                    if (data.ref_cues && data.ref_cues.length > 0) {
                        cuesContainer.innerHTML = data.ref_cues.map(c =>
                            `<div class="cue-item"><div class="cue-dot"></div><span>${c}</span></div>`
                        ).join('');
                    }
                }

                document.getElementById('header-target-label').innerText = 'REAL-TIME AUTO-DETECT';

                // Voice Button
                document.getElementById('voice-btn').innerText = data.voice_enabled ? 'Voice: ON' : 'Voice: OFF';

                // Exact Angles Table
                const tbody = document.getElementById('exact-table-body');
                if (data.exact_angles && data.exact_angles.length > 0) {
                    let html = '';
                    data.exact_angles.slice(0, 5).forEach(item => {
                        const isOk = item.status === 'ok';
                        const statClass = isOk ? 'stat-ok' : 'stat-dev';
                        const statText = isOk ? 'OK ✓' : (item.deviation > 0 ? `+${item.deviation}°` : `${item.deviation}°`);
                        html += `<tr>
                            <td style="color: #f1f5f9;">${item.joint.replace('Angle', '').trim()}</td>
                            <td class="${statClass}">${item.value}°</td>
                            <td style="color: #94a3b8;">${item.target}</td>
                            <td class="${statClass}">${statText}</td>
                        </tr>`;
                    });
                    tbody.innerHTML = html;
                }

                // Feedback
                const feedbackDiv = document.getElementById('feedback-list');
                if (data.feedback && data.feedback.length > 0) {
                    let html = '';
                    data.feedback.forEach(msg => {
                        html += `<div class="cue-item"><div class="cue-dot" style="background:#f59e0b;"></div><span>${msg}</span></div>`;
                    });
                    feedbackDiv.innerHTML = html;
                }
            } catch (err) {
                console.warn('Status poll error:', err);
            }
        }

        setInterval(fetchStatus, 300);

        async function changeTargetPose(pose) {
            const refImg = document.getElementById('ref-img');
            if (refImg) {
                refImg.src = '/api/reference_image/' + encodeURIComponent(pose);
            }
            await fetch('/api/control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'set_target', pose: pose })
            });
        }

        async function toggleMode() {
            await fetch('/api/control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'toggle_mode' })
            });
        }

        async function toggleVoice() {
            await fetch('/api/control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'toggle_voice' })
            });
        }
    </script>
</body>
</html>
"""


def main():
    global engine
    parser = argparse.ArgumentParser(description="Yoga Mitra Web Server & Live Studio")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Port to serve (default: 5000)")
    parser.add_argument("--profile", choices=["kaggle", "zenodo"], default="zenodo",
                        help="Model profile (default: zenodo)")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  🧘 YOGA MITRA — AI POSTURE WEB SERVER")
    print("=" * 60)
    print(f"  Profile : {args.profile.upper()}")
    print(f"  Camera  : Device #{args.camera}")
    print(f"  URL     : http://localhost:{args.port}")
    print("=" * 60 + "\n")

    engine = YogaMitraEngine(profile=args.profile)
    engine.camera_idx = args.camera
    engine.start()

    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
    finally:
        engine.stop()
        print("[Server] Stopped.")


if __name__ == "__main__":
    main()
