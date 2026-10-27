"""
src/app/overlay.py
────────────────────
Draws MediaPipe skeleton, pose label, confidence, correction messages, and
optional joint-angle values directly onto an OpenCV BGR frame.

All drawing helpers return the modified frame for convenience.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from src.pose import SKELETON_CONNECTIONS
from src.correction.feedback_engine import FeedbackItem
from src.models.inference import PosePrediction


# ── Colour palette ─────────────────────────────────────────────────────────────
CLR_GOOD    = (80,  200, 80)   # BGR green  — correct alignment
CLR_WARN    = (60,  160, 255)  # BGR orange — low confidence / occluded
CLR_CORRECT = (60,   80, 255)  # BGR red    — correction needed
CLR_INFO    = (200, 200, 200)  # BGR grey   — info text
CLR_PANEL   = (20,   20,  20)  # near-black panel background
CLR_BONE    = (180, 120,  60)  # skeleton connection colour
CLR_JOINT   = (60,  200, 255)  # joint dot colour
CLR_WHITE   = (255, 255, 255)


def draw_skeleton(
    frame: np.ndarray,
    landmarks: list[dict],
    show_landmarks: bool = True,
    show_skeleton: bool  = True,
    highlight_joint_ids: Optional[set[int]] = None,
) -> np.ndarray:
    """Draw joints and connections onto the frame (in-place, also returns frame)."""
    h, w = frame.shape[:2]
    if not landmarks:
        return frame

    pts: dict[int, tuple[int, int]] = {}
    for i, lm in enumerate(landmarks):
        x = int(lm["x"] * w)
        y = int(lm["y"] * h)
        if lm.get("unavailable"):
            continue  # expired / never observed: no dot and no bones to a guessed position
        pts[i] = (x, y)
        if show_landmarks:
            vis = float(lm.get("visibility", 0.0))
            if highlight_joint_ids and i in highlight_joint_ids:
                # Pulsing / prominent highlight for deviating joint
                cv2.circle(frame, (x, y), 9, CLR_CORRECT, -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 12, CLR_WHITE, 2, cv2.LINE_AA)
            elif lm.get("imputed"):
                # Estimated from recent history: hollow marker, never drawn as observed
                cv2.circle(frame, (x, y), 6, CLR_WARN, 1, cv2.LINE_AA)
            else:
                colour = CLR_JOINT if vis >= 0.30 else (80, 80, 80)
                cv2.circle(frame, (x, y), 5, colour, -1, cv2.LINE_AA)

    if show_skeleton:
        for (a, b) in SKELETON_CONNECTIONS:
            if a in pts and b in pts:
                # If both ends are highlighted, draw connecting line in amber
                line_col = CLR_WARN if (highlight_joint_ids and (a in highlight_joint_ids or b in highlight_joint_ids)) else CLR_BONE
                cv2.line(frame, pts[a], pts[b], line_col, 2, cv2.LINE_AA)

    return frame


def draw_angle_values(
    frame: np.ndarray,
    landmarks: list[dict],
    feature_dict: dict,
    angle_names: list[str],
) -> np.ndarray:
    """Render a small angle value next to each joint midpoint."""
    h, w = frame.shape[:2]
    from src.features import ANGLE_TRIPLETS

    for name in angle_names:
        val = feature_dict.get(name, np.nan)
        if not isinstance(val, float) or not np.isfinite(val):
            continue
        triplet = ANGLE_TRIPLETS.get(name)
        if triplet is None:
            continue
        _, b_id, _ = triplet
        if b_id >= len(landmarks):
            continue
        lm = landmarks[b_id]
        x = int(lm["x"] * w) + 8
        y = int(lm["y"] * h) - 8
        label = f"{val:.0f}"
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, CLR_WHITE, 1, cv2.LINE_AA)
    return frame


def draw_hud(
    frame: np.ndarray,
    prediction: PosePrediction,
    feedback_items: list[FeedbackItem],
    voice_enabled: bool = True,
    inference_interval: float = 0.5,
    last_inference_age: float = 0.0,
    show_angles: bool = False,
    font_scale: float = 0.55,
    max_lines: int = 3,
    target_pose: Optional[str] = None,
    landmarks: Optional[list[dict]] = None,
    alignment_score: Optional[float] = None,
    decision: Optional[object] = None,
    jev_status: Optional[str] = None,
) -> np.ndarray:
    """Draw a clean, minimalist, website-ready HUD."""
    h, w = frame.shape[:2]

    # ── 1. Check Framing / Full Body Visibility ──────────────────────────────
    framing_warning: Optional[str] = None
    is_floor_or_seated = prediction.pose_name in (
        "Vajrasana", "Parvatasana", "Bhujangasana", "Marjariasana",
        "Sarvangasana", "Viparita Karani", "Anantasana"
    )
    if landmarks and len(landmarks) >= 29:
        knees_ok = (landmarks[25].get("visibility", 0) >= 0.25) or (landmarks[26].get("visibility", 0) >= 0.25)
        ankles_ok = (landmarks[27].get("visibility", 0) >= 0.25) or (landmarks[28].get("visibility", 0) >= 0.25)
        if not knees_ok and not ankles_ok and not is_floor_or_seated:
            framing_warning = "Step back 2-3 meters so feet and legs are visible"
        elif not ankles_ok and not is_floor_or_seated:
            framing_warning = "Step back so full legs and feet are in view"

    # ── 2. Simple & Modern Pose Header (Real-Time Auto Mode) ──────────────────
    is_assessed = bool(decision is not None and getattr(decision, "assessment_allowed", False) and getattr(decision, "final_pose", None))
    conf_str = f"{prediction.confidence * 100:.0f}%"
    form_str = ""
    if hasattr(prediction, "correctness_label") and getattr(prediction, "correctness_label"):
        corr_lbl = getattr(prediction, "correctness_label").upper()
        form_str = f"  |  Form: {corr_lbl}"

    if prediction.status == "no_person":
        header_str = "STATUS: NO PERSON DETECTED"
        status_colour = CLR_CORRECT
    elif is_assessed:
        final_pose = decision.final_pose
        header_str = f"DETECTED: {final_pose} [{conf_str}]{form_str}"
        if alignment_score is not None and alignment_score > 0:
            header_str += f"  |  Score: {alignment_score:.0f}%"
        status_colour = CLR_GOOD
    elif prediction.status == "detected" and prediction.pose_name:
        header_str = f"DETECTING: {prediction.pose_name} [{conf_str}] (Hold steady...)"
        status_colour = CLR_WARN
    else:
        header_str = "STATUS: UNIDENTIFIED (Hold any asana)"
        status_colour = CLR_WARN

    # ── 3. Collect Clean Coaching Cues ────────────────────────────────────────
    lines: list[tuple[str, tuple[int, int, int]]] = []
    if framing_warning:
        lines.append((framing_warning, CLR_WARN))

    if is_assessed:
        for item in feedback_items[:max_lines]:
            col = {
                "correction": CLR_CORRECT,
                "warning":    CLR_WARN,
                "info":       CLR_GOOD,
            }.get(item.severity, CLR_INFO)
            lines.append((item.instruction, col))
    elif prediction.status == "detected" and prediction.pose_name:
        lines.append((f"Hold {prediction.pose_name} steady to confirm.", CLR_WARN))
        lines.append(("Real-time pose guidance will activate.", CLR_INFO))
    elif prediction.status != "no_person":
        lines.append(("Stand or hold posture clearly in view.", CLR_INFO))
        lines.append(("Real-time pose guidance will activate.", CLR_INFO))

    # ── 4. Render Floating Aesthetic Card at Top-Left ─────────────────────────
    card_x = 16
    card_y = 16
    row_h = 24

    # Calculate max text width for compact card
    (hw, hh), _ = cv2.getTextSize(header_str, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    max_text_w = hw
    for text, _ in lines:
        (lw, _), _ = cv2.getTextSize(f"• {text}", cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        if lw > max_text_w:
            max_text_w = lw

    card_w = min(max(max_text_w + 32, 240), w - 32)
    card_h = 32 + (len(lines) * row_h) + (6 if lines else 0)

    overlay_card = frame.copy()
    cv2.rectangle(overlay_card, (card_x, card_y), (card_x + card_w, card_y + card_h), (20, 24, 30), -1)
    cv2.rectangle(overlay_card, (card_x, card_y), (card_x + card_w, card_y + card_h), (55, 70, 85), 1)
    cv2.addWeighted(overlay_card, 0.78, frame, 0.22, 0, frame)

    # Header text
    cv2.putText(frame, header_str, (card_x + 14, card_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, status_colour, 1, cv2.LINE_AA)

    # Feedback lines
    curr_y = card_y + 22 + row_h
    for text, col in lines:
        cv2.putText(frame, f"• {text}", (card_x + 14, curr_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)
        curr_y += row_h

    return frame


def draw_calibration_overlay(
    frame: np.ndarray,
    is_active: bool,
    progress_pct: int,
    status_message: str,
) -> np.ndarray:
    """Draws an interactive calibration progress banner across the screen center."""
    if not is_active:
        return frame

    h, w = frame.shape[:2]
    box_w, box_h = min(560, w - 40), 100
    bx = (w - box_w) // 2
    by = (h - box_h) // 2

    # Translucent card background
    overlay = frame.copy()
    cv2.rectangle(overlay, (bx, by), (bx + box_w, by + box_h), (15, 18, 22), -1)
    cv2.rectangle(overlay, (bx, by), (bx + box_w, by + box_h), (60, 200, 255), 2)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Title & status
    cv2.putText(frame, "5-SECOND BODY BASELINE CALIBRATION (Tadasana)", (bx + 20, by + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, status_message, (bx + 20, by + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 220, 255), 1, cv2.LINE_AA)

    # Progress bar
    bar_x = bx + 20
    bar_y = by + 68
    bar_w = box_w - 40
    bar_h = 14
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 45, 55), -1)
    fill_w = int(bar_w * (max(0, min(100, progress_pct)) / 100.0))
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (80, 210, 120), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 90, 105), 1)

    return frame


def draw_key_hints(frame: np.ndarray) -> np.ndarray:
    """Hidden for a clean, minimalist website-ready HUD."""
    return frame


TIER_COLOURS = {"high": CLR_GOOD, "medium": CLR_WARN, "low": CLR_CORRECT}


def draw_decision_status(
    frame: np.ndarray,
    decision: Optional[object],
    landmark_tier: Optional[str] = None,
    correction_state: str = "",
    jev_status: Optional[str] = None,
    y_from_bottom: int = 30,
) -> np.ndarray:
    """Hidden for a clean, minimalist website-ready HUD."""
    return frame


def draw_diagnostics(
    frame: np.ndarray,
    lines: list[str],
    joint_tiers: Optional[dict[str, str]] = None,
    x0: Optional[int] = None,
    y0: int = 150,
) -> np.ndarray:
    """Right-aligned diagnostic panel (toggle with D)."""
    h, w = frame.shape[:2]
    rows = list(lines)
    tiers = joint_tiers or {}
    width = 360
    x0 = w - width - 8 if x0 is None else x0
    height = 18 * (len(rows) + len(tiers) + 1) + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + width, min(h - 40, y0 + height)), CLR_PANEL, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    y = y0 + 18
    cv2.putText(frame, "DIAGNOSTICS", (x0 + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_WHITE, 1, cv2.LINE_AA)
    for line in rows:
        y += 18
        cv2.putText(frame, line[:58], (x0 + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, CLR_INFO, 1, cv2.LINE_AA)
    for name, tier in tiers.items():
        y += 18
        cv2.putText(frame, f"{name.replace('_angle', '')}: {tier}", (x0 + 8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, TIER_COLOURS.get(tier, CLR_INFO), 1, cv2.LINE_AA)
    return frame
