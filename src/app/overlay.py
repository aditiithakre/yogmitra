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
    voice_enabled: bool,
    inference_interval: float,
    last_inference_age: float,          # seconds since last inference
    show_angles: bool = False,
    font_scale: float = 0.65,
    max_lines: int = 4,
    target_pose: Optional[str] = None,
    landmarks: Optional[list[dict]] = None,
    alignment_score: Optional[float] = None,
    decision: Optional[object] = None,
) -> np.ndarray:
    """Draw the full HUD panel at the top of the frame.

    *decision* (PoseDecision) makes the header agree with the correction engine:
    in auto-detect mode an unconfirmed label is shown as UNCERTAIN, not DETECTED.
    """
    h, w = frame.shape[:2]
    lh = 26   # line height in pixels
    x0 = 12

    # ── Check Framing / Full Body Visibility ──────────────────────────────────
    framing_warning: Optional[str] = None
    is_floor_or_seated = prediction.pose_name in (
        "Vajrasana", "Parvatasana", "Bhujangasana", "Marjariasana",
        "Sarvangasana", "Viparita Karani", "Anantasana"
    )
    if landmarks and len(landmarks) >= 29:
        knees_ok = (landmarks[25].get("visibility", 0) >= 0.25) or (landmarks[26].get("visibility", 0) >= 0.25)
        ankles_ok = (landmarks[27].get("visibility", 0) >= 0.25) or (landmarks[28].get("visibility", 0) >= 0.25)
        if not knees_ok and not ankles_ok and not is_floor_or_seated:
            framing_warning = "CAMERA WARNING: Knees & feet cut off! Step 2-3 meters back"
        elif not ankles_ok and not is_floor_or_seated:
            framing_warning = "CAMERA WARNING: Feet not visible! Step back so full legs are seen"

    # ── Dynamic Panel Height Calculation ──────────────────────────────────────
    has_seq = bool(
        hasattr(prediction, "sequence_state")
        and prediction.sequence_state
        and prediction.sequence_state.base_posture != "UNKNOWN"
    )
    extra_rows = 1  # Header
    if has_seq:
        extra_rows += 1
    if prediction.top3:
        extra_rows += 1
    if framing_warning:
        extra_rows += 1
    total_rows = extra_rows + min(len(feedback_items), max_lines)

    panel_h = max(115, total_rows * lh + 22)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), CLR_PANEL, -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    # ── Row 0: target / pose name + confidence + interval ─────────────────────
    interval_str = f"  |  Interval: {inference_interval:.1f}s"
    voice_str = "  |  Voice: ON" if voice_enabled else "  |  Voice: OFF"

    # Alignment score badge
    score_str = ""
    if alignment_score is not None and alignment_score > 0:
        score_str = f"  |  Score: {alignment_score:.0f}%"

    if target_pose:
        if prediction.pose_name == target_pose and prediction.status == "detected":
            header_str = f"PRACTICE: {target_pose} [Match {prediction.confidence*100:.0f}%]"
            status_colour = CLR_GOOD
        else:
            header_str = f"PRACTICE: {target_pose} (Aligning...)"
            status_colour = CLR_WARN
    else:
        pose_str = prediction.pose_name or "Detecting..."
        conf_str = f"{prediction.confidence * 100:.0f}%"
        form_str = ""
        if hasattr(prediction, "correctness_label") and getattr(prediction, "correctness_label"):
            corr_lbl = getattr(prediction, "correctness_label").upper()
            form_str = f"  |  Form: {corr_lbl}"
        header_str = f"DETECTED: {pose_str} [{conf_str}]{form_str}"
        status_colour = {
            "detected":      CLR_GOOD,
            "low_confidence":CLR_WARN,
            "no_person":     CLR_CORRECT,
        }.get(prediction.status, CLR_INFO)
        if decision is not None and prediction.status != "no_person":
            if not getattr(decision, "assessment_allowed", True):
                header_str = f"POSE UNCERTAIN (best guess {pose_str} {conf_str})"
                status_colour = CLR_WARN
            elif getattr(decision, "final_pose", None) and decision.final_pose != prediction.pose_name:
                header_str = f"DETECTED: {decision.final_pose} [via {decision.source}]{form_str}"

    cv2.putText(frame, f"{header_str}{score_str}{interval_str}{voice_str}",
                (x0, lh), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, status_colour, 1, cv2.LINE_AA)

    curr_row = 2

    # ── Row 1: Sequence Progression & Active Pose Candidates ──────────────────
    if has_seq:
        seq = prediction.sequence_state
        posture_lbl = seq.base_posture.replace("_", "/")
        fd = getattr(prediction, "filter_decision", None)
        if fd is not None and not fd.applied_sequence:
            # Show what the candidate filter actually did, not the tracker's unused pruning list
            if fd.skipped_reason:
                filt = f"filter off ({fd.skipped_reason})"
            elif fd.suppressed:
                filt = f"down-weighted: {', '.join(fd.suppressed[:3])}" + (f" +{len(fd.suppressed) - 3}" if len(fd.suppressed) > 3 else "")
            else:
                filt = "no classes down-weighted"
            seq_str = f"Base: {posture_lbl}  |  {filt}"
        else:
            cand_str = ", ".join(seq.active_candidates[:3])
            if len(seq.active_candidates) > 3:
                cand_str += f" +{len(seq.active_candidates) - 3}"
            seq_str = f"Step {seq.step}/{seq.step_total}: {seq.step_name}  |  Base: {posture_lbl}  |  Eligible: {cand_str}"
        cv2.putText(frame, seq_str,
                    (x0, curr_row * lh - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 0.70, (100, 220, 255), 1, cv2.LINE_AA)
        curr_row += 1

    # ── Row 2: Top candidate predictions ──────────────────────────────────────
    if prediction.top3:
        top_items = [f"{name} ({prob*100:.0f}%)" for name, prob in prediction.top3[:3]]
        top_str = "Match: " + "  |  ".join(top_items)
        cv2.putText(frame, top_str,
                    (x0, curr_row * lh - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 0.70, (180, 220, 180), 1, cv2.LINE_AA)
        curr_row += 1

    # ── Framing Warning Banner if user is too close ───────────────────────────
    if framing_warning:
        cv2.putText(frame, framing_warning,
                    (x0, curr_row * lh - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 0.80, (0, 140, 255), 2, cv2.LINE_AA)
        curr_row += 1

    # ── Rows: correction messages ─────────────────────────────────────────────
    for i, item in enumerate(feedback_items[:max_lines]):
        colour = {
            "correction": CLR_CORRECT,
            "warning":    CLR_WARN,
            "info":       CLR_GOOD,
        }.get(item.severity, CLR_INFO)
        cv2.putText(frame, item.instruction,
                    (x0, (curr_row + i) * lh - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 0.80, colour, 1, cv2.LINE_AA)

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
    """Tiny key-binding reminder at the bottom of the frame."""
    h, w = frame.shape[:2]
    hints = (
        "Q=Quit  T/N=Pose  M=Guided/Free  C=Calibrate  "
        "V=Voice  A=Angles  +/-=Interval  S=Shot  J=Jev check  D=Diagnostics"
    )
    cv2.putText(frame, hints, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, CLR_INFO, 1, cv2.LINE_AA)
    return frame


TIER_COLOURS = {"high": CLR_GOOD, "medium": CLR_WARN, "low": CLR_CORRECT}


def draw_decision_status(
    frame: np.ndarray,
    decision: Optional[object],
    landmark_tier: Optional[str],
    correction_state: str,
    jev_status: Optional[str] = None,
    y_from_bottom: int = 30,
) -> np.ndarray:
    """
    One-line status strip above the key hints:
      Local: <pose> <conf> | Jev: <pose> <conf> (<agreement>) | Landmarks: <tier> | Correction: <state>
    """
    if decision is None:
        return frame
    h, w = frame.shape[:2]
    local = f"Local: {decision.local_pose or '-'} {decision.local_confidence*100:.0f}%"
    if decision.jev_pose or (jev_status and jev_status not in ("disabled",)):
        jc = f" {decision.jev_confidence*100:.0f}%" if decision.jev_confidence is not None else ""
        jev = f"Jev: {decision.jev_pose or '-'}{jc} ({decision.agreement})"
        if jev_status and jev_status not in ("ok", "idle"):
            jev += f" [{jev_status}]"
    else:
        jev = "Jev: off"
    lt = (landmark_tier or "low").upper()
    text = f"{local}  |  {jev}  |  Landmarks: {lt}  |  Correction: {correction_state}"
    y = h - y_from_bottom
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y - 18), (w, y + 8), CLR_PANEL, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    colour = TIER_COLOURS.get(decision.tier if decision.assessment_allowed else "low", CLR_INFO)
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    if not decision.assessment_allowed or decision.tier != "high":
        cv2.putText(frame, f"Why: {decision.reason}", (10, y - 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, CLR_WARN, 1, cv2.LINE_AA)
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
