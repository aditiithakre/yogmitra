"""
src/features/__init__.py  /  src/features/geometry.py
───────────────────────────────────────────────────────
Reproduces ALL 39 features engineered during model training:
  • 10 joint angles
  •  6 normalised distances
  •  3 alignment angles
  •  3 symmetry diffs
  •  1 imputed-fraction + 16 per-joint imputed flags
        = 39 features (exactly matching feature_cols in preprocessing.joblib)

⚠  Do NOT change this file independently of the notebook's feature extraction.
   The training feature_cols list is the ground truth for ordering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

# ── Named landmark IDs (MediaPipe's 33-point body model) ─────────────────────
LANDMARK_IDS: dict[str, int] = {
    "left_shoulder": 11,  "right_shoulder": 12,
    "left_elbow": 13,     "right_elbow": 14,
    "left_wrist": 15,     "right_wrist": 16,
    "left_hip": 23,       "right_hip": 24,
    "left_knee": 25,      "right_knee": 26,
    "left_ankle": 27,     "right_ankle": 28,
    "left_heel": 29,      "right_heel": 30,
    "left_foot_index": 31,"right_foot_index": 32,
}

ANGLE_TRIPLETS: dict[str, tuple[int, int, int]] = {
    "left_elbow_angle":    (11, 13, 15),
    "right_elbow_angle":   (12, 14, 16),
    "left_shoulder_angle": (13, 11, 23),
    "right_shoulder_angle":(14, 12, 24),
    "left_hip_angle":      (11, 23, 25),
    "right_hip_angle":     (12, 24, 26),
    "left_knee_angle":     (23, 25, 27),
    "right_knee_angle":    (24, 26, 28),
    "left_ankle_angle":    (25, 27, 31),
    "right_ankle_angle":   (26, 28, 32),
}

DISTANCE_PAIRS: dict[str, tuple[int, int]] = {
    "left_wrist_to_shoulder": (15, 11),
    "right_wrist_to_shoulder":(16, 12),
    "wrist_to_wrist":         (15, 16),
    "ankle_to_ankle":         (27, 28),
    "shoulder_width":         (11, 12),
    "hip_width":              (23, 24),
}

VISIBILITY_THRESHOLD = 0.30


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex B in degrees (ba · bc)."""
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-8:
        return np.nan
    cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _calculate_distance(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return np.nan
    return float(np.linalg.norm(a - b))


def _line_angle(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """Angle of segment A→B relative to horizontal, in degrees."""
    if a is None or b is None:
        return np.nan
    dx, dy = b[0] - a[0], b[1] - a[1]
    if np.hypot(dx, dy) < 1e-8:
        return np.nan
    return float(np.degrees(np.arctan2(dy, dx)))


def _get_point(
    landmarks: list[dict],
    idx: int,
    width: int,
    height: int,
    threshold: float = VISIBILITY_THRESHOLD,
) -> Optional[np.ndarray]:
    """Return pixel coords only when the landmark is sufficiently visible."""
    if idx >= len(landmarks):
        return None
    lm = landmarks[idx]
    if float(lm.get("visibility", 0.0)) < threshold:
        return None
    x, y = float(lm["x"]), float(lm["y"])
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return np.array([x * width, y * height], dtype=float)


# ── Feature extraction (public API) ──────────────────────────────────────────

def extract_features(
    landmarks: list[dict],
    imputed_flags: list[int],
    width: int = 1280,
    height: int = 720,
) -> dict[str, float]:
    """
    Given 33 effective landmark dicts and their imputed flags, return the
    full 39-feature dict in training order.

    landmarks  – list of {x, y, z, visibility} in normalised [0,1] coords
    imputed_flags – list of 33 ints (1 = estimated, 0 = directly observed)
    width, height – frame dimensions (needed for pixel-space distances)
    """
    # ── Joint angles ──────────────────────────────────────────────────────────
    angle_features: dict[str, float] = {}
    for name, (a_id, b_id, c_id) in ANGLE_TRIPLETS.items():
        pts = [_get_point(landmarks, i, width, height) for i in (a_id, b_id, c_id)]
        if any(p is None for p in pts):
            angle_features[name] = np.nan
        else:
            angle_features[name] = _calculate_angle(*pts)

    # Ankle angle sanitization against camera bottom-edge truncation:
    # When a person is standing upright (shoulders above hips, hips above knees),
    # their feet rest flat on the floor (true anatomical angle ~75°-105°).
    # If the camera frame truncates the feet (ankles or toes near bottom y >= 0.86),
    # MediaPipe defaults the toes straight down, fabricating an artificial ~170°-180° angle
    # that confounds standing poses with inversion poses (Viparita Karani / Sarvangasana).
    if len(landmarks) >= 33:
        sy = (float(landmarks[11].get("y", 0.5)) + float(landmarks[12].get("y", 0.5))) / 2.0
        hy = (float(landmarks[23].get("y", 0.5)) + float(landmarks[24].get("y", 0.5))) / 2.0
        ky = (float(landmarks[25].get("y", 0.5)) + float(landmarks[26].get("y", 0.5))) / 2.0
        is_standing_upright = (hy - sy > 0.08) and (ky - hy > 0.06)
        if is_standing_upright:
            if float(landmarks[27].get("y", 0.0)) > 0.86 or float(landmarks[31].get("y", 0.0)) > 0.92:
                if angle_features.get("left_ankle_angle", 0) > 135.0:
                    angle_features["left_ankle_angle"] = 90.0
            if float(landmarks[28].get("y", 0.0)) > 0.86 or float(landmarks[32].get("y", 0.0)) > 0.92:
                if angle_features.get("right_ankle_angle", 0) > 135.0:
                    angle_features["right_ankle_angle"] = 90.0

    # ── Normalised distances ──────────────────────────────────────────────────
    named_pts: dict[str, Optional[np.ndarray]] = {
        name: _get_point(landmarks, idx, width, height)
        for name, idx in LANDMARK_IDS.items()
    }

    ls, rs = named_pts["left_shoulder"], named_pts["right_shoulder"]
    lh, rh = named_pts["left_hip"],      named_pts["right_hip"]

    shoulder_mid = (ls + rs) / 2 if ls is not None and rs is not None else None
    hip_mid      = (lh + rh) / 2 if lh is not None and rh is not None else None

    torso_length = _calculate_distance(shoulder_mid, hip_mid)
    if not (np.isfinite(torso_length) and torso_length > 1e-4):
        torso_length = _calculate_distance(ls, rs)
    if not (np.isfinite(torso_length) and torso_length > 1e-4):
        torso_length = 1.0  # safe floor

    distance_features: dict[str, float] = {}
    for key, (a_id, b_id) in DISTANCE_PAIRS.items():
        a = _get_point(landmarks, a_id, width, height)
        b = _get_point(landmarks, b_id, width, height)
        d = _calculate_distance(a, b)
        distance_features[key + "_norm"] = (
            d / torso_length
            if np.isfinite(d) and np.isfinite(torso_length) and torso_length > 1e-4
            else np.nan
        )

    # ── Alignment features ────────────────────────────────────────────────────
    shoulder_angle = _line_angle(ls, rs)
    hip_angle      = _line_angle(lh, rh)

    if shoulder_mid is not None and hip_mid is not None:
        ta = _line_angle(shoulder_mid, hip_mid)
        torso_lean = 90.0 - ta if np.isfinite(ta) else np.nan
    else:
        torso_lean = np.nan

    alignment_features = {
        "shoulder_line_angle_deg":       shoulder_angle,
        "hip_line_angle_deg":            hip_angle,
        "torso_lean_from_vertical_deg":  torso_lean,
    }

    # ── Symmetry features ─────────────────────────────────────────────────────
    def _asym(a_key: str, b_key: str) -> float:
        a, b = angle_features.get(a_key, np.nan), angle_features.get(b_key, np.nan)
        return abs(a - b) if np.isfinite(a) and np.isfinite(b) else np.nan

    symmetry_features = {
        "elbow_angle_asymmetry_deg": _asym("left_elbow_angle", "right_elbow_angle"),
        "knee_angle_asymmetry_deg":  _asym("left_knee_angle",  "right_knee_angle"),
        "hip_angle_asymmetry_deg":   _asym("left_hip_angle",   "right_hip_angle"),
    }

    # ── Imputation features ───────────────────────────────────────────────────
    frac = float(np.mean(imputed_flags)) if imputed_flags else 0.0
    imputation_features: dict[str, float] = {"imputed_landmark_fraction": frac}
    for name, idx in LANDMARK_IDS.items():
        imputation_features[f"{name}_imputed"] = (
            float(imputed_flags[idx]) if idx < len(imputed_flags) else 0.0
        )

    return {
        **angle_features,
        **distance_features,
        **alignment_features,
        **symmetry_features,
        **imputation_features,
    }


def features_to_series(feature_dict: dict, feature_cols: list[str]) -> pd.Series:
    """Convert a feature dict to a pandas Series with the exact training column order."""
    return pd.Series({col: feature_dict.get(col, np.nan) for col in feature_cols})
