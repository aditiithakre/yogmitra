"""
tests/test_landmark_tracking.py
────────────────────────────────
MediaPipe video-mode timestamps and per-landmark history (Fix 1 & Fix 2).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from src.features import extract_features
from src.pose.missing_landmarks import (
    IMPUTED_VISIBILITY,
    LandmarkHistory,
    STATUS_IMPUTED,
    STATUS_OBSERVED,
    STATUS_UNAVAILABLE,
)

ROOT = Path(__file__).resolve().parent.parent
TASK = ROOT / "pose_landmarker_heavy (1).task"


def _standing(vis: float = 0.95) -> list[dict]:
    lm = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": vis} for _ in range(33)]
    coords = {
        11: (0.42, 0.25), 12: (0.58, 0.25), 13: (0.38, 0.45), 14: (0.62, 0.45),
        15: (0.35, 0.65), 16: (0.65, 0.65), 23: (0.44, 0.55), 24: (0.56, 0.55),
        25: (0.44, 0.75), 26: (0.56, 0.75), 27: (0.44, 0.92), 28: (0.56, 0.92),
        29: (0.43, 0.94), 30: (0.57, 0.94), 31: (0.46, 0.96), 32: (0.54, 0.96),
    }
    for i, (x, y) in coords.items():
        lm[i] = {"x": x, "y": y, "z": 0.0, "visibility": vis}
    return lm


def _occlude(lm: list[dict], ids, vis: float = 0.05) -> list[dict]:
    out = [dict(p) for p in lm]
    for i in ids:
        out[i] = {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": vis}
    return out


# ── Fix 2: landmark history ───────────────────────────────────────────────────

def test_one_frame_occlusion_is_imputed_and_flagged():
    h = LandmarkHistory(max_age_ms=300)
    h.update(_standing(), 0)
    fr = h.update(_occlude(_standing(), [25]), 33)
    assert fr.status[25] == STATUS_IMPUTED
    assert fr.imputed_flags[25] == 1
    assert fr.effective[25]["imputed"] is True
    assert abs(fr.effective[25]["y"] - 0.75) < 1e-9  # last reliable position, not the garbage (0,0)
    assert fr.effective[25]["visibility"] == IMPUTED_VISIBILITY
    # classifier view keeps the joint hidden
    assert fr.observed_only[25]["visibility"] < 0.30


def test_persistent_occlusion_survives_beyond_one_frame_then_expires():
    h = LandmarkHistory(max_age_ms=300)
    h.update(_standing(), 0)
    occl = _occlude(_standing(), [25])
    # The legacy fallback lost the joint after one frame; history keeps it for max_age.
    for t in (33, 66, 100, 200, 300):
        fr = h.update(occl, t)
        assert fr.status[25] == STATUS_IMPUTED, t
    fr = h.update(occl, 334)
    assert fr.status[25] == STATUS_UNAVAILABLE
    assert fr.imputed_flags[25] == 0
    # expired joint must not yield an angle
    feats = extract_features(fr.effective, fr.imputed_flags)
    assert math.isnan(feats["left_knee_angle"])
    assert math.isnan(feats["left_hip_angle"])


def test_expired_history_is_not_revived():
    h = LandmarkHistory(max_age_ms=100)
    h.update(_standing(), 0)
    occl = _occlude(_standing(), [27])
    assert h.update(occl, 150).status[27] == STATUS_UNAVAILABLE
    assert h.update(occl, 160).status[27] == STATUS_UNAVAILABLE


def test_recovery_when_visibility_returns():
    h = LandmarkHistory(max_age_ms=300)
    h.update(_standing(), 0)
    h.update(_occlude(_standing(), [26]), 33)
    moved = _standing()
    moved[26] = {"x": 0.60, "y": 0.70, "z": 0.0, "visibility": 0.9}
    fr = h.update(moved, 66)
    assert fr.status[26] == STATUS_OBSERVED
    assert fr.imputed_flags[26] == 0
    assert fr.effective[26]["y"] == 0.70
    assert "imputed" not in fr.effective[26]


def test_imputed_never_indistinguishable_from_observed():
    h = LandmarkHistory()
    h.update(_standing(), 0)
    fr = h.update(_occlude(_standing(), [13, 15]), 33)
    for i in (13, 15):
        assert fr.effective[i].get("imputed") is True and fr.imputed_flags[i] == 1
    for i in (11, 12, 23):
        assert not fr.effective[i].get("imputed") and fr.imputed_flags[i] == 0


def test_reset_on_person_disappearing():
    h = LandmarkHistory()
    h.update(_standing(), 0)
    assert h.update(None, 33) is None
    fr = h.update(_occlude(_standing(), [25]), 66)
    assert fr.status[25] == STATUS_UNAVAILABLE


def test_reset_on_scene_change():
    h = LandmarkHistory(scene_change_norm=0.25)
    h.update(_standing(), 0)
    jumped = [dict(p, x=p["x"] - 0.4) for p in _standing()]
    jumped = _occlude(jumped, [25])
    fr = h.update(jumped, 33)
    assert fr.history_reset is True
    assert fr.status[25] == STATUS_UNAVAILABLE  # old person's knee must not be pasted in


def test_explicit_session_reset():
    h = LandmarkHistory()
    h.update(_standing(), 0)
    h.reset()
    assert h.update(_occlude(_standing(), [25]), 33).status[25] == STATUS_UNAVAILABLE


def test_visibility_decay_ghost_is_marked_estimated():
    """MediaPipe VIDEO mode decays visibility ~0.07/frame after occlusion (measured)."""
    h = LandmarkHistory(decay_drop=0.15, decay_window=3)
    for t, v in enumerate([0.95, 0.95, 0.88, 0.80, 0.72, 0.65]):
        lm = _standing()
        lm[27] = dict(lm[27], visibility=v)
        fr = h.update(lm, t * 33)
    assert 27 in fr.suspect_ids
    assert fr.status[27] == STATUS_IMPUTED and fr.imputed_flags[27] == 1


def test_stable_visibility_not_marked_suspect():
    h = LandmarkHistory()
    for t in range(6):
        fr = h.update(_standing(0.6 + 0.01 * (t % 2)), t * 33)
    assert fr.suspect_ids == []


# ── Fix 1: MediaPipe video-mode timestamps ────────────────────────────────────

class _FakeDetector:
    def __init__(self):
        self.timestamps = []
        self.closed = False

    def detect_for_video(self, _img, ts):
        self.timestamps.append(ts)

        class R:
            pose_landmarks = []
        return R()

    def detect(self, _img):
        class R:
            pose_landmarks = []
        return R()

    def close(self):
        self.closed = True


@pytest.fixture
def fake_wrapper(monkeypatch):
    from src import pose as pose_mod

    monkeypatch.setattr(pose_mod.PoseLandmarkerWrapper, "_create_detector", lambda self: _FakeDetector())
    return pose_mod.PoseLandmarkerWrapper("dummy.task", running_mode="video")


def test_video_timestamps_strictly_increasing(fake_wrapper):
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    for ts in (0, 33, 33, 20, float("nan"), -5, None, 100):
        assert fake_wrapper.detect(frame, timestamp_ms=ts) is None  # no person → None
    seen = fake_wrapper._detector.timestamps
    assert all(b > a for a, b in zip(seen, seen[1:])), seen
    assert seen[0] == 0 and seen[1] == 33
    assert fake_wrapper.timestamp_corrections >= 4


def test_wrapper_close_is_idempotent_and_blocks_use(fake_wrapper):
    det = fake_wrapper._detector
    fake_wrapper.close()
    fake_wrapper.close()
    assert det.closed and fake_wrapper.closed
    with pytest.raises(RuntimeError):
        fake_wrapper.detect(np.zeros((8, 8, 3), dtype=np.uint8))


def test_invalid_running_mode_rejected():
    from src.pose import PoseLandmarkerWrapper
    with pytest.raises(ValueError):
        PoseLandmarkerWrapper("x.task", running_mode="stream")


@pytest.mark.skipif(not TASK.exists(), reason="MediaPipe task file not present")
def test_real_mediapipe_video_and_image_modes_run_on_blank_frames():
    pytest.importorskip("mediapipe")
    from src.pose import PoseLandmarkerWrapper
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    try:
        for mode in ("video", "image"):
            with PoseLandmarkerWrapper(TASK, running_mode=mode) as det:
                for k in range(3):
                    assert det.detect(frame, timestamp_ms=k * 33) is None
            assert det.closed
    except OSError as exc:  # missing GPU/EGL system libraries on headless CI
        pytest.skip(f"MediaPipe runtime unavailable: {exc}")
