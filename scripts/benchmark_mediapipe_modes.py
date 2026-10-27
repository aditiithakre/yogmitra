"""
scripts/benchmark_mediapipe_modes.py
────────────────────────────────────
Compares MediaPipe PoseLandmarker IMAGE mode (previous implementation:
per-frame ``detect``) with VIDEO mode (``detect_for_video`` + timestamps)
on the *same* frames.

Measured per mode:
  • detection rate                – frames where a person was returned
  • latency mean / P95 (ms)       – wall-clock time of the detector call only
  • missing-landmark rate         – fraction of the 16 body landmarks used by the
                                    39-feature schema whose visibility < threshold
  • temporal jitter               – mean |second difference| of visible landmark
                                    positions, normalised by torso length
                                    (lower = steadier; includes real motion)
  • perturbation consistency      – (synthetic still-image sequence only) mean
                                    distance between landmarks mapped back through
                                    the known shift/scale and the IMAGE-mode result
                                    on the clean frame, normalised by torso length

None of these are ground-truth landmark accuracy: no annotated keypoints exist
for these inputs.  Sources are reported explicitly in the output JSON.

Usage:
    python scripts/benchmark_mediapipe_modes.py --video path/to/clip.mp4
    python scripts/benchmark_mediapipe_modes.py --still path/to/person.jpg --still-frames 120
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pose import PoseLandmarkerWrapper  # noqa: E402
from src.features import LANDMARK_IDS  # noqa: E402

BODY_IDS = sorted(LANDMARK_IDS.values())
VIS_T = 0.30


def _torso(lm: list[dict]) -> float:
    s = np.array([(lm[11]["x"] + lm[12]["x"]) / 2, (lm[11]["y"] + lm[12]["y"]) / 2])
    h = np.array([(lm[23]["x"] + lm[24]["x"]) / 2, (lm[23]["y"] + lm[24]["y"]) / 2])
    return max(float(np.linalg.norm(s - h)), 1e-3)


def _jitter(seq: list[Optional[list[dict]]]) -> Optional[float]:
    vals = []
    for a, b, c in zip(seq, seq[1:], seq[2:]):
        if a is None or b is None or c is None:
            continue
        scale = _torso(b)
        for i in BODY_IDS:
            if min(a[i]["visibility"], b[i]["visibility"], c[i]["visibility"]) < VIS_T:
                continue
            p = [np.array([f[i]["x"], f[i]["y"]]) for f in (a, b, c)]
            vals.append(float(np.linalg.norm(p[2] - 2 * p[1] + p[0])) / scale)
    return round(float(np.mean(vals)), 5) if vals else None


def run_mode(mode: str, frames: list[np.ndarray], fps: float, task: str) -> tuple[dict, list]:
    lat, outs = [], []
    with PoseLandmarkerWrapper(task, running_mode=mode) as det:
        for k, fr in enumerate(frames):
            t0 = time.perf_counter()
            out = det.detect(fr, timestamp_ms=int(k * 1000.0 / fps))
            lat.append((time.perf_counter() - t0) * 1000.0)
            outs.append(out)
    detected = [o for o in outs if o is not None]
    missing = (
        float(np.mean([[o[i]["visibility"] < VIS_T for i in BODY_IDS] for o in detected]))
        if detected else None
    )
    lat_arr = np.array(lat[1:] if len(lat) > 1 else lat)  # drop graph warm-up frame
    return {
        "frames": len(frames),
        "detection_rate": round(len(detected) / max(1, len(frames)), 4),
        "latency_mean_ms": round(float(lat_arr.mean()), 2),
        "latency_p95_ms": round(float(np.percentile(lat_arr, 95)), 2),
        "missing_body_landmark_rate": round(missing, 4) if missing is not None else None,
        "temporal_jitter_norm": _jitter(outs),
    }, outs


def load_video(path: Path, max_frames: int) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while len(frames) < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames, float(fps)


def synth_still_sequence(img: np.ndarray, n: int, seed: int = 0) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Random-walk shift/scale + sensor noise + a 15-frame lower-body occlusion."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    frames, mats = [], []
    tx = ty = 0.0
    s = 1.0
    for k in range(n):
        tx = float(np.clip(tx + rng.normal(0, 1.5), -0.04 * w, 0.04 * w))
        ty = float(np.clip(ty + rng.normal(0, 1.5), -0.04 * h, 0.04 * h))
        s = float(np.clip(s + rng.normal(0, 0.003), 0.95, 1.05))
        M = np.array([[s, 0, tx + (1 - s) * w / 2], [0, s, ty + (1 - s) * h / 2]], dtype=np.float32)
        fr = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        fr = np.clip(fr.astype(np.int16) + rng.normal(0, 4, fr.shape).astype(np.int16), 0, 255).astype(np.uint8)
        if n // 2 <= k < n // 2 + 15:  # occlude lower 30 % (legs) for 15 frames
            fr[int(h * 0.70):, :] = 30
        frames.append(fr)
        mats.append(M)
    return frames, mats


def consistency(outs: list, mats: list[np.ndarray], ref: list[dict], w: int, h: int, skip: Iterable[int]) -> Optional[float]:
    skip = set(skip)
    errs = []
    scale = _torso(ref)
    for k, (o, M) in enumerate(zip(outs, mats)):
        if o is None or k in skip:
            continue
        Minv = cv2.invertAffineTransform(M)
        for i in BODY_IDS:
            if o[i]["visibility"] < VIS_T or ref[i]["visibility"] < VIS_T:
                continue
            px = np.array([o[i]["x"] * w, o[i]["y"] * h, 1.0])
            back = Minv @ px
            errs.append(float(np.hypot(back[0] / w - ref[i]["x"], back[1] / h - ref[i]["y"])) / scale)
    return round(float(np.mean(errs)), 5) if errs else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="pose_landmarker_heavy (1).task")
    ap.add_argument("--video", type=Path, action="append", default=[])
    ap.add_argument("--still", type=Path, action="append", default=[])
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--still-frames", type=int, default=120)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "mediapipe_mode_comparison.json")
    args = ap.parse_args()

    report: dict = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task_file": str(args.task),
        "visibility_threshold": VIS_T,
        "note": "No annotated keypoints: metrics describe stability/coverage, not landmark accuracy.",
        "sources": {},
    }

    for vp in args.video:
        frames, fps = load_video(vp, args.max_frames)
        entry = {"kind": "real_video_proxy", "fps": fps}
        for mode in ("image", "video"):
            entry[mode], _ = run_mode(mode, frames, fps, args.task)
        report["sources"][vp.name] = entry
        print(vp.name, json.dumps(entry, indent=1))

    for sp in args.still:
        img = cv2.imread(str(sp))
        h, w = img.shape[:2]
        with PoseLandmarkerWrapper(args.task, running_mode="image") as det:
            ref = det.detect(img)
        if ref is None:
            print(f"[skip] no person in {sp}")
            continue
        frames, mats = synth_still_sequence(img, args.still_frames)
        occl = range(len(frames) // 2, len(frames) // 2 + 15)
        entry = {"kind": "synthetic_perturbation_of_real_still", "fps": 30.0, "occluded_frames": [occl.start, occl.stop]}
        for mode in ("image", "video"):
            m, outs = run_mode(mode, frames, 30.0, args.task)
            m["consistency_err_norm_unoccluded"] = consistency(outs, mats, ref, w, h, occl)
            occ_outs = [outs[k] for k in occl]
            m["occluded_frames_detected"] = sum(o is not None for o in occ_outs)
            m["occluded_frames_leg_missing_rate"] = (
                round(float(np.mean([[o[i]["visibility"] < VIS_T for i in (25, 26, 27, 28)]
                                     for o in occ_outs if o is not None])), 4)
                if any(o is not None for o in occ_outs) else None
            )
            entry[mode] = m
        report["sources"][sp.name] = entry
        print(sp.name, json.dumps(entry, indent=1))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
