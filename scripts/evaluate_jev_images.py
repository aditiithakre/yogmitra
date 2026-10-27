"""
scripts/evaluate_jev_images.py
──────────────────────────────
Reproducible comparison of the LOCAL pipeline, JEV, and the FUSED decision on
labelled still images (e.g. the original Zenodo / Kaggle photos).

Input (either):
  --images-dir DIR     with one sub-folder per class label (folder name == class name)
  --csv FILE           with columns image_path,label

For every image:
  MediaPipe (IMAGE mode) → 39 features → local classifier (+ configured filters)
  → local-only decision; then, if JEVMODEL_API_KEY is set and --jev is given,
  one Jev request (rate-limited) → fused decision via src.models.jev_fusion.decide

Reported: coverage and accuracy of issued labels for local-only, Jev and fused
decisions; local/Jev agreement; Jev status counts, request count, latency
mean/P95.  Jev latency is network + service time, *not* local model latency.

Without an API key the script still runs the local part and states that Jev
was not evaluated.  No image is written anywhere.

Usage:
    python scripts/evaluate_jev_images.py --images-dir path/to/test_images --profile zenodo
    JEVMODEL_API_KEY=... python scripts/evaluate_jev_images.py --images-dir ... --jev --max-images 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.app.webcam_app import load_config, load_fusion_config, project_path  # noqa: E402
from src.features import extract_features, features_to_series  # noqa: E402
from src.models.inference import DualTaskInference, FilterPolicy, PoseInference  # noqa: E402
from src.models.jev_client import JevClient, JevConfig  # noqa: E402
from src.models.jev_fusion import decide  # noqa: E402
from src.pose import PoseLandmarkerWrapper  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png"}


def list_images(args) -> list[tuple[Path, str]]:
    if args.csv:
        df = pd.read_csv(args.csv)
        return [(Path(p), str(l)) for p, l in zip(df["image_path"], df["label"])]
    out = []
    for d in sorted(p for p in args.images_dir.iterdir() if p.is_dir()):
        out += [(f, d.name) for f in sorted(d.iterdir()) if f.suffix.lower() in IMG_EXT]
    return out


def acc(rows, key):
    issued = [r for r in rows if r[key] is not None]
    return {
        "coverage": round(len(issued) / max(1, len(rows)), 4),
        "accuracy_of_issued": round(float(np.mean([r[key] == r["label"] for r in issued])), 4) if issued else None,
        "issued": len(issued),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", type=Path)
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--profile", choices=["zenodo", "kaggle"], default=None)
    ap.add_argument("--classifier", choices=["rf", "mlp"], default=None)
    ap.add_argument("--jev", action="store_true", help="also query Jev (needs JEVMODEL_API_KEY)")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--min-interval", type=float, default=0.6, help="seconds between Jev requests (≤120/min)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "jev_image_evaluation.json")
    args = ap.parse_args()
    if not (args.images_dir or args.csv):
        ap.error("give --images-dir or --csv")

    cfg = load_config("config.yaml")
    profile = args.profile or cfg["model"]["profile"]
    clf = args.classifier or cfg["model"]["active_classifier"]
    fcfg = dict(cfg["inference"].get("filters") or {})
    fcfg.update((cfg["inference"].get("filters_by_classifier") or {}).get(clf) or {})
    policy = FilterPolicy.from_config(fcfg)
    fusion = load_fusion_config(cfg, profile, clf)
    if profile == "zenodo":
        from src.models.model_loader import load_zenodo_artifacts
        art = load_zenodo_artifacts(project_path(cfg["model"]["dir"]), active_classifier=clf)
        inf = DualTaskInference(art, confidence_threshold=fusion.local_threshold, filter_policy=policy)
    else:
        from src.models.model_loader import load_artifacts
        art = load_artifacts(project_path(cfg["model"].get("kaggle_dir", "saved_yoga_model")))
        inf = PoseInference(art, active_classifier=clf, confidence_threshold=fusion.local_threshold, filter_policy=policy)

    client = JevClient(JevConfig.from_config(cfg.get("jev")))
    use_jev = args.jev and client.has_api_key
    images = list_images(args)
    if args.max_images:
        images = images[: args.max_images]
    print(f"{len(images)} images | profile={profile} clf={clf} threshold={fusion.local_threshold} | "
          f"Jev: {'ON' if use_jev else 'not evaluated' + (' (no API key)' if args.jev else '')}")

    rows, jev_status, jev_lat = [], Counter(), []
    last_req = 0.0
    with PoseLandmarkerWrapper(project_path(cfg["model"]["mediapipe_task"]), running_mode="image") as det:
        for path, label in images:
            img = cv2.imread(str(path))
            if img is None:
                continue
            lm = det.detect(img)
            row = {"image": str(path), "label": label, "person": lm is not None,
                   "local_top": None, "local_conf": 0.0, "local_decision": None, "jev": None, "fused": None}
            if lm is not None:
                h, w = img.shape[:2]
                feats = features_to_series(extract_features(lm, [0] * 33, w, h), art.feature_cols)
                inf.reset_history()
                pred = inf.predict(feats, landmarks=lm)
                vis = min(p["visibility"] for p in (lm[i] for i in (11, 12, 23, 24)))
                tier = "high" if vis >= 0.5 else ("medium" if vis >= 0.3 else "low")
                d_local = decide(pred, tier, "auto", art.class_names, time.monotonic(), fusion)
                row.update(local_top=pred.pose_name, local_conf=round(pred.confidence, 4),
                           local_decision=d_local.final_pose if d_local.assessment_allowed else None)
                jres = None
                if use_jev:
                    wait = args.min_interval - (time.monotonic() - last_req)
                    if wait > 0:
                        time.sleep(wait)
                    last_req = time.monotonic()
                    jres = client.classify(img, art.class_names, frame_timestamp=time.monotonic(), epoch=0)
                    jev_status[jres.status] += 1
                    if jres.ok:
                        jev_lat.append(jres.latency_ms)
                    row["jev"] = jres.label if jres.ok else None
                    row["jev_conf"] = jres.confidence
                d_fused = decide(pred, tier, "auto", art.class_names, time.monotonic(), fusion,
                                 jev=jres, current_epoch=0)
                row["fused"] = d_fused.final_pose if d_fused.assessment_allowed else None
                row["agreement"] = d_fused.agreement
            rows.append(row)
            print(f"  {path.name[:40]:40s} true={label:22s} local={row['local_decision']} jev={row['jev']} fused={row['fused']}")

    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "profile": profile, "classifier": clf, "local_threshold": fusion.local_threshold,
        "images": len(rows), "person_detected_rate": round(float(np.mean([r["person"] for r in rows])), 4) if rows else None,
        "local_only": acc(rows, "local_decision"),
        "local_top1_accuracy_all": round(float(np.mean([r["local_top"] == r["label"] for r in rows])), 4) if rows else None,
        "jev_evaluated": use_jev,
        "note": "accuracy_of_issued = accuracy among images where a label was issued (others reported as uncertain)",
    }
    if use_jev:
        both = [r for r in rows if r["jev"] and r["local_top"]]
        report.update({
            "jev": acc(rows, "jev"),
            "fused": acc(rows, "fused"),
            "local_jev_top1_agreement": round(float(np.mean([r["jev"] == r["local_top"] for r in both])), 4) if both else None,
            "jev_requests": int(sum(jev_status.values())),
            "jev_status_counts": dict(jev_status),
            "jev_latency_mean_ms": round(float(np.mean(jev_lat)), 1) if jev_lat else None,
            "jev_latency_p95_ms": round(float(np.percentile(jev_lat, 95)), 1) if jev_lat else None,
        })
    report["rows"] = rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
