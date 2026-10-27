"""
scripts/calibrate_confidence.py
───────────────────────────────
Chooses, per classifier and on the VALIDATION split only:
  1. the candidate-filter policy (highest validation stream macro-F1), and
  2. the confidence threshold for issuing an auto-detected pose label
     (lowest threshold whose validation selective accuracy ≥ --target-accuracy
     with coverage ≥ --min-coverage),
then reports the chosen settings on the held-out TEST split and writes
data/zenodo/confidence_calibration.json, which the webcam app reads.

Usage:
    python scripts/calibrate_confidence.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_runtime_pipeline import (  # noqa: E402
    RAW_CSV, CachedModel, build_inputs, load_split, metrics, run_protocol, summarize,
)
from src.models.inference import DualTaskInference, FilterPolicy, class_coarse_group  # noqa: E402
from src.models.model_loader import load_zenodo_artifacts  # noqa: E402

CANDIDATE_POLICIES = {
    "no_filters": FilterPolicy(use_gravity_prior=False, use_sequence_pruning=False),
    "prior_only_legacy": FilterPolicy(use_gravity_prior=True, use_sequence_pruning=False),
    "combined_legacy": FilterPolicy(),
    "safe_coarse": FilterPolicy(mode="safe", granularity="coarse", soft_floor=0.5, stable_frames=3),
    "safe_family_prior_only": FilterPolicy(mode="safe", granularity="family", use_sequence_pruning=False,
                                           soft_floor=0.05, stable_frames=1),
}
THRESHOLDS = [round(t, 2) for t in np.arange(0.25, 0.86, 0.05)]


def selective(rows, t):
    sel = [r for r in rows if r["conf"] >= t]
    return (len(sel) / max(1, len(rows)),
            float(np.mean([r["pred"] == r["truth"] for r in sel])) if sel else 0.0, sel)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classifiers", nargs="+", default=["rf", "mlp"])
    ap.add_argument("--target-accuracy", type=float, default=0.80)
    ap.add_argument("--min-coverage", type=float, default=0.50)
    ap.add_argument("--class-min-precision", type=float, default=0.50)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "zenodo" / "confidence_calibration.json")
    args = ap.parse_args()

    raw = pd.read_csv(RAW_CSV)
    raw = raw.drop(columns=[c for c in ("asana_label", "correctness_label", "combined_label", "group_id") if c in raw])
    splits = {s: load_split(s, raw, None) for s in ("val", "test")}
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "selection_split": "val", "report_split": "test",
           "target_selective_accuracy": args.target_accuracy, "min_coverage": args.min_coverage}

    for clf in args.classifiers:
        art = load_zenodo_artifacts(ROOT / "saved_yoga_model_zenodo", active_classifier=clf)
        classes = art.class_names
        cache = {}
        for name, df in splits.items():
            lms, feats = build_inputs(df, art.feature_cols)
            X = pd.DataFrame([f.values for f in feats], columns=art.feature_cols)
            X_imp = art.imputer.transform(X)
            X_eval = art.scaler.transform(X_imp) if art.model_type == "mlp" else X_imp
            cache[name] = (df, lms, feats, X_eval)
        base_a, base_b = art.asana_model, art.verifier_model

        def run(split, policy, stream):
            df, lms, feats, X_eval = cache[split]
            art.asana_model, art.verifier_model = CachedModel(base_a, X_eval), CachedModel(base_b, X_eval)
            inf = DualTaskInference(art, confidence_threshold=0.0, filter_policy=policy)
            if stream:
                sessions = df["session_group"].drop_duplicates().sample(frac=1.0, random_state=0).tolist()
                pos = {s: k for k, s in enumerate(sessions)}
                order = sorted(range(len(df)), key=lambda i: (pos[df.at[i, "session_group"]], i))
                return run_protocol(inf, df, lms, feats, order, reset_on_session=False)
            return run_protocol(inf, df, lms, feats, list(range(len(df))), reset_on_session=True)

        # 1. policy selection on val
        val_f1 = {k: summarize(run("val", p, True), classes)["macro_f1"] for k, p in CANDIDATE_POLICIES.items()}
        best = max(val_f1, key=val_f1.get)
        policy = CANDIDATE_POLICIES[best]
        print(f"[{clf}] val stream macro-F1 by policy: {val_f1} → {best}")

        # 2. threshold selection on val (session protocol = holding a pose)
        val_rows = run("val", policy, False)
        table = []
        chosen = None
        for t in THRESHOLDS:
            cov, acc, _ = selective(val_rows, t)
            table.append({"threshold": t, "coverage": round(cov, 4), "selective_accuracy": round(acc, 4)})
            if chosen is None and acc >= args.target_accuracy and cov >= args.min_coverage:
                chosen = t
        if chosen is None:
            chosen = max(table, key=lambda r: r["selective_accuracy"] if r["coverage"] >= args.min_coverage else 0)["threshold"]
        _, _, sel = selective(val_rows, chosen)
        per_cls = {}
        for c in classes:
            pred_c = [r for r in sel if r["pred"] == c]
            per_cls[c] = round(float(np.mean([r["truth"] == c for r in pred_c])), 3) if len(pred_c) >= 10 else 0.0

        # 3. confirmation on test
        test_rows = run("test", policy, False)
        test_stream = summarize(run("test", policy, True), classes)
        t_cov, t_acc, t_sel = selective(test_rows, chosen)
        trusted = [c for c, p in per_cls.items() if p >= args.class_min_precision]
        t_sel_trusted = [r for r in t_sel if r["pred"] in trusted]
        # 4. orientation veto: how often would it block correct vs wrong issued labels (test)?
        def veto(r):
            g = class_coarse_group(r["pred"])
            return r["coarse"] is not None and g is not None and g != r["coarse"]
        corr = [r for r in t_sel_trusted if r["pred"] == r["truth"]]
        wrong = [r for r in t_sel_trusted if r["pred"] != r["truth"]]
        veto_stats = {
            "correct_labels_vetoed_rate": round(float(np.mean([veto(r) for r in corr])), 4) if corr else None,
            "wrong_labels_vetoed_rate": round(float(np.mean([veto(r) for r in wrong])), 4) if wrong else None,
            "orientation_evidence_available_rate": round(float(np.mean([r["coarse"] is not None for r in test_rows])), 4),
        }
        kept = [r for r in t_sel_trusted if not veto(r)]

        out[clf] = {
            "artifact_dir": f"saved_yoga_model_zenodo/{clf}",
            "filter_policy_name": best,
            "filter_policy": policy.__dict__,
            "val_stream_macro_f1_by_policy": val_f1,
            "threshold": chosen,
            "class_min_precision": args.class_min_precision,
            "per_class_precision": per_cls,
            "untrusted_classes": [c for c in classes if c not in trusted],
            "val_threshold_table": table,
            "test": {
                "coverage_at_threshold": round(t_cov, 4),
                "selective_accuracy_at_threshold": round(t_acc, 4),
                "coverage_with_class_gate": round(len(t_sel_trusted) / max(1, len(test_rows)), 4),
                "selective_accuracy_with_class_gate": round(float(np.mean([r["pred"] == r["truth"] for r in t_sel_trusted])), 4) if t_sel_trusted else None,
                "stream_accuracy": test_stream["accuracy"],
                "stream_macro_f1": test_stream["macro_f1"],
                "orientation_veto": veto_stats,
                "coverage_with_class_gate_and_veto": round(len(kept) / max(1, len(test_rows)), 4),
                "selective_accuracy_with_class_gate_and_veto": round(float(np.mean([r["pred"] == r["truth"] for r in kept])), 4) if kept else None,
                "wrong_label_rate_before": round(float(np.mean([r["pred"] != r["truth"] for r in test_rows if r["conf"] >= 0.28])), 4),
            },
        }
        art.asana_model, art.verifier_model = base_a, base_b
        print(f"[{clf}] threshold={chosen} per-class precision={per_cls}")
        print(f"[{clf}] test: {json.dumps(out[clf]['test'])}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
