"""
scripts/evaluate_zenodo_models.py
─────────────────────────────────
Evaluation of the saved Zenodo models on the held-out TEST split.

Reported separately for every architecture found under saved_yoga_model_zenodo/
(rf/, mlp/) and for the un-suffixed "champion" directory, with the artifact's
model_type and file hash so it is unambiguous which classifier produced which
number:

  • Model A (10-asana classifier): accuracy, macro/weighted F1, per-class
    precision / recall / F1 / support, labelled confusion matrix
  • Model B (Right/Wrong form verifier): accuracy, balanced accuracy, precision,
    recall, F1, ROC-AUC, confusion matrix, per-asana accuracy
  • Latency of feature extraction and single-frame classifier inference measured
    on real held-out landmarks (not on degenerate dummy points)
  • SYNTHETIC occlusion stress test: real held-out landmarks with a random subset
    of the 16 body landmarks hidden; measures how the posture evaluator and the
    classifier degrade, and verifies that no correction is issued for an angle
    whose joints are hidden

These are bare-classifier, single-frame metrics on dataset landmarks.  They are
NOT landmark-estimation accuracy (no annotated keypoints exist) and NOT live
webcam accuracy.  Runtime-pipeline metrics (filters, smoothing, transitions)
are produced by scripts/evaluate_runtime_pipeline.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import ANGLE_TRIPLETS, LANDMARK_IDS, extract_features, features_to_series  # noqa: E402
from src.correction.posture_evaluator import evaluate as evaluate_posture  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("evaluate_zenodo")

RAW_CSV = ROOT / "zenodo_yoga_landmarks - zenodo_yoga_landmarks.csv"
BODY_IDS = sorted(LANDMARK_IDS.values())


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def to_landmarks(row: np.ndarray) -> list[dict]:
    return [{"x": float(row[4 * i]), "y": float(row[4 * i + 1]), "z": float(row[4 * i + 2]),
             "visibility": float(row[4 * i + 3])} for i in range(33)]


def load_test_landmarks(data_dir: Path) -> tuple[pd.DataFrame, list[list[dict]]]:
    test = pd.read_csv(data_dir / "test_features.csv", usecols=["image_path", "asana_label", "correctness_label"])
    raw = pd.read_csv(RAW_CSV, usecols=["image_path"] + [f"f_{i}" for i in range(132)])
    df = test.merge(raw, on="image_path", how="left")
    lms = [to_landmarks(r) for r in df[[f"f_{i}" for i in range(132)]].to_numpy(float)]
    return df, lms


def evaluate_trained_models(model_dir: Path, data_dir: Path) -> dict[str, Any]:
    """Evaluates Model A and Model B in *model_dir* on the held-out test features."""
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, classification_report,
        confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
    )

    model_a = joblib.load(model_dir / "zenodo_asana_classifier.joblib")
    model_b = joblib.load(model_dir / "zenodo_form_verifier.joblib")
    prep = joblib.load(model_dir / "zenodo_preprocessing.joblib")

    test_df = pd.read_csv(data_dir / "test_features.csv")
    feature_cols = prep["feature_cols"]
    asana_le = prep["asana_label_encoder"]
    corr_le = prep["correctness_label_encoder"]
    model_type = prep.get("model_type", "rf")

    X_test_imp = prep["imputer"].transform(test_df[feature_cols])
    X_eval = prep["scaler"].transform(X_test_imp) if model_type == "mlp" else X_test_imp

    labels = list(asana_le.classes_)
    y_a = asana_le.transform(test_df["asana_label"])
    pred_a = model_a.predict(X_eval)
    report_a = classification_report(y_a, pred_a, labels=list(range(len(labels))), target_names=labels,
                                     output_dict=True, zero_division=0)

    y_b = corr_le.transform(test_df["correctness_label"])
    corr_labels = list(corr_le.classes_)
    pos = corr_labels.index("Wrong") if "Wrong" in corr_labels else 1
    pred_b = model_b.predict(X_eval)
    proba_b = model_b.predict_proba(X_eval)[:, pos]

    per_asana = {}
    for asana in labels:
        mask = (test_df["asana_label"] == asana).to_numpy()
        if mask.sum():
            per_asana[asana] = {"samples": int(mask.sum()),
                                "accuracy": round(float(accuracy_score(y_b[mask], pred_b[mask])), 3)}

    return {
        "artifact_dir": str(model_dir.relative_to(ROOT)),
        "model_type": model_type,
        "estimator": type(model_a).__name__,
        "asana_classifier_md5": md5(model_dir / "zenodo_asana_classifier.joblib"),
        "model_a_asana": {
            "accuracy": round(float(accuracy_score(y_a, pred_a)), 4),
            "macro_f1": round(float(f1_score(y_a, pred_a, average="macro", zero_division=0)), 4),
            "weighted_f1": round(float(f1_score(y_a, pred_a, average="weighted", zero_division=0)), 4),
            "per_class": {c: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in report_a[c].items()}
                          for c in labels},
            "confusion_matrix": {"rows_true_cols_pred": labels,
                                 "matrix": confusion_matrix(y_a, pred_a, labels=list(range(len(labels)))).tolist()},
        },
        "model_b_form_verifier": {
            "positive_class": corr_labels[pos],
            "accuracy": round(float(accuracy_score(y_b, pred_b)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_b, pred_b)), 4),
            "precision": round(float(precision_score(y_b, pred_b, pos_label=pos, zero_division=0)), 4),
            "recall": round(float(recall_score(y_b, pred_b, pos_label=pos, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_b, pred_b, pos_label=pos, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_b == pos, proba_b)), 4),
            "confusion_matrix": {"rows_true_cols_pred": corr_labels, "matrix": confusion_matrix(y_b, pred_b).tolist()},
            "per_asana_accuracy": per_asana,
        },
    }


def benchmark_runtime(model_dir: Path, lms: list[list[dict]], n: int = 300) -> dict[str, float]:
    """Feature extraction + single-frame classifier latency on real held-out landmarks."""
    prep = joblib.load(model_dir / "zenodo_preprocessing.joblib")
    model = joblib.load(model_dir / "zenodo_asana_classifier.joblib")
    cols = prep["feature_cols"]
    feat_ms, clf_ms = [], []
    for k in range(n):
        lm = lms[k % len(lms)]
        t0 = time.perf_counter()
        s = features_to_series(extract_features(lm, [0] * 33, 1280, 720), cols)
        t1 = time.perf_counter()
        X = prep["imputer"].transform(pd.DataFrame([s.values], columns=cols))
        if prep.get("model_type") == "mlp":
            X = prep["scaler"].transform(X)
        model.predict_proba(X)
        t2 = time.perf_counter()
        feat_ms.append((t1 - t0) * 1000)
        clf_ms.append((t2 - t1) * 1000)
    return {
        "frames": n,
        "feature_extraction_mean_ms": round(float(np.mean(feat_ms)), 3),
        "feature_extraction_p95_ms": round(float(np.percentile(feat_ms, 95)), 3),
        "impute_scale_classify_mean_ms": round(float(np.mean(clf_ms)), 3),
        "impute_scale_classify_p95_ms": round(float(np.percentile(clf_ms, 95)), 3),
        "note": "excludes MediaPipe (see data/mediapipe_mode_comparison.json); single CPU thread of the build machine",
    }


def occlusion_stress_test(model_dir: Path, df: pd.DataFrame, lms: list[list[dict]], seed: int = 0) -> dict[str, Any]:
    """SYNTHETIC: hide k of the 16 body landmarks (visibility → 0.02) on real held-out frames."""
    prep = joblib.load(model_dir / "zenodo_preprocessing.joblib")
    model = joblib.load(model_dir / "zenodo_asana_classifier.joblib")
    cols, classes = prep["feature_cols"], prep["class_names"]
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {}
    for k in (0, 2, 4, 8):
        low_vis = unsupported = issued = unavailable = n_checks = correct = 0
        feats = []
        for i, lm in enumerate(lms):
            hidden = set(rng.choice(BODY_IDS, size=k, replace=False).tolist()) if k else set()
            occl = [dict(p, visibility=0.02) if j in hidden else p for j, p in enumerate(lm)]
            f = extract_features(occl, [0] * 33, 1280, 720)
            feats.append(features_to_series(f, cols).values)
            ev = evaluate_posture(df.at[i, "asana_label"], f, occl, [0] * 33)
            low_vis += ev.overall_status == "low_visibility"
            for c in ev.checks:
                n_checks += 1
                unavailable += c.status == "unavailable"
                if c.status in ("too_low", "too_high"):
                    issued += 1
                    if set(ANGLE_TRIPLETS.get(c.angle_name, ())) & hidden:
                        unsupported += 1
        X = prep["imputer"].transform(pd.DataFrame(feats, columns=cols))
        if prep.get("model_type") == "mlp":
            X = prep["scaler"].transform(X)
        pred = np.array(classes)[np.argmax(model.predict_proba(X), axis=1)]
        correct = float(np.mean(pred == df["asana_label"].to_numpy()))
        out[f"hidden_body_landmarks_{k}_of_16"] = {
            "frames": len(lms),
            "classifier_accuracy": round(correct, 4),
            "evaluator_low_visibility_rate": round(low_vis / len(lms), 4),
            "angle_checks_unavailable_rate": round(unavailable / max(1, n_checks), 4),
            "corrections_issued_per_frame": round(issued / len(lms), 3),
            "corrections_on_hidden_joints": int(unsupported),
        }
    return {"kind": "synthetic occlusion of real held-out MediaPipe landmarks (seeded)",
            "evaluated_against": "ground-truth asana reference ranges", "results": out}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Zenodo Models and Pipeline")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "saved_yoga_model_zenodo")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "zenodo")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--out-report", type=Path, default=ROOT / "data" / "zenodo_evaluation_report.json")
    args = parser.parse_args()

    if not (args.data_dir / "test_features.csv").exists():
        logger.error("Test data not found in %s. Run scripts/prepare_zenodo_dataset.py first.", args.data_dir)
        sys.exit(1)

    cfg = yaml.safe_load(args.config.read_text())
    active_clf = cfg["model"].get("active_classifier", "rf")
    split = json.loads((args.data_dir / "split_summary.json").read_text())

    dirs = {sub: args.model_dir / sub for sub in ("rf", "mlp") if (args.model_dir / sub / "zenodo_asana_classifier.joblib").exists()}
    if not dirs:
        logger.error("No trained Zenodo artifacts found under %s", args.model_dir)
        sys.exit(1)

    classifiers = {name: evaluate_trained_models(d, args.data_dir) for name, d in dirs.items()}
    root_prep = joblib.load(args.model_dir / "zenodo_preprocessing.joblib")
    root_md5 = md5(args.model_dir / "zenodo_asana_classifier.joblib")
    champion = {
        "artifact_dir": str(args.model_dir.relative_to(ROOT)),
        "model_type": root_prep.get("model_type"),
        "identical_to": [n for n, r in classifiers.items() if r["asana_classifier_md5"] == root_md5],
        "selection": root_prep.get("champion_selection"),
        "note": "the app loads <dir>/<active_classifier>/ when it exists, so this copy is NOT what the app runs",
    }

    df, lms = load_test_landmarks(args.data_dir)
    active_dir = dirs.get(active_clf, next(iter(dirs.values())))
    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "report_status": "measured",
        "active_profile": cfg["model"].get("profile"),
        "active_classifier": active_clf,
        "active_artifact_dir": str(active_dir.relative_to(ROOT)),
        "dataset": {
            "name": split.get("dataset_name"),
            "split_method": "group-aware split by recording burst session (no session in more than one split)",
            "evaluated_split": "test",
            "test_samples": split.get("test_samples"),
            "test_sessions": split.get("test_sessions"),
            "train_val_test_samples": [split.get("train_samples"), split.get("val_samples"), split.get("test_samples")],
            "used_for_training_or_tuning": "train (fitting); val (champion selection, thresholds, filter policy)",
        },
        "metric_scope": ("single-frame classification of stored MediaPipe landmarks; not landmark-estimation "
                         "accuracy (no annotated keypoints) and not live-webcam accuracy"),
        "classifiers": classifiers,
        "root_champion_copy": champion,
        "runtime_latency": {n: benchmark_runtime(d, lms) for n, d in dirs.items()},
        "occlusion_stress_test": occlusion_stress_test(active_dir, df, lms),
    }

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 78)
    print("  ZENODO MODEL EVALUATION (held-out test split)")
    print("=" * 78)
    print(f"  Active profile/classifier: {report['active_profile']} / {active_clf} ({report['active_artifact_dir']})")
    for n, r in classifiers.items():
        a, b = r["model_a_asana"], r["model_b_form_verifier"]
        print(f"  [{n.upper()}] Asana acc {a['accuracy']:.4f}  macro-F1 {a['macro_f1']:.4f} | "
              f"Form bal-acc {b['balanced_accuracy']:.4f}  AUC {b['roc_auc']:.4f}")
    print(f"  Root champion copy = {champion['model_type']} (identical to {champion['identical_to']})")
    for k, v in report["occlusion_stress_test"]["results"].items():
        print(f"  {k}: clf acc {v['classifier_accuracy']:.3f}, low-vis {v['evaluator_low_visibility_rate']:.3f}, "
              f"corrections on hidden joints {v['corrections_on_hidden_joints']}")
    print(f"  Report: {args.out_report}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
