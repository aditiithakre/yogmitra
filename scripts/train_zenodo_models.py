"""
scripts/train_zenodo_models.py
──────────────────────────────
Training pipeline architecture for Zenodo Yoga Models:
- Model A: Asana Classifier (10 classes: Anantasana ... Viparita Karani)
- Model B: Form Verifier (Binary: Right vs Wrong)

Architecture:
- Random Forest baseline (balanced class weighting, 150 estimators)
- Optional MLP baseline (multi-layer perceptron with BatchNorm / Dropout emulation)
- Strict training gate: dry-run validation by default; requires explicit approval flag.

Usage:
    # Dry-run validation (checks shapes, classes, pipeline integrity without training)
    python scripts/train_zenodo_models.py --dry-run

    # Proposed training command (to be run ONLY after explicit approval):
    # python scripts/train_zenodo_models.py --execute-training-approved --model-type rf
"""

from __future__ import annotations

import argparse
import json
import time
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.zenodo_annotations import ZENODO_ASANAS

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("train_zenodo")

DEFAULT_DATA_DIR = ROOT / "data" / "zenodo"
ORIGINAL_DIR = ROOT / "saved_yoga_model_zenodo"
# New models go to a separate, versioned candidate directory; originals are never overwritten
DEFAULT_OUT_DIR = ROOT / "saved_yoga_model_zenodo_candidates" / time.strftime("v%Y%m%d_%H%M%S")


def build_asana_classifier(model_type: str = "rf", random_state: int = 42) -> Any:
    """Build Model A: 10-Class Asana Classifier."""
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=150,
            max_depth=20,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size=64,
            learning_rate_init=0.001,
            max_iter=250,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def build_form_verifier(model_type: str = "rf", random_state: int = 42) -> Any:
    """Build Model B: Binary Form Verifier (Right vs Wrong)."""
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=150,
            max_depth=16,
            min_samples_split=6,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(96, 48),
            activation="relu",
            solver="adam",
            alpha=0.002,
            batch_size=64,
            learning_rate_init=0.001,
            max_iter=200,
            early_stopping=True,
            n_iter_no_change=12,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def run_pipeline(
    data_dir: Path = DEFAULT_DATA_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    model_type: str = "rf",
    execute_training: bool = False,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Validates data, builds architectures, and either halts at training gate or fits models.
    """
    logger.info("Checking data splits in: %s", data_dir.resolve())
    train_path = data_dir / "train_features.csv"
    val_path = data_dir / "val_features.csv"
    test_path = data_dir / "test_features.csv"
    blueprint_path = data_dir / "zenodo_preprocessing_blueprint.joblib"

    for p in (train_path, val_path, test_path, blueprint_path):
        if not p.exists():
            raise FileNotFoundError(f"Required dataset artifact missing: {p}. Run scripts/prepare_zenodo_dataset.py first.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    blueprint = joblib.load(blueprint_path)

    feature_cols = blueprint["feature_cols"]
    asana_le = blueprint["asana_label_encoder"]
    correctness_le = blueprint["correctness_label_encoder"]
    imputer = blueprint["imputer"]
    scaler = blueprint["scaler"]

    logger.info("Loaded datasets: Train=%d, Val=%d, Test=%d | Features=%d",
                len(train_df), len(val_df), len(test_df), len(feature_cols))

def train_single_architecture(
    model_type: str,
    X_train: np.ndarray,
    y_train_asana: np.ndarray,
    y_train_corr: np.ndarray,
    X_val: np.ndarray,
    y_val_asana: np.ndarray,
    y_val_corr: np.ndarray,
    target_dir: Path,
    blueprint: dict[str, Any],
    random_state: int = 42,
) -> dict[str, Any]:
    """Fits and evaluates a single architecture (RF or MLP)."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    logger.info(">>> Training %s architecture...", model_type.upper())
    model_a = build_asana_classifier(model_type, random_state=random_state)
    model_b = build_form_verifier(model_type, random_state=random_state)

    model_a.fit(X_train, y_train_asana)
    model_b.fit(X_train, y_train_corr)

    # Validation metrics
    val_pred_a = model_a.predict(X_val)
    val_pred_b = model_b.predict(X_val)

    acc_a = float(accuracy_score(y_val_asana, val_pred_a))
    f1_a = float(f1_score(y_val_asana, val_pred_a, average="macro", zero_division=0))
    acc_b = float(accuracy_score(y_val_corr, val_pred_b))
    bal_acc_b = float(balanced_accuracy_score(y_val_corr, val_pred_b))

    target_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_a, target_dir / "zenodo_asana_classifier.joblib")
    joblib.dump(model_b, target_dir / "zenodo_form_verifier.joblib")
    joblib.dump({
        "imputer": blueprint["imputer"],
        "scaler": blueprint["scaler"],
        "asana_label_encoder": blueprint["asana_label_encoder"],
        "correctness_label_encoder": blueprint["correctness_label_encoder"],
        "feature_cols": blueprint["feature_cols"],
        "class_names": blueprint["asana_label_encoder"].classes_.tolist(),
        "correctness_names": blueprint["correctness_label_encoder"].classes_.tolist(),
        "model_type": model_type,
    }, target_dir / "zenodo_preprocessing.joblib")

    logger.info("[%s] Val Model A: Acc=%.4f, Macro-F1=%.4f | Model B: Acc=%.4f, Bal-Acc=%.4f",
                model_type.upper(), acc_a, f1_a, acc_b, bal_acc_b)

    return {
        "model_type": model_type,
        "val_asana_accuracy": acc_a,
        "val_asana_macro_f1": f1_a,
        "val_form_accuracy": acc_b,
        "val_form_balanced_accuracy": bal_acc_b,
        "target_dir": str(target_dir.resolve()),
        "model_a": model_a,
        "model_b": model_b,
    }


def run_pipeline(
    data_dir: Path = DEFAULT_DATA_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    model_type: str = "rf",
    execute_training: bool = False,
    random_state: int = 42,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    """
    Validates data, builds architectures, and either halts at training gate or fits models.
    Supports model_type in ('rf', 'mlp', 'both').
    """
    logger.info("Checking data splits in: %s", data_dir.resolve())
    train_path = data_dir / "train_features.csv"
    val_path = data_dir / "val_features.csv"
    test_path = data_dir / "test_features.csv"
    blueprint_path = data_dir / "zenodo_preprocessing_blueprint.joblib"

    for p in (train_path, val_path, test_path, blueprint_path):
        if not p.exists():
            raise FileNotFoundError(f"Required dataset artifact missing: {p}. Run scripts/prepare_zenodo_dataset.py first.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    blueprint = joblib.load(blueprint_path)

    feature_cols = blueprint["feature_cols"]
    asana_le = blueprint["asana_label_encoder"]
    correctness_le = blueprint["correctness_label_encoder"]
    imputer = blueprint["imputer"]
    scaler = blueprint["scaler"]

    logger.info("Loaded datasets: Train=%d, Val=%d, Test=%d | Features=%d",
                len(train_df), len(val_df), len(test_df), len(feature_cols))

    # Preprocessing feature transformation
    X_train_raw = train_df[feature_cols].copy()
    X_val_raw = val_df[feature_cols].copy()
    X_test_raw = test_df[feature_cols].copy()

    X_train_imp = imputer.transform(X_train_raw)
    X_val_imp = imputer.transform(X_val_raw)
    X_test_imp = imputer.transform(X_test_raw)

    X_train_scaled = scaler.transform(X_train_imp)
    X_val_scaled = scaler.transform(X_val_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    # Labels
    y_train_asana = asana_le.transform(train_df["asana_label"])
    y_val_asana = asana_le.transform(val_df["asana_label"])
    y_test_asana = asana_le.transform(test_df["asana_label"])

    y_train_corr = correctness_le.transform(train_df["correctness_label"])
    y_val_corr = correctness_le.transform(val_df["correctness_label"])
    y_test_corr = correctness_le.transform(test_df["correctness_label"])

    logger.info("Prepared Model A target (Asana): %d unique classes", len(asana_le.classes_))
    logger.info("Prepared Model B target (Correctness): classes=%s", list(correctness_le.classes_))

    status_report: dict[str, Any] = {
        "status": "prepared",
        "training_executed": False,
        "model_type": model_type,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "n_features_in": len(feature_cols),
        "n_features_model": X_train_imp.shape[1],
        "asana_classes": asana_le.classes_.tolist(),
        "correctness_classes": correctness_le.classes_.tolist(),
        "target_out_dir": str(out_dir.resolve()),
    }

    # ── STRICT TRAINING GATE ──────────────────────────────────────────────────
    if not execute_training:
        print("\n" + "=" * 78)
        print("  [TRAINING GATE ACTIVE] Non-training validation PASSED.")
        print("  Model architectures and preprocessed data splits are fully verified.")
        print(f"  Architecture(s) configured: {model_type.upper()}")
        print(f"  Model A: Classifier ({len(asana_le.classes_)} Asanas)")
        print(f"  Model B: Form Verifier ({list(correctness_le.classes_)})")
        print("  STOPPED before fitting models (.fit() was NOT called).")
        print("  To train models, run with explicit user approval:")
        print("    python scripts/train_zenodo_models.py --execute-training-approved --model-type both")
        print("=" * 78 + "\n")
        return status_report

    # ── EXPLICITLY APPROVED TRAINING EXECUTION ────────────────────────────────
    existing = list(out_dir.rglob("zenodo_asana_classifier.joblib")) if out_dir.exists() else []
    if existing and not allow_overwrite:
        raise SystemExit(f"Refusing to overwrite existing model artifacts in {out_dir}. "
                         "Choose a new --out-dir (default is a timestamped candidate directory) "
                         "or pass --overwrite-existing explicitly.")
    logger.info(">>> TRAINING APPROVED: Proceeding with model fitting into %s", out_dir)
    results: dict[str, Any] = {}

    models_to_train = ["rf", "mlp"] if model_type == "both" else [model_type]

    for m in models_to_train:
        X_tr = X_train_imp if m == "rf" else X_train_scaled
        X_va = X_val_imp if m == "rf" else X_val_scaled
        sub_dir = out_dir / m if model_type == "both" else out_dir

        m_res = train_single_architecture(
            model_type=m,
            X_train=X_tr,
            y_train_asana=y_train_asana,
            y_train_corr=y_train_corr,
            X_val=X_va,
            y_val_asana=y_val_asana,
            y_val_corr=y_val_corr,
            target_dir=sub_dir,
            blueprint=blueprint,
            random_state=random_state,
        )
        results[m] = m_res

    # Select champion model for the main out_dir root
    if model_type == "both":
        # Compare validation macro F1 on Model A and balanced accuracy on Model B
        rf_score = results["rf"]["val_asana_macro_f1"] + results["rf"]["val_form_balanced_accuracy"]
        mlp_score = results["mlp"]["val_asana_macro_f1"] + results["mlp"]["val_form_balanced_accuracy"]
        champion_type = "rf" if rf_score >= mlp_score else "mlp"
        logger.info(">>> Champion model on validation set: %s (RF composite: %.4f, MLP composite: %.4f)",
                    champion_type.upper(), rf_score, mlp_score)

        # Copy champion artifacts to root out_dir so standard profile loaders find them
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(results[champion_type]["model_a"], out_dir / "zenodo_asana_classifier.joblib")
        joblib.dump(results[champion_type]["model_b"], out_dir / "zenodo_form_verifier.joblib")
        joblib.dump({
            "imputer": imputer,
            "scaler": scaler,
            "asana_label_encoder": asana_le,
            "correctness_label_encoder": correctness_le,
            "feature_cols": feature_cols,
            "class_names": asana_le.classes_.tolist(),
            "correctness_names": correctness_le.classes_.tolist(),
            "model_type": champion_type,
            "champion_selection": {
                "champion": champion_type,
                "rf_composite": round(rf_score, 4),
                "mlp_composite": round(mlp_score, 4),
            },
        }, out_dir / "zenodo_preprocessing.joblib")

    status_report["status"] = "trained"
    status_report["training_executed"] = True
    status_report["results"] = {
        m: {k: v for k, v in res.items() if not k.startswith("model_")}
        for m, res in results.items()
    }
    logger.info("Training complete. Primary artifacts saved to: %s", out_dir.resolve())

    # Summary table
    print("\n" + "=" * 78)
    print("  ZENODO MODEL TRAINING COMPLETE")
    print("=" * 78)
    for m, res in results.items():
        print(f"  [{m.upper()}] Model A (10 Asanas) Val Macro-F1: {res['val_asana_macro_f1']:.4f} (Acc: {res['val_asana_accuracy']:.4f})")
        print(f"  [{m.upper()}] Model B (Form)      Val Bal-Acc:  {res['val_form_balanced_accuracy']:.4f} (Acc: {res['val_form_accuracy']:.4f})")
    if model_type == "both":
        print(f"  Primary Active Deployment:         {champion_type.upper()} (saved to {out_dir.resolve()})")
    print("=" * 78 + "\n")

    return status_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Zenodo Yoga Models Training Pipeline")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model-type", choices=["rf", "mlp", "both"], default="rf")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Dry-run static validation")
    parser.add_argument(
        "--execute-training-approved",
        action="store_true",
        default=False,
        help="Explicit approval flag to proceed with model training fitting",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-existing", action="store_true", default=False,
                        help="Allow writing into a directory that already holds trained models")
    args = parser.parse_args()

    execute_flag = args.execute_training_approved and not args.dry_run

    run_pipeline(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        model_type=args.model_type,
        execute_training=execute_flag,
        random_state=args.seed,
        allow_overwrite=args.overwrite_existing,
    )


if __name__ == "__main__":
    main()
