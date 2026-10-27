"""
src/models/model_loader.py
───────────────────────────
Loads, validates, and exposes the trained classifiers and preprocessing
objects from saved_yoga_model/.

Raises clear RuntimeError messages when artifacts are missing or incompatible
so the application fails fast at startup rather than silently during inference.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_dir(path: str | Path) -> Path:
    """Relative artifact paths are resolved against the project root, not the shell cwd."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    return _PROJECT_ROOT / p


@dataclass
class YogaArtifacts:
    """Container for all loaded model artifacts."""
    imputer:       Any          # sklearn SimpleImputer
    scaler:        Any          # sklearn StandardScaler
    label_encoder: Any          # sklearn LabelEncoder
    feature_cols:  list[str]    # 39-element ordered feature list
    rf_model:      Any          # RandomForestClassifier
    mlp_model:     Optional[Any]  # tf.keras.Model or None
    class_names:   list[str]    # 47 pose names
    transition_matrix: np.ndarray  # 47×47 HMM matrix

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)


def load_artifacts(model_dir: str | Path) -> YogaArtifacts:
    """
    Load all model artifacts from *model_dir*.
    Validates shapes and class counts before returning.
    """
    model_dir = _resolve_dir(model_dir)
    _require(model_dir.is_dir(), f"Model directory not found: {model_dir}")

    # ── Preprocessing ─────────────────────────────────────────────────────────
    prep_path = model_dir / "yoga_preprocessing.joblib"
    _require(prep_path.exists(), f"Missing: {prep_path}")
    prep = joblib.load(prep_path)
    for key in ("imputer", "scaler", "label_encoder", "feature_cols"):
        _require(key in prep, f"yoga_preprocessing.joblib is missing key '{key}'")

    feature_cols: list[str] = prep["feature_cols"]
    n_raw_features = len(feature_cols)

    # After imputer.transform the feature matrix grows because add_indicator=True
    # adds missing-indicator columns.  We discover the expected width from the
    # imputer's own indicator_ attribute.
    imputer = prep["imputer"]
    n_model_features = n_raw_features
    if hasattr(imputer, "indicator_") and imputer.indicator_ is not None:
        n_model_features += imputer.indicator_.features_.shape[0]

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf_path = model_dir / "yoga_random_forest.joblib"
    _require(rf_path.exists(), f"Missing: {rf_path}")
    rf_model = joblib.load(rf_path)
    _require(
        rf_model.n_features_in_ == n_model_features,
        f"RF expects {rf_model.n_features_in_} features; "
        f"preprocessing produces {n_model_features}.  Retrain or resave."
    )

    # ── MLP (optional – may fail import if TensorFlow unavailable) ────────────
    mlp_model = None
    mlp_path = model_dir / "yoga_mlp.keras"
    if mlp_path.exists():
        try:
            import tensorflow as tf  # noqa: F401  – lazy import
            mlp_model = tf.keras.models.load_model(mlp_path)
        except Exception as exc:
            print(f"[ModelLoader] MLP skipped: {exc}")

    # ── HMM transition matrix ─────────────────────────────────────────────────
    hmm_path = model_dir / "yoga_hmm.npz"
    _require(hmm_path.exists(), f"Missing: {hmm_path}")
    hmm_data = np.load(hmm_path)
    transition_matrix = hmm_data["transition_matrix"]
    hmm_class_names: list[str] = hmm_data["class_names"].tolist()

    # ── Cross-validate class counts ───────────────────────────────────────────
    le_classes: list[str] = prep["label_encoder"].classes_.tolist()
    _require(
        len(le_classes) == len(hmm_class_names),
        "label_encoder classes and HMM class_names have different lengths."
    )
    _require(
        len(rf_model.classes_) == len(le_classes),
        "RF classes and label_encoder classes have different lengths."
    )

    return YogaArtifacts(
        imputer=imputer,
        scaler=prep["scaler"],
        label_encoder=prep["label_encoder"],
        feature_cols=feature_cols,
        rf_model=rf_model,
        mlp_model=mlp_model,
        class_names=le_classes,
        transition_matrix=transition_matrix,
    )


@dataclass
class YogaArtifactsZenodo:
    """Container for loaded Zenodo dual-task model artifacts."""
    imputer: Any
    scaler: Any
    asana_label_encoder: Any
    correctness_label_encoder: Any
    feature_cols: list[str]
    asana_model: Any                # Model A (10 Asanas)
    verifier_model: Any             # Model B (Binary Right/Wrong)
    class_names: list[str]          # 10 Asana names
    correctness_names: list[str]    # ["Right", "Wrong"]
    model_type: str = "rf"

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)


def load_zenodo_artifacts(
    model_dir: str | Path = "saved_yoga_model_zenodo",
    active_classifier: Optional[str] = None,
) -> YogaArtifactsZenodo:
    """
    Load Zenodo dual-task model artifacts from *model_dir*.
    If *active_classifier* ('rf' or 'mlp') is specified and a corresponding
    subdirectory exists, loads that sub-architecture; otherwise loads from *model_dir*.
    Raises RuntimeError if artifacts have not been trained yet.
    """
    model_dir = _resolve_dir(model_dir)
    if active_classifier:
        sub_dir = model_dir / active_classifier.lower()
        if sub_dir.is_dir() and (sub_dir / "zenodo_preprocessing.joblib").exists():
            model_dir = sub_dir

    _require(
        model_dir.is_dir(),
        f"Zenodo model directory not found: {model_dir}. "
        "Migration requires running the approved training pipeline (scripts/train_zenodo_models.py)."
    )

    prep_path = model_dir / "zenodo_preprocessing.joblib"
    _require(prep_path.exists(), f"Missing Zenodo preprocessing artifact: {prep_path}")
    prep = joblib.load(prep_path)

    for key in ("imputer", "scaler", "asana_label_encoder", "correctness_label_encoder", "feature_cols", "class_names"):
        _require(key in prep, f"zenodo_preprocessing.joblib missing key: '{key}'")

    asana_path = model_dir / "zenodo_asana_classifier.joblib"
    _require(asana_path.exists(), f"Missing Model A artifact: {asana_path}")
    asana_model = joblib.load(asana_path)

    verifier_path = model_dir / "zenodo_form_verifier.joblib"
    _require(verifier_path.exists(), f"Missing Model B artifact: {verifier_path}")
    verifier_model = joblib.load(verifier_path)

    return YogaArtifactsZenodo(
        imputer=prep["imputer"],
        scaler=prep["scaler"],
        asana_label_encoder=prep["asana_label_encoder"],
        correctness_label_encoder=prep["correctness_label_encoder"],
        feature_cols=prep["feature_cols"],
        asana_model=asana_model,
        verifier_model=verifier_model,
        class_names=prep["class_names"],
        correctness_names=prep.get("correctness_names", ["Right", "Wrong"]),
        model_type=prep.get("model_type", "rf"),
    )


def load_profile_artifacts(profile: str = "kaggle", model_dir: Optional[str | Path] = None) -> Any:
    """
    Factory loader supporting profile switching:
    - 'kaggle' (default): Loads 47-class Kaggle artifacts from saved_yoga_model/
    - 'zenodo': Loads 10-class dual-task Zenodo artifacts from saved_yoga_model_zenodo/
    """
    p = profile.lower()
    if p == "kaggle":
        target = model_dir or "saved_yoga_model"
        return load_artifacts(target)
    elif p == "zenodo":
        target = model_dir or "saved_yoga_model_zenodo"
        return load_zenodo_artifacts(target)
    else:
        raise ValueError(f"Unknown model profile: '{profile}'. Supported: 'kaggle', 'zenodo'")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"[ModelLoader] {message}")

