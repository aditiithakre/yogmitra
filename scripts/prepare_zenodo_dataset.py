"""
scripts/prepare_zenodo_dataset.py
─────────────────────────────────
Zenodo Yoga Landmarks dataset preparation pipeline:
1. Reconstructs MediaPipe 33-landmark structures from raw features f_0..f_131.
2. Computes the shared 39 biomechanical features with strict ordering parity.
3. Groups samples by burst session ({date}_{burst}_{group_id}) to prevent video/burst leakage.
4. Performs group-stratified splitting across train (70%), val (15%), test (15%), ensuring
   all 10 asanas and rare classes (Vajrasana) are represented in all partitions.
5. Fits preprocessing (imputation, scaling, label encoding) STRICTLY on the training set.
6. Computes candidate provisional reference angle ranges from the training partition's Right samples.
7. Saves preprocessed datasets, preprocessing blueprint, and split metadata.

Usage:
    python scripts/prepare_zenodo_dataset.py
    python scripts/prepare_zenodo_dataset.py --csv "zenodo_yoga_landmarks - zenodo_yoga_landmarks.csv" --out-dir "data/zenodo"
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import extract_features, features_to_series, ANGLE_TRIPLETS
from src.models.zenodo_annotations import ZENODO_ASANAS, parse_group_id

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("prepare_zenodo")


def landmark_vector_to_mediapipe_dicts(row_values: np.ndarray) -> list[dict[str, float]]:
    """
    Convert a 132-element numpy array (f_0..f_131) into 33 MediaPipe-style landmark dicts:
      f_{4i}   -> x
      f_{4i+1} -> y
      f_{4i+2} -> z
      f_{4i+3} -> visibility
    """
    if len(row_values) != 132:
        raise ValueError(f"Expected 132 landmark values, got {len(row_values)}")

    landmarks: list[dict[str, float]] = []
    for i in range(33):
        base = i * 4
        landmarks.append({
            "x": float(row_values[base]),
            "y": float(row_values[base + 1]),
            "z": float(row_values[base + 2]),
            "visibility": float(row_values[base + 3]),
        })
    return landmarks


def extract_burst_session(image_path: str, group_id: str) -> str:
    """
    Extract a leak-resistant session key from image path and group_id.
    Path example: .../20230317_072735.jpg
    Returns: '20230317_0727_{group_id}'
    """
    fn = Path(str(image_path)).name
    match = re.search(r"(\d{8}_\d{4})", fn)
    burst = match.group(1) if match else "noburst"
    clean_group = str(group_id).replace(" ", "_").replace("::", "__").lower()
    return f"{burst}__{clean_group}"


def stratified_group_split(
    df: pd.DataFrame,
    group_col: str,
    strat_col: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partitions df into train, val, test so that all rows with the same group_col
    belong to exactly one partition, while stratifying session groups by strat_col.
    """
    rng = np.random.RandomState(random_state)

    # Table of unique groups with their primary stratification label
    group_meta = (
        df.groupby(group_col)[strat_col]
        .agg(lambda x: x.mode().iloc[0] if not x.empty else "unknown")
        .reset_index()
    )

    train_groups: set[str] = set()
    val_groups: set[str] = set()
    test_groups: set[str] = set()

    # Stratify by each combined label
    for label, grp in group_meta.groupby(strat_col):
        g_list = grp[group_col].tolist()
        rng.shuffle(g_list)
        n = len(g_list)

        if n == 1:
            # Single group for this class (e.g. extremely rare) -> assign to train
            train_groups.update(g_list)
        elif n == 2:
            train_groups.add(g_list[0])
            test_groups.add(g_list[1])
        else:
            n_val = max(1, int(round(n * val_ratio)))
            n_test = max(1, int(round(n * test_ratio)))
            # Ensure at least 1 group remains for train
            if n_val + n_test >= n:
                n_val = max(1, (n - 1) // 2)
                n_test = max(1, n - 1 - n_val)
            n_train = n - n_val - n_test

            train_groups.update(g_list[:n_train])
            val_groups.update(g_list[n_train : n_train + n_val])
            test_groups.update(g_list[n_train + n_val :])

    train_df = df[df[group_col].isin(train_groups)].copy()
    val_df = df[df[group_col].isin(val_groups)].copy()
    test_df = df[df[group_col].isin(test_groups)].copy()

    # Verification: zero group overlap
    assert not (train_groups & val_groups), "Leakage detected between train and val groups!"
    assert not (train_groups & test_groups), "Leakage detected between train and test groups!"
    assert not (val_groups & test_groups), "Leakage detected between val and test groups!"

    return train_df, val_df, test_df


def derive_candidate_ranges(
    train_features_df: pd.DataFrame,
    asana_col: str = "asana_label",
    correctness_col: str = "correctness_label",
) -> dict[str, Any]:
    """
    Computes candidate joint angle ranges from correctly performed (Right) training samples.
    Saves distribution statistics (median, p10, p90, min, max, std, count).
    """
    right_df = train_features_df[train_features_df[correctness_col] == "Right"]
    candidate_ranges: dict[str, Any] = {}

    for asana in ZENODO_ASANAS:
        asana_samples = right_df[right_df[asana_col] == asana]
        n_samples = len(asana_samples)
        candidate_ranges[asana] = {
            "status": "provisional",
            "source": "Zenodo training partition Right samples",
            "sample_count": n_samples,
            "joint_ranges": {},
        }

        if n_samples < 5:
            logger.warning("Fewer than 5 Right training samples for asana '%s' (N=%d)", asana, n_samples)
            continue

        for angle_name in ANGLE_TRIPLETS.keys():
            if angle_name in asana_samples.columns:
                series = asana_samples[angle_name].dropna()
                if len(series) >= 5:
                    p10 = float(np.percentile(series, 10))
                    p90 = float(np.percentile(series, 90))
                    median = float(series.median())
                    std = float(series.std())
                    candidate_ranges[asana]["joint_ranges"][angle_name] = {
                        "min_recommended": round(p10, 1),
                        "max_recommended": round(p90, 1),
                        "median": round(median, 1),
                        "std": round(std, 1),
                        "valid_observations": int(len(series)),
                    }

    return candidate_ranges


def process_dataset(
    csv_path: Path,
    out_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Executes the complete preprocessing pipeline.
    """
    logger.info("Reading dataset from: %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Raw dataset shape: %s", df.shape)

    # 1. Validate required columns
    metadata_cols = ["image_path", "asana_label", "correctness_label", "combined_label", "group_id"]
    for col in metadata_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required metadata column: {col}")

    landmark_cols = [f"f_{i}" for i in range(132)]
    missing_lm_cols = [c for c in landmark_cols if c not in df.columns]
    if missing_lm_cols:
        raise KeyError(f"Missing {len(missing_lm_cols)} landmark columns: {missing_lm_cols[:5]}...")

    # 2. Extract shared 39 biomechanical features
    logger.info("Extracting 39 biomechanical features using runtime feature extractor...")
    lm_matrix = df[landmark_cols].values
    flags = [0] * 33

    feature_records: list[dict[str, float]] = []
    # Infer sample feature keys to establish fixed column order
    sample_lm = landmark_vector_to_mediapipe_dicts(lm_matrix[0])
    sample_feat = extract_features(sample_lm, flags, 1280, 720)
    ordered_feature_cols = list(sample_feat.keys())

    for idx in range(len(df)):
        lm_dicts = landmark_vector_to_mediapipe_dicts(lm_matrix[idx])
        feat_dict = extract_features(lm_dicts, flags, 1280, 720)
        feature_records.append(feat_dict)

    feat_df = pd.DataFrame(feature_records, columns=ordered_feature_cols)
    logger.info("Feature extraction complete. Feature matrix shape: %s", feat_df.shape)

    # 3. Attach metadata & burst session group
    combined_df = pd.concat([df[metadata_cols], feat_df], axis=1)
    combined_df["session_group"] = [
        extract_burst_session(p, g) for p, g in zip(df["image_path"], df["group_id"])
    ]
    logger.info("Constructed %d unique burst sessions.", combined_df["session_group"].nunique())

    # 4. Group-aware stratified splitting
    logger.info("Splitting dataset: Train=%.0f%%, Val=%.0f%%, Test=%.0f%% (Group-aware)...",
                train_ratio * 100, val_ratio * 100, test_ratio * 100)
    train_df, val_df, test_df = stratified_group_split(
        combined_df,
        group_col="session_group",
        strat_col="combined_label",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=seed,
    )

    logger.info("Split results: Train=%d rows, Val=%d rows, Test=%d rows",
                len(train_df), len(val_df), len(test_df))

    # Log class coverage across splits
    logger.info("Asana counts in Train:\n%s", train_df["asana_label"].value_counts().to_dict())
    logger.info("Asana counts in Val:\n%s", val_df["asana_label"].value_counts().to_dict())
    logger.info("Asana counts in Test:\n%s", test_df["asana_label"].value_counts().to_dict())

    # Verify all 10 asanas present in all partitions
    for split_name, s_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        missing = set(ZENODO_ASANAS) - set(s_df["asana_label"].unique())
        if missing:
            logger.warning("Partition '%s' missing asanas: %s", split_name, missing)

    # 5. Fit Preprocessing ONLY on Train partition
    logger.info("Fitting preprocessing objects (imputer, scaler, encoders) on TRAIN partition...")
    X_train_raw = train_df[ordered_feature_cols].copy()
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    imputer.fit(X_train_raw)

    X_train_imp = imputer.transform(X_train_raw)
    scaler = StandardScaler()
    scaler.fit(X_train_imp)

    # Asana LabelEncoder
    asana_le = LabelEncoder()
    asana_le.fit(sorted(ZENODO_ASANAS))

    # Correctness LabelEncoder ("Right" -> 1, "Wrong" -> 0)
    correctness_le = LabelEncoder()
    correctness_le.fit(["Wrong", "Right"])

    # 6. Compute candidate reference ranges from training Right examples
    logger.info("Deriving candidate reference angle ranges from Train partition...")
    candidate_ranges = derive_candidate_ranges(train_df)

    # 7. Save outputs
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train_features.csv"
    val_path = out_dir / "val_features.csv"
    test_path = out_dir / "test_features.csv"
    ranges_path = out_dir / "candidate_reference_ranges.json"
    prep_blueprint_path = out_dir / "zenodo_preprocessing_blueprint.joblib"
    summary_path = out_dir / "split_summary.json"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    with open(ranges_path, "w", encoding="utf-8") as f:
        json.dump(candidate_ranges, f, indent=2)

    blueprint = {
        "imputer": imputer,
        "scaler": scaler,
        "asana_label_encoder": asana_le,
        "correctness_label_encoder": correctness_le,
        "feature_cols": ordered_feature_cols,
        "class_names": asana_le.classes_.tolist(),
        "n_features_in": len(ordered_feature_cols),
    }
    joblib.dump(blueprint, prep_blueprint_path)

    summary: dict[str, Any] = {
        "dataset_name": "Zenodo Yoga Landmarks",
        "total_samples": len(df),
        "feature_count": len(ordered_feature_cols),
        "feature_cols": ordered_feature_cols,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "unique_sessions": combined_df["session_group"].nunique(),
        "train_sessions": train_df["session_group"].nunique(),
        "val_sessions": val_df["session_group"].nunique(),
        "test_sessions": test_df["session_group"].nunique(),
        "asana_classes": ZENODO_ASANAS,
        "train_asana_distribution": train_df["asana_label"].value_counts().to_dict(),
        "val_asana_distribution": val_df["asana_label"].value_counts().to_dict(),
        "test_asana_distribution": test_df["asana_label"].value_counts().to_dict(),
        "train_correctness_distribution": train_df["correctness_label"].value_counts().to_dict(),
        "val_correctness_distribution": val_df["correctness_label"].value_counts().to_dict(),
        "test_correctness_distribution": test_df["correctness_label"].value_counts().to_dict(),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Preprocessing complete! Files saved to: %s", out_dir.resolve())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Zenodo Yoga Landmarks Dataset")
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "zenodo_yoga_landmarks - zenodo_yoga_landmarks.csv",
        help="Path to raw Zenodo CSV file",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "zenodo",
        help="Output directory for preprocessed splits and metadata",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.csv.exists():
        logger.error("Dataset CSV not found at: %s", args.csv)
        sys.exit(1)

    process_dataset(
        csv_path=args.csv,
        out_dir=args.out_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
