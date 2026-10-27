"""
scripts/test_step_by_step_system.py
───────────────────────────────────
End-to-end evaluation of the full Yoga Mitra system across all 10 Zenodo asanas:
  1. Tests both Standing and Sitting/Kneeling yoga families in the dataset.
  2. Performs each yoga step-by-step (Step 1 -> Step 2 -> Step 3 -> ...).
  3. Verifies that the model correctly predicts the asana, base posture,
     and multi-step progression, dynamically pruning invalid candidates.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.model_loader import load_zenodo_artifacts
from src.models.inference import DualTaskInference
from src.correction.calibration import load_profile
from src.features import extract_features, features_to_series
from scripts.prepare_zenodo_dataset import landmark_vector_to_mediapipe_dicts


# Canonical ordered list of Zenodo asanas categorized by posture
YOGA_FAMILIES = {
    "STANDING": [
        "Tadasana",
        "Kati Chakrasana",
        "Ardhakati Chakrasana",
    ],
    "SEATED_KNEELING": [
        "Vajrasana",
        "Parvatasana",
    ],
    "FLOOR_PRONE_INVERTED": [
        "Bhujangasana",
        "Marjariasana",
        "Sarvangasana",
        "Viparita Karani",
        "Anantasana",
    ],
}


def parse_step_number(group_id_str: str) -> int:
    """Extract integer step number from group_id string like 'parvatasana::step 2 hand'."""
    m = re.search(r"step\s*(\d+)", str(group_id_str).lower())
    return int(m.group(1)) if m else 1


def run_step_by_step_evaluation(classifier: str = "rf", split: str = "test"):
    print("=" * 80)
    print("  YOGA MITRA — END-TO-END STEP-BY-STEP SYSTEM EVALUATION")
    print("=" * 80)

    # 1. Load Artifacts & Inferrer
    model_dir = ROOT / "saved_yoga_model_zenodo"
    print(f"\n[1] Loading Zenodo Dual-Task Model ({classifier}) from: {model_dir}")
    artifacts = load_zenodo_artifacts(model_dir, active_classifier=classifier)
    import yaml
    from src.models.inference import FilterPolicy
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    fcfg = dict(cfg["inference"].get("filters") or {})
    fcfg.update((cfg["inference"].get("filters_by_classifier") or {}).get(classifier) or {})
    policy = FilterPolicy.from_config(fcfg)
    inferrer = DualTaskInference(artifacts=artifacts, confidence_threshold=0.30, filter_policy=policy)
    print(f"    Classifier: {artifacts.model_type} | filters: {policy.mode}/{policy.granularity}")
    print(f"    Classes ({len(artifacts.class_names)}): {artifacts.class_names}")

    # 2. Load User Profile (Calibrated)
    profile_path = ROOT / "data" / "user_profile.json"
    user_profile = load_profile(profile_path)
    print(f"    User Profile: Calibrated={user_profile.calibrated}, "
          f"thigh_ratio={user_profile.thigh_to_torso_ratio:.3f}, leg_ratio={user_profile.leg_to_torso_ratio:.3f}")

    # 3. Load Dataset
    data_csv = ROOT / "zenodo_yoga_landmarks - zenodo_yoga_landmarks.csv"
    print(f"\n[2] Loading dataset from: {data_csv}")
    df = pd.read_csv(data_csv)
    if split != "all":
        keep = pd.read_csv(ROOT / "data" / "zenodo" / f"{split}_features.csv", usecols=["image_path"])["image_path"]
        df = df[df["image_path"].isin(set(keep))].sort_values("image_path").reset_index(drop=True)
    print(f"    Samples in '{split}' split: {len(df)} (held-out evaluation only when split=test)")

    f_cols = [f"f_{i}" for i in range(132)]

    overall_results: dict[str, Any] = {}
    total_frames_evaluated = 0
    total_correct_predictions = 0

    # 4. Iterate over each family and each yoga
    for family_name, asanas in YOGA_FAMILIES.items():
        print("\n" + "#" * 80)
        print(f"  POSTURE FAMILY: {family_name}")
        print("#" * 80)

        for asana in asanas:
            print("\n" + "-" * 78)
            print(f"  >> ASANA: {asana.upper()}")
            print("-" * 78)

            asana_df = df[df["asana_label"] == asana].copy()
            asana_df["step_num"] = asana_df["group_id"].apply(parse_step_number)

            available_steps = sorted(asana_df["step_num"].unique())
            print(f"    Available Steps in Dataset: {available_steps}")

            # Reset inferrer for this yoga session (clean state progression)
            inferrer.reset_history()

            asana_summary = {
                "family": family_name,
                "steps": {},
                "total_tested": 0,
                "total_correct": 0,
            }

            # Evaluate each step in temporal order (Step 1 -> Step 2 -> ...)
            for step_num in available_steps:
                step_df = asana_df[asana_df["step_num"] == step_num]
                # Prioritize Right form samples
                right_samples = step_df[step_df["correctness_label"] == "Right"]
                eval_subset = right_samples if len(right_samples) >= 5 else step_df

                # Take up to 25 representative frames per step
                sample_frames = eval_subset.iloc[:25]

                step_correct = 0
                step_total = len(sample_frames)
                top_predicted_poses: dict[str, int] = defaultdict(int)
                detected_bases: dict[str, int] = defaultdict(int)
                detected_steps: dict[int, int] = defaultdict(int)
                confidences: list[float] = []
                form_correct_count = 0

                # Sample for detailed trace display
                first_pred_info: dict[str, Any] = {}

                for idx, (_, row) in enumerate(sample_frames.iterrows()):
                    row_vals = row[f_cols].values.astype(float)
                    lm = landmark_vector_to_mediapipe_dicts(row_vals)
                    flags = [0] * 33
                    feat_dict = extract_features(lm, flags, 1280, 720)
                    feat_s = features_to_series(feat_dict, artifacts.feature_cols)

                    pred = inferrer.predict(feat_s, landmarks=lm, user_profile=user_profile)
                    seq = pred.sequence_state

                    is_correct = (pred.pose_name == asana)
                    # In Steps 1 & 2 of Vajrasana/Parvatasana, sharing base posture is expected
                    is_valid_candidate = asana in (seq.active_candidates if seq else [pred.pose_name])

                    if is_correct:
                        step_correct += 1

                    top_predicted_poses[pred.pose_name] += 1
                    confidences.append(pred.confidence)
                    if pred.is_form_correct:
                        form_correct_count += 1

                    if seq:
                        detected_bases[seq.base_posture] += 1
                        detected_steps[seq.step] += 1

                    if idx == 0:
                        first_pred_info = {
                            "pred_pose": pred.pose_name,
                            "confidence": pred.confidence,
                            "form": pred.correctness_label,
                            "base_posture": seq.base_posture if seq else "N/A",
                            "step_num": seq.step if seq else 1,
                            "step_name": seq.step_name if seq else "N/A",
                            "active_candidates": seq.active_candidates if seq else [],
                            "pruned_count": len(seq.pruned_poses) if seq else 0,
                        }

                step_acc = (step_correct / step_total) * 100.0 if step_total > 0 else 0.0
                mean_conf = float(np.mean(confidences)) * 100.0 if confidences else 0.0
                dominant_pred = max(top_predicted_poses.items(), key=lambda x: x[1])[0] if top_predicted_poses else "None"
                dominant_base = max(detected_bases.items(), key=lambda x: x[1])[0] if detected_bases else "UNKNOWN"
                status_icon = "OK" if step_acc >= 70.0 or asana in first_pred_info.get("active_candidates", []) else "WARN"

                print(f"    Step {step_num:d} [{status_icon:4s}]: Evaluated {step_total:2d} frames | "
                      f"Acc: {step_acc:5.1f}% | Top Pred: {dominant_pred:20s} (conf: {mean_conf:4.1f}%) | "
                      f"Base: {dominant_base:15s}")
                print(f"            Trace: Predicted Step {first_pred_info.get('step_num')} ({first_pred_info.get('step_name')}) | "
                      f"Active: {first_pred_info.get('active_candidates')[:3]} | Pruned: {first_pred_info.get('pruned_count')} poses")

                asana_summary["steps"][f"step_{step_num}"] = {
                    "frames_tested": step_total,
                    "accuracy_pct": round(step_acc, 1),
                    "mean_confidence_pct": round(mean_conf, 1),
                    "dominant_prediction": dominant_pred,
                    "dominant_base": dominant_base,
                    "active_candidates": first_pred_info.get("active_candidates", []),
                }

                asana_summary["total_tested"] += step_total
                asana_summary["total_correct"] += step_correct
                total_frames_evaluated += step_total
                total_correct_predictions += step_correct

            overall_asana_acc = (asana_summary["total_correct"] / asana_summary["total_tested"]) * 100.0 if asana_summary["total_tested"] > 0 else 0.0
            asana_summary["overall_accuracy_pct"] = round(overall_asana_acc, 1)
            overall_results[asana] = asana_summary
            print(f"    --> {asana} OVERALL ACCURACY: {overall_asana_acc:.1f}% ({asana_summary['total_correct']}/{asana_summary['total_tested']})\n")

    # 5. Grand Summary Table
    print("\n" + "=" * 80)
    print("  GRAND SUMMARY: MODEL & SEQUENCE TRACKER PERFORMANCE ACROSS ALL YOGAS")
    print("=" * 80)
    print(f"  {'Asana':24s} | {'Family':16s} | {'Steps':6s} | {'Frames':8s} | {'Accuracy':10s} | Status")
    print("  " + "-" * 76)

    for asana, res in overall_results.items():
        fam = res["family"]
        n_steps = len(res["steps"])
        n_frames = res["total_tested"]
        acc = res["overall_accuracy_pct"]
        status = "PASSED" if acc >= 75.0 else ("GOOD" if acc >= 60.0 else "REVIEW")
        print(f"  {asana:24s} | {fam:16s} | {n_steps:6d} | {n_frames:8d} | {acc:9.1f}% | {status}")

    grand_accuracy = (total_correct_predictions / total_frames_evaluated) * 100.0 if total_frames_evaluated > 0 else 0.0
    print("  " + "-" * 76)
    print(f"  TOTAL SYSTEM PERFORMANCE: {grand_accuracy:.1f}% Top-1 Accuracy across {total_frames_evaluated} multi-step frames\n")

    # Save structured report
    report_path = ROOT / "data" / "step_by_step_test_report.json"
    with open(report_path, "w") as f:
        json.dump({"classifier": artifacts.model_type, "split": split,
                   "filter_policy": policy.__dict__, "grand_accuracy_pct": round(grand_accuracy, 1),
                   "frames": total_frames_evaluated, "asanas": overall_results}, f, indent=2)
    print(f"[App] Detailed JSON report saved to: {report_path}\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", choices=["rf", "mlp"], default="rf")
    ap.add_argument("--split", choices=["test", "val", "all"], default="test")
    a = ap.parse_args()
    run_step_by_step_evaluation(a.classifier, a.split)
