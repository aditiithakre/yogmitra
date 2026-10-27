"""
scripts/evaluate_runtime_pipeline.py
────────────────────────────────────
Reproducible evaluation of the *runtime* Zenodo pipeline (not just the bare
classifier) on held-out data, using the real ``DualTaskInference.predict``:

  features (training-consistent, from stored MediaPipe landmarks)
    → classifier (RF or MLP)
    → candidate filters (none / prior / sequence / both / safe)
    → EMA smoothing → confidence status

Protocols
  • session : frames of each held-out session in temporal order; history reset at
              every session start (user holding one asana).
  • stream  : all held-out sessions concatenated in a seeded random order with
              no reset (user moving between asanas) → measures recovery after
              transitions and wrongly persisting restrictions.

Ablation (FIX 4), confidence gating (FIX 3) and missing-landmark
representation (FIX 5) experiments are all written to one JSON report.
Thresholds are *selected on the validation split* and *reported on test*.

Usage:
    python scripts/evaluate_runtime_pipeline.py                 # val + test, rf + mlp
    python scripts/evaluate_runtime_pipeline.py --quick         # subsample for smoke runs
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import extract_features, features_to_series  # noqa: E402
from src.models.inference import DualTaskInference, FilterPolicy  # noqa: E402
from src.models.model_loader import load_zenodo_artifacts  # noqa: E402
from src.models.sequence_tracker import BASE_ORIENTATION_MAP  # noqa: E402
from src.pose.missing_landmarks import LandmarkHistory  # noqa: E402

RAW_CSV = ROOT / "zenodo_yoga_landmarks - zenodo_yoga_landmarks.csv"
SPLIT_DIR = ROOT / "data" / "zenodo"

FILTER_CONFIGS: dict[str, FilterPolicy] = {
    "no_filters": FilterPolicy(use_gravity_prior=False, use_sequence_pruning=False),
    "prior_only_legacy": FilterPolicy(use_gravity_prior=True, use_sequence_pruning=False),
    "sequence_only_legacy": FilterPolicy(use_gravity_prior=False, use_sequence_pruning=True),
    "combined_legacy": FilterPolicy(use_gravity_prior=True, use_sequence_pruning=True),
}
# Safe-policy grid: tuned on the validation split only.
for _gran in ("coarse", "family"):
    for _floor in (0.05, 0.2, 0.5):
        for _stable in (1, 3):
            FILTER_CONFIGS[f"safe_{_gran}_floor{_floor}_stable{_stable}"] = FilterPolicy(
                mode="safe", granularity=_gran, soft_floor=_floor, stable_frames=_stable)


class CachedModel:
    """Wraps a fitted sklearn model; predict_proba is served from a per-row cache."""

    def __init__(self, model, X: np.ndarray):
        self._model = model
        self._cache = {row.tobytes(): p for row, p in zip(X, model.predict_proba(X))}
        self.classes_ = model.classes_

    def predict_proba(self, X):
        out = []
        for row in np.asarray(X, dtype=float):
            p = self._cache.get(row.tobytes())
            out.append(p if p is not None else self._model.predict_proba(row[None, :])[0])
        return np.vstack(out)


def to_landmarks(row: np.ndarray) -> list[dict]:
    return [{"x": float(row[4 * i]), "y": float(row[4 * i + 1]), "z": float(row[4 * i + 2]),
             "visibility": float(row[4 * i + 3])} for i in range(33)]


def load_split(name: str, raw: pd.DataFrame, limit_sessions: int | None) -> pd.DataFrame:
    split = pd.read_csv(SPLIT_DIR / f"{name}_features.csv", usecols=["image_path", "asana_label", "session_group"])
    df = split.merge(raw, on="image_path", how="left", suffixes=("", "_raw"))
    df = df.sort_values(["session_group", "image_path"]).reset_index(drop=True)
    if limit_sessions:
        keep = df["session_group"].drop_duplicates().sample(limit_sessions, random_state=0)
        df = df[df["session_group"].isin(keep)].reset_index(drop=True)
    return df


def build_inputs(df: pd.DataFrame, feature_cols: list[str]) -> tuple[list, list]:
    f_cols = [f"f_{i}" for i in range(132)]
    lms, feats = [], []
    for row in df[f_cols].to_numpy(dtype=float):
        lm = to_landmarks(row)
        lms.append(lm)
        feats.append(features_to_series(extract_features(lm, [0] * 33, 1280, 720), feature_cols))
    return lms, feats


def metrics(y_true: list[str], y_pred: list[str], classes: list[str]) -> dict[str, Any]:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)), 4),
        "per_class": {c: {"precision": round(float(p[i]), 3), "recall": round(float(r[i]), 3),
                          "f1": round(float(f[i]), 3), "support": int(s[i])} for i, c in enumerate(classes)},
    }


def run_protocol(inferrer: DualTaskInference, df: pd.DataFrame, lms: list, feats: list,
                 order: list[int], reset_on_session: bool) -> list[dict]:
    rows = []
    prev_session = None
    for i in order:
        sess = df.at[i, "session_group"]
        if reset_on_session and sess != prev_session:
            inferrer.reset_history()
        pred = inferrer.predict(feats[i], landmarks=lms[i], user_profile=None)
        raw_top = inferrer._art.class_names[int(np.argmax(pred.raw_probabilities))]
        rows.append({
            "idx": i, "session": sess, "truth": df.at[i, "asana_label"],
            "pred": pred.pose_name, "conf": pred.confidence, "status": pred.status,
            "raw_top": raw_top,
            "truth_suppressed": df.at[i, "asana_label"] in (pred.filter_decision.suppressed or []),
            "filtered_top": inferrer._art.class_names[int(np.argmax(
                pred.probabilities if pred.probabilities is not None else pred.raw_probabilities))],
            "skipped": pred.filter_decision.skipped_reason,
            "base": pred.filter_decision.base_posture,
            "coarse": pred.filter_decision.coarse_group,
        })
        prev_session = sess
    return rows


def summarize(rows: list[dict], classes: list[str]) -> dict[str, Any]:
    y = [r["truth"] for r in rows]
    out = metrics(y, [r["pred"] for r in rows], classes)
    out["false_suppression_rate"] = round(float(np.mean([r["truth_suppressed"] for r in rows])), 4)
    out["harmful_filter_frames"] = int(sum(r["raw_top"] == r["truth"] and r["pred"] != r["truth"] and r["truth_suppressed"] for r in rows))
    out["helpful_filter_frames"] = int(sum(r["raw_top"] != r["truth"] and r["pred"] == r["truth"] for r in rows))
    out["filter_skipped_rate"] = round(float(np.mean([r["skipped"] is not None for r in rows])), 4)
    # Orientation correctness of the base-posture detector (for classes with a known family)
    fam = [(BASE_ORIENTATION_MAP.get(r["truth"]), r["base"]) for r in rows]
    known = [(t, b) for t, b in fam if t and b != "UNKNOWN"]
    out["base_posture_agreement"] = round(float(np.mean([t == b for t, b in known])), 4) if known else None
    out["base_posture_unknown_rate"] = round(float(np.mean([b == "UNKNOWN" for _, b in fam])), 4)
    return out


def transition_stats(rows: list[dict]) -> dict[str, Any]:
    """Frames needed after an asana change until the prediction first matches."""
    delays, never = [], 0
    i = 0
    while i < len(rows):
        j = i
        while j < len(rows) and rows[j]["session"] == rows[i]["session"]:
            j += 1
        if i > 0 and rows[i - 1]["truth"] != rows[i]["truth"]:
            hit = next((k - i for k in range(i, j) if rows[k]["pred"] == rows[k]["truth"]), None)
            if hit is None:
                never += 1
            else:
                delays.append(hit)
        i = j
    n = len(delays) + never
    return {
        "asana_changes": n,
        "mean_frames_to_recover": round(float(np.mean(delays)), 2) if delays else None,
        "never_recovered_sessions": never,
        "never_recovered_rate": round(never / n, 4) if n else None,
    }


def gating_table(rows: list[dict], thresholds: list[float]) -> list[dict]:
    out = []
    for t in thresholds:
        sel = [r for r in rows if r["conf"] >= t]
        out.append({
            "threshold": t,
            "coverage": round(len(sel) / max(1, len(rows)), 4),
            "selective_accuracy": round(float(np.mean([r["pred"] == r["truth"] for r in sel])), 4) if sel else None,
            "wrong_label_issued_rate": round(sum(r["pred"] != r["truth"] for r in sel) / max(1, len(rows)), 4),
        })
    return out


def missing_landmark_experiment(inferrer: DualTaskInference, df: pd.DataFrame, lms: list, feature_cols: list[str],
                                rng_seed: int = 0) -> dict[str, Any]:
    """
    Synthetic occlusion stress test (no filters):
      a chain of joints is hidden for a burst of frames inside each session and the
      classifier receives either
        • observed_only : training-consistent representation (hidden joints → NaN, flags 0)
        • carry_forward : last observed position carried forward with imputed flags = 1
                          (the representation the old app fed to the classifier)
        • carry_forward_flags_zeroed : carried positions, flags 0 (as in Zenodo training data)
    """
    chains = {"left_leg": (25, 27, 29, 31), "right_arm": (14, 16), "both_legs": (25, 26, 27, 28, 29, 30, 31, 32)}
    rng = np.random.default_rng(rng_seed)
    res: dict[str, Any] = {}
    for chain_name, ids in chains.items():
        correct = {"observed_only": [], "carry_forward": [], "carry_forward_flags_zeroed": [], "unoccluded_reference": []}
        for _sess, g in df.groupby("session_group", sort=False):
            idxs = list(g.index)
            if len(idxs) < 4:
                continue
            start = int(rng.integers(1, len(idxs) - 2))
            burst = set(idxs[start:start + 3])
            hist = LandmarkHistory(max_age_ms=10_000, decay_drop=0)  # carry within the burst
            for k, i in enumerate(idxs):
                lm = lms[i]
                if i in burst:
                    lm = [dict(p, visibility=0.02) if j in ids else p for j, p in enumerate(lm)]
                fr = hist.update(lm, k * 33.0)
                if i not in burst:
                    continue
                truth = df.at[i, "asana_label"]
                for mode in correct:
                    if mode == "observed_only":
                        f = extract_features(fr.observed_only, [0] * 33, 1280, 720)
                    elif mode == "carry_forward":
                        f = extract_features(fr.effective, fr.imputed_flags, 1280, 720)
                    elif mode == "carry_forward_flags_zeroed":
                        f = extract_features(fr.effective, [0] * 33, 1280, 720)
                    else:
                        f = extract_features(lms[i], [0] * 33, 1280, 720)
                    inferrer.reset_history()
                    p = inferrer.predict(features_to_series(f, feature_cols), landmarks=None)
                    correct[mode].append(p.pose_name == truth)
        res[chain_name] = {m: round(float(np.mean(v)), 4) for m, v in correct.items() if v}
        res[chain_name]["frames"] = len(correct["observed_only"])
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classifiers", nargs="+", default=["rf", "mlp"])
    ap.add_argument("--quick", action="store_true", help="evaluate a 15-session subsample per split")
    ap.add_argument("--only-missing", action="store_true", help="run only the missing-landmark experiment")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "runtime_pipeline_evaluation.json")
    args = ap.parse_args()

    t_start = time.time()
    raw = pd.read_csv(RAW_CSV)
    raw = raw.drop(columns=[c for c in ("asana_label", "correctness_label", "combined_label", "group_id") if c in raw])
    limit = 15 if args.quick else None
    splits = {s: load_split(s, raw, limit) for s in ("val", "test")}

    report: dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "Zenodo yoga landmarks (stored MediaPipe landmarks; no images)",
        "splits": {s: {"frames": len(d), "sessions": int(d["session_group"].nunique())} for s, d in splits.items()},
        "feature_representation": "training-consistent: visibility<0.30 joints → NaN, imputed flags 0",
        "notes": [
            "session protocol resets inference history per held-out session; stream protocol never resets",
            "consecutive Zenodo frames are ~1 s burst photos, not 30 fps video",
            "thresholds/policies are chosen on 'val' and reported on 'test'",
            "user_profile=None (uncalibrated) for all runs",
        ],
        "classifiers": {},
    }

    for clf in args.classifiers:
        art = load_zenodo_artifacts(ROOT / "saved_yoga_model_zenodo", active_classifier=clf)
        classes = art.class_names
        clf_rep: dict[str, Any] = {"artifact_dir": f"saved_yoga_model_zenodo/{clf}", "model_type": art.model_type}
        for split_name, df in splits.items():
            lms, feats = build_inputs(df, art.feature_cols)
            X = pd.DataFrame([f.values for f in feats], columns=art.feature_cols)
            X_imp = art.imputer.transform(X)
            X_eval = art.scaler.transform(X_imp) if art.model_type == "mlp" else X_imp
            art.asana_model = CachedModel(art.asana_model, X_eval)
            art.verifier_model = CachedModel(art.verifier_model, X_eval)

            split_rep: dict[str, Any] = {}
            bare = [classes[int(k)] for k in np.argmax(art.asana_model.predict_proba(X_eval), axis=1)]
            split_rep["bare_classifier_single_frame"] = metrics(list(df["asana_label"]), bare, classes)
            split_rep["bare_classifier_single_frame"]["confusion_matrix"] = {
                "labels": classes, "matrix": confusion_matrix(df["asana_label"], bare, labels=classes).tolist()}

            sess_order = list(range(len(df)))
            sessions = df["session_group"].drop_duplicates().sample(frac=1.0, random_state=0).tolist()
            pos = {s: k for k, s in enumerate(sessions)}
            stream_order = sorted(range(len(df)), key=lambda i: (pos[df.at[i, "session_group"]], i))

            for cfg_name, policy in ({} if args.only_missing else FILTER_CONFIGS).items():
                inf = DualTaskInference(art, confidence_threshold=0.28, filter_policy=policy)
                r_sess = run_protocol(inf, df, lms, feats, sess_order, reset_on_session=True)
                inf.reset_history()
                r_stream = run_protocol(inf, df, lms, feats, stream_order, reset_on_session=False)
                split_rep[cfg_name] = {
                    "session": summarize(r_sess, classes),
                    "stream": {**summarize(r_stream, classes), "transitions": transition_stats(r_stream)},
                    "confidence_gating_session": gating_table(r_sess, [0.28, 0.40, 0.50, 0.60, 0.70]),
                }
                print(f"[{clf}/{split_name}] {cfg_name:22s} session acc={split_rep[cfg_name]['session']['accuracy']:.3f} "
                      f"F1={split_rep[cfg_name]['session']['macro_f1']:.3f} | stream acc={split_rep[cfg_name]['stream']['accuracy']:.3f} "
                      f"F1={split_rep[cfg_name]['stream']['macro_f1']:.3f} falseSupp={split_rep[cfg_name]['stream']['false_suppression_rate']:.3f}")

            inf = DualTaskInference(art, confidence_threshold=0.28, filter_policy=FILTER_CONFIGS["no_filters"])
            split_rep["missing_landmark_representation"] = missing_landmark_experiment(inf, df, lms, art.feature_cols)
            print(f"[{clf}/{split_name}] missing-landmark: {split_rep['missing_landmark_representation']}")
            art.asana_model = art.asana_model._model
            art.verifier_model = art.verifier_model._model
            clf_rep[split_name] = split_rep
        report["classifiers"][clf] = clf_rep

    # ── Policy selection on validation (stream macro-F1), confirmed on test ──
    for clf, rep in report["classifiers"].items():
        val = rep.get("val", {})
        cands = {k: v["stream"]["macro_f1"] for k, v in val.items() if k.startswith("safe_")}
        if cands:
            best = max(cands, key=cands.get)
            base = val["no_filters"]["stream"]["macro_f1"]
            rep["policy_selection"] = {
                "criterion": "validation stream macro-F1",
                "best_safe_policy": best,
                "best_safe_val_macro_f1": cands[best],
                "no_filters_val_macro_f1": base,
                "safe_beats_no_filters_on_val": cands[best] > base,
                "test_macro_f1": {k: rep["test"][k]["stream"]["macro_f1"] for k in ("no_filters", "combined_legacy", best)},
                "test_accuracy": {k: rep["test"][k]["stream"]["accuracy"] for k in ("no_filters", "combined_legacy", best)},
            }
            print(f"[{clf}] selection:", json.dumps(rep["policy_selection"]))

    report["runtime_seconds"] = round(time.time() - t_start, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
