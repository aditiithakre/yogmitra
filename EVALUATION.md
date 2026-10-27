# Yoga Mitra — Evaluation Report

Every number below was produced by a script in `scripts/` and is stored in the JSON file named next to it.
Three kinds of evidence are kept separate:

| Label | Meaning |
|---|---|
| **MEASURED** | Real model predictions on held-out data (sessions never used for training or tuning) |
| **SYNTHETIC** | Stress tests built from real data with an artificial perturbation (occlusion, shifting, noise) |
| **QUALITATIVE** | Replays or spot checks of the live app on a few clips; illustrative, not statistics |

What these metrics are **not**:
- **Not landmark-estimation accuracy.** Neither dataset has annotated keypoints, so MediaPipe's own accuracy cannot be measured. Only its stability and coverage are reported.
- **Not live-webcam accuracy.** The Zenodo data are burst photographs (about 1 frame per second), not 30 fps video, and no labelled webcam recordings were available.

Build machine for the measurements in this revision: cloud Linux VM, CPU only, Python 3.12, scikit-learn 1.9.0, MediaPipe 1.1.1.
Latency numbers are machine-specific.

## Reproduce

```bash
python scripts/evaluate_zenodo_models.py        # → data/zenodo_evaluation_report.json   (bare classifiers, RF and MLP)
python scripts/evaluate_runtime_pipeline.py     # → data/runtime_pipeline_evaluation.json (filters, smoothing, transitions)
python scripts/calibrate_confidence.py          # → data/zenodo/confidence_calibration.json (thresholds chosen on val)
python scripts/evaluate_system.py               # → data/evaluation_metrics.json          (Kaggle 47-class)
python scripts/test_step_by_step_system.py      # → data/step_by_step_test_report.json    (held-out test, per step)
python scripts/benchmark_mediapipe_modes.py --video clip.mp4 --still person.jpg  # → data/mediapipe_mode_comparison.json
python scripts/evaluate_jev_images.py --images-dir labelled_images/ [--jev]      # → data/jev_image_evaluation.json
python -m pytest tests/ -q
```

---

## 1. Data and splits (Zenodo profile)

- **Source:** 11,118 stored MediaPipe landmark vectors (`zenodo_yoga_landmarks - zenodo_yoga_landmarks.csv`) covering 10 asanas, each labelled Right or Wrong form.
- **Split:** group-aware by recording-burst session (329 sessions), so no session appears in more than one split.
  - Train: 7,993 frames
  - Val: 1,556 frames / 50 sessions
  - Test: 1,569 frames / 50 sessions
- **How each split was used:**
  - **train** fitted the models (original training run; not retrained here).
  - **val** chose the RF-vs-MLP champion (original run), and in this revision the confidence thresholds and the filter policy.
  - **test** is used only for the numbers reported below.

## 2. Bare classifiers on the test split — MEASURED
`data/zenodo_evaluation_report.json` · single frame, no filters or smoothing

The app runs the classifier named in `config.yaml` (`model.active_classifier: rf`) from `saved_yoga_model_zenodo/rf/`.
The un-suffixed `saved_yoga_model_zenodo/` copy is the **MLP** "champion". It is byte-identical to `mlp/` and is *not* what the app runs.

| Model A (10 asanas) | RF (`rf/`, active) | MLP (`mlp/`) |
|---|---|---|
| Accuracy | 0.7597 | 0.7693 |
| Macro F1 | 0.6278 | 0.6246 |
| Weighted F1 | 0.7628 | 0.7637 |

Per-class precision / recall / F1 (support):

| Asana | RF P / R / F1 | MLP P / R / F1 | n |
|---|---|---|---|
| Anantasana | 0.52 / 0.28 / 0.36 | 0.55 / 0.38 / 0.45 | 141 |
| Ardhakati Chakrasana | 0.90 / 0.82 / 0.86 | 0.90 / 0.87 / 0.88 | 114 |
| Bhujangasana | 0.94 / 0.84 / 0.89 | 0.89 / 0.74 / 0.81 | 133 |
| Kati Chakrasana | 0.94 / 0.83 / 0.88 | 0.80 / 0.80 / 0.80 | 87 |
| Marjariasana | 0.92 / 1.00 / 0.96 | 0.87 / 1.00 / 0.93 | 218 |
| Parvatasana | 0.92 / 0.96 / 0.94 | 0.91 / 0.97 / 0.94 | 417 |
| **Sarvangasana** | **0.00 / 0.00 / 0.00** | **0.01 / 0.01 / 0.01** | 74 |
| Tadasana | 0.93 / 0.97 / 0.95 | 0.93 / 0.93 / 0.93 | 231 |
| **Vajrasana** | 0.43 / **0.17** / 0.24 | 0.33 / **0.08** / 0.13 | 36 |
| **Viparita Karani** | 0.18 / **0.22** / 0.20 | 0.34 / **0.37** / 0.36 | 118 |

Labelled confusion matrices are in the JSON. Most Sarvangasana frames are predicted as Viparita Karani, and Vajrasana as Parvatasana.

**Model B (form verifier, positive class = "Wrong")**

| Metric | RF | MLP |
|---|---|---|
| Accuracy | 0.7425 | 0.7259 |
| Balanced accuracy | 0.7434 | 0.7292 |
| Precision | 0.7758 | 0.7776 |
| Recall | 0.7316 | 0.6853 |
| F1 | 0.7531 | 0.7285 |
| ROC-AUC | 0.8393 | 0.8054 |

The form verifier is shown only as an informational "Form: RIGHT/WRONG" tag. It never drives joint corrections; those come from the reference-range evaluator.

## 3. Gravitational prior and sequence pruning ablation — MEASURED
`data/runtime_pipeline_evaluation.json` · real `DualTaskInference.predict` + EMA smoothing over held-out sessions

- **session** protocol: history is reset for each session (the user holds one asana).
- **stream** protocol: all sessions are concatenated in a seeded random order with no reset (the user moves between asanas).

Test split, stream protocol. "False suppression" means the true class was down-weighted by the filters.

| Configuration | RF acc | RF macro F1 | RF false suppression | MLP acc | MLP macro F1 | MLP false suppression |
|---|---|---|---|---|---|---|
| No filters | 0.772 | 0.629 | 0 % | 0.781 | 0.633 | 0 % |
| Prior only (legacy ×0.005) | 0.718 | 0.587 | 12.8 % | **0.804** | **0.656** | 5.0 % |
| Sequence pruning only (legacy ×0.001) | 0.522 | 0.384 | 28.1 % | 0.715 | 0.567 | 6.8 % |
| **Prior + sequence (shipped before this revision)** | **0.535** | **0.398** | **31.9 %** | 0.710 | 0.570 | 13.1 % |
| Safe / coarse (new; reliability-gated, soft) | 0.772 | 0.629 | 0.8 % | 0.782 | 0.633 | 0.5 % |

**Shipped legacy filters, RF per-class recall on test (no filters → legacy combined):**
- Tadasana: 0.99 → **0.28**
- Bhujangasana: 0.87 → **0.37**
- Marjariasana: 0.995 → **0.54**

On RF test frames, the filters turned 347 correct classifier outputs into wrong ones and fixed 43.

**Root cause.** The rule-based orientation detector reads upright poses well:
- STANDING is detected 95–100 % of the time for the standing asanas.
- SEATED is detected 74–88 % of the time for the seated asanas.

It cannot separate the floor families from each other:
- Bhujangasana is classed INVERTED 43 % of the time.
- Sarvangasana is classed PRONE 41 % of the time.
- Marjariasana is classed INVERTED 32 % of the time.

Near-zero multipliers then erase the correct class.

**Transitions (stream protocol, 44 asana changes).** Mean frames to recover after a change:

| | No filters | Legacy combined | Safe |
|---|---|---|---|
| RF | 0.76 | 2.23 | 0.76 |
| MLP | 1.49 | 0.64 | 1.49 |

**Policy now deployed.** It was chosen on **val** by `scripts/calibrate_confidence.py` and confirmed on test above.
- **RF:** `safe` / `coarse`. Orientation evidence is used only when torso and legs are directly observed with visibility ≥ 0.5 and the orientation has been stable for 3 inferences. It is used only at the upright-vs-floor level, as a single ×0.5 soft weight. A class the classifier believes at ≥ 0.6 is never suppressed. Measured as neutral; legacy was strongly harmful.
- **MLP:** legacy **prior only**, because the evidence supports it (val macro F1 0.594 → 0.641, test 0.633 → 0.656). Sequence pruning stays off for both classifiers.

## 4. Confidence gating for auto-detect — MEASURED
`data/zenodo/confidence_calibration.json`

The previous threshold, 0.28, accepted almost every frame. On test, a wrong label was presented on **20.8 % (RF) / 19.4 % (MLP)** of the frames it accepted, and the app used that label for highlighting, scoring and progress logging.

Thresholds were chosen on **val**: the lowest value whose selective accuracy is ≥ 0.80 with coverage ≥ 0.5.
Classes whose val precision at that threshold is below 0.5 are not auto-assessed.

| Test split (session protocol) | RF | MLP |
|---|---|---|
| Threshold (chosen on val) | 0.50 | 0.50 |
| Classes not auto-assessed (val precision < 0.5) | Sarvangasana, Vajrasana, Viparita Karani | the same |
| Coverage / accuracy of issued labels at threshold | 0.821 / 0.836 | 0.945 / 0.840 |
| + class gate | 0.709 / **0.953** | 0.875 / **0.881** |
| + orientation veto | 0.699 / 0.953 | 0.859 / 0.880 |
| Correct labels blocked by the orientation veto | 1.5 % | 1.9 % |

Guided mode is unaffected: it always assesses the user's chosen target, limited only by landmark reliability.

## 5. Missing-landmark representation — SYNTHETIC
`data/missing_landmark_representation.json`

For 3 consecutive frames in each held-out session, a joint chain was hidden, and the classifier received one of three inputs:
- **(a)** the hidden joints as NaN, the same representation used in training;
- **(b)** the last observation carried forward, with imputed flags set to 1;
- **(c)** carried forward, with flags set to 0.

| Test accuracy | Hidden: left leg | Hidden: right arm | Hidden: both legs |
|---|---|---|---|
| RF, NaN | 0.430 | 0.548 | 0.296 |
| RF, carry-forward | **0.541** | **0.556** | **0.659** |
| MLP, NaN | 0.489 | 0.541 | 0.304 |
| MLP, carry-forward | **0.578** | **0.578** | **0.585** |
| Unoccluded reference (RF / MLP) | 0.615 / 0.667 | 0.644 / 0.689 | 0.682 / 0.696 |

Two findings:
- Short-term carry-forward clearly helps, so it is kept, now bounded by `landmark_history.max_age_ms` (default 300 ms).
- The Zenodo models were trained with every `*_imputed` flag equal to 0. Setting the flags to 1 or 0 gave **identical** results for both models, so the app zeroes them for the Zenodo profile to match training. The Kaggle models were trained with flags, so they are kept there.

## 6. Occlusion stress test — SYNTHETIC
`data/zenodo_evaluation_report.json → occlusion_stress_test`

k of the 16 body landmarks were hidden at random on all 1,569 real held-out frames. RF classifier; the posture evaluator was checked against the ground-truth asana.

| Hidden landmarks | Classifier accuracy | Evaluator "low visibility" | Corrections issued on a hidden joint |
|---|---|---|---|
| 0 / 16 | 0.760 | 0.212 (side views hide joints natively) | 0 |
| 2 / 16 | 0.512 | 0.637 | 0 |
| 4 / 16 | 0.381 | 0.912 | 0 |
| 8 / 16 | 0.280 | 0.999 | 0 |

The joint-level guarantee holds: no correction was ever issued for an angle whose joints were hidden.
Classification degrades quickly when joints are lost without any history, which is why bounded carry-forward (Section 5) matters.

## 7. MediaPipe IMAGE vs VIDEO mode — SYNTHETIC / proxy
`data/mediapipe_mode_comparison.json`

Both modes ran on the same frames. No yoga video with ground-truth keypoints was available, so two proxies were used:
- OpenCV's public pedestrian clip `vtest.avi` (small, distant people);
- MediaPipe's `pose.jpg` (Warrior II) and `business-person.png`, with a random-walk shift/scale, sensor noise and a 15-frame occlusion of the lower 30 %.

| | IMAGE (old) | VIDEO (new) |
|---|---|---|
| vtest.avi: detection rate | 0.02 | **0.46** |
| vtest.avi: jitter | 0.591 | **0.168** |
| vtest.avi: latency, mean (P95) | 17 (30) ms | 39 (81) ms |
| pose.jpg sequence: jitter | 0.022 | **0.013** |
| pose.jpg sequence: missing body landmarks | 6.3 % | **2.1 %** |
| pose.jpg sequence: latency, mean | 81 ms | **69 ms** |
| pose.jpg sequence: shift-consistency error | **0.0096** | 0.0143 |
| Legs reported missing during 15-frame occlusion | **100 %** | 25 % |

On vtest.avi, IMAGE mode is faster mostly because it finds nobody.

**Trade-offs, not a free win:**
- VIDEO mode tracks more robustly and with far less jitter, but it lags slightly behind real motion.
- VIDEO mode **smooths visibility over time.** An occluded joint keeps its old position while its visibility decays by about 0.07 per frame, so it stays above the 0.30 threshold for about 8 frames.

The landmark history detects that decay pattern and treats such joints as estimated, never as observed (`tests/test_landmark_tracking.py`).

In the occlusion replay, VIDEO mode sometimes also *re-synthesises* plausible but wrong limbs (straight "ghost" legs inside a blacked-out region) at visibility 0.3–0.55.
- These reach the **medium** tier at most, so corrections based on them are worded as qualified suggestions.
- At the inferences inside the occlusion the frame tier was low, and corrections were suppressed.

Visibility alone cannot expose every such ghost. This is a residual risk of VIDEO mode.
`mediapipe.running_mode: image` restores the old behaviour.

## 8. Kaggle 47-class profile — MEASURED
`data/evaluation_metrics.json`

- **Data:** 2,582 still images. The stratified 80/20 split (seed 42) is identical to the training notebook. All rows are still images, so there is no video-frame leakage.
- **Results (517 test images):**

| | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| RF | 82.79 % | 0.7607 | 0.8203 |
| MLP (Keras) | 77.56 % | 0.7082 | 0.7703 |

- **Missing features (RF), SYNTHETIC.** Random *engineered features* (not landmarks) set to NaN:

| Features masked | 0 % | 10 % | 20 % | 30 % |
|---|---|---|---|---|
| Accuracy | 82.79 % | 79.50 % | 73.89 % | 67.89 % |

## 9. Step-by-step evaluation — MEASURED
`data/step_by_step_test_report.json`

Up to 25 Right-form frames per step, **test split only**, RF with the deployed filters: **71.3 % top-1 over 585 frames**.
Sarvangasana scored 0 %, Viparita Karani 22 % and Anantasana 39 %.
The previous version of this script sampled the whole dataset, including training frames, used the MLP copy and the legacy filters, so its numbers are not comparable.

## 10. Latency on this build machine — MEASURED

| Stage | Mean | P95 |
|---|---|---|
| Feature extraction | 0.6–1.3 ms | — |
| RF impute + classify (one frame) | 52 ms | 64 ms |
| MLP impute + classify (one frame) | 2.0 ms | 2.9 ms |
| MediaPipe Heavy, VIDEO mode, 1000×667 | 69 ms | 85 ms |

Classification runs every `inference.interval_seconds` (0.5 s), not every frame. Camera FPS is therefore bounded mainly by MediaPipe.
Earlier figures in this file (RF 32 ms, 30 FPS) were measured on the author's laptop (RTX 3050) and were not re-measured here.

## 11. Live-pipeline replays — QUALITATIVE
`run_app.py --headless --video … --log-decisions …`

A 5-second replay of `pose.jpg` (Warrior II) with motion and a leg occlusion, 10 inferences per run:

| Run | With the original 0.28 threshold and no gating | After |
|---|---|---|
| Zenodo / RF auto-detect (Warrior II is not a Zenodo class) | 7 / 10 inferences gave "qualified" corrections for "Ardhakati Chakrasana" | **0 / 10**: reported as uncertain |
| Zenodo / MLP auto-detect | MLP says "Anantasana" (a floor pose) at 96 %; assessed on 8 / 10 inferences | **1 / 10** wrong assessments; 5 blocked by the orientation veto, the rest by label stability |
| Kaggle / RF auto-detect | — | Virabhadrasana Two recognised (74–77 %), no false corrections; assessment suppressed while the legs were occluded |
| Zenodo guided "Tadasana" | — | Target preserved; corrections issued against Tadasana (the person really is not in Tadasana); suppressed during occlusion |

A false-correction rate on real practice sessions needs reviewed recordings of the target users. That remains to be done; see the checklist in the README.

## 12. Jev image assessment — NOT EVALUATED LIVE

`jevmodel.org` was not reachable from the build environment, and no `JEVMODEL_API_KEY` was available.

- **Verified with mocked HTTP responses** (`tests/test_jev_integration.py`, 38 tests):
  - request schema;
  - response validation;
  - auth / credit / timeout / outage handling;
  - bounded retries, throttling, single-flight concurrency;
  - stale-answer rejection;
  - local-only fallback;
  - the fusion policy.
- **Not verified:**
  - whether Jev recognises yoga asanas in webcam stills;
  - its real latency, cost and rate behaviour;
  - whether the documented image opt-in header is still required.

Run `scripts/evaluate_jev_images.py --jev` on labelled photos before citing any Jev accuracy. **No claim is made here that Jev improves recognition.** It cannot fix landmark detection, because joint-angle corrections always come from MediaPipe landmarks.

## 13. Corrections to earlier versions of this report

| Earlier claim | Status |
|---|---|
| "Occlusion suppression 100 % at 25 % and 50 %" | **Wrong.** The test put all 33 landmarks on one point, so every angle was NaN before any occlusion. The JSON correctly showed 0 % at 25 %. Replaced by Section 6. |
| `zenodo_evaluation_report.json` top-level metrics, labelled "Champion Active Model" | They were **MLP** metrics, while the app runs RF. The new report lists RF and MLP separately, with artifact hashes. |
| "Gravitational posture prior physically separates standing from prone/inverted poses" | Partly true for upright poses. Together with sequence pruning it **lowered** RF held-out accuracy from 0.77 to 0.53 (Section 3). |
| Step-by-step accuracy | Previously included training frames. Now test split only (Section 9). |
| "Temporal fallback restores missing limbs" | The old fallback lasted one frame only. Now bounded multi-frame history (Section 5). |
| "RF weighted F1 75.83 %" (Zenodo) | Re-measured as 0.7628. |
| "Occlusion 10/20/30 %" (Kaggle) | These masked engineered features, not landmarks. Relabelled. |
| RF 32 ms / 30 FPS | Laptop measurement, not reproduced on this machine (Section 10). |

## 14. Known limitations

1. The existing models do not recognise **Sarvangasana, Vajrasana or Viparita Karani** reliably (recall 0–0.37). Auto-detect now reports "uncertain" for them instead of guessing. Retraining with more and better-separated data would be needed; the original artifacts were not changed.
2. Training and evaluation landmarks come from burst photographs. Live webcam conditions (camera angle, distance, clothing, lighting) are untested by these numbers.
3. 2D monocular geometry: foreshortened limbs and side-on views reduce angle reliability. About 21 % of held-out frames already lack enough visible joints for a full assessment.
4. The MLP can be confidently wrong on poses outside its 10 classes. The orientation veto and label-stability rule catch most such cases in the replay, but not all.
5. Thresholds were tuned on 50 validation sessions. They should be re-checked on recordings of the actual users.
