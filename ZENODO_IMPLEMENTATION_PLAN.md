# Zenodo Dataset Migration, Missing Features, and PPT-Aligned Implementation Plan

**Project:** Yoga Mitra: AI-Powered Real-Time Personalized Yoga Posture Detection and Correction System  
**Author:** Senior Computer Vision & ML Engineering Team  
**Date:** October 2026  
**Status:** Implementation Ready (Pre-Training Phase — Strict Training Gate Enforced)

---

## 1. Executive Summary & Audit of Existing Implementation

A rigorous inspection of the current repository against the project presentation (`btp2 (2).pptx`) and source code was performed.

### 1.1 What Already Works
- **MediaPipe Landmark Extraction & Fallback:** `src/pose/landmark_extraction.py` and `src/pose/missing_landmarks.py` implement MediaPipe Tasks API pose extraction with single-frame temporal imputation (`apply_temporal_fallback`) for missing keypoints.
- **Biomechanical Feature Extraction (39 Features):** `src/features/__init__.py` reliably computes 10 joint angles, 6 normalized distances, 3 alignment angles, 3 symmetry differences, and 17 imputation indicators (1 fraction + 16 per-joint flags).
- **Kaggle 47-Class Baseline Inference:** `src/models/model_loader.py` and `src/models/inference.py` successfully load and run the Kaggle Random Forest classifier (`yoga_random_forest.joblib`) with rolling majority-vote smoothing.
- **Kaggle Posture Evaluation:** `src/correction/posture_evaluator.py` evaluates 12 Kaggle poses with bilateral front/rear leg adaptation (e.g. Warrior I & II) and calculates a 0–100% alignment score.
- **Audio & Visual Feedback HUD:** `src/correction/feedback_engine.py`, `src/app/voice.py` (threaded SAPI5), and `src/app/overlay.py` deliver natural coaching cues with per-joint cooldowns and HUD visualization.
- **Reference Display:** `src/app/reference_display.py` provides a dual-panel interface with stability filtering and fallback cards for 8 guided Kaggle poses.
- **Basic Session Logging:** `src/correction/progress_tracker.py` writes session duration, frame count, average/peak alignment score, and raw joint error counts to SQLite (`data/yoga_sessions.db`).
- **Core Test Suite:** 26 automated unit tests pass in `tests/test_pipeline.py`.

### 1.2 What is Partially Implemented
- **Personalization (Calibration):** `src/correction/calibration.py` records limb proportions (`arm_span_ratio`, `leg_to_torso_ratio`, `shoulder_width_ratio`) in Mountain Pose, but `posture_evaluator.py` only consumes a scalar `tolerance_multiplier` (default 1.0). The measured anatomical ratios do not functionally adapt joint angle ranges.
- **Reliability Assessment:** The evaluator assigns "high", "medium", or "low" tiers based solely on static visibility thresholds (<0.30, <0.50) and imputation flags. It lacks temporal continuity, frame-to-frame velocity/jitter tracking, and viewpoint stability as mandated by Slide 7 of the PPT.
- **Progress Tracking:** SQLite records raw counts, but lacks longitudinal analytics, improvement trends over time, pose-wise learning curves, error frequency deltas, and measurement-grounded personalized recommendations.
- **Evaluation Pipeline:** `scripts/evaluate_system.py` is hardcoded to the 47-class Kaggle model and synthetic random vectors, without support for form verification, confusion matrices, or occlusion stress tests.

### 1.3 What is Missing for Zenodo Migration
- **Zenodo Vocabulary & Target Ranges:** Zero reference angle ranges or metadata exist for the 10 Zenodo asanas (Anantasana, Ardhakati Chakrasana, Bhujangasana, Kati Chakrasana, Marjariasana, Parvatasana, Sarvangasana, Tadasana, Vajrasana, Viparita Karani).
- **Dual-Task Model Architecture:** The system only supports single-label multiclass pose classification. It lacks the dual-task architecture (Model A: 10-class Asana Classifier, Model B: Binary Right/Wrong Form Verifier) required by the Zenodo dataset.
- **Reusable Zenodo Dataset Pipeline:** No preprocessing script exists to convert the 137-column Zenodo CSV into the 39 runtime biomechanical features with group-aware leakage-free splitting.
- **Rare Class Handling & Group Leakage Protection:** The Zenodo dataset exhibits severe class imbalance (Vajrasana has only 149 rows) and high temporal autocorrelation (burst frames 1 second apart). Naive splitting causes complete class loss or severe train/test data leakage.
- **Model Profile Abstraction:** The codebase tightly couples `load_artifacts` and `PoseInference` to `saved_yoga_model/` without a clean mechanism to switch between the legacy 47-class Kaggle profile and the new 10-class Zenodo profile.

---

## 2. PPT Alignment Matrix

| PPT Topic & Slide | Slide Goal | Existing State | Plan & Implementation Action |
| :--- | :--- | :--- | :--- |
| **Personalized Assessment** (Slide 5) | Replace universal posture standards with body-adaptive joint-angle ranges based on limb lengths, proportions, and mobility. | Calibration records ratios to JSON, but evaluator only applies a scalar multiplier. | Enhance `UserProfile` schema to v2.0; implement anatomical proportion-adapted target ranges where limb lengths directly modify reachable angle thresholds; add mobility variance estimation. |
| **Recognition ≠ Correction** (Slide 6) | Detect affected joint, deviation magnitude, severity tier (mild/moderate/major), and specific actionable directional cues. | Implemented for 12 Kaggle poses with bilateral rules. | Extend `REFERENCE_RANGES` and `posture_evaluator.py` to all 10 Zenodo asanas with directional hints, severity tiers, and focus-joint HUD highlighting. |
| **Confidence-Aware Feedback** (Slide 7) | Combine landmark confidence, joint visibility, and temporal stability to assess reliability and prevent false corrections under occlusion or jitter. | Only static MediaPipe visibility and imputation flags are checked. | Create `TemporalReliabilityTracker` computing rolling landmark velocity/jitter; suppress corrections on low-reliability frames; gate form verifier output. |
| **Six-Step Pipeline** (Slides 10–11) | Pose estimation → Landmark extraction → Pose recognition → Alignment analysis → Error detection → Real-time feedback. | Implemented for 47-class Kaggle model. | Maintain 6-step architecture while adding Model B (Form Verification) as a parallel verification stream in Step 3. |
| **Progress & Longitudinal Improvement** (Slide 11, Step 6) | Store history, track pose-wise performance, detect recurring mistakes, show improvement trends, provide personalized recommendations. | Basic table of past sessions with raw error counts. | Add `ProgressAnalytics` module computing score trends, error frequency deltas, pose-specific mastery rates, and data-grounded tips. |
| **Evaluation & Robustness** (Slide 15) | Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix, FPS, latency, occlusion robustness. | Stale Kaggle-only synthetic script. | Build `scripts/evaluate_zenodo_models.py` supporting dual-task metrics, occlusion masking stress tests, latency benchmarks, and dry-run execution. |

---

## 3. Comprehensive Zenodo Dataset Audit

### 3.1 Dataset Overview
- **Location:** `zenodo_yoga_landmarks - zenodo_yoga_landmarks.csv` (Root directory, 21.4 MB)
- **Shape:** 11,118 rows × 137 columns
- **Column Composition:**
  - 5 Metadata columns: `image_path`, `asana_label`, `correctness_label`, `combined_label`, `group_id`
  - 132 Landmark columns: `f_0` through `f_131`
- **Data Integrity:**
  - Null / NaN values: **0** across all 137 columns
  - Infinite values: **0** across all 137 columns
  - Exact duplicate rows (all columns): **0**

### 3.2 Landmark Structure Verification
MediaPipe's Pose Landmarker model outputs 33 landmarks with 4 channels per landmark: $(x, y, z, \text{visibility})$.
- `f_0, f_4, ..., f_128`: Landmark $x$ coordinates (range: $-0.4509$ to $1.1089$, mean $0.4982$)
- `f_1, f_5, ..., f_129`: Landmark $y$ coordinates (range: $0.0169$ to $1.0822$, mean $0.5005$)
- `f_2, f_6, ..., f_130`: Landmark $z$ coordinates (range: $-3.0756$ to $2.4886$, mean $-0.0511$)
- `f_3, f_7, ..., f_131`: Landmark visibility confidence (range: $0.0006$ to $1.0000$, mean $0.8134$)
- **Ordering Confirmation:** The 132 features map precisely to MediaPipe landmark IDs 0 to 32 in standard order: $f_{4i} = x_i, f_{4i+1} = y_i, f_{4i+2} = z_i, f_{4i+3} = \text{vis}_i$.

### 3.3 Class Distributions & Imbalance Analysis

#### Task A: Asana Distribution (10 Classes)
| Asana Label | Count | Percentage | Observations |
| :--- | :--- | :--- | :--- |
| **Parvatasana** | 2,754 | 24.77% | Dominant majority class (Mountain / Downward-Dog variant) |
| **Tadasana** | 1,571 | 14.13% | Standing Mountain Pose |
| **Marjariasana** | 1,551 | 13.95% | Cat / Cow Pose |
| **Anantasana** | 1,150 | 10.34% | Side-reclining Sleeping Vishnu Pose |
| **Bhujangasana** | 1,105 | 9.94% | Cobra Pose |
| **Ardhakati Chakrasana** | 924 | 8.31% | Lateral Side-Bend Wheel Pose |
| **Viparita Karani** | 693 | 6.23% | Inverted Lake / Legs-Up Pose |
| **Sarvangasana** | 676 | 6.08% | Supported Shoulder Stand |
| **Kati Chakrasana** | 545 | 4.90% | Standing Waist Twist Pose |
| **Vajrasana** | 149 | 1.34% | **Critical Minority Class** (~18x smaller than Parvatasana) |
| **Total** | **11,118** | **100.0%** | |

#### Task B: Form Correctness Distribution
- **Right:** 5,913 (53.18%)
- **Wrong:** 5,205 (46.82%)
- Overall form verification is well balanced (~53:47).

#### Asana × Correctness Cross-Tabulation
Every single asana contains both Right and Wrong instances:
- Anantasana: 624 Right / 526 Wrong (54.3% Right)
- Ardhakati Chakrasana: 506 Right / 418 Wrong (54.8% Right)
- Bhujangasana: 647 Right / 458 Wrong (58.6% Right)
- Kati Chakrasana: 320 Right / 225 Wrong (58.7% Right)
- Marjariasana: 833 Right / 718 Wrong (53.7% Right)
- Parvatasana: 1,369 Right / 1,385 Wrong (49.7% Right)
- Sarvangasana: 339 Right / 337 Wrong (50.1% Right)
- Tadasana: 854 Right / 717 Wrong (54.4% Right)
- Vajrasana: 65 Right / 84 Wrong (43.6% Right)
- Viparita Karani: 356 Right / 337 Wrong (51.4% Right)

### 3.4 Group ID, Step, and Fault Semantics
- **Unique Group IDs:** 87 distinct strings in format `{asana}::step {N} [fault_description]`.
- **Step distribution:**
  - Step 1: 3,637 (32.7%)
  - Step 2: 3,059 (27.5%)
  - Step 3: 2,244 (20.2%)
  - Step 4: 1,652 (14.9%)
  - Step 5: 526 (4.7%) — *Proves steps exceed 1–4, extending through step 5.*
- **Fault Tags:**
  - For Right poses: no fault tag (`[None/Normal]`, 5,913 rows).
  - For Wrong poses: explicit anatomical error tags such as `legs` (1,275), `leg and hand` (1,033), `hand` (511), `neck` (454), `hand and legs` (323), `knee` (268), `standing` (179), `head` (139), `back overbend` (35), `back bending` (35), `ankle` (28).
  - These provide valuable ground-truth validation for the joint-level correction engine.

### 3.5 Recording Sessions, Data Leakage Risk & Splitting Strategy
- **Image Path Metadata:** Paths point to Google Drive: `/content/drive/MyDrive/Yoga_Project/Images/{Asana}/{Right|Wrong} Steps/.../{YYYYMMDD_HHMMSS}.jpg`.
- **Recording Dates:** 5 recording dates exist:
  - 2023-03-09: 149 samples (100% Vajrasana)
  - 2023-03-10: 3,299 samples (Kati Chakrasana, Parvatasana)
  - 2023-03-11: 2,475 samples (Ardhakati Chakrasana, Marjariasana)
  - 2023-03-12: 2,676 samples (Bhujangasana, Tadasana)
  - 2023-03-17: 2,519 samples (Anantasana, Sarvangasana, Viparita Karani)
- **Critical Finding — Date Splitting is Impossible:** Each asana was recorded exclusively on 1 or 2 specific dates. Splitting by recording date would result in 100% train/test class disjointness!
- **Burst / Sequence Autocorrelation:** Within each asana and date, images are recorded in temporal bursts 1–2 seconds apart (`YYYYMMDD_HHMM`). There are 2,187 identical landmark vectors across 3,792 rows due to static video frames.
- **Leakage Prevention Strategy:**
  - We define a **Burst Session Group**: `session_group = {date}_{hour_minute}_{group_id}`.
  - There are 258 distinct burst sessions.
  - We use a **Stratified Group Split** over `session_group` stratified by `combined_label` (`{asana}_{Right|Wrong}`):
    - Train: 70% of sessions
    - Validation: 15% of sessions
    - Test: 15% of sessions
  - **Rare Class Guarantee:** Vajrasana has 29 bursts (17 Right, 12 Wrong). Group stratification guarantees all splits contain both Right and Wrong Vajrasana samples without leaking continuous frames across splits.

---

## 4. Proposed Architecture & System Design

```
                                 LIVE CAMERA FRAME (1280x720)
                                              │
                                              ▼
                               MediaPipe Pose Landmarker (33 keypoints)
                                              │
                                              ▼
                            Temporal Landmark Fallback (Missing Keypoints)
                                              │
                                              ▼
                         Temporal Reliability Tracker (Jitter / Stability)
                                              │
                                              ▼
                      Biomechanical Feature Extraction (geometry.py)
                         39 Unified Features (Angles, Dims, Leans, Flags)
                                              │
                   ┌──────────────────────────┴──────────────────────────┐
                   ▼                                                     ▼
        Model Profile: KAGGLE (47-Class)                      Model Profile: ZENODO (10-Class)
        (Legacy Mode — Active Baseline)                       (Prepared Architecture — Awaiting Training)
                   │                                                     │
        ┌──────────┴──────────┐                               ┌──────────┴──────────┐
        ▼                     ▼                               ▼                     ▼
  Random Forest              MLP                         Model A (Classifier)  Model B (Form Verifier)
(47-Class Prob)       (47-Class Prob)                   (10-Class Asana Prob) (Right vs Wrong Prob)
        │                     │                               │                     │
        └──────────┬──────────┘                               └──────────┬──────────┘
                   │                                                     │
                   └──────────────────────────┬──────────────────────────┘
                                              │
                                              ▼
                              Posture Evaluator (posture_evaluator.py)
                           Personalized Reference Angle Comparison
                                (v2.0 Biomechanical Calibration)
                                              │
                                              ▼
                               Feedback Engine (feedback_engine.py)
                             Confidence-Aware Joint-Level Guidance
                                              │
                                              ▼
                           HUD Overlay + Dual-Panel Reference Display
                               + Non-blocking Audio Coach
```

### 4.1 Model Profiles & Dual-Task Abstraction
We introduce an explicit `ModelProfile` abstraction in `src/models/model_loader.py` and `src/models/inference.py`:
1. `KaggleProfile`:
   - Single multiclass model (47 classes).
   - Artifacts: `saved_yoga_model/yoga_random_forest.joblib`, `yoga_preprocessing.joblib`, etc.
   - Remains the working default!
2. `ZenodoProfile`:
   - Dual-task model:
     - `asana_model`: Predicts 1 of 10 asanas + confidence.
     - `verifier_model`: Predicts `Right` vs `Wrong` + probability.
   - Target artifact directory: `saved_yoga_model_zenodo/`.
   - When artifacts are not yet trained, loading cleanly reports status as `AWAITING_TRAINING` and falls back gracefully.

### 4.2 Strict Feature Parity Guarantee
Training pipelines must NOT train on raw 132 landmark coordinates. Both training and inference strictly compute the shared 39 biomechanical features through `src.features.extract_features` with identical column ordering, imputation, and scaling.

---

## 5. Implementation Roadmap & Order of Execution

1. **Step 1:** Complete audit and save `ZENODO_IMPLEMENTATION_PLAN.md` (Current step).
2. **Step 2:** Build `scripts/prepare_zenodo_dataset.py`:
   - Converts 132 raw features into MediaPipe 33-point landmark dicts.
   - Computes 39 runtime features via `extract_features`.
   - Performs group-aware stratified train/val/test split (no burst leakage, preserving Vajrasana).
   - Computes candidate provisional reference ranges from training `Right` samples.
3. **Step 3:** Build `scripts/train_zenodo_models.py` and `scripts/evaluate_zenodo_models.py`:
   - Implements Random Forest and MLP architectures for Model A and Model B.
   - Enforces the strict training gate: provides `--dry-run` and parameter verification; halts before `.fit()`.
4. **Step 4:** Refactor `src/models/model_loader.py` and `src/models/inference.py`:
   - Implement `YogaArtifactsZenodo`, `DualTaskInference`, and profile switching (`kaggle` vs `zenodo`).
   - Guarantee backwards compatibility with the 47-class Kaggle artifacts.
5. **Step 5:** Extend `src/correction/reference_ranges.py` and `posture_evaluator.py`:
   - Add reference ranges and metadata for all 10 Zenodo poses.
   - Add bilateral and pose-specific adaptation rules.
6. **Step 6:** Upgrade `src/correction/calibration.py`:
   - Store v2.0 profile with timestamp, measurement stability, and proportion ratios.
   - Connect arm span and leg ratios directly into pose evaluator tolerances and reachable thresholds.
7. **Step 7:** Implement `src/correction/reliability_tracker.py`:
   - Stateful temporal jitter and velocity calculator.
   - Combine landmark confidence, imputation, and jitter to classify frame reliability into High, Medium, Low.
8. **Step 8:** Upgrade `src/app/reference_display.py` and `src/correction/progress_tracker.py`:
   - Add cue cards and metadata for 10 Zenodo asanas.
   - Add longitudinal analytics, score trends, and personalized recommendations.
9. **Step 9:** Add test cases in `tests/test_zenodo_pipeline.py` and verify all tests pass.
10. **Step 10:** Document results, review limitations, present exact training commands, and pause at the training gate.

---

## 6. Strict Training Gate Notice

**CRITICAL RULE:** In accordance with user instructions, no model fitting (`.fit()`), network training, or generation of new trained artifact files will be performed during this stage. The code, pipeline, tests, and interfaces will be fully prepared and statically validated. The training commands will be provided for explicit user approval before execution.
