# Implementation Audit — Yoga Mitra Posture Correction System

Date: 2026-10-15  
Repository: `e:\Yoga_posture_project`  
Auditor: Senior Python & CV/ML Engineer  

---

## 1. Executive Implementation Audit Table

| Feature | Status | Evidence | Required Action |
|:---|:---:|:---|:---|
| **Voice feedback** | **Broken / Unreliable** | In `src/app/voice.py`, `pyttsx3.init()` was called on the main thread during `__init__`, while `engine.say()` / `engine.runAndWait()` ran inside a secondary daemon worker thread `_worker()`. On Windows SAPI5/COM, cross-thread calls cause silent COM deadlock / hanging. Also, cooldown in `feedback_engine.py` is a static 12s lock with robotic strings and obsolete queued messages are not discarded. | Refactor `VoiceFeedback` to initialize SAPI5 engine inside its dedicated worker thread with proper `pythoncom.CoInitialize()`. Implement dynamic per-error cooldowns, speech queue clearing on posture change/mute, natural human-friendly voice cues, and `--test-voice` diagnostic. |
| **Reference pose image** | **Missing** | `src/app/overlay.py` and `src/app/webcam_app.py` render only a single video stream showing the live camera feed with an OpenCV skeleton. There is no dual-panel layout or persistent reference target image for the user to compare against. | Implement a responsive two-panel UI: Left panel = live webcam with HUD & skeleton; Right panel = reference asana image, title, key alignment cues, and highlighted focus joint. Add `assets/pose_references/` with high-quality illustrations/schematics for all guided targets, and graceful fallback card rendering. |
| **Confidence-aware feedback** | **Partial** | `src/correction/posture_evaluator.py` checks individual landmark `visibility >= 0.30` and `imputed_fraction > 0.25`, but lacks a structured multi-tier reliability model (High, Medium, Low) and temporal jitter / stability tracking. Low visibility merely aborts rather than offering qualified gentle feedback or specific advice to step back/adjust lighting. | Implement 3-tier feedback policy (High: actionable angle deviation; Medium: qualified suggestion; Low: suppress angle math and guide camera positioning). Separate temporal tracking fallback from feedback reliability. |
| **User calibration** | **Missing** | No baseline calibration workflow, limb length measurement, or user profile persistence exists in the repository. Target angles are hardcoded in `src/correction/reference_ranges.py` with no personalization for limb proportions. | Build `src/correction/calibration.py` offering an optional guided calibration flow (e.g. 30 frames in neutral stance/Tadasana). Estimate user limb ratios and torso scaling. Persist profile to `user_profile.json` and load on startup with safe fallback defaults. |
| **Progress tracking** | **Missing** | No session history database or tracking persistence exists in the repository. Sessions end without recording score trajectories, mistakes, or progress. | Implement `src/correction/progress_tracker.py` using SQLite (`data/session_history.db`) to record session timestamps, duration, asana practised, alignment scores, recurring joint errors, and persistent corrections. Expose CLI `--history` summary and clean exit summary. |
| **Joint deviation and severity** | **Partial** | `AngleCheck` in `posture_evaluator.py` calculates degrees outside range and `status` ("too_low" / "too_high" / "ok" / "unavailable"), but lacks `severity` categorization (mild / moderate / major), directional movement phrasing ("bend", "straighten", "raise", "lower"), and an overall alignment score (0–100%). | Extend `AngleCheck` and `PostureEvaluation` to include severity classifications, natural directional verbs, reliable joint weights, and a validated 0–100% alignment score. |
| **Performance evaluation** | **Partial** | `tests/test_pipeline.py` contains 14 unit tests, but there are no automated latency benchmarks, FPS measurements, or an `EVALUATION.md` report documenting held-out model metrics and system benchmarks. | Build an automated benchmark script `evaluate_system.py` measuring model precision/recall/F1, inference latency, camera FPS, and generate `EVALUATION.md`. |

---

## 2. Detailed Root-Cause Analysis for Priority One (Voice)

### Root Cause
1. **Windows SAPI5 COM Thread Affinity**:
   When `pyttsx3.init()` is invoked on the main application thread during `VoiceFeedback.__init__`, the underlying Windows Speech API COM object is bound to that thread. When the daemon thread later calls `engine.runAndWait()`, Windows COM throws an RPC error or indefinitely deadlocks waiting for the main thread message pump.
2. **Verification**:
   Executing `pyttsx3.init()` on the main thread and calling `runAndWait()` inside a `threading.Thread` deadlocks 100% of the time on Windows. Executing `pyttsx3.init()` *inside* the worker thread executes and completes with zero errors.
3. **Queue Invalidation**:
   The existing queue accumulates messages and plays them sequentially. If a user adjusts their posture while a message is queued, the obsolete message still plays seconds later, giving outdated corrections.

---

## 3. Implementation Plan

1. **Priority 1 (Voice)**:
   - Rewrite `src/app/voice.py`: dedicated worker thread initialization, safe lifecycle, queue clearing on mute/posture change.
   - Refactor `feedback_engine.py`: human-like natural phrasing ("Bend front knee slightly", "Straighten rear leg", "Arms up"), intelligent cooldown.
   - Add CLI `--test-voice` in `run_app.py`.
2. **Priority 2 (Reference Asana Two-Panel UI)**:
   - Create `assets/pose_references/` with reference cards for the 8 core asanas.
   - Build dual-panel layout in `src/app/overlay.py` and `src/app/webcam_app.py`.
   - Update target image immediately on `T` / `N`.
3. **Priority 3 (Confidence-Aware Feedback)**:
   - Implement High / Medium / Low feedback policy in `posture_evaluator.py` & `feedback_engine.py`.
4. **Priority 4 (User Calibration)**:
   - Implement `src/correction/calibration.py` with profile saving/loading to `user_profile.json`.
5. **Priority 5 (Joint Deviation & Scoring)**:
   - Add `severity` ("mild", "moderate", "major") and 0–100% overall alignment score.
6. **Priority 6 (Session History & Tracking)**:
   - Implement `src/correction/progress_tracker.py` with SQLite storage in `data/session_history.db`.
7. **Priority 7 (Testing & Evaluation)**:
   - Add comprehensive tests covering all new modules.
   - Create `evaluate_system.py` and generate `EVALUATION.md`.

---

## 4. Reliability audit, 2026-10-27: confirmed root causes and fixes

Every item was confirmed by running the code or measuring held-out data. Evidence is in [EVALUATION.md](EVALUATION.md).

| # | Root cause (file) | Evidence | Fix |
|---|---|---|---|
| 1 | MediaPipe ran in IMAGE mode, so every webcam frame got a fresh person detection with no tracking (`src/pose/__init__.py`) | Same frames, VIDEO vs IMAGE: detection 0.46 vs 0.02 on small subjects; jitter 2–3× lower | VIDEO mode with monotonic ms timestamps. IMAGE mode kept (`mediapipe.running_mode`). Deterministic `close()` in a `finally` block |
| 2 | Temporal fallback used only the previous raw frame, so an occlusion longer than one frame lost the joint. Imputed visibility 0.35 passed the 0.30 threshold, so imputed and observed joints were indistinguishable (`src/pose/missing_landmarks.py`) | Unit tests; synthetic occlusion | `LandmarkHistory`: per-joint last reliable observation, max age, `imputed`/`unavailable` flags, resets on person loss or scene change, detection of VIDEO-mode visibility-decay "ghosts" |
| 3 | Auto-detect evaluated the posture of a `low_confidence` label. That drove joint highlights, alignment score, progress logging and the reference panel even though the text was suppressed. The confidence threshold of 0.28 accepted about 97 % of frames, with about 20 % wrong (`src/app/webcam_app.py`) | Held-out test; Warrior II replay | `src/models/jev_fusion.decide` gates assessment on: the calibrated threshold (0.50, chosen on val), a per-class gate, orientation consistency, and label stability. One decision object drives evaluator, feedback, voice, HUD and reference panel |
| 4 | Gravitational prior (×0.005) and sequence pruning (×0.001) were applied together on every frame (`src/models/inference.py`, `src/models/sequence_tracker.py`) | RF test accuracy 0.772 → 0.535; Tadasana recall 0.99 → 0.28 | `FilterPolicy` with a `safe` mode (reliability- and stability-gated, upright/floor only, soft weight, high-confidence override). Per-classifier policy chosen on val (MLP keeps the prior, which helps it) |
| 5 | The evaluator let the reliability tracker *override* its own tier, so an imputed joint could be "high" (`src/correction/posture_evaluator.py`) | Unit test | Tier = minimum of visibility, imputation, tracker and decision tier. Unavailable-majority frames become `low_visibility` |
| 6 | `load_profile()` dropped `thigh_to_torso_ratio`, so the calibrated 0.836 became the default 0.85 (`src/correction/calibration.py`) | Round-trip test | Field loaded |
| 7 | With `profile: kaggle`, the config `model.dir` pointed at the Zenodo directory, and artifact / task paths depended on the shell cwd (`webcam_app.py`, `model_loader.py`) | Running the tests from another cwd failed | `model.kaggle_dir`; all paths resolved against the project root |
| 8 | The Zenodo report labelled MLP metrics as the active champion while the app runs RF. The occlusion test used 33 identical landmarks, and EVALUATION.md claimed 100 % suppression at 25 % where the JSON said 0 % (`scripts/evaluate_zenodo_models.py`) | Re-run | Per-architecture report with artifact hashes; occlusion test on real held-out landmarks |
| 9 | The step-by-step test included training frames and used the MLP copy with legacy filters (`scripts/test_step_by_step_system.py`) | Code review | Test split only; active classifier and policy recorded |
| 10 | The training script's default output directory was the shipped model directory (`scripts/train_zenodo_models.py`) | Code review | Timestamped candidate directory; refuses to overwrite without `--overwrite-existing` |

Not changed, because nothing in the evidence justified it:
- the 39-feature schema;
- the trained models;
- calibration and anthropometric personalisation logic (apart from bug 6);
- reference ranges, voice, and progress tracking.
