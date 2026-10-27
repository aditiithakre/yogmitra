# Yoga Mitra: Real-Time Personalized Yoga Posture Detection and Correction

YogaMitra is an intelligent, real-time computer vision system that monitors yoga practice through a standard webcam, evaluates body biomechanics, and delivers instant posture corrections both visually and through voice coaching.

---

## Key Capabilities

1. **Dual-Panel Visual Interface:** Left panel displays the live webcam feed with skeletal overlays, color-coded joint angle pills, and alignment HUD; right panel displays high-resolution target asana reference cards with key alignment cues.
2. **Dual Interfaces:**
   - **Native Desktop App:** Low-latency OpenCV window with HUD, status strip, and real-time diagnostics (`run_app.py` / `run_desktop.bat`).
   - **Modern Web Dashboard:** Full-featured browser interface on `http://localhost:5000` with live MJPEG streaming, real-time telemetry, pose selector, and interactive controls (`server.py`).
3. **Biomechanical Evaluation (39 Features):** Computes 10 critical joint angles, 6 normalized distances, 3 alignment angles, 3 symmetry differences, and 17 imputation flags using MediaPipe PoseLandmarker Heavy in video stream mode.
4. **Dual-Task Zenodo Machine Learning Pipeline:**
   - **Model A (Asana Classifier):** Classifies the current asana across 10 core yoga poses.
   - **Model B (Form Verifier):** Binary verification checking whether the pose execution meets anatomical correctness.
   - Also supports a 47-class Kaggle Random Forest / MLP model profile.
5. **Multi-Tier Reliability & Temporal Smoothing:** Joint-level confidence gating (High, Medium, Low) prevents false corrections when body parts are occluded or out of frame.
6. **Natural Voice Coaching:** Threaded SAPI5 / speech engine delivers human-like directional guidance ("Straighten your left knee", "Raise your arms") with per-joint cooldowns and posture-change queue clearing.
7. **Session Progress Analytics:** Persists alignment scores, session durations, joint error trajectories, and recommendations in a local SQLite database (`data/yoga_sessions.db`).

---

## Architecture

```mermaid
graph TD
    A[Webcam Frame] --> B[MediaPipe PoseLandmarker Heavy<br/>VIDEO mode + Monotonic Timestamps]
    B --> C[Landmark History & Fallback<br/>Temporal Imputation + Scene-Change Reset]
    C --> D[Temporal Reliability Tracker<br/>Visibility / Imputation / Jitter Tiers]
    C --> E[39 Biomechanical Features<br/>Angles, Distances, Symmetry]
    E --> F{Model Profile}
    F -->|Zenodo Profile| G1[Model A: 10-Asana Classifier<br/>Model B: Binary Form Verifier]
    F -->|Kaggle Profile| G2[47-Class Classifier RF/MLP]
    G1 --> H[Reliability-Gated Candidate Filters + EMA Smoothing]
    G2 --> H
    H --> J[Pose Decision<br/>Calibrated Threshold, Class Gate, Orientation Check]
    D --> J
    J -->|High / Medium Reliability| K[Posture Evaluator<br/>Joint-Level Range Checks & Severity Classification]
    J -->|Uncertain / Low Tracking| U[Uncertain State + Camera Guidance]
    K --> L[Feedback Engine<br/>High: Specific Cue | Medium: Qualified Suggestion]
    U --> L
    L --> M1[Desktop OpenCV HUD & Reference Card]
    L --> M2[Web Dashboard Telemetry & MJPEG Stream]
    L --> N[Voice Engine<br/>Dynamic Cooldown & Queue Management]
    K --> P[Session Progress Tracker<br/>SQLite Storage & Longitudinal Analytics]
```

---

## Installation

### Prerequisites
- Python **3.10 – 3.12** (tested on Python 3.12)
- Webcam (built-in or USB)
- ~1.5 GB disk space (for models and dataset features)

### Setup

```bash
# Clone the repository
git clone https://github.com/anshu-0132/yogmitra && cd yogmitra

# Create and activate virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate

# macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional — for Kaggle 47-class MLP (Keras) model:
pip install -r requirements-optional.txt

# Linux only (for system voice output):
sudo apt install espeak-ng
```

---

## Running the Application

### Option A: Desktop Application
Run directly using Python or the launcher script:

```bash
# Launch default configuration (Zenodo profile, Random Forest, auto-detect)
python run_app.py

# Or on Windows, double-click run_desktop.bat
run_desktop.bat
```

#### Common Desktop Options:
```bash
python run_app.py --target-pose "Tadasana"           # Guided practice on a specific pose
python run_app.py --free                             # Free practice (auto-detect)
python run_app.py --profile kaggle                   # Use the 47-class Kaggle model profile
python run_app.py --classifier mlp                   # Use MLP instead of Random Forest
python run_app.py --diagnostics                      # Enable live diagnostic overlay
python run_app.py --no-voice                         # Run with visual feedback only
python run_app.py --camera 1                         # Use secondary external webcam
python run_app.py --calibrate                        # Run 5-second body calibration on start
python run_app.py --history                          # Display session history and past scores
python run_app.py --test-voice                       # Verify text-to-speech audio output
```

### Option B: Interactive Web Dashboard
YogaMitra includes a standalone Flask web server providing a browser-based dashboard:

```bash
python server.py
```
Open your browser and navigate to:
```
http://localhost:5000
```

**Web Dashboard Features:**
- Live video stream with skeletal landmarks, angle badges, and target pose card
- Real-time biomechanical telemetry showing joint angles and deviation indicators
- Form verification status (Model B verification)
- Interactive pose selector for switching between Free Mode and Guided Asanas
- Audio toggle and session summary stats

---

## Keyboard Controls (Desktop App)

| Key | Action |
|:---:|:---|
| `T` / `N` | Cycle through guided target poses |
| `M` | Toggle between Guided and Auto-Detect (Free) mode |
| `C` | Start or cancel personalized body calibration |
| `D` | Toggle live diagnostics panel |
| `V` | Mute / unmute voice feedback |
| `A` | Toggle joint-angle numerical labels |
| `+` / `-` | Increase / decrease inference frequency |
| `R` | Reset tracking, reliability, and landmark histories |
| `S` | Capture a screenshot to `screenshots/` |
| `Q` / `Esc` | Safely quit and display session summary |

---

## User Interface Elements

- **Header HUD:** Displays detected asana, confidence percentage, and form verification badge when confidence passes validation gates. Displays helpful guidance when pose is uncertain.
- **Side-by-Side Target Reference:** Shows an illustrated reference card, key anatomical focus points, and alignment cues for the selected pose.
- **Skeleton Visualization:**
  - Solid color-coded lines and joint circles indicate actively tracked landmarks.
  - Orange hollow markers denote short-term temporally estimated joints.
  - Red / amber highlights pinpoint misaligned joints requiring correction.
- **Angle Pills:** Real-time numerical degree readouts positioned next to evaluated joints.
- **Status Strip:** Displays model prediction, tracking reliability tier (`HIGH`, `MEDIUM`, `LOW`), and current coaching feedback.

---

## Supported Asanas

### Zenodo 10-Asana Dataset:
1. **Tadasana** (Mountain Pose)
2. **Kati Chakrasana** (Standing Spinal Twist)
3. **Ardhakati Chakrasana** (Lateral Side Bend)
4. **Vajrasana** (Thunderbolt Pose)
5. **Parvatasana** (Seated Mountain Pose)
6. **Bhujangasana** (Cobra Pose)
7. **Marjariasana** (Cat-Cow Pose)
8. **Sarvangasana** (Shoulder Stand)
9. **Viparita Karani** (Legs-Up-the-Wall Pose)
10. **Anantasana** (Side-Reclining Leg Lift)

### Kaggle Profile Reference Poses:
Virabhadrasana I & II (Warrior I & II), Utkatasana (Chair Pose), Trikonasana (Triangle Pose), Phalakasana (Plank), Adho Mukha Svanasana (Downward-Facing Dog), and Vrksasana (Tree Pose).

---

## Project Structure

```
yogmitra/
├── assets/
│   └── pose_references/        # Target asana reference cards & diagrams
├── config.yaml                 # Central application & pipeline configuration
├── data/
│   ├── yoga_sessions.db        # SQLite session history and performance logs
│   └── zenodo/                 # Dataset features, splits, and calibration data
├── run_app.py                  # Main desktop OpenCV application entrypoint
├── run_desktop.bat             # Windows one-click desktop launcher
├── server.py                   # Flask web server & browser dashboard
├── saved_yoga_model/           # Kaggle model artifacts (RF joblib & MLP keras)
├── saved_yoga_model_zenodo/    # Zenodo dual-task models (Classifier + Form Verifier)
├── scripts/                    # Training, evaluation, and calibration utilities
├── src/
│   ├── app/                    # UI overlay, reference card display, voice engine, webcam loop
│   ├── correction/             # Biomechanical ranges, posture evaluator, feedback, calibration
│   ├── features/               # 39-feature biomechanical extraction engine
│   ├── models/                 # Model loader, inference engine, sequence tracker
│   └── pose/                   # MediaPipe wrapper & missing landmark interpolation
└── tests/                      # Automated unit and integration test suite
```

---

## Testing & Verification

Run the automated test suite with pytest:

```bash
python -m pytest tests/ -q
```

The test suite covers:
- Core feature extraction and geometrical invariants
- Zenodo dual-task pipeline and model loading
- Video timestamp continuity and occlusion recovery
- Confidence calibration and reliability gating policies
- Posture evaluator severity calculations and angle range limits
