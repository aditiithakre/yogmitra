"""
run_app.py  –  launcher script (project root)

Usage (from the project root, or any directory):
    python run_app.py
    python run_app.py --interval 1 --classifier rf
    python run_app.py --no-voice
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# If launched under Python 3.13+ where ML dependencies (MediaPipe/sklearn) are missing,
# automatically delegate to the installed Python 3.12 environment via Windows py launcher
if sys.version_info >= (3, 13):
    py_launcher = shutil.which("py")
    if py_launcher:
        chk = subprocess.run([py_launcher, "-3.12", "-c", "import sys; sys.exit(0)"], capture_output=True)
        if chk.returncode == 0:
            print("[YogMitra] Python 3.13 detected; launching seamlessly via Python 3.12 environment...")
            sys.exit(subprocess.call([py_launcher, "-3.12", str(Path(__file__).resolve())] + sys.argv[1:]))

# Ensure src/ is importable when launched from project root
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import yaml

env_file = ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.strip().split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def _patch_config(cfg: dict, args: argparse.Namespace) -> dict:
    if args.interval is not None:
        cfg["inference"]["interval_seconds"] = args.interval
    if args.classifier is not None:
        cfg["model"]["active_classifier"] = args.classifier
    if args.profile is not None:
        cfg["model"]["profile"] = args.profile
    if args.no_voice:
        cfg["voice"]["enabled"] = False
    if args.camera is not None:
        cfg["camera"]["device_index"] = args.camera
    if "practice" not in cfg:
        cfg["practice"] = {}
    cfg["practice"]["enabled"] = False
    if getattr(args, "video", None):
        cfg["camera"]["source"] = args.video
    if getattr(args, "max_frames", None):
        cfg["camera"]["max_frames"] = args.max_frames
    if getattr(args, "headless", False):
        cfg["display"]["headless"] = True
    if getattr(args, "diagnostics", False):
        cfg["display"]["diagnostics"] = True
    if getattr(args, "log_decisions", None):
        cfg.setdefault("logging", {})["decision_log"] = args.log_decisions
    if getattr(args, "jev", False):
        cfg.setdefault("jev", {})["enabled"] = True
    if getattr(args, "no_jev", False):
        cfg.setdefault("jev", {})["enabled"] = False
    if getattr(args, "local_primary", False):
        cfg.setdefault("jev", {})["primary"] = False
    if getattr(args, "jev_every", None) is not None:
        cfg.setdefault("jev", {})["periodic_interval_seconds"] = args.jev_every
    if getattr(args, "calibrate", False):
        if "calibration" not in cfg:
            cfg["calibration"] = {}
        cfg["calibration"]["force_startup"] = True
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Yoga Pose Correction — Real-Time Webcam App"
    )
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config YAML (default: config.yaml)")
    parser.add_argument("--profile", choices=["kaggle", "zenodo"], default=None,
                        help="Active model profile: 'kaggle' (47-class default) or 'zenodo' (10-class dual-task)")
    parser.add_argument("--interval", type=float, default=None,
                        help="Inference interval in seconds (overrides config)")
    parser.add_argument("--classifier", choices=["rf", "mlp"], default=None,
                        help="Active classifier (overrides config)")
    parser.add_argument("--target-pose", default=None,
                        help="Target pose to practice (default: Virabhadrasana Two or Tadasana)")
    parser.add_argument("--free", action="store_true",
                        help="Start in free auto-detect mode")
    parser.add_argument("--no-voice", action="store_true",
                        help="Disable voice feedback")
    parser.add_argument("--test-voice", action="store_true",
                        help="Run audio speaker diagnostic test and exit")
    parser.add_argument("--history", action="store_true",
                        help="Display practice session history table from SQLite and exit")
    parser.add_argument("--progress", action="store_true",
                        help="Display comprehensive longitudinal progress and personalization report and exit")
    parser.add_argument("--camera", type=int, default=None,
                        help="Camera device index (overrides config)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Force 5-second body baseline calibration on startup")
    parser.add_argument("--video", default=None,
                        help="Replay a video file instead of the webcam (verification / demos)")
    parser.add_argument("--headless", action="store_true",
                        help="Do not open a window (use with --video for automated checks)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after this many frames")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Show the live diagnostic panel (also toggled with D)")
    parser.add_argument("--log-decisions", default=None,
                        help="Write one JSON line per inference (prediction, decision, feedback) to this file")
    parser.add_argument("--jev", action="store_true",
                        help="Enable Jev image assessment (requires JEVMODEL_API_KEY)")
    parser.add_argument("--no-jev", action="store_true",
                        help="Force local-only operation")
    parser.add_argument("--jev-every", type=float, default=None,
                        help="Send a frame to Jev every N seconds (0 = only on ambiguity / J)")
    parser.add_argument("--local-primary", action="store_true",
                        help="Use the local classifier + angle rules for decisions even when Jev is on")
    args = parser.parse_args()

    if args.test_voice:
        from src.app.voice import test_diagnostic
        success = test_diagnostic()
        sys.exit(0 if success else 1)

    if args.history:
        from src.correction.progress_tracker import ProgressTracker
        tracker = ProgressTracker()
        print("\n" + tracker.format_history_table(limit=15) + "\n")
        sys.exit(0)

    if args.progress:
        from src.correction.progress_tracker import ProgressTracker
        tracker = ProgressTracker()
        print("\n" + tracker.format_progress_report(limit=15) + "\n")
        sys.exit(0)


    cfg_path = ROOT / args.config
    if not cfg_path.exists():
        sys.exit(f"Config file not found: {cfg_path}")

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    cfg = _patch_config(cfg, args)

    from src.app.webcam_app import run
    run(cfg)


if __name__ == "__main__":
    main()
