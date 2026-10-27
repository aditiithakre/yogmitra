"""
src/app/reference_display.py
────────────────────────────
Renders the dedicated right-hand Reference Asana panel for Yoga Mitra.

Features:
- Seamless two-panel interface integration (Right panel: Reference Asana & Cues)
- Displays target pose name (Sanskrit & English), reference illustration, key alignment cues
- Highlights active focus joint when a deviation is detected by the posture evaluator
- Temporal stability filter for Free Auto-Detect mode to prevent noisy pose flickering
- High-performance caching to preserve 30+ FPS rendering without repeated disk/resize ops
- Elegant fallback card when a reference image is unavailable
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.correction.reference_ranges import POSE_METADATA

logger = logging.getLogger(__name__)

# UI Color constants (BGR)
CLR_PANEL_BG   = (20, 24, 28)       # Dark slate
CLR_CARD_BG    = (30, 36, 44)       # Card background
CLR_BORDER     = (50, 60, 72)       # Border outline
CLR_TITLE      = (255, 255, 255)    # White
CLR_SUBTITLE   = (160, 200, 240)    # Soft cyan/blue
CLR_CUE_TEXT   = (210, 215, 225)    # Light gray text
CLR_ACCENT     = (80, 210, 120)     # Green accent
CLR_ALERT_BG   = (30, 30, 70)       # Dark red-amber banner
CLR_ALERT_TEXT = (90, 140, 255)     # Amber-orange highlight
CLR_MUTED      = (120, 130, 145)    # Muted secondary


class ReferenceDisplay:
    """
    Manages loading, caching, stability filtering, and rendering of the
    Reference Asana panel.
    """

    def __init__(
        self,
        assets_dir: Path | str = "assets/pose_references",
        panel_width: int = 380,
    ):
        self.assets_dir = Path(assets_dir)
        self.panel_width = panel_width
        self._image_cache: dict[str, np.ndarray] = {}

        # Free auto-detect stability filter
        self._detect_history: deque[str] = deque(maxlen=3)
        self._stable_pose: Optional[str] = None
        self._stability_threshold: int = 2

        # Last rendered panel cache (pose_name, focus_joint, height) -> panel
        self._cached_key: Optional[tuple[str, Optional[str], int]] = None
        self._cached_panel: Optional[np.ndarray] = None

    def reset_stability(self) -> None:
        """Clear auto-detection smoothing history."""
        self._detect_history.clear()
        self._stable_pose = None
        self._cached_key = None

    def update_auto_detected_pose(
        self,
        predicted_pose: Optional[str],
        confidence: float,
        min_confidence: float = 0.25,
    ) -> Optional[str]:
        """
        Smooths auto-detected pose predictions to prevent image flickering.
        Returns stable pose name if consistent, otherwise the previous stable pose.
        """
        if not predicted_pose or predicted_pose == "—" or confidence < min_confidence:
            self._detect_history.append("none")
        else:
            self._detect_history.append(predicted_pose)

        # Assessment no longer available for consecutive updates → stop showing a stale pose
        recent = list(self._detect_history)[-self._stability_threshold:]
        if len(recent) >= self._stability_threshold and all(p == "none" for p in recent):
            self._stable_pose = None
            return None

        # Count occurrences of the latest candidate
        if len(self._detect_history) >= self._stability_threshold:
            candidates = [p for p in self._detect_history if p != "none"]
            if candidates:
                from collections import Counter
                most_common, count = Counter(candidates).most_common(1)[0]
                if count >= self._stability_threshold:
                    self._stable_pose = most_common

        return self._stable_pose or (predicted_pose if confidence >= min_confidence else None)

    def _load_asset(self, asset_filename: str, max_size: int = 300) -> Optional[np.ndarray]:
        """Load and cache asset image from disk."""
        if asset_filename in self._image_cache:
            return self._image_cache[asset_filename]

        file_path = self.assets_dir / asset_filename
        if not file_path.exists():
            return None

        img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning("[ReferenceDisplay] Failed to read asset: %s", file_path)
            return None

        # Resize keeping aspect ratio
        ih, iw = img.shape[:2]
        scale = min(max_size / iw, max_size / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        self._image_cache[asset_filename] = resized
        return resized

    def render_panel(
        self,
        pose_name: Optional[str],
        height: int,
        focus_joint: Optional[str] = None,
        is_guided: bool = True,
        confidence: Optional[float] = None,
    ) -> np.ndarray:
        """
        Render the right reference panel (panel_width x height).
        Uses caching to avoid redraws unless pose or focus joint changes.
        """
        conf_key = int(round((confidence or 0.0) * 100)) if not is_guided else None
        cache_key = (str(pose_name), str(focus_joint), height, is_guided, conf_key)
        if self._cached_key == cache_key and self._cached_panel is not None:
            return self._cached_panel.copy()

        w = self.panel_width
        panel = np.zeros((height, w, 3), dtype=np.uint8)
        panel[:] = CLR_PANEL_BG

        # Left separator line
        cv2.line(panel, (0, 0), (0, height), CLR_BORDER, 2)

        meta = POSE_METADATA.get(pose_name) if pose_name else None

        # ── 1. Top Section: Mode Badge & Title ─────────────────────────────────
        badge_y = 28
        if not pose_name:
            mode_text = "READY — STANDING BY"
            badge_color = CLR_SUBTITLE
            english_name = "Waiting for Pose..."
            sanskrit_name = "Step into view & perform any asana"
        else:
            mode_text = "REAL-TIME REFERENCE"
            badge_color = CLR_ACCENT
            english_name = meta["english_name"] if meta else pose_name
            sanskrit_name = meta["sanskrit_name"] if meta else pose_name

        cv2.putText(panel, mode_text, (20, badge_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, badge_color, 1, cv2.LINE_AA)

        if confidence is not None and pose_name:
            conf_str = f"Match: {confidence*100:.0f}%"
            cv2.putText(panel, conf_str, (w - 110, badge_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, CLR_MUTED, 1, cv2.LINE_AA)

        title_y = badge_y + 30
        cv2.putText(panel, english_name, (20, title_y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.72, CLR_TITLE, 1, cv2.LINE_AA)
        cv2.putText(panel, sanskrit_name, (20, title_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, CLR_SUBTITLE, 1, cv2.LINE_AA)

        # ── 2. Reference Image or Fallback Card ────────────────────────────────
        img_box_top = title_y + 36
        img_box_height = 280
        img_box_w = w - 40
        card_rect = (20, img_box_top, img_box_w, img_box_height)

        cv2.rectangle(panel, (card_rect[0], card_rect[1]),
                      (card_rect[0] + card_rect[2], card_rect[1] + card_rect[3]),
                      CLR_CARD_BG, -1)
        cv2.rectangle(panel, (card_rect[0], card_rect[1]),
                      (card_rect[0] + card_rect[2], card_rect[1] + card_rect[3]),
                      CLR_BORDER, 1)

        asset_drawn = False
        if meta and "asset_name" in meta:
            asset_img = self._load_asset(meta["asset_name"], max_size=img_box_height - 16)
            if asset_img is not None:
                ah, aw = asset_img.shape[:2]
                ox = card_rect[0] + (card_rect[2] - aw) // 2
                oy = card_rect[1] + (card_rect[3] - ah) // 2
                if asset_img.ndim == 3 and asset_img.shape[2] == 4:
                    alpha = asset_img[:, :, 3].astype(float) / 255.0
                    for c in range(3):
                        panel[oy:oy+ah, ox:ox+aw, c] = (
                            alpha * asset_img[:, :, c] + (1.0 - alpha) * panel[oy:oy+ah, ox:ox+aw, c]
                        ).astype(np.uint8)
                else:
                    panel[oy:oy+ah, ox:ox+aw] = asset_img[:, :, :3]
                asset_drawn = True

        if not asset_drawn:
            # Fallback placeholder
            msg1 = "Perform any yoga asana"
            msg2 = "in view of the camera."
            msg3 = "Real-time AI will detect & guide."
            cv2.putText(panel, msg1, (card_rect[0] + 30, card_rect[1] + 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, CLR_MUTED, 1, cv2.LINE_AA)
            cv2.putText(panel, msg2, (card_rect[0] + 30, card_rect[1] + 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, CLR_MUTED, 1, cv2.LINE_AA)
            cv2.putText(panel, msg3, (card_rect[0] + 30, card_rect[1] + 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_BORDER, 1, cv2.LINE_AA)

        # ── 3. Active Correction Focus Banner ──────────────────────────────────
        cues_top = img_box_top + img_box_height + 18
        if focus_joint:
            banner_h = 32
            cv2.rectangle(panel, (20, cues_top), (w - 20, cues_top + banner_h), CLR_ALERT_BG, -1)
            cv2.rectangle(panel, (20, cues_top), (w - 20, cues_top + banner_h), CLR_ALERT_TEXT, 1)
            alert_msg = f"! FOCUS ON: {focus_joint.upper()}"
            cv2.putText(panel, alert_msg, (28, cues_top + 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, CLR_ALERT_TEXT, 1, cv2.LINE_AA)
            cues_top += banner_h + 16

        # ── 4. Key Alignment Cues ──────────────────────────────────────────────
        cv2.putText(panel, "KEY ALIGNMENT CUES", (20, cues_top),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (140, 160, 180), 1, cv2.LINE_AA)
        cues_top += 22

        cues = meta.get("cues", []) if meta else [
            "Maintain steady breathing throughout pose",
            "Root firmly through feet / contact points",
            "Keep spine long and core engaged",
        ]

        for cue in cues[:4]:
            # Draw small bullet
            cv2.circle(panel, (26, cues_top - 4), 3, CLR_ACCENT, -1, cv2.LINE_AA)

            # Wrap text if cue is long
            words = cue.split(" ")
            line1, line2 = "", ""
            for word in words:
                test_line = f"{line1} {word}".strip()
                if len(test_line) * 7.5 > (w - 55):
                    line2 += f" {word}"
                else:
                    line1 = test_line

            cv2.putText(panel, line1, (36, cues_top),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, CLR_CUE_TEXT, 1, cv2.LINE_AA)
            cues_top += 18
            if line2:
                cv2.putText(panel, line2.strip(), (36, cues_top),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, CLR_CUE_TEXT, 1, cv2.LINE_AA)
                cues_top += 18

        # Cache this panel
        self._cached_key = cache_key
        self._cached_panel = panel
        return panel.copy()


# Guided practice pose collections by model profile
KAGGLE_GUIDED_POSES: list[str] = [
    "Virabhadrasana Two",
    "Virabhadrasana One",
    "Utkatasana",
    "Trikonasana",
    "Phalakasana",
    "Adho Mukha Svanasana",
    "Vrksasana",
    "Uttanasana",
    "Anjaneyasana",
]

ZENODO_GUIDED_POSES: list[str] = [
    "Tadasana",
    "Bhujangasana",
    "Marjariasana",
    "Parvatasana",
    "Ardhakati Chakrasana",
    "Kati Chakrasana",
    "Vajrasana",
    "Anantasana",
    "Sarvangasana",
    "Viparita Karani",
]


def get_target_poses_for_profile(profile: str = "kaggle") -> list[str]:
    """Return available guided practice poses for the specified model profile."""
    if profile.lower() == "zenodo":
        return list(ZENODO_GUIDED_POSES)
    return list(KAGGLE_GUIDED_POSES)

