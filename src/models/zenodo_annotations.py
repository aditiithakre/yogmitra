"""
src/models/zenodo_annotations.py
─────────────────────────────────
Utilities for parsing Zenodo Yoga Landmarks metadata annotations:
- Asana labels and Sanskrit/English canonical names
- Correctness labels ("Right" / "Wrong")
- Combined labels ("{Asana}_{Correctness}")
- Group ID semantics: step numbering (1 to 5) and body part fault tags
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# Canonical 10 Zenodo Asanas
ZENODO_ASANAS: list[str] = [
    "Anantasana",
    "Ardhakati Chakrasana",
    "Bhujangasana",
    "Kati Chakrasana",
    "Marjariasana",
    "Parvatasana",
    "Sarvangasana",
    "Tadasana",
    "Vajrasana",
    "Viparita Karani",
]

# Body part fault mapping from group_id substrings to canonical joint identifiers
FAULT_KEYWORD_MAP: dict[str, list[str]] = {
    "leg": ["left_knee_angle", "right_knee_angle", "left_hip_angle", "right_hip_angle"],
    "legs": ["left_knee_angle", "right_knee_angle", "left_hip_angle", "right_hip_angle"],
    "knee": ["left_knee_angle", "right_knee_angle"],
    "knees": ["left_knee_angle", "right_knee_angle"],
    "hand": ["left_elbow_angle", "right_elbow_angle", "left_shoulder_angle", "right_shoulder_angle"],
    "hands": ["left_elbow_angle", "right_elbow_angle", "left_shoulder_angle", "right_shoulder_angle"],
    "arm": ["left_elbow_angle", "right_elbow_angle"],
    "arms": ["left_elbow_angle", "right_elbow_angle"],
    "neck": ["torso_lean_from_vertical_deg"],
    "head": ["torso_lean_from_vertical_deg"],
    "palms": ["left_wrist_to_shoulder_norm", "right_wrist_to_shoulder_norm"],
    "back": ["torso_lean_from_vertical_deg", "hip_line_angle_deg"],
    "ankle": ["left_ankle_angle", "right_ankle_angle"],
}


@dataclass
class ZenodoAnnotation:
    """Parsed semantic representation of a Zenodo metadata row."""
    asana: str
    correctness: str                    # "Right" or "Wrong"
    step: Optional[int] = None          # 1 to 5 (or None if unstated)
    raw_fault: str = ""                 # e.g., "leg and hand", "neck"
    affected_body_parts: list[str] = field(default_factory=list)
    candidate_angles: list[str] = field(default_factory=list)


def parse_combined_label(combined_label: str) -> tuple[str, str]:
    """
    Split combined_label (e.g. 'Parvatasana_Right') into (asana, correctness).
    """
    if "_" in combined_label:
        parts = combined_label.rsplit("_", 1)
        return parts[0], parts[1]
    return combined_label, "Unknown"


def parse_group_id(group_id: str, asana_hint: Optional[str] = None) -> ZenodoAnnotation:
    """
    Parse group_id into structured annotation.
    Examples:
      'parvatasana::step 2' -> asana='Parvatasana', step=2, fault=''
      'parvatasana::step 3 leg and hand' -> step=3, fault='leg and hand'
      'tadasana::step 5 standing' -> step=5, fault='standing'
    """
    raw = str(group_id).strip()
    asana = asana_hint or "Unknown"
    step: Optional[int] = None
    fault = ""

    if "::" in raw:
        parts = raw.split("::", 1)
        asana_str = parts[0].strip().title()
        if asana_str:
            asana = asana_str
        rest = parts[1].strip()
    else:
        rest = raw

    # Match step number
    step_match = re.search(r"step\s*(\d+)", rest, re.IGNORECASE)
    if step_match:
        step = int(step_match.group(1))

    # Clean out the step token to isolate fault text
    fault_text = re.sub(r"step\s*\d+\s*[-:]*\s*", "", rest, flags=re.IGNORECASE).strip()
    fault = fault_text.lower() if fault_text else ""

    # Map fault text to affected body parts and candidate angles
    affected: list[str] = []
    candidate_angles: list[str] = []
    if fault:
        for kw, angles in FAULT_KEYWORD_MAP.items():
            if re.search(rf"\b{kw}\b", fault):
                if kw not in affected:
                    affected.append(kw)
                for ang in angles:
                    if ang not in candidate_angles:
                        candidate_angles.append(ang)

    correctness = "Wrong" if fault else "Right"

    return ZenodoAnnotation(
        asana=asana,
        correctness=correctness,
        step=step,
        raw_fault=fault,
        affected_body_parts=affected,
        candidate_angles=candidate_angles,
    )
