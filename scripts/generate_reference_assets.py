"""
scripts/generate_reference_assets.py
────────────────────────────────────
Generates professional anatomical schematic reference images for all key yoga poses
and saves them to assets/pose_references/.
"""

import math
from pathlib import Path
import cv2
import numpy as np

OUTPUT_DIR = Path("assets/pose_references")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Aesthetic palette
BG_COLOR = (24, 28, 32)         # Deep slate dark BGR
BONE_COLOR = (210, 190, 160)     # Warm bone cyan-tint
JOINT_COLOR = (60, 210, 255)     # Glowing amber/cyan
HEAD_COLOR = (220, 220, 220)
ACCENT_COLOR = (80, 220, 120)    # Green alignment highlight
GRID_COLOR = (35, 40, 48)
TEXT_COLOR = (240, 240, 240)


def create_base_canvas(title: str, subtitle: str) -> np.ndarray:
    w, h = 600, 600
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = BG_COLOR

    # Subtle background grid
    for x in range(0, w, 40):
        cv2.line(img, (x, 0), (x, h), GRID_COLOR, 1)
    for y in range(0, h, 40):
        cv2.line(img, (0, y), (w, y), GRID_COLOR, 1)

    # Floor line
    cv2.line(img, (50, 480), (550, 480), (70, 80, 95), 2, cv2.LINE_AA)

    # Header
    cv2.putText(img, title, (30, 45), cv2.FONT_HERSHEY_DUPLEX, 0.85, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(img, subtitle, (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 180, 200), 1, cv2.LINE_AA)
    return img


def draw_skeleton(img: np.ndarray, joints: dict[str, tuple[int, int]], bones: list[tuple[str, str]], cues: list[tuple[str, tuple[int, int], str]]):
    # Draw bones
    for a, b in bones:
        if a in joints and b in joints:
            cv2.line(img, joints[a], joints[b], BONE_COLOR, 4, cv2.LINE_AA)

    # Draw head
    if "head" in joints:
        cv2.circle(img, joints["head"], 24, HEAD_COLOR, -1, cv2.LINE_AA)
        cv2.circle(img, joints["head"], 24, (100, 110, 120), 2, cv2.LINE_AA)

    # Draw joints
    for name, pt in joints.items():
        if name != "head":
            cv2.circle(img, pt, 7, JOINT_COLOR, -1, cv2.LINE_AA)
            cv2.circle(img, pt, 9, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw alignment cues / angle labels
    for text, pos, color_type in cues:
        col = ACCENT_COLOR if color_type == "good" else (60, 160, 255)
        # Small backdrop for text readability
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        x, y = pos
        cv2.rectangle(img, (x - 4, y - th - 4), (x + tw + 4, y + 4), (20, 22, 26), -1)
        cv2.rectangle(img, (x - 4, y - th - 4), (x + tw + 4, y + 4), col, 1)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)


# Standard bone connections
STANDARD_BONES = [
    ("head", "neck"),
    ("neck", "left_shoulder"), ("neck", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("neck", "mid_hip"),
    ("mid_hip", "left_hip"), ("mid_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]


def gen_warrior_two():
    img = create_base_canvas("Warrior II", "Virabhadrasana II")
    joints = {
        "head": (290, 180),
        "neck": (290, 220),
        "left_shoulder": (270, 230),
        "right_shoulder": (310, 230),
        "left_elbow": (190, 230),
        "left_wrist": (120, 230),
        "right_elbow": (390, 230),
        "right_wrist": (480, 230),
        "mid_hip": (290, 330),
        "left_hip": (270, 330),
        "right_hip": (310, 330),
        "left_knee": (210, 400),
        "left_ankle": (210, 480),
        "right_knee": (380, 400),
        "right_ankle": (450, 480),
    }
    cues = [
        ("Front Knee 90 deg", (150, 420), "good"),
        ("Rear Leg Straight (180 deg)", (380, 440), "good"),
        ("Arms Parallel to Floor", (220, 205), "good"),
        ("Torso Upright", (310, 300), "good"),
    ]
    draw_skeleton(img, joints, STANDARD_BONES, cues)
    cv2.imwrite(str(OUTPUT_DIR / "virabhadrasana_two.png"), img)


def gen_warrior_one():
    img = create_base_canvas("Warrior I", "Virabhadrasana I")
    joints = {
        "head": (280, 200),
        "neck": (280, 240),
        "left_shoulder": (265, 245),
        "right_shoulder": (295, 245),
        "left_elbow": (260, 150),
        "left_wrist": (260, 90),
        "right_elbow": (300, 150),
        "right_wrist": (300, 90),
        "mid_hip": (280, 340),
        "left_hip": (260, 340),
        "right_hip": (300, 340),
        "left_knee": (200, 400),
        "left_ankle": (200, 480),
        "right_knee": (370, 410),
        "right_ankle": (440, 480),
    }
    cues = [
        ("Arms Overhead (180 deg)", (210, 75), "good"),
        ("Front Knee 90 deg", (140, 420), "good"),
        ("Back Leg Straight", (370, 440), "good"),
        ("Chest Forward", (300, 280), "good"),
    ]
    draw_skeleton(img, joints, STANDARD_BONES, cues)
    cv2.imwrite(str(OUTPUT_DIR / "virabhadrasana_one.png"), img)


def gen_chair():
    img = create_base_canvas("Chair Pose", "Utkatasana")
    joints = {
        "head": (260, 190),
        "neck": (275, 230),
        "left_shoulder": (265, 235),
        "right_shoulder": (285, 235),
        "left_elbow": (230, 150),
        "left_wrist": (200, 100),
        "right_elbow": (250, 150),
        "right_wrist": (220, 100),
        "mid_hip": (350, 340),
        "left_hip": (340, 340),
        "right_hip": (360, 340),
        "left_knee": (270, 400),
        "left_ankle": (300, 480),
        "right_knee": (280, 400),
        "right_ankle": (310, 480),
    }
    cues = [
        ("Arms Extended Overhead", (140, 85), "good"),
        ("Knees Bent 70-110 deg", (180, 420), "good"),
        ("Hips Sunk Back", (370, 340), "good"),
        ("Weight in Heels", (330, 495), "good"),
    ]
    draw_skeleton(img, joints, STANDARD_BONES, cues)
    cv2.imwrite(str(OUTPUT_DIR / "utkatasana.png"), img)


def gen_triangle():
    img = create_base_canvas("Triangle Pose", "Trikonasana")
    joints = {
        "head": (240, 310),
        "neck": (260, 330),
        "left_shoulder": (250, 340),
        "right_shoulder": (270, 320),
        "left_elbow": (220, 400),
        "left_wrist": (200, 460),
        "right_elbow": (290, 240),
        "right_wrist": (320, 170),
        "mid_hip": (340, 360),
        "left_hip": (320, 360),
        "right_hip": (360, 360),
        "left_knee": (260, 420),
        "left_ankle": (200, 480),
        "right_knee": (390, 420),
        "right_ankle": (440, 480),
    }
    cues = [
        ("Top Arm Reaching Up", (300, 150), "good"),
        ("Both Legs Straight (180 deg)", (270, 450), "good"),
        ("Bottom Hand to Shin/Floor", (140, 440), "good"),
        ("Open Chest Sideways", (280, 300), "good"),
    ]
    draw_skeleton(img, joints, STANDARD_BONES, cues)
    cv2.imwrite(str(OUTPUT_DIR / "trikonasana.png"), img)


def gen_plank():
    img = create_base_canvas("Plank Pose", "Phalakasana")
    joints = {
        "head": (140, 310),
        "neck": (170, 330),
        "left_shoulder": (180, 340),
        "right_shoulder": (180, 340),
        "left_elbow": (180, 410),
        "left_wrist": (180, 480),
        "right_elbow": (180, 410),
        "right_wrist": (180, 480),
        "mid_hip": (310, 370),
        "left_hip": (310, 370),
        "right_hip": (310, 370),
        "left_knee": (410, 415),
        "left_ankle": (500, 480),
        "right_knee": (410, 415),
        "right_ankle": (500, 480),
    }
    cues = [
        ("Wrists Under Shoulders", (110, 460), "good"),
        ("Straight Line: Head to Heels", (240, 340), "good"),
        ("Hips Neutral (No Sag)", (330, 350), "good"),
        ("Legs Straight & Active", (410, 440), "good"),
    ]
    draw_skeleton(img, joints, STANDARD_BONES, cues)
    cv2.imwrite(str(OUTPUT_DIR / "phalakasana.png"), img)


def gen_downward_dog():
    img = create_base_canvas("Downward Dog", "Adho Mukha Svanasana")
    joints = {
        "head": (210, 380),
        "neck": (220, 360),
        "left_shoulder": (220, 360),
        "right_shoulder": (220, 360),
        "left_elbow": (180, 420),
        "left_wrist": (140, 480),
        "right_elbow": (180, 420),
        "right_wrist": (140, 480),
        "mid_hip": (300, 240),
        "left_hip": (300, 240),
        "right_hip": (300, 240),
        "left_knee": (380, 360),
        "left_ankle": (450, 480),
        "right_knee": (380, 360),
        "right_ankle": (450, 480),
    }
    cues = [
        ("Inverted V-Shape", (320, 220), "good"),
        ("Straight Arms & Spine", (150, 340), "good"),
        ("Hips High Toward Sky", (330, 250), "good"),
        ("Legs Straight, Heels Grounding", (380, 420), "good"),
    ]
    draw_skeleton(img, joints, STANDARD_BONES, cues)
    cv2.imwrite(str(OUTPUT_DIR / "adho_mukha_svanasana.png"), img)


def gen_tree():
    img = create_base_canvas("Tree Pose", "Vrksasana")
    joints = {
        "head": (300, 150),
        "neck": (300, 190),
        "left_shoulder": (275, 205),
        "right_shoulder": (325, 205),
        "left_elbow": (260, 250),
        "left_wrist": (290, 260),
        "right_elbow": (340, 250),
        "right_wrist": (310, 260),
        "mid_hip": (300, 320),
        "left_hip": (285, 320),
        "right_hip": (315, 320),
        "left_knee": (285, 400),
        "left_ankle": (285, 480),
        "right_knee": (390, 370),
        "right_ankle": (315, 385),
    }
    cues = [
        ("Hands at Heart (Prayer)", (250, 240), "good"),
        ("Standing Leg Straight (180 deg)", (160, 420), "good"),
        ("Bent Knee Open Sideways", (380, 350), "good"),
        ("Foot on Calf or Thigh", (330, 400), "good"),
    ]
    draw_skeleton(img, joints, STANDARD_BONES, cues)
    cv2.imwrite(str(OUTPUT_DIR / "vrksasana.png"), img)


def gen_forward_fold():
    img = create_base_canvas("Forward Fold", "Uttanasana")
    joints = {
        "head": (270, 400),
        "neck": (275, 370),
        "left_shoulder": (270, 360),
        "right_shoulder": (280, 360),
        "left_elbow": (260, 420),
        "left_wrist": (255, 480),
        "right_elbow": (260, 420),
        "right_wrist": (255, 480),
        "mid_hip": (330, 260),
        "left_hip": (320, 260),
        "right_hip": (340, 260),
        "left_knee": (315, 370),
        "left_ankle": (310, 480),
        "right_knee": (325, 370),
        "right_ankle": (320, 480),
    }
    cues = [
        ("Hinge Deeply at Hips", (350, 250), "good"),
        ("Spine Hanging Long", (160, 370), "good"),
        ("Legs Straight (Micro-bend OK)", (340, 390), "good"),
        ("Hands to Floor or Shins", (150, 470), "good"),
    ]
    draw_skeleton(img, joints, STANDARD_BONES, cues)
    cv2.imwrite(str(OUTPUT_DIR / "uttanasana.png"), img)


def gen_low_lunge():
    img = create_base_canvas("Low Lunge", "Anjaneyasana")
    joints = {
        "head": (270, 180),
        "neck": (270, 220),
        "left_shoulder": (255, 230),
        "right_shoulder": (285, 230),
        "left_elbow": (240, 150),
        "left_wrist": (230, 90),
        "right_elbow": (290, 150),
        "right_wrist": (300, 90),
        "mid_hip": (280, 350),
        "left_hip": (270, 350),
        "right_hip": (290, 350),
        "left_knee": (200, 400),
        "left_ankle": (200, 480),
        "right_knee": (400, 470),
        "right_ankle": (480, 480),
    }
    cues = [
        ("Front Knee Stacked at 90 deg", (130, 420), "good"),
        ("Back Knee Resting on Mat", (370, 450), "good"),
        ("Hips Sinking Forward & Down", (310, 340), "good"),
        ("Chest & Arms Lifted", (270, 120), "good"),
    ]
    draw_skeleton(img, joints, STANDARD_BONES, cues)
    cv2.imwrite(str(OUTPUT_DIR / "anjaneyasana.png"), img)


def main():
    gen_warrior_two()
    gen_warrior_one()
    gen_chair()
    gen_triangle()
    gen_plank()
    gen_downward_dog()
    gen_tree()
    gen_forward_fold()
    gen_low_lunge()
    print(f"Generated 9 reference pose images in {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
