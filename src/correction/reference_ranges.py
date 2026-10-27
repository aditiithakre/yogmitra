"""
src/correction/reference_ranges.py
────────────────────────────────────
Pose-specific joint-angle reference ranges, descriptive metadata,
and key alignment cues for real-time guidance.
"""

from __future__ import annotations

# Canonical 10 Zenodo guided asanas
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

# ── Reference angle ranges ───────────────────────────────────────────────────

REFERENCE_RANGES: dict[str, dict[str, tuple[float, float]]] = {

    "Virabhadrasana One": {          # Warrior 1
        "left_knee_angle":      (85, 115),
        "right_knee_angle":     (160, 185),
        "left_elbow_angle":     (160, 185),
        "right_elbow_angle":    (160, 185),
        "left_shoulder_angle":  (150, 185),
        "right_shoulder_angle": (150, 185),
    },

    "Virabhadrasana Two": {          # Warrior 2
        "left_knee_angle":      (85, 115),
        "right_knee_angle":     (160, 185),
        "left_elbow_angle":     (160, 185),
        "right_elbow_angle":    (160, 185),
    },

    "Virabhadrasana Three": {        # Warrior 3 / T-balance
        "left_knee_angle":      (165, 185),
        "right_hip_angle":      (85, 105),
        "left_elbow_angle":     (160, 185),
        "right_elbow_angle":    (160, 185),
    },

    "Utkatasana": {                  # Chair Pose
        "left_knee_angle":      (70, 115),
        "right_knee_angle":     (70, 115),
        "left_hip_angle":       (60, 105),
        "right_hip_angle":      (60, 105),
        "left_elbow_angle":     (160, 185),
        "right_elbow_angle":    (160, 185),
    },

    "Phalakasana": {                 # High Plank
        "left_elbow_angle":     (165, 185),
        "right_elbow_angle":    (165, 185),
        "left_knee_angle":      (165, 185),
        "right_knee_angle":     (165, 185),
        "left_hip_angle":       (160, 195),
        "right_hip_angle":      (160, 195),
    },

    "Adho Mukha Svanasana": {        # Downward-Facing Dog
        "left_knee_angle":      (155, 185),
        "right_knee_angle":     (155, 185),
        "left_shoulder_angle":  (145, 185),
        "right_shoulder_angle": (145, 185),
    },

    "Trikonasana": {                 # Triangle Pose
        "left_knee_angle":      (160, 185),
        "right_knee_angle":     (160, 185),
        "left_hip_angle":       (60, 105),
        "right_hip_angle":      (60, 105),
    },

    "Vrksasana": {                   # Tree Pose
        "left_knee_angle":      (165, 185),
        "right_knee_angle":     (35, 75),
    },

    "Uttanasana": {                  # Standing Forward Bend
        "left_knee_angle":      (155, 185),
        "right_knee_angle":     (155, 185),
    },

    "Utthita Parsvakonasana": {      # Extended Side Angle
        "left_knee_angle":      (85, 115),
        "right_knee_angle":     (160, 185),
        "left_elbow_angle":     (155, 185),
        "right_elbow_angle":    (155, 185),
    },

    "Parsva Virabhadrasana": {       # Reverse Warrior
        "left_knee_angle":      (85, 115),
        "right_knee_angle":     (160, 185),
    },

    "Balasana": {                    # Child's Pose
        "left_knee_angle":      (0, 50),
        "right_knee_angle":     (0, 50),
        "left_hip_angle":       (0, 60),
        "right_hip_angle":      (0, 60),
    },

    "Anjaneyasana": {                # Low Lunge
        "left_knee_angle":      (85, 115),
        "right_knee_angle":     (160, 195),
    },

    # ── Zenodo 10-Asana Vocabulary (Provisional Biomechanical Baselines) ──────
    "Anantasana": {                  # Side-Reclining Sleeping Vishnu
        "left_elbow_angle":     (120, 180),
        "right_elbow_angle":    (120, 180),
        "left_knee_angle":      (150, 185),
        "right_knee_angle":     (145, 185),
        "left_hip_angle":       (120, 180),
        "right_hip_angle":      (120, 180),
    },

    "Ardhakati Chakrasana": {        # Lateral Side Bend / Half Wheel
        "left_elbow_angle":     (150, 185),
        "right_elbow_angle":    (150, 185),
        "left_shoulder_angle":  (40, 180),
        "right_shoulder_angle": (40, 180),
        "left_knee_angle":      (160, 185),
        "right_knee_angle":     (160, 185),
    },

    "Bhujangasana": {                # Cobra Pose
        "left_elbow_angle":     (120, 170),
        "right_elbow_angle":    (120, 170),
        "left_shoulder_angle":  (30, 80),
        "right_shoulder_angle": (30, 80),
        "left_knee_angle":      (160, 185),
        "right_knee_angle":     (160, 185),
        "left_hip_angle":       (140, 180),
        "right_hip_angle":      (140, 180),
    },

    "Kati Chakrasana": {             # Standing Waist Rotating Twist
        "left_elbow_angle":     (80, 165),
        "right_elbow_angle":    (80, 165),
        "left_knee_angle":      (160, 185),
        "right_knee_angle":     (160, 185),
        "left_shoulder_angle":  (35, 120),
        "right_shoulder_angle": (35, 120),
    },

    "Marjariasana": {                # Cat-Cow Pose / Tabletop
        "left_elbow_angle":     (155, 185),
        "right_elbow_angle":    (155, 185),
        "left_shoulder_angle":  (70, 110),
        "right_shoulder_angle": (70, 110),
        "left_hip_angle":       (75, 110),
        "right_hip_angle":      (75, 110),
        "left_knee_angle":      (75, 115),
        "right_knee_angle":     (75, 115),
    },

    "Parvatasana": {                 # Seated Mountain Pose (arms overhead, seated/kneeling)
        "left_elbow_angle":     (150, 185),
        "right_elbow_angle":    (150, 185),
        "left_shoulder_angle":  (120, 185),
        "right_shoulder_angle": (120, 185),
    },

    "Sarvangasana": {                # Supported Shoulder Stand
        "left_knee_angle":      (165, 185),
        "right_knee_angle":     (165, 185),
        "left_hip_angle":       (160, 185),
        "right_hip_angle":      (160, 185),
        "left_elbow_angle":     (60, 110),
        "right_elbow_angle":    (60, 110),
    },

    "Tadasana": {                    # Mountain Pose
        "left_knee_angle":      (165, 185),
        "right_knee_angle":     (165, 185),
        "left_hip_angle":       (165, 185),
        "right_hip_angle":      (165, 185),
        "left_elbow_angle":     (155, 185),
        "right_elbow_angle":    (155, 185),
    },

    "Vajrasana": {                   # Thunderbolt / Kneeling Pose
        "left_elbow_angle":     (120, 185),
        "right_elbow_angle":    (120, 185),
        "left_shoulder_angle":  (35, 95),
        "right_shoulder_angle": (35, 95),
    },

    "Viparita Karani": {             # Inverted Lake / Legs-Up Pose
        "left_knee_angle":      (160, 185),
        "right_knee_angle":     (160, 185),
        "left_hip_angle":       (70, 115),
        "right_hip_angle":      (70, 115),
        "left_elbow_angle":     (140, 185),
        "right_elbow_angle":    (140, 185),
    },
}


# ── Joint Display Names ───────────────────────────────────────────────────────
JOINT_DISPLAY_NAMES: dict[str, str] = {
    "left_knee_angle":      "left knee",
    "right_knee_angle":     "right knee",
    "left_elbow_angle":     "left arm",
    "right_elbow_angle":    "right arm",
    "left_shoulder_angle":  "left shoulder",
    "right_shoulder_angle": "right shoulder",
    "left_hip_angle":       "left hip",
    "right_hip_angle":      "right hip",
    "left_ankle_angle":     "left ankle",
    "right_ankle_angle":    "right ankle",
}

# ── Pose Metadata & Reference Cues ───────────────────────────────────────────
POSE_METADATA: dict[str, dict] = {
    "Virabhadrasana Two": {
        "english_name": "Warrior II",
        "sanskrit_name": "Virabhadrasana II",
        "asset_name": "virabhadrasana_two.png",
        "cues": [
            "Bend front knee to 90 deg directly over ankle",
            "Keep back leg straight and grounded",
            "Extend arms parallel to floor at shoulder height",
            "Keep torso centered and upright over pelvis",
        ],
        "key_joints": ["front knee", "back leg", "arms", "torso"],
    },
    "Virabhadrasana One": {
        "english_name": "Warrior I",
        "sanskrit_name": "Virabhadrasana I",
        "asset_name": "virabhadrasana_one.png",
        "cues": [
            "Deep front knee bend at 90 deg",
            "Extend back leg straight with foot anchored",
            "Reach both arms high overhead with palms facing",
            "Square chest and hips toward the front",
        ],
        "key_joints": ["front knee", "back leg", "arms overhead"],
    },
    "Utkatasana": {
        "english_name": "Chair Pose",
        "sanskrit_name": "Utkatasana",
        "asset_name": "utkatasana.png",
        "cues": [
            "Bend knees deeply and sit hips back",
            "Keep weight centered in heels",
            "Extend arms overhead alongside ears",
            "Lengthen spine and lift chest forward",
        ],
        "key_joints": ["knees", "hips", "arms"],
    },
    "Trikonasana": {
        "english_name": "Triangle Pose",
        "sanskrit_name": "Trikonasana",
        "asset_name": "trikonasana.png",
        "cues": [
            "Both legs straight without locking knees",
            "Reach bottom hand down toward shin or floor",
            "Extend top arm straight up toward sky",
            "Keep torso in line with front leg, chest open",
        ],
        "key_joints": ["legs", "arms", "side body"],
    },
    "Phalakasana": {
        "english_name": "Plank Pose",
        "sanskrit_name": "Phalakasana",
        "asset_name": "phalakasana.png",
        "cues": [
            "Arms straight with wrists under shoulders",
            "Body forms a straight diagonal line from head to heels",
            "Engage core without letting hips sag or pike",
            "Keep legs firm and heels pressing back",
        ],
        "key_joints": ["arms", "hips", "legs"],
    },
    "Adho Mukha Svanasana": {
        "english_name": "Downward Dog",
        "sanskrit_name": "Adho Mukha Svanasana",
        "asset_name": "adho_mukha_svanasana.png",
        "cues": [
            "Hands shoulder-width apart, fingers spread wide",
            "Press hips high creating an inverted V-shape",
            "Straighten arms and press chest toward thighs",
            "Lengthen legs and draw heels toward floor",
        ],
        "key_joints": ["shoulders", "spine", "legs"],
    },
    "Vrksasana": {
        "english_name": "Tree Pose",
        "sanskrit_name": "Vrksasana",
        "asset_name": "vrksasana.png",
        "cues": [
            "Stand tall on standing leg, root into ground",
            "Place other foot on inner calf or thigh (avoid knee)",
            "Hands at chest in prayer or extended overhead",
            "Find a steady gaze point for balance",
        ],
        "key_joints": ["standing leg", "bent knee", "balance"],
    },
    "Uttanasana": {
        "english_name": "Forward Fold",
        "sanskrit_name": "Uttanasana",
        "asset_name": "uttanasana.png",
        "cues": [
            "Hinge forward at hip joints, not waist",
            "Keep spine long and fold torso over legs",
            "Legs straight with gentle micro-bend if tight",
            "Let head and neck hang completely relaxed",
        ],
        "key_joints": ["hips", "hamstrings", "spine"],
    },
    "Anjaneyasana": {
        "english_name": "Low Lunge",
        "sanskrit_name": "Anjaneyasana",
        "asset_name": "anjaneyasana.png",
        "cues": [
            "Front knee stacked directly above ankle at 90 deg",
            "Back knee rested gently on mat, leg extended",
            "Sweep arms overhead and lift chest",
            "Sink hips forward and down comfortably",
        ],
        "key_joints": ["front knee", "hips", "arms"],
    },

    # ── Zenodo 10 Asanas Metadata ─────────────────────────────────────────────
    "Anantasana": {
        "english_name": "Sleeping Vishnu Pose",
        "sanskrit_name": "Anantasana",
        "asset_name": "anantasana.png",
        "cues": [
            "Lie on your side with bottom arm bent supporting your head",
            "Keep torso and bottom leg in a straight aligned axis",
            "Raise top leg upward and hold toe or maintain vertical reach",
            "Engage core to maintain side balance without rolling back",
        ],
        "key_joints": ["top leg", "bottom arm", "torso line"],
        "provisional": True,
    },

    "Ardhakati Chakrasana": {
        "english_name": "Lateral Side Bend",
        "sanskrit_name": "Ardhakati Chakrasana",
        "asset_name": "ardhakati_chakrasana.png",
        "cues": [
            "Stand tall with feet together and weight evenly distributed",
            "Raise one arm straight up alongside your ear",
            "Bend torso laterally to the opposite side while exhaling",
            "Keep both knees straight and hips facing forward",
        ],
        "key_joints": ["raised arm", "lateral torso", "knees"],
        "provisional": True,
    },

    "Bhujangasana": {
        "english_name": "Cobra Pose",
        "sanskrit_name": "Bhujangasana",
        "asset_name": "bhujangasana.png",
        "cues": [
            "Lie prone with palms under shoulders and elbows tucked close",
            "Inhale and lift chest smoothly using back muscles",
            "Keep elbows slightly bent, shoulders rolled down away from ears",
            "Keep tops of feet, thighs, and pelvis firmly grounded",
        ],
        "key_joints": ["elbows", "shoulders", "chest lift", "lower back"],
        "provisional": True,
    },

    "Kati Chakrasana": {
        "english_name": "Standing Spinal Twist",
        "sanskrit_name": "Kati Chakrasana",
        "asset_name": "kati_chakrasana.png",
        "cues": [
            "Stand with feet shoulder-width apart and arms relaxed",
            "Rotate torso smoothly from the waist toward one side",
            "Wrap front arm across chest to opposite shoulder",
            "Keep hips relatively stable and turn head gently with the twist",
        ],
        "key_joints": ["torso twist", "shoulders", "knees"],
        "provisional": True,
    },

    "Marjariasana": {
        "english_name": "Cat-Cow Pose",
        "sanskrit_name": "Marjariasana",
        "asset_name": "marjariasana.png",
        "cues": [
            "Begin on hands and knees in tabletop position",
            "Stack wrists directly under shoulders and knees under hips",
            "Exhale to arch spine upward toward ceiling (Cat)",
            "Inhale to drop belly and lift chest and tailbone (Cow)",
        ],
        "key_joints": ["shoulders", "spine curve", "hips", "knees"],
        "provisional": True,
    },

    "Parvatasana": {
        "english_name": "Mountain / Seated Mountain Pose",
        "sanskrit_name": "Parvatasana",
        "asset_name": "parvatasana.png",
        "cues": [
            "Sit in Padmasana or stand tall, lengthening through spine",
            "Interlock fingers and extend arms straight up overhead",
            "Turn palms upward and press toward ceiling",
            "Keep elbows straight and shoulders relaxed away from neck",
        ],
        "key_joints": ["arms overhead", "elbows", "spine lengthening"],
        "provisional": True,
    },

    "Sarvangasana": {
        "english_name": "Shoulder Stand",
        "sanskrit_name": "Sarvangasana",
        "asset_name": "sarvangasana.png",
        "cues": [
            "Lift legs, hips, and lower back vertically off mat",
            "Support middle/upper back firmly with palms, elbows grounded",
            "Straighten legs toward ceiling, drawing feet over shoulders",
            "Keep weight on shoulders and upper arms, never on neck",
        ],
        "key_joints": ["vertical legs", "elbow support", "spine vertical"],
        "provisional": True,
    },

    "Tadasana": {
        "english_name": "Mountain Pose",
        "sanskrit_name": "Tadasana",
        "asset_name": "tadasana.png",
        "cues": [
            "Stand with feet together or hip-width, grounded through all four corners",
            "Engage quadriceps and lift kneecaps without locking",
            "Lengthen spine upward and roll shoulders back and down",
            "Arms rest comfortably by sides or extend overhead with palms facing",
        ],
        "key_joints": ["knees straight", "neutral hips", "upright spine"],
        "provisional": True,
    },

    "Vajrasana": {
        "english_name": "Thunderbolt Pose",
        "sanskrit_name": "Vajrasana",
        "asset_name": "Vajrasana.png",
        "cues": [
            "Kneel on mat with big toes touching and heels parted",
            "Sit buttocks comfortably in cavity between heels",
            "Keep spine, neck, and head in a straight vertical line",
            "Rest palms on thighs with elbows relaxed",
        ],
        "key_joints": ["knee flexion", "hip flexion", "upright spine"],
        "provisional": True,
    },

    "Viparita Karani": {
        "english_name": "Inverted Lake / Legs-Up Pose",
        "sanskrit_name": "Viparita Karani",
        "asset_name": "viparita_karani.png",
        "cues": [
            "Lie on back and elevate legs vertically (against wall or supported)",
            "Keep legs straight and relaxed together",
            "Rest arms out to sides with palms facing upward",
            "Allow pelvis and lower back to rest comfortably on mat or bolster",
        ],
        "key_joints": ["legs vertical", "relaxed hips", "grounded torso"],
        "provisional": True,
    },
}

