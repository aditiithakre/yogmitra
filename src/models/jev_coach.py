"""
src/models/jev_coach.py
───────────────────────
Jev-driven posture coaching: per-pose checkpoints asked as Jev questions, and the
mapping from Jev's answers back to on-screen / spoken instructions.

Each checkpoint is one narrow Choice question about one body part ("Is the back
upright, slouching or leaning back?").  Every option carries the instruction to
give when Jev picks it (None = nothing to correct).  A shared "not_visible" option
lets Jev say the body part cannot be judged instead of guessing.

The checkpoints for the pose being practised are sent in the same request as the
asana question, so they share the image's input cost.  Jev judges directions from
the picture; it does not measure angles, so instructions say *what* to move and
*which way*, and the overall Score question gives a 0–100 form rating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# MediaPipe landmark ids, used to highlight the body part an instruction is about
NOSE = [0]
SHOULDERS = [11, 12]
ELBOWS = [13, 14]
WRISTS = [15, 16]
ARMS = ELBOWS + WRISTS
HIPS = [23, 24]
KNEES = [25, 26]
ANKLES = [27, 28]
TORSO = SHOULDERS + HIPS
LEGS = KNEES + ANKLES

NOT_VISIBLE_KEY = "not_visible"
NOT_VISIBLE_TEXT = "this part of the body cannot be seen clearly in the photo"

FORM_SCORE_LEVELS = [
    "poor: the posture is far from the intended pose",
    "fair: the pose is recognisable but has clear alignment problems",
    "good: the pose is mostly correct with small alignment issues",
    "excellent: the pose matches the intended alignment closely",
]


@dataclass(frozen=True)
class Checkpoint:
    key: str                       # question id sent to Jev (unique within a pose)
    body_part: str                 # display name, e.g. "back"
    joints: tuple[int, ...]        # landmark ids to highlight when a correction is issued
    question: str                  # what Jev is asked about the person in `photo`
    options: dict[str, tuple[str, Optional[str]]]   # option key → (criterion for Jev, instruction or None)


def _cp(key: str, body_part: str, joints: list[int], question: str,
        **options: tuple[str, Optional[str]]) -> Checkpoint:
    return Checkpoint(key, body_part, tuple(joints), question, options)


CHECKPOINTS: dict[str, list[Checkpoint]] = {
    "Tadasana": [
        _cp("tad_spine", "back", TORSO, "Is the body standing tall and vertical?",
            ok=("body vertical, head over shoulders over hips", None),
            leaning_forward=("slouching or leaning forward", "Stand tall: lift your chest and bring your shoulders back over your hips."),
            leaning_back=("leaning backward", "Shift your weight forward slightly so your body is vertical."),
            tilted=("tilting to one side", "Straighten up and spread your weight evenly over both feet.")),
        _cp("tad_knees", "knees", KNEES, "Are both knees straight?",
            ok=("both legs straight", None),
            bent=("one or both knees visibly bent", "Straighten both knees.")),
        _cp("tad_arms", "arms", ARMS, "How are the arms held?",
            ok=("arms straight, relaxed at the sides or stretched fully overhead", None),
            bent=("elbows bent", "Straighten your arms fully."),
            uneven=("one arm higher than the other", "Hold both arms at the same height.")),
        _cp("tad_feet", "feet", ANKLES, "How are the feet placed?",
            ok=("feet together or hip-width apart, flat on the floor", None),
            too_wide=("feet much wider than hip width", "Bring your feet closer together.")),
    ],
    "Kati Chakrasana": [
        _cp("kati_twist", "waist", TORSO, "How far is the upper body twisted?",
            ok=("upper body clearly rotated to one side", None),
            too_little=("barely twisted, chest still facing the camera", "Twist further from your waist, turning your chest and head to the side."),
            hips_turning=("the hips turn along with the chest", "Keep your hips facing forward and twist only from the waist.")),
        _cp("kati_spine", "back", TORSO, "Is the back upright during the twist?",
            ok=("spine upright", None),
            leaning=("leaning forward, backward or sideways", "Keep your spine upright while you twist; don't lean.")),
        _cp("kati_arms", "arms", ARMS, "Where are the arms?",
            ok=("wrapped around the torso: one hand on the opposite shoulder, the other behind the back", None),
            loose=("arms hanging or held away from the body", "Wrap your arms around your body: one hand on the opposite shoulder, the other behind your back.")),
        _cp("kati_feet", "feet", ANKLES, "How are the feet placed?",
            ok=("feet about shoulder-width apart, flat on the floor", None),
            too_close=("feet together or very close", "Step your feet a little wider apart, about shoulder width."),
            heels_lifted=("heels lifted off the floor", "Keep both feet flat on the floor.")),
    ],
    "Ardhakati Chakrasana": [
        _cp("ardha_arm", "raised arm", ARMS, "How is the raised arm held?",
            ok=("raised arm straight, upper arm close to the ear", None),
            bent=("raised arm bent at the elbow", "Straighten your raised arm fully."),
            away_from_ear=("raised arm held away from the head", "Bring your raised arm closer to your ear.")),
        _cp("ardha_bend", "side bend", TORSO, "How does the torso bend?",
            ok=("torso bends clearly sideways", None),
            too_little=("almost no sideways bend", "Bend a little further sideways, sliding your lower hand down your leg."),
            bending_forward=("bending forward instead of to the side", "Bend straight to the side, not forward; keep your chest open.")),
        _cp("ardha_knees", "knees", KNEES, "Are both knees straight?",
            ok=("both legs straight", None),
            bent=("one or both knees bent", "Keep both knees straight.")),
    ],
    "Vajrasana": [
        _cp("vaj_spine", "back", TORSO, "Is the back upright?",
            ok=("back upright and vertical", None),
            slouching=("back rounded or slouching forward", "Sit up straight and lift your chest."),
            leaning_back=("leaning backward", "Bring your upper body forward slightly so your back is vertical.")),
        _cp("vaj_hips", "hips", HIPS + KNEES, "Is the person sitting on the heels?",
            ok=("sitting fully back on the heels", None),
            lifted=("hips raised above the heels", "Sit back fully onto your heels.")),
        _cp("vaj_head", "head", NOSE + SHOULDERS, "Where is the head facing?",
            ok=("head straight, looking ahead", None),
            head_down=("head dropped, looking down", "Lift your chin and look straight ahead.")),
        _cp("vaj_hands", "hands", WRISTS, "Where are the hands?",
            ok=("hands resting on the thighs or knees", None),
            elsewhere=("hands not on the thighs", "Rest your hands on your thighs.")),
    ],
    "Parvatasana": [
        _cp("parv_arms", "arms", ARMS, "How are the arms raised?",
            ok=("both arms straight and stretched fully overhead", None),
            elbows_bent=("elbows bent", "Straighten your elbows and stretch your arms fully overhead."),
            too_low=("arms forward or not fully raised", "Raise your arms higher, close to your ears.")),
        _cp("parv_palms", "palms", WRISTS, "Are the palms joined above the head?",
            ok=("palms joined together above the head", None),
            apart=("hands apart", "Join your palms together above your head.")),
        _cp("parv_spine", "back", TORSO, "Is the back upright?",
            ok=("spine long and upright", None),
            slouching=("back rounded or slouching", "Lengthen your spine and sit tall."),
            leaning=("leaning backward or to one side", "Keep your body vertical; don't lean.")),
    ],
    "Bhujangasana": [
        _cp("bhuj_chest", "chest", SHOULDERS, "How is the chest lifted?",
            ok=("chest lifted, shoulders down and back", None),
            too_low=("chest barely off the floor", "Lift your chest a little higher using your back muscles."),
            shoulders_hunched=("shoulders hunched up toward the ears", "Roll your shoulders back and away from your ears.")),
        _cp("bhuj_elbows", "elbows", ELBOWS, "How are the elbows held?",
            ok=("elbows slightly bent and close to the body", None),
            locked=("arms locked fully straight", "Bend your elbows slightly; don't lock your arms."),
            flared=("elbows pointing out to the sides", "Keep your elbows close to your body.")),
        _cp("bhuj_hips", "hips and legs", HIPS + LEGS, "Where are the hips and legs?",
            ok=("hips, thighs and legs on the floor, legs together", None),
            hips_lifted=("hips lifted off the floor", "Keep your hips and thighs on the floor."),
            legs_apart=("legs spread apart", "Bring your legs together.")),
        _cp("bhuj_head", "head", NOSE + SHOULDERS, "How is the head held?",
            ok=("neck long, gaze forward or slightly up", None),
            thrown_back=("head thrown far back", "Don't throw your head back; keep your neck long.")),
    ],
    "Marjariasana": [
        _cp("marj_hands", "hands", ARMS + SHOULDERS, "Where are the hands?",
            ok=("hands directly under the shoulders, arms straight", None),
            forward=("hands far in front of the shoulders", "Place your hands directly under your shoulders."),
            elbows_bent=("elbows bent", "Straighten your arms.")),
        _cp("marj_knees", "knees", KNEES + HIPS, "Where are the knees?",
            ok=("knees directly under the hips", None),
            misplaced=("knees far in front of or behind the hips", "Bring your knees directly under your hips.")),
        _cp("marj_spine", "back", TORSO, "How is the spine moving?",
            ok=("spine clearly arched down or rounded up", None),
            flat=("back flat, no arch or curve", "Move your spine: arch it down as you inhale, round it up as you exhale.")),
    ],
    "Sarvangasana": [
        _cp("sarv_legs", "legs", LEGS, "How are the legs raised?",
            ok=("legs straight, together and pointing vertically up", None),
            angled=("legs tilted well away from vertical", "Bring your legs up to vertical, in line with your body."),
            knees_bent=("knees bent", "Straighten your knees."),
            apart=("legs apart", "Keep your legs together.")),
        _cp("sarv_torso", "back", TORSO, "How is the torso held?",
            ok=("torso vertical, hips stacked over the shoulders", None),
            hips_back=("hips sagging back, torso slanted", "Push your hips up so your back is straighter and more vertical.")),
        _cp("sarv_elbows", "elbows", ELBOWS + WRISTS, "How are the arms supporting the back?",
            ok=("hands support the lower back, elbows close together", None),
            elbows_wide=("elbows spread wide or hands off the back", "Bring your elbows closer and support your back with your hands.")),
    ],
    "Viparita Karani": [
        _cp("vip_legs", "legs", LEGS, "How are the legs raised?",
            ok=("legs straight and together, raised up", None),
            knees_bent=("knees bent", "Straighten your legs."),
            too_low=("legs well below vertical", "Raise your legs higher toward the ceiling.")),
        _cp("vip_hips", "hips", HIPS + WRISTS, "How are the hips supported?",
            ok=("hips low and supported by the hands or the floor", None),
            unsupported=("hips lifted with no support", "Support your hips with your hands.")),
        _cp("vip_head", "head and shoulders", NOSE + SHOULDERS, "Where are the head and shoulders?",
            ok=("head and shoulders resting on the floor", None),
            lifted=("head or shoulders lifted off the floor", "Rest your head and shoulders on the floor.")),
    ],
    "Anantasana": [
        _cp("anan_leg", "top leg", LEGS, "How is the top leg raised?",
            ok=("top leg raised high and straight", None),
            knee_bent=("top knee bent", "Straighten your raised leg."),
            too_low=("top leg only slightly lifted", "Lift your top leg higher.")),
        _cp("anan_line", "body line", TORSO, "Is the body in one line on its side?",
            ok=("body in one straight line from head to feet, lying on its side", None),
            rolling=("rolling forward or backward", "Stay on your side; don't roll forward or back."),
            hips_bent=("bent at the hips", "Keep your body in one straight line from head to feet.")),
        _cp("anan_head", "head", NOSE + ELBOWS, "How is the head supported?",
            ok=("head resting on the hand, elbow on the floor", None),
            unsupported=("head not resting on the hand", "Rest your head on your hand.")),
    ],
}

MAX_QUESTIONS = 8   # TypeSafe evaluates up to 8 questions per request together


def checkpoints_for(pose: Optional[str]) -> list[Checkpoint]:
    return list(CHECKPOINTS.get(pose or "", []))


def build_questions(pose: str, hint: str, max_checkpoints: int) -> dict[str, Any]:
    """Checkpoint (Choice) questions plus one overall form Score for ``pose``."""
    lead = f"The person in `photo` is attempting {pose} ({hint})."
    questions: dict[str, Any] = {}
    for cp in checkpoints_for(pose)[:max_checkpoints]:
        criteria = {k: crit for k, (crit, _) in cp.options.items()}
        criteria[NOT_VISIBLE_KEY] = NOT_VISIBLE_TEXT
        questions[cp.key] = {"type": "choice", "instructions": f"{lead} {cp.question}", "criteria": criteria}
    questions["form_score"] = {
        "type": "score",
        "instructions": f"{lead} How well does the posture in `photo` match a correct {pose}?",
        "criteria": list(FORM_SCORE_LEVELS),
    }
    return questions


@dataclass
class JevCorrection:
    checkpoint: str
    body_part: str
    choice: str
    instruction: Optional[str]     # None when this body part is fine
    confidence: float
    joints: tuple[int, ...] = ()

    @property
    def needs_change(self) -> bool:
        return self.instruction is not None


@dataclass
class JevCoaching:
    pose: str
    corrections: list[JevCorrection] = field(default_factory=list)   # every answered checkpoint
    form_score: Optional[float] = None       # 0–100, from the Score question
    form_confidence: Optional[float] = None


def parse_coaching(answers: dict[str, Any], pose: str, max_checkpoints: int) -> JevCoaching:
    """Validate checkpoint answers; unknown or malformed answers are dropped."""
    out = JevCoaching(pose=pose)
    for cp in checkpoints_for(pose)[:max_checkpoints]:
        a = answers.get(cp.key)
        if not isinstance(a, dict) or a.get("type") != "choice":
            continue
        choice = a.get("choice")
        conf = _prob(a.get("confidence"))
        if conf is None or (choice not in cp.options and choice != NOT_VISIBLE_KEY):
            continue
        if choice == NOT_VISIBLE_KEY:
            continue
        out.corrections.append(JevCorrection(cp.key, cp.body_part, choice, cp.options[choice][1], conf, cp.joints))
    s = answers.get("form_score")
    if isinstance(s, dict) and s.get("type") == "score":
        try:
            val = float(s.get("score"))
        except (TypeError, ValueError):
            val = None
        if val is not None and 0.0 <= val <= len(FORM_SCORE_LEVELS) - 1:
            out.form_score = round(100.0 * val / (len(FORM_SCORE_LEVELS) - 1), 1)
            out.form_confidence = _prob(s.get("confidence"))
    return out


def _prob(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if 0.0 <= v <= 1.0 else None
