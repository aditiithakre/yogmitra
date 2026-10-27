"""Jev-primary coaching: checkpoint questions, answer parsing, decisions and feedback."""

from __future__ import annotations

import json

import numpy as np

from src.correction.feedback_engine import FeedbackEngine
from src.models.jev_client import JevClient, JevConfig, JevResult
from src.models.jev_coach import (CHECKPOINTS, FORM_SCORE_LEVELS, MAX_QUESTIONS, NOT_VISIBLE_KEY,
                                  build_questions, parse_coaching)
from src.models.jev_fusion import decide_jev_primary
from src.models.zenodo_annotations import ZENODO_ASANAS

FRAME = np.full((480, 640, 3), 127, np.uint8)
CLASSES = ["Tadasana", "Vajrasana", "Bhujangasana"]


def test_every_zenodo_asana_has_valid_checkpoints():
    assert set(CHECKPOINTS) == set(ZENODO_ASANAS)
    for pose, cps in CHECKPOINTS.items():
        assert 3 <= len(cps) <= MAX_QUESTIONS - 2, pose          # asana + form score share the request
        assert len({cp.key for cp in cps}) == len(cps)
        for cp in cps:
            assert "ok" in cp.options and cp.options["ok"][1] is None
            assert NOT_VISIBLE_KEY not in cp.options
            assert all(text for k, (_, text) in cp.options.items() if k != "ok"), cp.key
            assert cp.joints and all(0 <= j <= 32 for j in cp.joints)


def test_questions_reference_photo_and_add_not_visible_and_score():
    qs = build_questions("Vajrasana", "kneeling", max_checkpoints=4)
    assert qs["form_score"]["type"] == "score" and qs["form_score"]["criteria"] == FORM_SCORE_LEVELS
    choice_qs = [q for q in qs.values() if q["type"] == "choice"]
    assert len(choice_qs) == 4
    for q in choice_qs:
        assert "`photo`" in q["instructions"] and "Vajrasana" in q["instructions"]
        assert NOT_VISIBLE_KEY in q["criteria"]


def test_parse_coaching_maps_choices_to_instructions():
    answers = {
        "vaj_spine": {"type": "choice", "choice": "slouching", "confidence": 0.8},
        "vaj_hips": {"type": "choice", "choice": "ok", "confidence": 0.9},
        "vaj_head": {"type": "choice", "choice": NOT_VISIBLE_KEY, "confidence": 0.9},
        "vaj_hands": {"type": "choice", "choice": "made_up", "confidence": 0.9},      # not offered → dropped
        "form_score": {"type": "score", "score": 2.0, "confidence": 0.7},
    }
    c = parse_coaching(answers, "Vajrasana", max_checkpoints=4)
    by = {x.checkpoint: x for x in c.corrections}
    assert set(by) == {"vaj_spine", "vaj_hips"}
    assert by["vaj_spine"].instruction == "Sit up straight and lift your chest." and by["vaj_spine"].needs_change
    assert not by["vaj_hips"].needs_change
    assert c.form_score == round(100 * 2.0 / (len(FORM_SCORE_LEVELS) - 1), 1)


def test_client_sends_checkpoints_only_for_focus_pose():
    client = JevClient(JevConfig(coach_checkpoints=3), api_key="sk-test")
    body, _ = client.build_payload(client.encode_image(FRAME), CLASSES)
    assert set(body["questions"]) == {"asana"}
    body, _ = client.build_payload(client.encode_image(FRAME), CLASSES, focus_pose="Vajrasana")
    assert set(body["questions"]) == {"asana", "vaj_spine", "vaj_hips", "vaj_head", "form_score"}
    ref = body["state"]["photo"]
    assert json.dumps(body).count(ref) == 2                    # image still sent once, referenced by path
    assert client.coaching_pose("Not A Pose") is None
    assert JevClient(JevConfig(coach_checkpoints=0), api_key="k").coaching_pose("Vajrasana") is None


def test_parse_response_attaches_coaching():
    data = {"answers": {
        "asana": {"type": "choice", "choice": "vajrasana", "confidence": 0.9},
        "vaj_spine": {"type": "choice", "choice": "leaning_back", "confidence": 0.75},
    }}
    res = JevClient.parse_response(data, {"vajrasana": "Vajrasana"}, coach_pose="Vajrasana", coach_checkpoints=4)
    assert res.ok and res.focus_pose == "Vajrasana"
    assert [c.choice for c in res.coaching.corrections] == ["leaning_back"]


def _jev(label="Vajrasana", conf=0.9, unclear=False, ts=10.0, epoch=0, coaching=None):
    return JevResult(ok=True, status="ok", label=None if unclear else label, confidence=conf, unclear=unclear,
                     frame_timestamp=ts, epoch=epoch, coaching=coaching)


def test_decide_jev_primary_needs_no_landmarks():
    d = decide_jev_primary(_jev(), None, "auto", CLASSES, now=10.5, current_epoch=0)
    assert d.assessment_allowed and d.final_pose == "Vajrasana" and d.tier == "high" and d.source == "jev"
    d = decide_jev_primary(_jev(conf=0.75), None, "auto", CLASSES, now=10.5, current_epoch=0)
    assert d.assessment_allowed and d.tier == "medium"
    assert not decide_jev_primary(_jev(unclear=True), None, "auto", CLASSES, 10.5, current_epoch=0).assessment_allowed
    assert not decide_jev_primary(_jev(conf=0.5), None, "auto", CLASSES, 10.5, current_epoch=0).assessment_allowed
    assert not decide_jev_primary(_jev(), None, "auto", CLASSES, now=30.0, current_epoch=0).assessment_allowed
    assert decide_jev_primary(None, None, "auto", CLASSES, 10.5).reason == "waiting for Jev's first answer"


def test_decide_jev_primary_guided_requires_target():
    d = decide_jev_primary(_jev(label="Tadasana"), None, "guided", CLASSES, 10.5,
                           target_pose="Vajrasana", current_epoch=0)
    assert not d.assessment_allowed and d.user_message == "Move into Vajrasana."
    d = decide_jev_primary(_jev(), None, "guided", CLASSES, 10.5, target_pose="Vajrasana", current_epoch=0)
    assert d.assessment_allowed and d.final_pose == "Vajrasana"


def test_feedback_from_jev_shows_and_speaks_persistent_corrections():
    answers = {"vaj_spine": {"type": "choice", "choice": "slouching", "confidence": 0.8},
               "vaj_hips": {"type": "choice", "choice": "lifted", "confidence": 0.4}}     # below threshold
    coaching = parse_coaching(answers, "Vajrasana", 4)
    d = decide_jev_primary(_jev(coaching=coaching), None, "auto", CLASSES, 10.5, current_epoch=0)
    fe = FeedbackEngine(cooldown_seconds=0.0, min_persistence_frames=2)
    items = fe.update_from_jev(d, coaching, fresh=True)
    assert [i.instruction for i in items] == ["Sit up straight and lift your chest."]
    assert fe.pop_speech() is None                              # first sighting: shown, not yet spoken
    fe.update_from_jev(d, coaching, fresh=False)                # display refresh does not count
    assert fe.pop_speech() is None
    fe._last_spoken_time = 0.0
    fe.update_from_jev(d, coaching, fresh=True)
    assert fe.pop_speech() == "Sit up straight and lift your chest."


def test_feedback_from_jev_good_form_and_suppression():
    good = parse_coaching({"vaj_spine": {"type": "choice", "choice": "ok", "confidence": 0.9}}, "Vajrasana", 4)
    d = decide_jev_primary(_jev(coaching=good), None, "auto", CLASSES, 10.5, current_epoch=0)
    fe = FeedbackEngine()
    assert fe.update_from_jev(d, good, fresh=True)[0].issue == "ok"
    assert fe.update_from_jev(d, None, fresh=True)[0].issue == "checking"     # coaching not in yet
    blocked = decide_jev_primary(_jev(unclear=True), None, "auto", CLASSES, 10.5, current_epoch=0)
    assert fe.update_from_jev(blocked, good, fresh=True) == []
