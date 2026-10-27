"""
tests/test_jev_integration.py
─────────────────────────────
Jev client, worker and decision-fusion tests.  All HTTP traffic is mocked:
no API key or network access is required.
"""

from __future__ import annotations

import base64
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from src.models.jev_client import (
    UNCLEAR_KEY,
    JevClient,
    JevConfig,
    JevResult,
    option_key,
)
from src.models.jev_fusion import FusionConfig, decide
from src.models.jev_worker import JevWorker

CLASSES = ["Tadasana", "Vajrasana", "Parvatasana", "Bhujangasana", "Viparita Karani"]
FRAME = np.full((480, 640, 3), 127, dtype=np.uint8)


class FakeResponse:
    def __init__(self, status=200, payload=None, headers=None, text=""):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def ok_payload(choice, conf=0.9, probs=None, form=None):
    ans = {"asana": {"type": "choice", "choice": choice, "confidence": conf,
                     "probabilities": probs or {choice: conf}}}
    if form:
        ans["form"] = {"type": "choice", "choice": form, "confidence": 0.8}
    return {"id": "req_123", "answers": ans}


def make_client(responses, **cfg):
    config = JevConfig(max_retries=cfg.pop("max_retries", 1), **cfg)
    return JevClient(config, api_key="sk-test-SECRET", session=FakeSession(responses))


# ── Request construction ──────────────────────────────────────────────────────

def test_payload_uses_documented_schema():
    client = make_client([FakeResponse(payload=ok_payload("tadasana"))])
    res = client.classify(FRAME, CLASSES, frame_id=7, frame_timestamp=1.0, epoch=3)
    call = client._session.calls[0]
    body, headers = call["json"], call["headers"]
    assert call["url"] == "https://api.typesafe.ai/v1/systemone"
    assert body["model"] == "jev-preview"
    (ref, asset), = body["assets"].items()
    assert asset["type"] == "image" and asset["format"] == "jpg"
    assert base64.b64decode(asset["data_base64"])[:2] == b"\xff\xd8"  # JPEG magic
    q = body["questions"]["asana"]
    assert q["type"] == "choice"
    assert set(q["criteria"]) == {option_key(c) for c in CLASSES} | {UNCLEAR_KEY}
    # the image reference appears exactly once, in the state; the question refers to it by path
    assert body["state"]["photo"] == ref
    assert json.dumps(body).count(ref) == 2  # the assets key + the state value
    assert "`photo`" in q["instructions"]
    assert headers["TypeSafe-Danger-Zone-Images"] == "true"
    assert headers["Authorization"] == "Bearer sk-test-SECRET"
    assert headers["Content-Type"] == "application/json"
    assert headers["Idempotency-Key"]
    assert res.ok and res.label == "Tadasana" and res.confidence == 0.9
    assert (res.frame_id, res.frame_timestamp, res.epoch) == (7, 1.0, 3)


def test_image_resized_into_documented_bounds():
    client = make_client([])
    import cv2
    big = np.zeros((3000, 2000, 3), dtype=np.uint8)
    small = np.zeros((100, 150, 3), dtype=np.uint8)
    for img in (big, small):
        dec = cv2.imdecode(np.frombuffer(client.encode_image(img), np.uint8), cv2.IMREAD_COLOR)
        assert 280 <= min(dec.shape[:2]) and max(dec.shape[:2]) <= 1568


def test_form_question_only_when_enabled():
    c1 = make_client([FakeResponse(payload=ok_payload("vajrasana"))])
    c1.classify(FRAME, CLASSES)
    assert "form" not in c1._session.calls[0]["json"]["questions"]
    c2 = make_client([FakeResponse(payload=ok_payload("vajrasana", form="visible_misalignment"))],
                     request_form_assessment=True)
    res = c2.classify(FRAME, CLASSES)
    assert "form" in c2._session.calls[0]["json"]["questions"]
    assert res.form_assessment == "visible_misalignment"


# ── Response parsing ──────────────────────────────────────────────────────────

def test_parse_maps_option_keys_back_to_labels():
    key_map = {option_key(c): c for c in CLASSES}
    res = JevClient.parse_response(ok_payload("viparita_karani", 0.8,
                                              {"viparita_karani": 0.8, "tadasana": 0.2}), key_map)
    assert res.ok and res.label == "Viparita Karani"
    assert res.probabilities == {"Viparita Karani": 0.8, "Tadasana": 0.2}


def test_parse_confidence_falls_back_to_probability():
    key_map = {option_key(c): c for c in CLASSES}
    data = {"answers": {"asana": {"choice": "tadasana", "probabilities": {"tadasana": 0.66}}}}
    assert JevClient.parse_response(data, key_map).confidence == 0.66
    data = {"answers": {"asana": {"choice": "tadasana"}}}
    assert JevClient.parse_response(data, key_map).confidence is None  # never invented


@pytest.mark.parametrize("data", [
    None, {}, {"answers": {}}, {"answers": {"asana": "x"}},
    {"answers": {"asana": {"choice": "warrior_three"}}},         # not an offered option
    {"answers": {"asana": {"choice": 5}}},
])
def test_parse_rejects_invalid_responses(data):
    res = JevClient.parse_response(data, {option_key(c): c for c in CLASSES})
    assert not res.ok and res.status == "invalid_response"


def test_parse_drops_out_of_range_probabilities():
    data = ok_payload("tadasana", conf=1.7, probs={"tadasana": 1.7, "vajrasana": float("nan"), "bogus": 0.1})
    res = JevClient.parse_response(data, {option_key(c): c for c in CLASSES})
    assert res.ok and res.confidence is None and res.probabilities == {}


def test_unclear_choice_has_no_label():
    res = make_client([FakeResponse(payload=ok_payload(UNCLEAR_KEY, 0.7))]).classify(FRAME, CLASSES)
    assert res.ok and res.unclear and res.label is None


# ── Failure handling ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("code,status", [
    (401, "auth_error"), (402, "insufficient_credits"), (422, "invalid_request"),
])
def test_non_retryable_http_errors(code, status):
    client = make_client([FakeResponse(code, {"error": {"message": "nope"}})], max_retries=3)
    res = client.classify(FRAME, CLASSES)
    assert not res.ok and res.status == status and res.http_status == code
    assert len(client._session.calls) == 1


def test_retry_then_success_reuses_idempotency_key(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    client = make_client([FakeResponse(429, {}, headers={"Retry-After": "0"}),
                          FakeResponse(payload=ok_payload("bhujangasana"))], max_retries=2)
    res = client.classify(FRAME, CLASSES)
    assert res.ok and res.label == "Bhujangasana" and res.attempts == 2
    keys = {c["headers"]["Idempotency-Key"] for c in client._session.calls}
    assert len(keys) == 1


def test_retries_are_bounded(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    client = make_client([FakeResponse(502, {})] * 5, max_retries=2)
    res = client.classify(FRAME, CLASSES)
    assert res.status == "service_error" and len(client._session.calls) == 3


def test_timeout_and_outage(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)

    class Timeout(Exception):
        pass

    class ConnectionError(Exception):
        pass

    assert make_client([Timeout(), Timeout()]).classify(FRAME, CLASSES).status == "timeout"
    assert make_client([ConnectionError(), ConnectionError()]).classify(FRAME, CLASSES).status == "network_error"


def test_malformed_json_is_invalid_response():
    res = make_client([FakeResponse(200, ValueError("bad json"))]).classify(FRAME, CLASSES)
    assert res.status == "invalid_response"


def test_missing_api_key_sends_nothing(monkeypatch):
    monkeypatch.delenv("JEVMODEL_API_KEY", raising=False)
    session = FakeSession([])
    client = JevClient(JevConfig(), session=session)
    res = client.classify(FRAME, CLASSES)
    assert res.status == "no_api_key" and session.calls == []


def test_api_key_never_exposed(caplog):
    client = make_client([FakeResponse(401, {"error": "invalid key"})])
    with caplog.at_level("DEBUG"):
        res = client.classify(FRAME, CLASSES)
    assert "SECRET" not in repr(client)
    assert "SECRET" not in json.dumps(res.__dict__, default=str)
    assert "SECRET" not in caplog.text


# ── Worker: throttling, bounded concurrency, staleness tags ───────────────────

class SlowClient:
    def __init__(self, delay=0.05):
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self.calls = 0
        self.lock = threading.Lock()

    def classify(self, frame, cands, frame_id=None, frame_timestamp=None, epoch=None, focus_pose=None):
        with self.lock:
            self.active += 1
            self.calls += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(self.delay)
        with self.lock:
            self.active -= 1
        return JevResult(ok=True, status="ok", label=cands[0], confidence=0.9,
                         frame_id=frame_id, frame_timestamp=frame_timestamp, epoch=epoch, latency_ms=50)


def _wait_for(worker, timeout=2.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = worker.poll()
        if r is not None:
            return r
        time.sleep(0.01)
    return None


def test_worker_rate_limits_and_never_runs_concurrently():
    client = SlowClient()
    w = JevWorker(client, min_interval_seconds=10.0)
    try:
        assert w.submit(FRAME, CLASSES, 1, 0.0, 0)
        for k in range(10):
            assert not w.submit(FRAME, CLASSES, k + 2, 0.0, 0)  # throttled
        res = _wait_for(w)
        assert res is not None and res.frame_id == 1
        assert client.calls == 1 and client.max_active == 1
        assert w.stats.throttled == 10
    finally:
        w.shutdown()


def test_worker_manual_force_is_still_single_flight():
    client = SlowClient(delay=0.1)
    w = JevWorker(client, min_interval_seconds=10.0)
    try:
        for k in range(5):
            w.submit(FRAME, CLASSES, k, 0.0, 0, force=True)
        time.sleep(0.5)
        assert client.max_active == 1
        assert client.calls <= 2  # in-flight + at most one replacement
        assert w.stats.replaced >= 3
    finally:
        w.shutdown()


def test_worker_does_not_block_caller():
    w = JevWorker(SlowClient(delay=0.5), min_interval_seconds=0)
    try:
        t0 = time.perf_counter()
        w.submit(FRAME, CLASSES, 1, 0.0, 0)
        assert time.perf_counter() - t0 < 0.05
    finally:
        w.shutdown()


# ── Fusion policy ─────────────────────────────────────────────────────────────

def pred(name, conf, status=None, top3=None):
    return SimpleNamespace(pose_name=name, confidence=conf,
                           status=status or ("detected" if conf >= 0.28 else "low_confidence"),
                           top3=top3 or [(name, conf)])


def jev_res(label, conf=0.9, ts=10.0, epoch=1, ok=True, unclear=False):
    return JevResult(ok=ok, status="ok" if ok else "timeout", label=label, confidence=conf,
                     frame_timestamp=ts, epoch=epoch, unclear=unclear)


CFG = FusionConfig(local_threshold=0.28, local_high_confidence=0.6, jev_min_confidence=0.7, jev_max_age_seconds=4.0)


def test_local_only_operation_without_jev():
    d = decide(pred("Tadasana", 0.8), "high", "auto", CLASSES, now=10, cfg=CFG)
    assert d.assessment_allowed and d.final_pose == "Tadasana" and d.tier == "high" and d.source == "local"
    d = decide(pred("Tadasana", 0.4), "high", "auto", CLASSES, now=10, cfg=CFG)
    assert d.assessment_allowed and d.tier == "medium"


def test_low_confidence_auto_prediction_is_not_assessed():
    d = decide(pred("Vajrasana", 0.2), "high", "auto", CLASSES, now=10, cfg=CFG)
    assert not d.assessment_allowed and d.final_pose is None
    assert "threshold" in d.reason and d.user_message


def test_agreement_keeps_pose():
    d = decide(pred("Vajrasana", 0.5), "high", "auto", CLASSES, now=10, cfg=CFG,
               jev=jev_res("Vajrasana"), current_epoch=1)
    assert d.final_pose == "Vajrasana" and d.agreement == "agree" and d.tier == "high"


def test_disagreement_does_not_trust_jev_over_confident_local():
    d = decide(pred("Tadasana", 0.85), "high", "auto", CLASSES, now=10, cfg=CFG,
               jev=jev_res("Vajrasana"), current_epoch=1)
    assert d.final_pose == "Tadasana" and d.agreement == "disagree" and d.tier == "medium"


def test_unresolved_disagreement_reports_uncertainty():
    d = decide(pred("Tadasana", 0.4), "high", "auto", CLASSES, now=10, cfg=CFG,
               jev=jev_res("Vajrasana"), current_epoch=1)
    assert not d.assessment_allowed and d.final_pose is None and d.agreement == "disagree"


def test_jev_resolves_only_among_local_candidates_and_capped_medium():
    top = [("Vajrasana", 0.22), ("Parvatasana", 0.20), ("Tadasana", 0.1)]
    d = decide(pred("Vajrasana", 0.22, top3=top), "high", "auto", CLASSES, now=10, cfg=CFG,
               jev=jev_res("Parvatasana"), current_epoch=1)
    assert d.assessment_allowed and d.final_pose == "Parvatasana" and d.tier == "medium"
    assert d.source == "jev_resolved"
    d = decide(pred("Vajrasana", 0.22, top3=top), "high", "auto", CLASSES, now=10, cfg=CFG,
               jev=jev_res("Bhujangasana"), current_epoch=1)
    assert not d.assessment_allowed


def test_stale_low_confidence_unclear_or_failed_jev_ignored():
    p = pred("Vajrasana", 0.22, top3=[("Vajrasana", 0.22), ("Parvatasana", 0.2)])
    for j, epoch, expect in [
        (jev_res("Parvatasana", ts=1.0), 1, "jev_stale"),            # 9 s old
        (jev_res("Parvatasana", epoch=0), 1, "jev_stale"),           # different pose epoch
        (jev_res("Parvatasana", conf=0.5), 1, "jev_low_confidence"),
        (jev_res(None, unclear=True), 1, "jev_unclear"),
        (jev_res("Parvatasana", ok=False), 1, "jev_unavailable"),
    ]:
        d = decide(p, "high", "auto", CLASSES, now=10, cfg=CFG, jev=j, current_epoch=epoch)
        assert not d.assessment_allowed, expect
        assert d.agreement == expect


def test_unreliable_landmarks_block_assessment_even_if_jev_confident():
    d = decide(pred("Tadasana", 0.9), "low", "auto", CLASSES, now=10, cfg=CFG,
               jev=jev_res("Tadasana", conf=0.99), current_epoch=1)
    assert not d.assessment_allowed and d.tier == "low"


def test_guided_target_preserved_and_never_replaced_by_jev():
    d = decide(pred("Vajrasana", 0.1), "high", "guided", CLASSES, now=10, cfg=CFG,
               target_pose="Tadasana", jev=jev_res("Bhujangasana", conf=0.99), current_epoch=1)
    assert d.final_pose == "Tadasana" and d.assessment_allowed and d.source == "guided_target"
    assert d.tier == "medium"  # strong contrary evidence lowers certainty only


def test_guided_low_unrelated_confidence_is_not_proof_of_target():
    d = decide(pred("Vajrasana", 0.1), "medium", "guided", CLASSES, now=10, cfg=CFG, target_pose="Tadasana")
    assert d.final_pose == "Tadasana" and d.tier == "medium"  # bounded by landmarks, not raised
    d = decide(pred("Tadasana", 0.99), "medium", "guided", CLASSES, now=10, cfg=CFG, target_pose="Tadasana")
    assert d.tier == "medium"


def test_guided_low_landmarks_suppressed():
    d = decide(pred("Tadasana", 0.9), "low", "guided", CLASSES, now=10, cfg=CFG, target_pose="Tadasana")
    assert not d.assessment_allowed and d.final_pose == "Tadasana"


def test_untrusted_class_requires_more_evidence():
    cfg = FusionConfig(class_min_precision=0.5, class_precision={"Tadasana": 0.9, "Vajrasana": 0.2})
    assert decide(pred("Tadasana", 0.5), "high", "auto", CLASSES, now=0, cfg=cfg).assessment_allowed
    assert not decide(pred("Vajrasana", 0.5), "high", "auto", CLASSES, now=0, cfg=cfg).assessment_allowed


def test_no_person():
    d = decide(pred("—", 0.0, status="no_person"), "low", "auto", CLASSES, now=0, cfg=CFG)
    assert not d.assessment_allowed and "No person" in d.user_message
