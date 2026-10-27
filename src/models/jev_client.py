"""
src/models/jev_client.py
────────────────────────
Client for Jev (TypeSafe "System One" decision model) used as an *optional*
image-based second opinion on which asana is visible in a still frame.

Jev never replaces MediaPipe landmarks or the biomechanical correction engine;
see ``src/models/jev_fusion.py`` for how its answer is (conservatively) used.

API surface implemented (from TypeSafe's docs at https://docs.typesafe.ai/,
pages "System One", "Images" and "Request rules and errors", checked against the
live API on 2026-10-27):

  POST {endpoint}                     default https://api.typesafe.ai/v1/systemone
  Authorization: Bearer $JEVMODEL_API_KEY   (a TypeSafe console key)
  Content-Type: application/json
  Idempotency-Key: <uuid>             same key reused when retrying
  TypeSafe-Danger-Zone-Images: true   required on every request with an image
  body = {
    "model": "jev-preview",
    "state": {"context": "...", "photo": "<ref uuid>"},
    "assets": {"<ref uuid>": {"type": "image", "format": "jpg", "data_base64": "<base64>"}},
    "questions": {
      "asana": {"type": "choice", "instructions": "...",
                "criteria": {"<option_key>": "<description>", ...}}
    }
  }
  response: {"answers": {"asana": {"type": "choice", "choice": "<option_key>",
                                    "confidence": 0.93,
                                    "probabilities": {"<option_key>": 0.93, ...}}}, ...}

Documented limits honoured here: PNG/JPEG stills (we send 1 JPEG, capped at
4 MB; the API allows 16 MiB), 280–1568 px per side, every asset referenced.
Documented errors: 400 (bad image / images not turned on), 401 (bad key),
413 (too large), 422 (invalid request), 429 (rate limit), 529 (overloaded).
429/5xx/529/timeouts are retried a bounded number of times with exponential
backoff.

The API key is read from the environment only and is never logged or returned.
"""

from __future__ import annotations

import base64
import logging
import math
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

from src.models.jev_coach import JevCoaching, build_questions, checkpoints_for, parse_coaching

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-preview"
API_KEY_ENV = "JEVMODEL_API_KEY"
UNCLEAR_KEY = "unclear"
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MIN_SIDE, MAX_SIDE = 280, 1568
RETRYABLE_HTTP = {429, 500, 502, 503, 504, 529}

# Short visual descriptions help a choice question; names alone are ambiguous.
ASANA_VISUAL_HINTS: dict[str, str] = {
    "Tadasana": "standing upright on both feet, legs straight, arms at the sides or stretched overhead (mountain pose)",
    "Kati Chakrasana": "standing with feet apart and the upper body twisted to one side, arms wrapped around the torso (standing spinal twist)",
    "Ardhakati Chakrasana": "standing, one arm raised overhead and the torso bending sideways away from it (lateral standing side bend)",
    "Vajrasana": "kneeling and sitting back on the heels with an upright back, hands resting on the thighs (thunderbolt pose)",
    "Parvatasana": "sitting or kneeling upright with both arms stretched straight overhead, palms joined",
    "Bhujangasana": "lying face down with legs on the floor, chest and head lifted by the arms (cobra pose)",
    "Marjariasana": "on hands and knees in a tabletop position, spine arched or rounded (cat-cow pose)",
    "Sarvangasana": "lying on the back with hips and straight legs lifted vertically, body supported on the shoulders (shoulder stand)",
    "Viparita Karani": "lying on the back with both legs raised up toward the ceiling while the hips stay low or supported (legs-up inversion)",
    "Anantasana": "lying on one side, head resting on one hand, top leg lifted upward (side-reclining leg lift)",
}

FORM_CRITERIA = {
    "visibly_aligned": "the posture looks well formed: limbs extended and positioned as the pose requires",
    "visible_misalignment": "a clear form problem is visible, such as bent limbs that should be straight or a collapsed back",
    "cannot_judge": "the body is partly out of frame, blurred, or too small to judge form",
}


def option_key(label: str) -> str:
    """Stable option key for a class label (lowercase, underscores)."""
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


@dataclass
class JevConfig:
    endpoint: str = DEFAULT_ENDPOINT
    model: str = DEFAULT_MODEL
    timeout_seconds: float = 8.0
    max_retries: int = 1
    jpeg_quality: int = 80
    max_image_side: int = 1024
    request_form_assessment: bool = False
    coach_checkpoints: int = 4       # checkpoint questions per request for the focus pose (0 = no coaching)
    image_headers: dict[str, str] = field(default_factory=lambda: {"TypeSafe-Danger-Zone-Images": "true"})
    api_key_env: str = API_KEY_ENV
    api_key: Optional[str] = None

    @classmethod
    def from_config(cls, cfg: Optional[dict]) -> "JevConfig":
        cfg = cfg or {}
        out = cls()
        for k in ("endpoint", "model", "timeout_seconds", "max_retries", "jpeg_quality",
                  "max_image_side", "request_form_assessment", "coach_checkpoints", "api_key_env"):
            if k in cfg and cfg[k] is not None:
                setattr(out, k, type(getattr(out, k))(cfg[k]))
        if "api_key" in cfg and cfg["api_key"] is not None:
            out.api_key = str(cfg["api_key"])
        if isinstance(cfg.get("image_headers"), dict):
            out.image_headers = {str(k): str(v) for k, v in cfg["image_headers"].items()}
        return out


@dataclass
class JevResult:
    """Validated outcome of one Jev request (success or failure)."""
    ok: bool
    status: str                      # ok | disabled | no_api_key | auth_error | insufficient_credits |
                                     # invalid_request | rate_limited | timeout | network_error |
                                     # service_error | invalid_response
    label: Optional[str] = None      # mapped candidate class, None if "unclear" or failure
    confidence: Optional[float] = None
    probabilities: dict[str, float] = field(default_factory=dict)   # class label → probability
    unclear: bool = False
    form_assessment: Optional[str] = None
    form_confidence: Optional[float] = None
    latency_ms: float = 0.0
    http_status: Optional[int] = None
    error: Optional[str] = None
    request_id: Optional[str] = None
    attempts: int = 0
    frame_id: Optional[int] = None
    frame_timestamp: Optional[float] = None   # monotonic seconds of the analysed frame
    epoch: Optional[int] = None               # app-defined pose epoch at submission
    candidates: list[str] = field(default_factory=list)
    focus_pose: Optional[str] = None          # pose whose checkpoints were asked, if any
    coaching: Optional[JevCoaching] = None    # checkpoint answers + form score for focus_pose


def _status_for_http(code: int) -> str:
    if code in (401, 403):
        return "auth_error"
    if code == 402:
        return "insufficient_credits"
    if code in (400, 404, 413, 415, 422):
        return "invalid_request"
    if code == 429:
        return "rate_limited"
    return "service_error"


def _safe_error_text(resp: Any) -> str:
    try:
        data = resp.json()
        err = data.get("error", data) if isinstance(data, dict) else data
        if isinstance(err, dict):
            err = err.get("message") or err.get("type") or str(err)
        return str(err)[:200]
    except Exception:
        return (getattr(resp, "text", "") or "")[:200]


def _valid_prob(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and 0.0 <= v <= 1.0 else None


class JevClient:
    """Synchronous Jev client. Use :class:`src.models.jev_worker.JevWorker` from the UI loop."""

    def __init__(self, config: Optional[JevConfig] = None, api_key: Optional[str] = None, session: Any = None):
        self.config = config or JevConfig()
        self._api_key = api_key if api_key is not None else (os.environ.get(self.config.api_key_env, "") or self.config.api_key or "")
        self._session = session

    def __repr__(self) -> str:  # never leak the key
        return f"JevClient(endpoint={self.config.endpoint!r}, model={self.config.model!r}, key={'set' if self._api_key else 'missing'})"

    @property
    def has_api_key(self) -> bool:
        return bool(self._api_key)

    def _http(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    # ── Request construction ──────────────────────────────────────────────────
    def encode_image(self, frame_bgr: np.ndarray) -> bytes:
        """Resize into the documented 280–1568 px window and JPEG-encode (≤ 4 MB)."""
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            raise ValueError("empty frame")
        h, w = frame_bgr.shape[:2]
        max_side = min(MAX_SIDE, max(MIN_SIDE, int(self.config.max_image_side)))
        scale = min(1.0, max_side / max(h, w))
        if min(h, w) * scale < MIN_SIDE:
            scale = MIN_SIDE / min(h, w)
        img = frame_bgr if abs(scale - 1.0) < 1e-6 else cv2.resize(
            frame_bgr, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        if max(img.shape[:2]) > MAX_SIDE:  # extreme aspect ratios cannot satisfy both bounds
            raise ValueError("image aspect ratio outside supported range")
        quality = int(np.clip(self.config.jpeg_quality, 30, 95))
        while True:
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                raise ValueError("JPEG encoding failed")
            data = buf.tobytes()
            if len(data) <= MAX_IMAGE_BYTES or quality <= 30:
                return data
            quality -= 15

    def coaching_pose(self, focus_pose: Optional[str]) -> Optional[str]:
        """The pose whose checkpoints are asked, or None when coaching is off / not defined."""
        if focus_pose and self.config.coach_checkpoints > 0 and checkpoints_for(focus_pose):
            return focus_pose
        return None

    def build_payload(self, jpeg: bytes, candidates: list[str],
                      focus_pose: Optional[str] = None) -> tuple[dict, dict[str, str]]:
        """Return (json body, option_key → class label map).

        With ``focus_pose`` the request also carries that pose's coaching checkpoints
        and an overall form score, answered against the same image.
        """
        key_map = {option_key(c): c for c in candidates}
        criteria = {k: f"{c}: {ASANA_VISUAL_HINTS.get(c, 'the yoga pose ' + c)}" for k, c in key_map.items()}
        criteria[UNCLEAR_KEY] = ("no person is clearly visible, the body is largely cut off or blurred, "
                                 "or the person is not in any of the listed postures")
        questions: dict[str, Any] = {
            "asana": {
                "type": "choice",
                "instructions": ("Which yoga posture is the person in `photo` holding? "
                                 "Choose 'unclear' rather than guessing when the posture cannot be seen well."),
                "criteria": criteria,
            }
        }
        if self.config.request_form_assessment:
            questions["form"] = {
                "type": "choice",
                "instructions": "How does the visible overall form of the posture in `photo` look?",
                "criteria": dict(FORM_CRITERIA),
            }
        coach = self.coaching_pose(focus_pose)
        if coach:
            hint = ASANA_VISUAL_HINTS.get(coach, "the yoga pose " + coach)
            questions.update(build_questions(coach, hint, self.config.coach_checkpoints))
        # The reference is matched as plain text anywhere in the request, so use a fresh UUID
        # and place it once, in the state; questions refer to it by path (`photo`).
        ref = str(uuid.uuid4())
        body = {
            "model": self.config.model,
            "state": {
                "context": ("A single still photo from a home yoga-practice webcam. "
                            "Judge only what is visible in the photo."),
                "photo": ref,
            },
            "assets": {ref: {"type": "image", "format": "jpg",
                             "data_base64": base64.b64encode(jpeg).decode("ascii")}},
            "questions": questions,
        }
        return body, key_map

    # ── Response parsing ──────────────────────────────────────────────────────
    @staticmethod
    def parse_response(data: Any, key_map: dict[str, str], want_form: bool = False,
                       coach_pose: Optional[str] = None, coach_checkpoints: int = 0) -> JevResult:
        """Validate a decoded JSON response; never trusts undocumented fields."""
        if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
            return JevResult(ok=False, status="invalid_response", error="missing 'answers' object")
        ans = data["answers"].get("asana")
        if not isinstance(ans, dict):
            return JevResult(ok=False, status="invalid_response", error="missing 'asana' answer")
        choice = ans.get("choice")
        valid_keys = set(key_map) | {UNCLEAR_KEY}
        if not isinstance(choice, str) or choice not in valid_keys:
            return JevResult(ok=False, status="invalid_response", error="choice is not one of the offered options")

        probs: dict[str, float] = {}
        raw_probs = ans.get("probabilities")
        if isinstance(raw_probs, dict):
            for k, v in raw_probs.items():
                p = _valid_prob(v)
                if k in valid_keys and p is not None:
                    probs[k] = p
        conf = _valid_prob(ans.get("confidence"))
        if conf is None:
            conf = probs.get(choice)  # fall back to the chosen option's probability, if given

        res = JevResult(
            ok=True, status="ok",
            label=key_map.get(choice) if choice != UNCLEAR_KEY else None,
            unclear=(choice == UNCLEAR_KEY),
            confidence=conf,
            probabilities={key_map[k]: v for k, v in probs.items() if k in key_map},
            request_id=str(data.get("id")) if data.get("id") is not None else None,
        )
        if want_form:
            f = data["answers"].get("form")
            if isinstance(f, dict) and f.get("choice") in FORM_CRITERIA:
                res.form_assessment = f["choice"]
                res.form_confidence = _valid_prob(f.get("confidence"))
        if coach_pose:
            res.focus_pose = coach_pose
            res.coaching = parse_coaching(data["answers"], coach_pose, coach_checkpoints)
        return res

    # ── Request execution ─────────────────────────────────────────────────────
    def classify(
        self,
        frame_bgr: np.ndarray,
        candidates: list[str],
        frame_id: Optional[int] = None,
        frame_timestamp: Optional[float] = None,
        epoch: Optional[int] = None,
        focus_pose: Optional[str] = None,
    ) -> JevResult:
        meta = dict(frame_id=frame_id, frame_timestamp=frame_timestamp, epoch=epoch, candidates=list(candidates))
        if not self._api_key:
            return JevResult(ok=False, status="no_api_key", error=f"{self.config.api_key_env} not set", **meta)
        if not candidates:
            return JevResult(ok=False, status="invalid_request", error="no candidate classes", **meta)
        coach = self.coaching_pose(focus_pose)
        try:
            body, key_map = self.build_payload(self.encode_image(frame_bgr), candidates, focus_pose=coach)
        except ValueError as exc:
            return JevResult(ok=False, status="invalid_request", error=str(exc), **meta)

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Idempotency-Key": str(uuid.uuid4()),
            **self.config.image_headers,
        }
        t0 = time.perf_counter()
        attempts = 0
        result: Optional[JevResult] = None
        for attempt in range(max(0, int(self.config.max_retries)) + 1):
            attempts += 1
            retry_after: Optional[float] = None
            try:
                resp = self._http().post(self.config.endpoint, json=body, headers=headers,
                                         timeout=self.config.timeout_seconds)
            except Exception as exc:  # requests.Timeout / ConnectionError / others
                name = type(exc).__name__
                status = "timeout" if "Timeout" in name else "network_error"
                result = JevResult(ok=False, status=status, error=name, **meta)
                retryable = True
            else:
                code = int(getattr(resp, "status_code", 0))
                if 200 <= code < 300:
                    try:
                        data = resp.json()
                    except Exception:
                        data = None
                    result = self.parse_response(data, key_map, self.config.request_form_assessment,
                                                 coach_pose=coach, coach_checkpoints=self.config.coach_checkpoints)
                    for k, v in meta.items():
                        setattr(result, k, v)
                    result.http_status = code
                    retryable = False
                else:
                    result = JevResult(ok=False, status=_status_for_http(code), http_status=code,
                                       error=_safe_error_text(resp), **meta)
                    retryable = code in RETRYABLE_HTTP
                    try:
                        retry_after = float(resp.headers.get("Retry-After"))
                    except Exception:
                        retry_after = None
            if not retryable or attempt >= self.config.max_retries:
                break
            time.sleep(min(4.0, retry_after if retry_after is not None else 0.5 * (2 ** attempt)))

        assert result is not None
        result.attempts = attempts
        result.latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        logger.info("[Jev] status=%s http=%s latency=%.0fms attempts=%d", result.status,
                    result.http_status, result.latency_ms, attempts)
        return result
