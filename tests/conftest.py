"""
Shared test fixtures.

Design goals:
- Tests must run deterministically WITHOUT a GPU, a webcam, the 94 MB model
  weights, or external credentials.
- The heavy ML stack (torch / torchvision / cv2 model internals) is stubbed in
  the `model` module; the FastAPI app itself is imported for real so API,
  streak, tutor and database behaviour is fully exercised.
- Cloudflare credentials are unset so the LLM path deterministically falls
  back (missing-config short-circuit → instant, no network call).
"""

import copy
import os
import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --- deterministic environment (must happen before any project import) ----
os.environ.pop("CLOUDFLARE_ACCOUNT_ID", None)  # unconfigured → deterministic fallback
os.environ.pop("CLOUDFLARE_API_TOKEN", None)   # no credentials in tests
os.environ.setdefault("DB_PATH", str(ROOT / "test_runtime.db"))

CLASS_NAMES = [
    "Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral",
]


def _probs(emotion: str, confidence: float) -> dict:
    others = (1.0 - confidence) / (len(CLASS_NAMES) - 1)
    return {c: round(others, 6) if c != emotion else round(confidence, 6) for c in CLASS_NAMES}


def _predict_image(model, img):
    e = FAKE["emotion"]
    c = FAKE["confidence"]
    return {
        "label": e,
        "confidence": c,
        "emoji": "X",
        "probs": _probs(e, c),
    }


def _predict_frame(model, frame):
    res = _predict_image(model, frame)
    res["face"] = {"x": 0, "y": 0, "w": 640, "h": 480}
    return [res]


def _load_model(weights_path: str = "unused.pth"):
    return types.SimpleNamespace(fake=True, weights=weights_path)


# Safe to call — main.py only uses load_model/predict_image/predict_frame.
FAKE = {"emotion": "Neutral", "confidence": 0.9}  # mutated by the set_emotion fixture

model_stub = types.ModuleType("model")
model_stub.CLASS_NAMES = list(CLASS_NAMES)
model_stub.load_model = _load_model
model_stub.predict_image = _predict_image
model_stub.predict_frame = _predict_frame
model_stub.EMOTION_EMOJIS = {}
model_stub.get_model_info = lambda: {
    "model_name": "emotion-resnet50",
    "model_version": "v1",
    "architecture": "resnet50",
    "num_classes": len(CLASS_NAMES),
    "classes": list(CLASS_NAMES),
    "git_commit": "test",
    "metric_status": "test-stub",
}
model_stub._git_commit = lambda: "test"
sys.modules.setdefault("model", model_stub)

import database  # noqa: E402  (now has DB_PATH from env)
import main      # noqa: E402
import tutor_trigger  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_global_state():
    """Fresh module state + clean DB between tests."""
    main.device_results.clear()
    tutor_trigger.reset_all()
    conn = database.get_connection()
    conn.execute("DELETE FROM predictions")
    conn.execute("DELETE FROM tutor_feedback")
    conn.commit()
    conn.close()
    yield


@pytest.fixture
def set_emotion():
    """Dynamically control what the fake model predicts."""
    def _set(emotion: str, confidence: float = 0.92):
        FAKE["emotion"] = emotion
        FAKE["confidence"] = confidence
        FAKE["probs"] = _probs(emotion, confidence)
    yield _set


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    with TestClient(main.app) as c:
        yield c