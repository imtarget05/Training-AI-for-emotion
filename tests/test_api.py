"""API-level tests via FastAPI TestClient (Phase 3, 4, 7).

The fake emotion model in conftest lets us drive exact emotion sequences so
the streak/cooldown/tutor behaviour is exercised end-to-end through real HTTP.
"""

import io
from PIL import Image

import main


def _jpeg_bytes(size=(64, 64), color=(120, 40, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def _predict(client, device="emulator", image=None, emotion=None, confidence=None):
    del emotion, confidence  # sequence is controlled via set_emotion fixture
    files = {"file": ("frame.jpg", image or _jpeg_bytes(), "image/jpeg")}
    data = {"device_id": device}
    return client.post("/predict", files=files, data=data)


# ── Phase 3: /predict contract & backward compatibility ────────────────────

def test_predict_returns_backward_compatible_fields(client, set_emotion):
    set_emotion("Happiness")
    r = _predict(client)
    assert r.status_code == 200
    body = r.json()
    # original contract unchanged
    for key in ("device_id", "timestamp", "label", "confidence", "probs", "emoji"):
        assert key in body
    assert body["label"] == "Happiness"
    assert body["device_id"] == "emulator"
    assert "tutor_feedback" not in body  # optional field stays absent


def test_predict_invalid_image_returns_400(client):
    r = client.post("/predict", files={"file": ("bad.jpg", b"not-an-image", "image/jpeg")})
    assert r.status_code == 400
    assert "error" in r.json()


def test_latest_emotion_404_then_200(client, set_emotion):
    assert client.get("/latest/emotion").status_code == 404
    set_emotion("Neutral")
    _predict(client)
    latest = client.get("/latest/emotion")
    assert latest.status_code == 200
    assert latest.json()["label"] == "Neutral"


# ── Phase 4: tutor trigger cases over real HTTP ────────────────────────────

def test_one_support_frame_no_tutor(client, set_emotion):
    set_emotion("Sadness")
    body = _predict(client).json()
    assert "tutor_feedback" not in body


def test_two_support_frames_no_tutor(client, set_emotion):
    set_emotion("Sadness")
    _predict(client)
    body2 = _predict(client).json()
    assert "tutor_feedback" not in body2


def test_third_support_frame_triggers_fallback(client, set_emotion):
    set_emotion("Sadness", confidence=0.85)
    for i in range(2):
        _predict(client)
    body3 = _predict(client).json()
    fb = body3.get("tutor_feedback")
    assert fb is not None
    assert fb["source"] == "fallback"  # Cloudflare unconfigured in tests
    assert fb["emotion"] == "Sadness"
    assert fb["message"]
    assert set(fb.keys()) >= {"message", "source", "emotion", "generated_at", "latency_ms"}


def test_non_support_emotion_resets_streak(client, set_emotion):
    set_emotion("Sadness")
    _predict(client)
    set_emotion("Fear")
    _predict(client)
    set_emotion("Sadness")
    body = _predict(client).json()
    assert "tutor_feedback" not in body  # streak was broken by Fear


def test_cooldown_prevents_second_feedback(client, set_emotion):
    set_emotion("Sadness")
    for _ in range(3):
        _predict(client)  # 3rd → trigger
    hist1 = client.get("/tutor/history", params={"device_id": "emulator"}).json()["history"]
    assert len(hist1) == 1
    # 3 more consecutive support frames → cooldown blocks a second trigger
    for _ in range(3):
        body = _predict(client).json()
        assert "tutor_feedback" not in body
    hist2 = client.get("/tutor/history", params={"device_id": "emulator"}).json()["history"]
    assert len(hist2) == 1


# ── Phase 7: on-demand endpoint + history API ──────────────────────────────

def test_tutor_feedback_on_demand(client, set_emotion):
    set_emotion("Fear")
    r = client.post("/tutor/feedback", json={
        "device_id": "manual", "emotion": "Fear", "confidence": 0.7, "lang": "vi",
    })
    assert r.status_code == 200
    fb = r.json()
    assert set(fb.keys()) >= {"message", "source", "emotion", "generated_at", "latency_ms"}
    assert fb["emotion"] == "Fear"
    assert fb["source"] == "fallback"

    hist = client.get("/tutor/history", params={"device_id": "manual"}).json()["history"]
    assert len(hist) == 1 and hist[0]["trigger_emotion"] == "Fear"


def test_tutor_history_empty_and_filtered(client):
    assert client.get("/tutor/history").json() == {"history": []}
    client.post("/tutor/feedback", json={"emotion": "Anger", "device_id": "dev-filter"})
    rows = client.get("/tutor/history", params={"device_id": "dev-filter"}).json()["history"]
    assert len(rows) == 1
    # unknown device returns empty without error
    assert client.get("/tutor/history", params={"device_id": "ghost"}).json()["history"] == []


def test_root_lists_tutor_endpoints(client):
    eps = client.get("/").json()["endpoints"]
    assert "tutor_feedback_on_demand" in eps
    assert "tutor_history" in eps


def test_model_info_endpoint(client):
    r = client.get("/info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_name"] == "emotion-resnet50"
    assert body["model_version"] == "v1"
    assert body["architecture"] == "resnet50"
    assert body["num_classes"] == 7
    assert len(body["classes"]) == 7
    # No sensitive/internal leakage
    assert "weights_file" not in body or body.get("weights_file")
    for forbidden in ("/Users", "secret", "env", "DB_PATH"):
        assert forbidden.lower() not in str(body).lower()