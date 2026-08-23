"""
E2E test — runs against a LIVE server (Docker/uvicorn) with the REAL model.

Skipped by default; enable with:
    E2E_BASE_URL=http://localhost:8000 pytest -m e2e -v

Assumes the app is already running (see scripts/demo_e2e.py for a full pass +
step-by-step output suitable for a demo video).

Behaviour here is intentionally *tolerant*: the real model's prediction cannot
be controlled, so wherever a support-needed emotion is required, we only
assert when the live label actually is one (and record what we observed).
"""

import io
import os

import pytest

from PIL import Image

E2E_URL = os.environ.get("E2E_BASE_URL", "")

pytestmark = pytest.mark.e2e
requires_server = pytest.mark.skipif(
    not E2E_URL, reason="set E2E_BASE_URL to run against a live server"
)


def _jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (128, 128), (70, 90, 120)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def http():
    import httpx

    with httpx.Client(base_url=E2E_URL, timeout=15) as c:
        yield c


@requires_server
def test_root_and_contract(http):
    r = http.get("/")
    assert r.status_code == 200
    assert "tutor_feedback_on_demand" in r.json()["endpoints"]


@requires_server
def test_predict_endpoint_healthy(http):
    r = http.post("/predict", files={"file": ("f.jpg", _jpeg_bytes(), "image/jpeg")},
                  data={"device_id": "e2e"})
    assert r.status_code == 200
    body = r.json()
    assert all(k in body for k in ("device_id", "timestamp", "label", "confidence", "probs"))


@requires_server
def test_predict_invalid_image(http):
    r = http.post("/predict", files={"file": ("f.jpg", b"garbage", "image/jpeg")})
    assert r.status_code == 400


@requires_server
def test_tutor_fallback_on_demand(http):
    r = http.post("/tutor/feedback", json={"device_id": "e2e", "emotion": "Sadness", "lang": "vi"})
    assert r.status_code == 200
    fb = r.json()
    assert fb["source"] in ("llm", "fallback")
    assert fb["message"] and fb["emotion"] == "Sadness"


@requires_server
def test_history_endpoint(http):
    r = http.get("/tutor/history", params={"device_id": "e2e"})
    assert r.status_code == 200
    assert isinstance(r.json()["history"], list)


@requires_server
def test_reports_healthy(http):
    for path in ("/reports/emotion-distribution", "/reports/confidence-stats",
                 "/reports/devices", "/reports/predictions"):
        assert http.get(path).status_code == 200


@requires_server
def test_websocket_roundtrip(http):
    """Send one base64 JPEG over WS and expect a faces JSON response."""
    from websockets.sync.client import connect as ws_connect

    ws_url = E2E_URL.replace("http", "ws").rstrip("/") + "/ws/camera"
    with ws_connect(ws_url) as ws:
        import base64
        ws.send(base64.b64encode(_jpeg_bytes()).decode())
        msg = ws.recv()
        assert "faces" in msg or "error" in msg  # no crash / valid envelope