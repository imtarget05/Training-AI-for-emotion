"""Reliability tests for §6: retry 429/502/503/504, exhausted-429 → 503.

Covers tutor._call_cloudflare retry loop + API mapping in main.py.
All provider HTTP is faked — no credentials, no network.
"""

import asyncio
import types

import pytest

import tutor


class _FakeResponse:
    def __init__(self, status_code, body=None):
        self.status_code = status_code
        self._body = body or {}

    def raise_for_status(self):
        if 400 <= self.status_code:
            import httpx

            req = httpx.Request("POST", "http://test")
            resp = httpx.Response(self.status_code, request=req)
            raise httpx.HTTPStatusError("err", request=req, response=resp)

    def json(self):
        return self._body


class _FakeClient:
    """Fake httpx.AsyncClient serving a scripted response sequence."""

    script = []
    calls = 0

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, *a, **k):
        type(self).calls += 1
        idx = min(type(self).calls - 1, len(type(self).script) - 1)
        item = type(self).script[idx]
        if isinstance(item, BaseException):
            raise item
        return item


def _patch_env(monkeypatch, max_retries=2):
    monkeypatch.setattr(tutor, "CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setattr(tutor, "CLOUDFLARE_API_TOKEN", "tok")
    monkeypatch.setattr(tutor, "CLOUDFLARE_MAX_RETRIES", max_retries)
    monkeypatch.setattr(tutor, "CLOUDFLARE_RETRY_BACKOFF_SECONDS", 0.0)

    async def _no_sleep(_):
        return None

    monkeypatch.setattr(tutor.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(tutor.httpx, "AsyncClient", _FakeClient)
    _FakeClient.calls = 0


def _ok(text="Hi, let's take it step by step."):
    return _FakeResponse(200, {"result": {"response": text}, "success": True})


def test_retry_then_success(monkeypatch):
    _patch_env(monkeypatch)
    _FakeClient.script = [_FakeResponse(503), _ok()]
    out = asyncio.run(tutor._call_cloudflare("prompt"))
    assert out is not None and "step by step" in out
    assert _FakeClient.calls == 2


def test_retryable_statuses_all_retried(monkeypatch):
    import itertools

    for status in (429, 502, 503, 504):
        _patch_env(monkeypatch, max_retries=1)
        _FakeClient.script = [_FakeResponse(status), _ok("recovered")]
        out = asyncio.run(tutor._call_cloudflare("p"))
        assert out == "recovered", f"status {status} should be retried"
        assert _FakeClient.calls == 2


def test_exhausted_429_raises(monkeypatch):
    _patch_env(monkeypatch, max_retries=2)
    _FakeClient.script = [_FakeResponse(429)]
    with pytest.raises(tutor.CloudflareRateLimitExhausted):
        asyncio.run(tutor._call_cloudflare("p"))
    assert _FakeClient.calls == 3  # 1 initial + 2 retries


def test_exhausted_503_falls_back_no_raise(monkeypatch):
    _patch_env(monkeypatch, max_retries=2)
    _FakeClient.script = [_FakeResponse(503)]
    out = asyncio.run(tutor._call_cloudflare("p"))
    assert out is None
    assert _FakeClient.calls == 3


def test_401_no_retry_immediate_fallback(monkeypatch):
    _patch_env(monkeypatch)
    _FakeClient.script = [_FakeResponse(401)]
    out = asyncio.run(tutor._call_cloudflare("p"))
    assert out is None
    assert _FakeClient.calls == 1


def test_transport_error_retried_then_success(monkeypatch):
    import httpx

    _patch_env(monkeypatch, max_retries=2)
    _FakeClient.script = [httpx.ConnectError("down"), _ok("back")]
    out = asyncio.run(tutor._call_cloudflare("p"))
    assert out == "back"
    assert _FakeClient.calls == 2


def test_generate_propagates_rate_limit(monkeypatch):
    async def _raise(*a, **k):
        raise tutor.CloudflareRateLimitExhausted("429 persisted")

    monkeypatch.setattr(tutor, "_call_cloudflare", _raise)
    with pytest.raises(tutor.CloudflareRateLimitExhausted):
        asyncio.run(tutor.generate_tutor_feedback("Sadness", 0.8))


def test_on_demand_503_mapping(client, monkeypatch):
    import main

    async def _raise(*a, **k):
        raise tutor.CloudflareRateLimitExhausted("429 persisted")

    monkeypatch.setattr(main, "generate_tutor_feedback", _raise)
    r = client.post(
        "/tutor/feedback",
        json={"device_id": "d1", "emotion": "Sadness", "confidence": 0.8},
    )
    assert r.status_code == 503
    assert "rate limit" in r.json()["detail"].lower()


def test_predict_background_rate_limit_keeps_200(client, set_emotion, monkeypatch):
    import main

    async def _raise(*a, **k):
        raise tutor.CloudflareRateLimitExhausted("429 persisted")

    monkeypatch.setattr(main, "generate_tutor_feedback", _raise)
    set_emotion("Sadness")
    import io

    from PIL import Image

    for _ in range(2):
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (10, 10, 10)).save(buf, format="JPEG")
        client.post(
            "/predict",
            files={"file": ("f.jpg", buf.getvalue(), "image/jpeg")},
            data={"device_id": "rl-dev"},
        )
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (10, 10, 10)).save(buf, format="JPEG")
    r = client.post(
        "/predict",
        files={"file": ("f.jpg", buf.getvalue(), "image/jpeg")},
        data={"device_id": "rl-dev"},
    )
    assert r.status_code == 200
    fb = r.json().get("tutor_feedback")
    assert fb is not None
    assert fb.get("rate_limited") is True
    assert fb.get("retryable") is True
