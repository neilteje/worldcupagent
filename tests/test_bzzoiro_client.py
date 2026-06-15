"""BZZOIRO client behaviour (spec §13/§29) — auth, retries, errors, pagination,
snapshots, shadow-only default. All network is mocked via httpx.MockTransport."""
from __future__ import annotations

import httpx
import pytest

import config
import data.bzzoiro as bz
from data.bzzoiro import BzzoiroAPIClient
from data.bzzoiro_snapshots import create_bzzoiro_snapshot, generate_payload_hash


@pytest.fixture(autouse=True)
def _enable_bzzoiro(monkeypatch):
    monkeypatch.setattr(config, "BZZOIRO_ENABLED", True)
    monkeypatch.setattr(config, "BZZOIRO_KEY", "test-token")
    monkeypatch.setattr(config, "BZZOIRO_MAX_RETRIES", 2)
    # Never actually sleep during retry/backoff tests.
    monkeypatch.setattr(bz.time, "sleep", lambda *_: None)


def _client_with(handler):
    c = BzzoiroAPIClient()
    c._client = httpx.Client(base_url=c.base_url, headers=c.headers,
                             transport=httpx.MockTransport(handler))
    return c


def test_auth_header_sent():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"id": 1})

    c = _client_with(handler)
    c.get_event(1)
    assert seen["auth"] == "Token test-token"


def test_404_returns_structured_error_not_empty():
    c = _client_with(lambda r: httpx.Response(404, json={"detail": "nope"}))
    out = c.get_event(7)
    assert out.get("error") == "HTTP 404"


def test_500_retries_then_structured_error():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(500, json={"err": "boom"})

    c = _client_with(handler)
    out = c.get_event(7)
    assert calls["n"] == config.BZZOIRO_MAX_RETRIES + 1   # initial + retries
    assert out.get("error") == "HTTP 500"


def test_429_respects_retry_after_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={})
        return httpx.Response(200, json={"id": 1, "ok": True})

    c = _client_with(handler)
    out = c.get_event(1)
    assert calls["n"] == 2
    assert out["ok"] is True


def test_request_error_returns_structured_error():
    def handler(request):
        raise httpx.ConnectError("down", request=request)

    c = _client_with(handler)
    out = c.get_event(1)
    assert out.get("error") == "RequestError"


def test_pagination_follows_next():
    pages = [
        httpx.Response(200, json={"results": [{"id": 1}, {"id": 2}], "next": "x"}),
        httpx.Response(200, json={"results": [{"id": 3}], "next": None}),
    ]
    state = {"i": 0}

    def handler(request):
        resp = pages[state["i"]]
        state["i"] += 1
        return resp

    c = _client_with(handler)
    teams = c.search_teams("Alpha")
    assert [t["id"] for t in teams] == [1, 2, 3]


def test_malformed_payload_surfaces_error():
    c = _client_with(lambda r: httpx.Response(200, content=b"not json"))
    out = c.get_event(1)
    assert "error" in out


def test_disabled_client_returns_error(monkeypatch):
    monkeypatch.setattr(config, "BZZOIRO_ENABLED", False)
    c = _client_with(lambda r: httpx.Response(200, json={"id": 1}))
    assert c.get_event(1)["error"] == "BZZOIRO_ENABLED is False"


def test_snapshot_hashing_stable_and_sensitive():
    p1 = {"a": 1, "updated_at": "2026-06-14T00:00:00Z"}
    snap = create_bzzoiro_snapshot("event", "42", p1, success=True)
    assert snap.provider == "bzzoiro"
    assert snap.payload_hash == generate_payload_hash(p1)
    assert snap.provider_updated_at is not None
    assert generate_payload_hash(p1) != generate_payload_hash({"a": 2})


def test_external_model_shadow_only_by_default():
    # Spec §17/§31: BZZOIRO model is shadow-only with zero weight by default.
    assert config.BZZOIRO_MODEL_SHADOW_ONLY is True
    assert config.BZZOIRO_MODEL_WEIGHT == 0.0
