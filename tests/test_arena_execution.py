from agent.config import load_settings
from agent.run_cycle import _safe_order
from reasoning.ledger_builder import LedgerBuilder


class _Resp:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> dict:
        return self._payload


def test_safe_order_uses_arena_orders_route_and_polls(monkeypatch):
    settings = type(load_settings())(
        arena_key="test-key",
        dry_run=False,
        order_poll_attempts=2,
        order_poll_seconds=0.0,
    )
    calls: list[tuple[str, str]] = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(("POST", url))
        assert url.endswith("/api/v1/arena/orders")
        return _Resp(200, {"order_id": "ord_123", "status": "pending", "size_usdc_locked": "1.00"})

    def fake_get(url, headers=None, timeout=None):
        calls.append(("GET", url))
        return _Resp(
            200,
            {
                "order_id": "ord_123",
                "status": "filled",
                "size_usdc_filled": "1.00",
                "open_fills": [{"tx_hash": "0xabc", "clob_order_id": "clob-1"}],
            },
        )

    monkeypatch.setattr("agent.run_cycle.httpx.post", fake_post)
    monkeypatch.setattr("agent.run_cycle.httpx.get", fake_get)

    result = _safe_order(
        settings,
        {
            "fixture_code": "F",
            "team_code": "HOME",
            "usd_size": "1.00",
            "limit_price": 0.55,
            "time_in_force_seconds": 30,
            "idempotency_key": "idem-1",
        },
    )

    assert result["submitted"] is True
    assert result["final_status"] == "filled"
    assert result["tx_hash"] == "0xabc"
    assert result["clob_order_id"] == "clob-1"
    assert calls == [
        ("POST", "https://stair-ai.com/api/v1/arena/orders"),
        ("GET", "https://stair-ai.com/api/v1/arena/orders/ord_123"),
    ]


def test_ledger_wire_records_match_batch_contract_shape():
    records = LedgerBuilder("F", "PRE_MATCH", load_settings(True)).build_standard_trace(
        prediction={"fixture_code": "F", "window": "PRE_MATCH", "probabilities": {"home": 0.44, "draw": 0.30, "away": 0.26}, "confidence": 0.70},
        order={"action_type": "skip", "reason": "skip"},
        reflection={"decision": "skip", "data_complete": 0.8},
    )
    sample = records[0]
    assert "schema_version" in sample
    assert "session_id" in sample
    assert "record_id" in sample
    assert "behavior" in sample
    assert "client_ts_utc" in sample
    assert "upstream_record_id" in sample
    assert "parent_ids" not in sample
    assert "timestamp" not in sample
