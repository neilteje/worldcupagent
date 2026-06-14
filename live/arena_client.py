"""
Per-agent arena API client (1 API key = 1 agent identity).

Implements the 2026-06-10 arena API contract (jupyter/release_note/
ARENA_RELEASE_NOTES_20260610.md):

  - GET  /v1/arena/matches[/{fixture_id}]      — window timing + server_ts_utc
  - GET  /v1/arena/agents/me                   — wallet: available/locked USDC
  - POST /v1/arena/orders  (fixture_id field!) — buy-YES limit order, 30s TIF
  - GET  /v1/arena/orders[/{order_id}]         — order list / polling
  - POST /v1/arena/orders/{order_id}/close     — close an open order/position
  - GET  /v1/arena/exposure                    — live token holdings
  - GET  /v1/arena/polymarket/markets/{fid}/settlement — resolved prices
  - POST /v1/arena/ledger/... handled by ledger/client.py (per-key sessions)

Every method is a thin, typed wrapper; callers decide what failures mean.
"""
from __future__ import annotations
import time
import uuid
from typing import Any

import httpx

import config

_BASE = f"{config.ARENA_BASE}/api/v1/arena"

# Order terminal states (polling stops on these).
TERMINAL_ORDER_STATES = {"filled", "closed", "rejected", "cancelled", "expired"}
ORDERS_CLOSEABLE_AT_HALFTIME = {
    "open", "pending", "accepted", "submitted", "partially_filled", "filled"
}


def _first_present(data: dict, *keys: str) -> Any:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return None


def _to_float(value) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fill_report(order: dict | None) -> dict:
    """Extract actual fill accounting from known arena order response shapes."""
    order = order or {}
    fills = order.get("open_fills") or order.get("fills") or []
    if not isinstance(fills, list):
        fills = []

    filled_notional = _to_float(_first_present(
        order, "size_usdc_filled", "filled_usdc", "filled_notional", "filled_notional_usdc"
    ))
    filled_shares = _to_float(_first_present(
        order, "filled_shares", "shares_filled", "quantity_filled", "size_tokens_filled"
    ))
    avg_price = _to_float(_first_present(
        order, "average_fill_price", "avg_fill_price", "avg_price", "filled_avg_price"
    ))
    if avg_price is None and filled_notional is not None and filled_shares:
        avg_price = filled_notional / filled_shares

    requested = _to_float(_first_present(order, "size_usdc", "usd_size", "requested_usdc"))
    unfilled = _to_float(_first_present(order, "size_usdc_unfilled", "unfilled_usdc"))
    if unfilled is None and requested is not None and filled_notional is not None:
        unfilled = max(0.0, requested - filled_notional)

    status = str(order.get("status") or "").lower()
    partial_state = "unfilled"
    if filled_notional and filled_notional > 0:
        partial_state = "filled" if status == "filled" else "partial"
    if status in ("rejected", "cancelled", "expired"):
        partial_state = status

    return {
        "requested_stake_usdc": requested,
        "submitted_limit_price": _to_float(_first_present(order, "limit_price")),
        "actual_average_fill_price": avg_price,
        "filled_shares": filled_shares,
        "filled_notional_usdc": filled_notional,
        "unfilled_usdc": unfilled,
        "fees_usdc": _to_float(_first_present(order, "fees_usdc", "fee_usdc", "total_fees_usdc")),
        "timestamp": _first_present(order, "updated_at", "created_at", "timestamp"),
        "order_id": _first_present(order, "order_id", "id"),
        "market_id": _first_present(order, "market_id", "condition_id", "clob_market_id"),
        "partial_fill_state": partial_state,
        "raw_fill_count": len(fills),
    }


class ArenaClient:
    """All arena operations for ONE agent identity (one x-api-key)."""

    def __init__(self, api_key: str | None = None, name: str = "agent"):
        self.api_key = api_key or config.ARENA_KEY
        self.name = name
        self._h = {"x-api-key": self.api_key, "Content-Type": "application/json"}

    # ── Identity / wallet ────────────────────────────────────────────────

    def me(self) -> dict:
        resp = httpx.get(f"{_BASE}/agents/me", headers=self._h, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def wallet(self) -> dict:
        """
        Returns {"available": float, "locked": float, "address": str|None}.
        20260610 shape: agents/me → {"wallet": {available_balance_usdc, ...}};
        falls back to the legacy flat wallet_balance_usd field.
        """
        info = self.me()
        w = info.get("wallet") or {}
        if w:
            return {
                "available": float(w.get("available_balance_usdc") or 0),
                "locked":    float(w.get("locked_balance_usdc") or 0),
                "address":   w.get("address"),
            }
        return {"available": float(info.get("wallet_balance_usd") or 0),
                "locked": 0.0, "address": None}

    # ── Match timing (authoritative window state) ────────────────────────

    def matches(self) -> list[dict]:
        resp = httpx.get(f"{_BASE}/matches", headers=self._h, timeout=20)
        resp.raise_for_status()
        body = resp.json()
        return body.get("matches") or body.get("data") or (body if isinstance(body, list) else [])

    def match(self, fixture_id: int | str) -> dict | None:
        """
        Match timing for one fixture: kickoff, window boundaries,
        `current_window` (the window open NOW, or None), `server_ts_utc`.
        Align to server_ts_utc, never the local clock. Returns None on 404.
        """
        resp = httpx.get(f"{_BASE}/matches/{fixture_id}", headers=self._h, timeout=20)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    # ── Orders ───────────────────────────────────────────────────────────

    def place_order(
        self,
        fixture_id: int | str,
        team_code: str,
        usd_size: float,
        limit_price: float,
        tif_seconds: int | None = None,
    ) -> dict:
        """
        POST one buy-YES limit order. usd_size is a string on the wire.
        Returns the raw response dict; {"status": "not_live", ...} on 404.
        """
        payload = {
            "fixture_id":            str(fixture_id),
            "team_code":             team_code,
            "usd_size":              f"{round(float(usd_size), 2):.2f}",
            "limit_price":           round(float(limit_price), 4),
            "time_in_force_seconds": tif_seconds or config.DEFAULT_TIF_SECONDS,
            "idempotency_key":       str(uuid.uuid4()),
        }
        resp = httpx.post(f"{_BASE}/orders", headers=self._h, json=payload, timeout=60)
        if resp.status_code == 404:
            return {"status": "not_live", "payload": payload}
        if not resp.is_success:
            return {"status": "rejected_http", "http_status": resp.status_code,
                    "body": resp.text[:400], "payload": payload}
        out = resp.json()
        out["payload"] = payload
        return out

    def get_order(self, order_id: str) -> dict | None:
        resp = httpx.get(f"{_BASE}/orders/{order_id}", headers=self._h, timeout=15)
        if not resp.is_success:
            return None
        return resp.json()

    def orders(self, status: str | None = None) -> list[dict]:
        params = {"status": status} if status else None
        resp = httpx.get(f"{_BASE}/orders", headers=self._h, params=params, timeout=20)
        resp.raise_for_status()
        body = resp.json()
        return body.get("orders") or body.get("data") or (body if isinstance(body, list) else [])

    def poll_order(self, order_id: str, attempts: int = 6, interval: float = 5.0) -> dict:
        """
        Poll an order to a terminal state (≤ attempts × interval ≈ the 30s TIF).
        Returns {"final_status", "tx_hash", "clob_order_id", "reject_reason",
                 "filled_usdc", "fill_report"}.
        """
        out = {"final_status": None, "tx_hash": None,
               "clob_order_id": None, "reject_reason": None, "filled_usdc": None,
               "fill_report": _fill_report(None)}
        for _ in range(attempts):
            time.sleep(interval)
            d = self.get_order(order_id)
            if not d:
                continue
            out["final_status"] = d.get("status")
            out["reject_reason"] = d.get("rejection_reason") or out["reject_reason"]
            out["filled_usdc"] = d.get("size_usdc_filled") or out["filled_usdc"]
            out["fill_report"] = _fill_report(d)
            fills = d.get("open_fills") or []
            if fills:
                out["tx_hash"] = fills[0].get("tx_hash") or out["tx_hash"]
                out["clob_order_id"] = fills[0].get("clob_order_id") or out["clob_order_id"]
            if out["final_status"] in TERMINAL_ORDER_STATES:
                break
        return out

    def close_order(self, order_id: str) -> dict:
        """
        Ask the arena to close an order/position.

        The close endpoint is intentionally separate from settlement; callers
        use it for explicit exits such as the halftime flatten-before-retrade
        workflow. Returns a normalized failure dict instead of raising on
        non-2xx so the live loop can keep moving.
        """
        resp = httpx.post(f"{_BASE}/orders/{order_id}/close", headers=self._h, timeout=60)
        if not resp.is_success:
            return {
                "status": "close_rejected_http",
                "order_id": order_id,
                "http_status": resp.status_code,
                "body": resp.text[:400],
            }
        out = resp.json()
        out.setdefault("order_id", order_id)
        return out

    def close_fixture_orders(self, fixture_id: int | str) -> list[dict]:
        """
        Close all orders for this fixture whose status may still represent
        active exposure. Best-effort: one failed close does not block the rest.
        """
        results: list[dict] = []
        for order in self.orders():
            if str(order.get("fixture_id") or order.get("fixture_code")) != str(fixture_id):
                continue
            status = str(order.get("status") or "").lower()
            if status not in ORDERS_CLOSEABLE_AT_HALFTIME:
                continue
            order_id = order.get("order_id") or order.get("id")
            if not order_id:
                continue
            try:
                result = self.close_order(str(order_id))
            except Exception as exc:
                result = {"status": "close_error", "order_id": order_id, "error": repr(exc)}
            result.setdefault("previous_status", status)
            results.append(result)
        return results

    @staticmethod
    def execution_status_for(final_status: str | None, tx_hash: str | None,
                             submitted_ok: bool) -> str:
        """
        Map a polled order outcome to the ledger Acting.execution_status enum
        (sample-agent mapping): filled / closed+tx → confirmed;
        closed-without-tx / rejected → failed; non-terminal → pending.
        """
        if final_status == "filled" or (final_status == "closed" and tx_hash):
            return "confirmed"
        if final_status in ("closed", "rejected", "cancelled", "expired"):
            return "failed"
        return "pending" if submitted_ok else "failed"

    # ── Exposure / settlement ────────────────────────────────────────────

    def exposure(self) -> list[dict]:
        """Live token holdings per outcome (the 20260610 /exposure meaning)."""
        resp = httpx.get(f"{_BASE}/exposure", headers=self._h, timeout=20)
        resp.raise_for_status()
        body = resp.json()
        return body.get("positions") or body.get("data") or (body if isinstance(body, list) else [])

    def settlement(self, fixture_id: int | str) -> dict | None:
        """Resolved Polymarket prices for a fixture (winner token → 1)."""
        resp = httpx.get(f"{_BASE}/polymarket/markets/{fixture_id}/settlement",
                         headers=self._h, timeout=20)
        if not resp.is_success:
            return None
        return resp.json()

    # ── Connectivity ─────────────────────────────────────────────────────

    def check(self) -> dict[str, Any]:
        """One-shot health check used by `python -m live test`."""
        out: dict[str, Any] = {"name": self.name, "ok": False}
        try:
            info = self.me()
            w = self.wallet()
            out.update(ok=True,
                       agent_id=info.get("agent_id"),
                       display_name=info.get("display_name"),
                       phase=info.get("lifecycle_phase"),
                       available=w["available"], locked=w["locked"])
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {e}"
        return out
