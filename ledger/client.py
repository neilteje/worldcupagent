"""
Reasoning Ledger client — builds and batch-submits the trace.

Schema v0.3 (StairAI/Reasoning-Ledger). Key wire contract:
  - agent_id is NOT set client-side; the arena injects it from x-api-key
  - session_id is a client-supplied string grouping one decision cycle
  - client_ts_utc is epoch milliseconds (not seconds)
  - Batch POST: json={"records": [...]}  → {records: [...enriched], errors: [...]}
  - Predictions: Acting record with action_type="prediction",
    parameters={"fixture_code": str(fixture_id), "outcome": str, "probability": float}

Base record fields:
  schema_version, session_id, record_id, behavior, client_ts_utc

Behavior-specific fields:
  Observing    : trigger_source, trigger_type, trigger_description, trigger_payload_summary
  ToolCalling  : tool_meta, description, input_payload, output_payload, success, upstream_record_id
  Thinking     : model_invocation, prompt, inputs, output_payload, upstream_record_id
  Acting       : action_type, target_system, action_summary, parameters, dry_run, execution_status, execution_id
"""
from __future__ import annotations
import uuid
import time
import json
from typing import Any
import httpx
import config

_BATCH_URL = f"{config.ARENA_BASE}/api/v1/arena/ledger/records/batch"
_HEADERS = {
    "x-api-key": config.ARENA_KEY,
    "Content-Type": "application/json",
}
_SCHEMA_VERSION = "0.3"


# ── Helpers ────────────────────────────────────────────────────────────────

def _trunc(obj: Any, limit: int = 30000) -> str:
    """
    JSON-stringify + truncate to stay under per-record size limits.
    Always returns a string (Thinking.output_payload ≤ 32 KB; record ≤ 64 KB).
    Matches the notebook's _trunc() which always returns a string.
    """
    s = obj if isinstance(obj, str) else json.dumps(obj, default=str)
    if len(s) <= limit:
        return s
    return s[:limit] + f"…[truncated, was {len(s)} chars]"


def _new_record(session_id: str, behavior: str, **fields) -> dict:
    """
    Compose the BaseRecord envelope plus behavior-specific fields.
    agent_id intentionally omitted — the arena fills it from x-api-key.
    """
    rec = {
        "schema_version": _SCHEMA_VERSION,
        "session_id":     session_id,
        "record_id":      str(uuid.uuid4()),
        "behavior":       behavior,
        "client_ts_utc":  int(time.time() * 1000),  # milliseconds
    }
    rec.update({k: v for k, v in fields.items() if v is not None})
    return rec


def _build_mi(
    provider: str,
    model_name: str,
    internal_reasoning: str = "",
    tokens_in: int | None = None,
    tokens_out: int | None = None,
) -> dict:
    """Build a ModelInvocation dict."""
    mi: dict[str, Any] = {
        "provider":   provider,
        "model_name": model_name,
        "tokens_in":  tokens_in,
        "tokens_out": tokens_out,
    }
    if internal_reasoning:
        mi["internal_reasoning"] = internal_reasoning
    return mi


# ── Session builder ────────────────────────────────────────────────────────

class LedgerSession:
    """
    Accumulates records for one decision cycle (fixture × window).
    Call .submit() at end of session.
    """

    def __init__(self, fixture_id: int, fixture_name: str, window: str):
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self.session_id = f"{window.lower()}:{fixture_id}:{ts}"
        self.fixture_id = fixture_id
        self.fixture_name = fixture_name
        self.window = window
        self._records: list[dict] = []

    def _add(self, rec: dict) -> dict:
        self._records.append(rec)
        return rec

    def last_id(self) -> str | None:
        return self._records[-1]["record_id"] if self._records else None

    # ── Observing ──────────────────────────────────────────────────────────

    def trigger(self, source: str = "cron") -> dict:
        """Record the trigger that woke this agent run."""
        return self._add(_new_record(
            self.session_id, "Observing",
            trigger_source=source,
            trigger_type="cron_trigger",
            trigger_description=(
                f"{self.window} prediction run for fixture {self.fixture_id} "
                f"({self.fixture_name})"
            ),
            trigger_payload_summary=(
                f"fixture_id={self.fixture_id}; window={self.window}"
            ),
        ))

    # ── ToolCalling ────────────────────────────────────────────────────────

    def tool_call(
        self,
        name: str,
        endpoint: str,
        description: str,
        input_payload: Any,
        output_payload: Any,
        success: bool = True,
        via: str = "arena.proxy",
        upstream_ids: list[str] | None = None,
    ) -> dict:
        """Record an external API/data call."""
        return self._add(_new_record(
            self.session_id, "ToolCalling",
            upstream_record_id=upstream_ids or ([self.last_id()] if self.last_id() else []),
            tool_meta={"name": name, "endpoint": endpoint, "via": via},
            description=description,
            input_payload=_trunc(input_payload),
            output_payload=_trunc(output_payload),
            success=success,
        ))

    # ── Thinking ───────────────────────────────────────────────────────────

    def thinking(
        self,
        prompt_system: str,
        inputs: list[dict],
        output_payload: Any,
        provider: str,
        model_name: str,
        internal_reasoning: str = "",
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        upstream_ids: list[str] | None = None,
    ) -> dict:
        """Record an LLM reasoning step with chain-of-thought capture."""
        mi = _build_mi(provider, model_name, internal_reasoning, tokens_in, tokens_out)
        return self._add(_new_record(
            self.session_id, "Thinking",
            upstream_record_id=upstream_ids or ([self.last_id()] if self.last_id() else []),
            model_invocation=mi,
            prompt=_trunc(prompt_system, limit=16000),
            inputs=[
                {"input_record_id": inp.get("record_id"), "input_payload": _trunc(inp.get("payload"))}
                for inp in inputs
            ],
            output_payload=_trunc(output_payload),
        ))

    # ── Acting — prediction (scored by arena) ─────────────────────────────

    def acting_prediction(
        self,
        outcome: str,           # "home_win" | "draw" | "away_win"
        probability: float,     # clamped to [0.001, 0.999]
        upstream_ids: list[str] | None = None,
    ) -> dict:
        """
        Submit a prediction to the arena for PSL scoring.
        The arena validates + scores the latest Acting record with
        action_type="prediction" per window.
        """
        prob = max(0.001, min(0.999, float(probability)))
        return self._add(_new_record(
            self.session_id, "Acting",
            upstream_record_id=upstream_ids or ([self.last_id()] if self.last_id() else []),
            action_type="prediction",
            target_system="arena",
            action_summary=(
                f"Predict {outcome} @ p={prob:.3f} for fixture {self.fixture_id}"
            ),
            parameters={
                "fixture_code": str(self.fixture_id),
                "outcome":      outcome,
                "probability":  prob,
            },
            dry_run=False,
            execution_status="confirmed",
        ))

    # ── Acting — order ─────────────────────────────────────────────────────

    def acting_order(
        self,
        direction: str,
        outcome: str,
        size_usdc: float,
        limit_price: float,
        order_payload: dict,
        execution_status: str = "pending",
        execution_id: str | None = None,
        upstream_ids: list[str] | None = None,
    ) -> dict:
        """Record placing a bet order."""
        return self._add(_new_record(
            self.session_id, "Acting",
            upstream_record_id=upstream_ids or ([self.last_id()] if self.last_id() else []),
            action_type="open_order",
            target_system="arena",
            action_summary=(
                f"Open {direction} ${size_usdc:.2f} on {outcome} @ ≤{limit_price}"
            ),
            parameters=order_payload,
            dry_run=False,
            execution_status=execution_status,
            execution_id=execution_id,
        ))

    # ── Submit ─────────────────────────────────────────────────────────────

    def submit(self) -> dict:
        """
        Batch-submit all records for this session.
        Wire format: json={"records": [...]}
        Response: {"records": [...enriched], "errors": [...]}
        """
        resp = httpx.post(
            _BATCH_URL,
            headers=_HEADERS,
            json={"records": self._records},
            timeout=60,
        )
        if resp.status_code == 404:
            return {"status": "404_not_yet_live", "records_built": len(self._records)}
        resp.raise_for_status()
        return resp.json()

    def record_count(self) -> int:
        return len(self._records)

    def records(self) -> list[dict]:
        return list(self._records)
