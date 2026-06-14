"""
Reasoning Ledger client — builds and batch-submits the trace.

Schema v0.3 (StairAI/Reasoning-Ledger, per jupyter/arena_batch_submit_schema.md
and the 2026-06-10 arena release notes). Key wire contract:
  - agent_id is NOT set client-side; the arena injects it from x-api-key
  - session_id is a client-supplied string grouping one decision cycle
  - client_ts_utc is epoch milliseconds (not seconds)
  - Batch POST: json={"records": [...], "fixture_id": "..."} — the top-level
    fixture_id binds every session in the batch to the fixture so predictions
    can be scored. Response: {records: [...enriched], errors: [...]}
  - Predictions: Acting record with action_type="prediction",
    parameters={"fixture_id": str(fixture_id), "outcome": str, "probability": float}
    (fixture_code was RENAMED to fixture_id in the 20260610 release — sending
    the old name fails the prediction)
  - Dry-run validation: POST /ledger/records/validate (same checks, no persist)

Behavior-specific fields (only schema-valid fields are emitted):
  Observing    : trigger_source, trigger_type, trigger_description, trigger_payload_summary
  ToolCalling  : tool_meta, description, input_payload, output_payload, success
  Planning     : goal, steps[{index, description, depends_on}], contingencies
  Thinking     : prompt, inputs[{input_payload, input_record_id}], output_payload
  Acting       : action_type, target_system, action_summary, parameters, dry_run,
                 execution_status, execution_id
  Reflecting   : inputs[{input_payload, input_record_id}], output_payload
"""
from __future__ import annotations
import uuid
import time
import json
from pathlib import Path
from typing import Any
import httpx
import config

_BATCH_URL = f"{config.ARENA_BASE}/api/v1/arena/ledger/records/batch"
_VALIDATE_URL = f"{config.ARENA_BASE}/api/v1/arena/ledger/records/validate"
_SCHEMA_VERSION = "0.3"

# Local dump of every batch payload before POSTing (replay / forensics).
_PAYLOAD_DIR = Path(__file__).resolve().parent.parent / "storage" / "ledgers"


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
    """
    Build a ModelInvocation dict. tokens_in/tokens_out are OPTIONAL in the
    schema but, when present, must be numbers — null is rejected. So we only
    emit them when the provider actually returned a count.
    """
    mi: dict[str, Any] = {
        "provider":   provider,
        "model_name": model_name,
    }
    if isinstance(tokens_in, (int, float)):
        mi["tokens_in"] = tokens_in
    if isinstance(tokens_out, (int, float)):
        mi["tokens_out"] = tokens_out
    if internal_reasoning:
        mi["internal_reasoning"] = internal_reasoning
    return mi


# ── Session builder ────────────────────────────────────────────────────────

class LedgerSession:
    """
    Accumulates records for one decision cycle (fixture × window).
    Call .submit() at end of session.
    """

    def __init__(self, fixture_id: int, fixture_name: str, window: str,
                 api_key: str | None = None, agent_tag: str = ""):
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        tag = f"{agent_tag}:" if agent_tag else ""
        self.session_id = f"{window.lower()}:{fixture_id}:{tag}{ts}"
        self.fixture_id = fixture_id
        self.fixture_name = fixture_name
        self.window = window
        self.api_key = api_key or config.ARENA_KEY
        self.agent_tag = agent_tag
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
        provider: str = "",
        model_name: str = "",
        internal_reasoning: str = "",
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        upstream_ids: list[str] | None = None,
    ) -> dict:
        """Record an LLM reasoning step with chain-of-thought capture."""
        # ModelInvocation requires non-empty provider+model_name; omit otherwise
        # (e.g. a deterministic step logged as Thinking, or a failed LLM call).
        mi = (_build_mi(provider, model_name, internal_reasoning, tokens_in, tokens_out)
              if provider and model_name else None)
        wire_inputs = []
        for inp in inputs:
            item = {"input_payload": _trunc(inp.get("payload"))}
            if inp.get("record_id"):     # optional UUID; never send null
                item["input_record_id"] = inp["record_id"]
            wire_inputs.append(item)
        return self._add(_new_record(
            self.session_id, "Thinking",
            upstream_record_id=upstream_ids or ([self.last_id()] if self.last_id() else []),
            model_invocation=mi,
            prompt=_trunc(prompt_system, limit=16000),
            inputs=wire_inputs,
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
                "fixture_id":  str(self.fixture_id),
                "outcome":     outcome,
                "probability": prob,
            },
            dry_run=False,
            execution_status="confirmed",
        ))

    # ── Planning (goal decomposition — schema requires goal + steps) ────────

    def planning(
        self,
        goal: str,
        steps: list[str | dict],
        contingencies: list[str] | None = None,
        notes: str | None = None,
        upstream_ids: list[str] | None = None,
    ) -> dict:
        """
        Record a goal-decomposition step. Schema v0.3 Planning requires
        `goal` (string) + `steps` (PlanningStep[]); free-form payloads are NOT
        valid here — use thinking() for analysis output.

        `steps` items may be plain strings (auto-indexed) or full PlanningStep
        dicts {index, description, depends_on}.
        """
        step_objs = []
        for i, s in enumerate(steps):
            if isinstance(s, dict):
                step_objs.append({"index": s.get("index", i),
                                  "description": str(s.get("description", "")),
                                  **({"depends_on": s["depends_on"]} if s.get("depends_on") else {})})
            else:
                step_objs.append({"index": i, "description": str(s),
                                  **({"depends_on": [i - 1]} if i > 0 else {})})
        return self._add(_new_record(
            self.session_id, "Planning",
            upstream_record_id=upstream_ids or ([self.last_id()] if self.last_id() else []),
            goal=goal,
            steps=step_objs,
            contingencies=contingencies,
            notes=_trunc(notes, limit=2000) if notes else None,
        ))

    # ── Reflecting (post-decision retrospective — what we'd do differently) ──

    def reflecting(
        self,
        inputs: list[dict],
        output_payload: Any,
        provider: str = "",
        model_name: str = "",
        internal_reasoning: str = "",
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        upstream_ids: list[str] | None = None,
    ) -> dict:
        """
        Record a closing reflection over the whole session. Schema v0.3
        Reflecting requires `inputs` (ReflectingInput[]) + `output_payload`
        (string). Each input: {"record_id": ..., "payload": ...}.
        """
        fields: dict[str, Any] = {
            "upstream_record_id": upstream_ids or ([self.last_id()] if self.last_id() else []),
            "inputs": [
                {"input_record_id": inp.get("record_id"),
                 "input_payload": _trunc(inp.get("payload"))}
                for inp in (inputs or [{"payload": "(session trace)"}])
            ],
            "output_payload": _trunc(output_payload),
        }
        # Strip null input_record_id (schema: optional UUID, must not be null)
        for i in fields["inputs"]:
            if i.get("input_record_id") is None:
                i.pop("input_record_id", None)
        if model_name:
            fields["model_invocation"] = _build_mi(
                provider, model_name, internal_reasoning, tokens_in, tokens_out)
        return self._add(_new_record(self.session_id, "Reflecting", **fields))

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
        action_type: str = "open_order",
        upstream_ids: list[str] | None = None,
    ) -> dict:
        """Record placing a bet order."""
        if action_type == "close_order":
            summary = f"Close {outcome} order/position for fixture {self.fixture_id}"
        else:
            summary = f"Open {direction} ${size_usdc:.2f} on {outcome} @ ≤{limit_price}"
        return self._add(_new_record(
            self.session_id, "Acting",
            upstream_record_id=upstream_ids or ([self.last_id()] if self.last_id() else []),
            action_type=action_type,
            target_system="arena",
            action_summary=summary,
            parameters=order_payload,
            dry_run=False,
            execution_status=execution_status,
            execution_id=execution_id,
        ))

    # ── Submit ─────────────────────────────────────────────────────────────

    def validate(self) -> dict | None:
        """
        Dry-run the batch through /ledger/records/validate (same checks as
        submit, nothing persisted). Non-blocking: returns the response dict,
        or None if the endpoint is unreachable.
        """
        try:
            resp = httpx.post(
                _VALIDATE_URL,
                headers={"x-api-key": self.api_key, "Content-Type": "application/json"},
                json={"records": self._records},
                timeout=30,
            )
            if resp.status_code == 404:
                return None
            return resp.json()
        except Exception:
            return None

    def submit(self, validate_first: bool = True) -> dict:
        """
        Batch-submit all records for this session.

        Wire format: json={"records": [...], "fixture_id": "..."} — the
        top-level fixture_id binds the session to the fixture server-side so
        the prediction can be scored (20260610 release: batch session binding).

        Persists the payload locally before POSTing, pre-validates via the
        dry-run endpoint (non-blocking), then submits.
        Response: {"records": [...enriched], "errors": [...]}
        """
        payload = {"records": self._records, "fixture_id": str(self.fixture_id)}

        # Local dump for replay / forensics — written before any network call.
        try:
            _PAYLOAD_DIR.mkdir(parents=True, exist_ok=True)
            fname = self.session_id.replace(":", "_").replace("/", "_") + ".json"
            (_PAYLOAD_DIR / fname).write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8")
        except Exception:
            pass

        if validate_first:
            v = self.validate()
            if v and (v.get("errors") or not v.get("valid", True)):
                print(f"  [ledger] validate flagged {len(v.get('errors') or [])} issue(s): "
                      f"{json.dumps((v.get('errors') or [])[:3], default=str)[:400]}")

        resp = httpx.post(
            _BATCH_URL,
            headers={"x-api-key": self.api_key, "Content-Type": "application/json"},
            json=payload,
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
