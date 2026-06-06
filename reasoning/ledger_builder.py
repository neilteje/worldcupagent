from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json, time, uuid
import httpx
from agent.config import Settings
from reasoning.schemas import LedgerRecord

class LedgerBuilder:
    def __init__(self, fixture_code: str, window: str, settings: Settings):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self.session_id = f"{fixture_code}-{window}-{stamp}-{uuid.uuid4().hex[:8]}"
        self.fixture_code, self.window, self.settings = fixture_code, window, settings
        self.records: list[LedgerRecord] = []

    def add(self, behavior: str, label: str, payload: dict | None = None, parents: list[str] | None = None, **extra) -> str:
        body = {"label": label, **(payload or {}), **extra}
        rec = LedgerRecord(self.session_id, behavior, parent_ids=parents or [], payload=body)
        self.records.append(rec)
        return rec.record_id

    def build_standard_trace(self, *, kickoff_time=None, lock_time=None, sportmonks=None, supabase=None, polymarket=None, bookmaker=None, lineup=None, halftime=None, probability=None, consensus=None, edge=None, risk=None, prediction=None, order=None, reflection=None) -> list[dict]:
        obs = self.add("Observing", "scheduled window trigger", {"fixture_code": self.fixture_code, "window": self.window, "kickoff_time": kickoff_time, "lock_time": lock_time})
        plan = self.add("Planning", "gather data, compute probabilities, detect edge, submit prediction/order", parents=[obs])
        sm = self.add("ToolCalling", "Sportmonks fixture/pre-match/live data", {"output_payload": sportmonks or {}}, parents=[plan])
        sb = self.add("ToolCalling", "Supabase priors/live checkpoint", {"output_payload": supabase or {}}, parents=[plan])
        pm = self.add("ToolCalling", "Polymarket mapping/Gamma/CLOB mids", {"output_payload": polymarket or {}}, parents=[plan])
        bk = self.add("ToolCalling", "Bookmaker odds parsing", {"output_payload": bookmaker or {}}, parents=[sm])
        lu = self.add("Thinking", "lineup delta", {"output_payload": lineup or {}}, parents=[sm])
        ht = None
        if self.window.upper() == "HT":
            ht = self.add("Thinking", "halftime scoreline luck model", {"output_payload": halftime or {}}, parents=[sm, sb, pm])
        prob_parents = [sm, sb, lu] + ([ht] if ht else [])
        pr = self.add("Thinking", "base probability model", {"output_payload": probability or {}}, parents=prob_parents)
        co = self.add("Thinking", "consensus triangle", {"output_payload": consensus or {}}, parents=[pr, bk, pm])
        ed = self.add("Thinking", "edge engine", {"output_payload": edge or {}}, parents=[co, pr, pm])
        risk_parents = [ed, lu] + ([ht] if ht else [])
        ra = self.add("Thinking", "risk audit / sanity checks", {"output_payload": risk or {}}, parents=risk_parents)
        pred = self.add("Acting", "prediction", {"action_type": "prediction", "target_system": "arena", "parameters": prediction or {}}, parents=[ra])
        od = self.add("Acting", "order or skip", {"action_type": (order or {}).get("action_type", "skip"), "target_system": "arena", "parameters": order or {}}, parents=[pred, ed])
        self.add("Reflecting", "run reflection", {"output_payload": reflection or {}}, parents=[pred, od])
        return [r.to_wire() for r in self.records]

    def validate_dag(self) -> bool:
        seen = set()
        for r in self.records:
            if any(pid not in seen for pid in r.parent_ids):
                return False
            seen.add(r.record_id)
        return True

class LedgerAdapter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.url = f"{settings.arena_base}/v1/arena/ledger/records"

    def save_local(self, session_id: str, records: list[dict], suffix: str = "") -> Path:
        path = self.settings.storage_dir / "runs" / f"{session_id}{suffix}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"session_id": session_id, "records": records}, indent=2), encoding="utf-8")
        return path

    def submit(self, session_id: str, records: list[dict], retries: int = 2) -> dict:
        self.save_local(session_id, records)
        if self.settings.dry_run:
            return {"submitted": False, "reason": "dry_run_local_only", "records_built": len(records)}
        if not self.settings.arena_key:
            return {"submitted": False, "reason": "ARENA_KEY missing", "records_built": len(records)}
        last = None
        for attempt in range(retries + 1):
            try:
                r = httpx.post(self.url, headers=self.settings.headers, json={"records": records}, timeout=30)
                if r.status_code == 404:
                    return {"submitted": False, "reason": "ledger endpoint unavailable", "records_built": len(records)}
                if r.is_success:
                    return {"submitted": True, "response": r.json(), "records_built": len(records)}
                last = f"HTTP {r.status_code}: {r.text[:200]}"
            except Exception as exc:
                last = repr(exc)
            time.sleep(0.25 * (2 ** attempt))
        self.save_local(session_id, records, ".failed")
        return {"submitted": False, "reason": last, "records_built": len(records)}
