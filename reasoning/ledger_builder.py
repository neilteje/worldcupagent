from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json, time, uuid
import httpx
from agent.config import Settings
from reasoning.schemas import LedgerRecord
from reasoning.trace_quality import evaluate_trace, model_invocation, reasoning_trace, summarize_reasoning


def _trunc(obj, limit: int = 30000) -> str:
    text = obj if isinstance(obj, str) else json.dumps(obj, sort_keys=True, default=str)
    return text if len(text) <= limit else text[:limit] + f"...[truncated, was {len(text)} chars]"


def _outcome_from_prediction(prediction: dict | None) -> tuple[str | None, float | None]:
    probs = (prediction or {}).get("probabilities") or {}
    if not probs:
        return None, None
    outcome = max(("home", "draw", "away"), key=lambda k: float(probs.get(k, 0.0) or 0.0))
    return outcome, float(probs.get(outcome, 0.0) or 0.0)

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
        probability = probability or {}
        risk = risk or {}
        prediction = prediction or {}
        order = order or {}
        reflection = reflection or {}
        obs = self.add(
            "Observing",
            "scheduled window trigger",
            {
                "trigger_source": "agent.scheduler",
                "trigger_type": "fixture_window",
                "trigger_description": f"{self.window} prediction cycle for {self.fixture_code}",
                "trigger_payload_summary": f"fixture_code={self.fixture_code}; window={self.window}; kickoff={kickoff_time}; lock={lock_time}",
                "fixture_code": self.fixture_code,
                "window": self.window,
                "kickoff_time": kickoff_time,
                "lock_time": lock_time,
                "reasoning_trace": reasoning_trace(
                    step_type="observation",
                    objective="Start a bounded prediction-market decision cycle for one fixture/window.",
                    inputs_used=["fixture_code", "window", "kickoff_time", "lock_time"],
                    method="Scheduler-triggered observation with explicit lock-window metadata.",
                    checks=["window normalized to PRE_MATCH or HT"],
                    output_summary=f"Opened session {self.session_id}.",
                    evidence_refs=["fixture_code", "lock_time"],
                    risk_controls=["one session per fixture/window"],
                ),
            },
        )
        plan = self.add(
            "Planning",
            "gather data, compute probabilities, detect edge, submit prediction/order",
            {
                "plan_steps": [
                    "fetch authoritative and market data",
                    "extract/cap weak LLM-derived claims",
                    "blend probabilities deterministically",
                    "calibrate and audit risk",
                    "emit prediction Acting record",
                    "emit order or skip Acting record",
                    "reflect on trace quality",
                ],
                "reasoning_trace": reasoning_trace(
                    step_type="plan",
                    objective="Produce a scoreable prediction and auditable order/skip decision.",
                    inputs_used=["Sportmonks", "Supabase", "Polymarket", "bookmaker", "lineup", "optional LLM feature extraction"],
                    method="Deterministic model composition with LLMs limited to feature extraction, analysis, and critique.",
                    assumptions=["Market data is a useful prior.", "Official lineup/Sportmonks data outranks weak text claims."],
                    uncertainties=["fixture data freshness", "lineup confirmation", "market liquidity"],
                    decision_rule="Never allow an LLM to be the final forecaster or authorize orders.",
                    evidence_refs=[obs],
                    risk_controls=["DRY_RUN gate", "duplicate protection", "blocking risk flags"],
                ),
            },
            parents=[obs],
        )
        sm = self.add("ToolCalling", "Sportmonks fixture/pre-match/live data", {"tool_meta": {"name": "sportmonks_proxy", "via": "arena"}, "description": "Fetch fixture detail, predictions, lineups, and live data when available.", "input_payload": _trunc({"fixture_code": self.fixture_code, "window": self.window}), "output_payload": sportmonks or {}, "success": bool(sportmonks), "reasoning_trace": reasoning_trace(step_type="tool_call", objective="Collect authoritative football data.", inputs_used=["fixture_id"], method="Arena Sportmonks proxy or synthetic fallback.", output_summary=f"Sportmonks fields: {list((sportmonks or {}).keys())}", evidence_refs=[plan], risk_controls=["missing source lowers confidence"])}, parents=[plan])
        sb = self.add("ToolCalling", "Supabase priors/live checkpoint", {"tool_meta": {"name": "supabase_world_cup_arena", "via": "supabase"}, "description": "Fetch historical priors and HT checkpoints.", "input_payload": _trunc({"fixture_code": self.fixture_code}), "output_payload": supabase or {}, "success": bool(supabase), "reasoning_trace": reasoning_trace(step_type="tool_call", objective="Collect historical priors/live checkpoints.", method="Supabase REST fallback-aware read.", output_summary=f"Supabase fields: {list((supabase or {}).keys())}", evidence_refs=[plan], risk_controls=["missing priors redistributed in blender"])}, parents=[plan])
        pm = self.add("ToolCalling", "Polymarket mapping/Gamma/CLOB mids", {"tool_meta": {"name": "polymarket_proxy", "via": "arena"}, "description": "Fetch market prices and previous snapshot.", "input_payload": _trunc({"fixture_code": self.fixture_code}), "output_payload": polymarket or {}, "success": bool((polymarket or {}).get("normalized_probs")), "reasoning_trace": reasoning_trace(step_type="tool_call", objective="Collect market-implied probabilities.", method="Arena Polymarket proxy or demo/synthetic fallback.", output_summary=f"Market complete={bool((polymarket or {}).get('complete'))}", evidence_refs=[plan], risk_controls=["orders blocked if market data missing"])}, parents=[plan])
        bk = self.add("ToolCalling", "Bookmaker odds parsing", {"tool_meta": {"name": "bookmaker_parser", "via": "sportmonks"}, "description": "Parse bookmaker probabilities from fixture detail.", "input_payload": _trunc({"sportmonks_record": sm}), "output_payload": bookmaker or {}, "success": bool(bookmaker), "reasoning_trace": reasoning_trace(step_type="tool_call", objective="Extract non-market reference odds.", method="Normalize bookmaker odds to 3-way probabilities.", output_summary=f"Bookmaker available={bool(bookmaker)}", evidence_refs=[sm], risk_controls=["source disagreement is audited"])}, parents=[sm])
        lu = self.add(
            "Thinking",
            "lineup delta",
            {
                "model_invocation": model_invocation(
                    provider="deterministic",
                    model_name="lineup_delta_v1",
                    internal_reasoning=summarize_reasoning(
                        label="Official lineup delta evaluation",
                        inputs={"lineup": lineup},
                        outputs={"delta": (lineup or {}).get("probability_delta"), "risk_flags": (lineup or {}).get("risk_flags")},
                        rule="Confirmed lineup shocks can move probabilities within hard caps; unconfirmed lineups reduce confidence.",
                    ),
                ),
                "prompt": "deterministic_lineup_delta(expected_starters, confirmed_starters, formations)",
                "inputs": [{"input_record_id": sm, "input_payload": _trunc(sportmonks or {})}],
                "output_payload": lineup or {},
                "reasoning_trace": reasoning_trace(
                    step_type="deterministic_reasoning",
                    objective="Convert lineup availability into capped probability deltas and risk flags.",
                    inputs_used=["expected_lineups", "confirmed_lineups", "formations"],
                    method="Compare expected vs confirmed starters and goalkeeper/formation changes.",
                    uncertainties=["lineup_unconfirmed" if "lineup_unconfirmed" in (lineup or {}).get("risk_flags", []) else "none"],
                    decision_rule="Official lineup data overrides web claims; unconfirmed lineup cannot support lineup-driven orders.",
                    output_summary=str((lineup or {}).get("reason", "")),
                    evidence_refs=[sm],
                    risk_controls=(lineup or {}).get("risk_flags", []),
                ),
            },
            parents=[sm],
        )
        ht = None
        if self.window.upper() == "HT":
            ht = self.add("Thinking", "halftime scoreline luck model", {"model_invocation": model_invocation(provider="deterministic", model_name="halftime_scoreline_luck_v1", internal_reasoning=summarize_reasoning(label="HT scoreline/xG update", inputs={"halftime": halftime}, outputs={"ht_probs": (halftime or {}).get("ht_probs"), "confidence": (halftime or {}).get("confidence")}, rule="Use live score, xG, shot pressure, and cards to update prematch probabilities.")), "prompt": "deterministic_halftime_update(score, xg, shots, cards)", "inputs": [{"input_record_id": sm, "input_payload": _trunc(sportmonks or {})}, {"input_record_id": sb, "input_payload": _trunc(supabase or {})}, {"input_record_id": pm, "input_payload": _trunc(polymarket or {})}], "output_payload": halftime or {}, "reasoning_trace": reasoning_trace(step_type="deterministic_reasoning", objective="Update probabilities from halftime state.", inputs_used=["score", "xG", "shots", "cards"], method="Scoreline luck and performance-signal model.", output_summary=str((halftime or {}).get("reason", "")), evidence_refs=[sm, sb, pm], risk_controls=["confidence cap", "uncertainty floor"])}, parents=[sm, sb, pm])
        claim_record = None
        claims = (probability or {}).get("claim_extraction") or {}
        if claims.get("called") or claims.get("claims") or claims.get("dropped_claims"):
            usage = claims.get("usage") or {}
            claim_record = self.add("Thinking", "structured LLM claim extraction", {"model_invocation": model_invocation(provider=claims.get("provider", "anthropic" if claims.get("called") else "deterministic"), model_name=claims.get("model", "prestructured_claim_validator"), internal_reasoning=summarize_reasoning(label="Structured claim extraction", inputs={"claim_count": len(claims.get("claims", [])), "dropped": len(claims.get("dropped_claims", []))}, outputs={"claims": claims.get("claims", [])[:4], "risk_flags": claims.get("risk_flags", [])}, rule="LLM may extract typed claims only; JSON validation, dedupe, delta caps, and official overrides are deterministic."), tokens_in=usage.get("input_tokens"), tokens_out=usage.get("output_tokens")), "prompt": "extract typed claims: injury/suspension/lineup/weather/tactical_change/motivation/market_stale/contradiction; strict JSON", "inputs": [{"input_record_id": plan, "input_payload": "text sources if configured"}], "output_payload": claims, "reasoning_trace": reasoning_trace(step_type="llm_feature_extraction", objective="Extract typed non-forecast claims from text sources.", method="Anthropic structured JSON extraction followed by local validation and caps.", decision_rule="Claims become scored signals only; they do not directly forecast or authorize orders.", output_summary=f"claims={len(claims.get('claims', []))}; dropped={len(claims.get('dropped_claims', []))}", evidence_refs=[plan, lu], risk_controls=["strict JSON validation", "delta cap", "official source override"])}, parents=[plan, lu])
        prob_parents = [sm, sb, lu] + ([ht] if ht else [])
        if claim_record:
            prob_parents.append(claim_record)
        pr = self.add("Thinking", "deterministic probability blender and calibration", {"model_invocation": model_invocation(provider="deterministic", model_name="probability_blender_calibration_v1", internal_reasoning=summarize_reasoning(label="Probability blend and calibration", inputs={"weights": probability.get("weights"), "source_contribution": probability.get("source_contribution")}, outputs={"probabilities": probability.get("probabilities"), "confidence": probability.get("confidence"), "uncertainty": probability.get("uncertainty"), "risk_flags": probability.get("risk_flags")}, rule="Blend only validated sources, redistribute missing weights, apply calibration and draw floor, then expose source contribution.")), "prompt": "deterministic_blend(sportmonks, bookmaker, polymarket, supabase, lineup, ht, draw_model, structured_claims)", "inputs": [{"input_record_id": pid, "input_payload": "see upstream record"} for pid in prob_parents], "output_payload": probability or {}, "reasoning_trace": reasoning_trace(step_type="deterministic_reasoning", objective="Produce calibrated 3-way probabilities.", inputs_used=["Sportmonks", "bookmaker", "Polymarket", "Supabase", "lineup", "HT", "draw_model", "structured_claims"], method="Weighted deterministic blender with calibration, shrinkage, and contribution accounting.", assumptions=["Source weights are redistributed when missing.", "Weak LLM claims are capped before blending."], uncertainties=probability.get("missing_sources", []), decision_rule="Final forecast must be valid normalized probabilities; no LLM final forecaster.", output_summary=f"final_probs={probability.get('probabilities')}; confidence={probability.get('confidence')}; uncertainty={probability.get('uncertainty')}", evidence_refs=prob_parents, risk_controls=probability.get("risk_flags", []))}, parents=prob_parents)
        co = self.add("Thinking", "consensus triangle", {"model_invocation": model_invocation(provider="deterministic", model_name="consensus_triangle_v1", internal_reasoning=summarize_reasoning(label="Consensus comparison", inputs={"model": probability.get("probabilities"), "bookmaker": bookmaker, "market": (polymarket or {}).get("normalized_probs")}, outputs=consensus, rule="Compare top picks and agreement cases; do not override probability model.")), "prompt": "consensus_triangle(model_probs, bookmaker_probs, market_probs)", "inputs": [{"input_record_id": pr, "input_payload": "model probabilities"}, {"input_record_id": bk, "input_payload": "bookmaker"}, {"input_record_id": pm, "input_payload": "market"}], "output_payload": consensus or {}, "reasoning_trace": reasoning_trace(step_type="deterministic_reasoning", objective="Identify source agreement/disagreement.", method="Compare model, bookmaker, and market top picks.", decision_rule="Disagreement modifies confidence/risk; it does not authorize orders.", output_summary=f"consensus={(consensus or {}).get('case')}", evidence_refs=[pr, bk, pm], risk_controls=["confidence modifier", "bet size modifier"])}, parents=[pr, bk, pm])
        ed = self.add("Thinking", "edge engine", {"model_invocation": model_invocation(provider="deterministic", model_name="edge_engine_v1", internal_reasoning=summarize_reasoning(label="Edge evaluation", inputs={"model": probability.get("probabilities"), "market": (polymarket or {}).get("normalized_probs"), "confidence": probability.get("confidence")}, outputs=edge, rule="Edge must exceed tier thresholds and survive confidence/uncertainty checks.")), "prompt": "evaluate_edge(model_probs, market_probs, bookmaker_probs, confidence, uncertainty, consensus)", "inputs": [{"input_record_id": co, "input_payload": "consensus"}, {"input_record_id": pr, "input_payload": "model"}, {"input_record_id": pm, "input_payload": "market"}], "output_payload": edge or {}, "reasoning_trace": reasoning_trace(step_type="deterministic_reasoning", objective="Compute trade edge and tier.", method="Compare calibrated model probability with market probability per outcome.", decision_rule="Only medium/strong or high-confidence supported soft edges may proceed to risk audit.", output_summary=str((edge or {}).get("reason", "")), evidence_refs=[co, pr, pm], risk_controls=["edge threshold", "confidence threshold", "uncertainty threshold"])}, parents=[co, pr, pm])
        risk_parents = [ed, lu] + ([ht] if ht else [])
        llm_analysis = reflection.get("llm_analysis")
        analyst_record = None
        if llm_analysis:
            parsed = llm_analysis.get("parsed") or {}
            usage = llm_analysis.get("usage") or {}
            analyst_reasoning = llm_analysis.get("internal_reasoning") or summarize_reasoning(label="LLM analyst critique of deterministic signals", inputs={"edge": edge, "risk": risk}, outputs={"recommendation": parsed.get("recommendation"), "risk_posture": parsed.get("risk_posture"), "additional_risk_flags": parsed.get("additional_risk_flags")}, rule="LLM analyst may veto or add caution but cannot approve or place orders.")
            analyst_record = self.add("Thinking", "LLM signal analyst", {"model_invocation": model_invocation(provider=llm_analysis.get("provider", "anthropic"), model_name=llm_analysis.get("model", "unknown"), internal_reasoning=analyst_reasoning, tokens_in=usage.get("input_tokens"), tokens_out=usage.get("output_tokens")), "prompt": "Analyze deterministic signals. Return JSON. No order authority.", "inputs": [{"input_record_id": ed, "input_payload": "edge"}, {"input_record_id": pr, "input_payload": "probability"}], "output_payload": llm_analysis, "reasoning_trace": reasoning_trace(step_type="llm_critic", objective="Second-pass critique of deterministic signal bundle.", method="Structured Anthropic JSON analysis.", decision_rule="LLM can add blocking caution/veto; deterministic gates remain authoritative.", output_summary=f"recommendation={parsed.get('recommendation')}; posture={parsed.get('risk_posture')}", evidence_refs=[ed, pr], risk_controls=["order_authorization_allowed=false"])}, parents=[ed, pr])
            risk_parents.append(analyst_record)
        ra = self.add("Thinking", "risk audit / sanity checks", {"model_invocation": model_invocation(provider="deterministic", model_name="risk_audit_v1", internal_reasoning=summarize_reasoning(label="Risk audit", inputs={"edge": edge, "dry_run": self.settings.dry_run}, outputs=risk, rule="Orders are allowed only when no blocking flags remain.")), "prompt": "audit_decision(probabilities, edge, confidence, uncertainty, dry_run, market_complete, lineup, consensus)", "inputs": [{"input_record_id": pid, "input_payload": "see upstream risk input"} for pid in risk_parents], "output_payload": risk or {}, "reasoning_trace": reasoning_trace(step_type="deterministic_reasoning", objective="Apply hard safety and validity gates.", method="Sanity checks, duplicate guard, dry-run guard, confidence/uncertainty guard.", decision_rule="order_allowed iff blocking_risk_flags is empty.", output_summary=f"order_allowed={risk.get('order_allowed')}; blocking={risk.get('blocking_risk_flags')}", evidence_refs=risk_parents, risk_controls=risk.get("blocking_risk_flags", []))}, parents=risk_parents)
        outcome, probability_value = _outcome_from_prediction(prediction)
        prediction_params = {**prediction, "outcome": outcome, "probability": probability_value}
        pred = self.add("Acting", "prediction", {"action_type": "prediction", "target_system": "arena", "action_summary": f"Submit latest {self.window} probability distribution for scoring.", "parameters": prediction_params, "dry_run": self.settings.dry_run, "execution_status": "dry_run" if self.settings.dry_run else "ready", "reasoning_trace": reasoning_trace(step_type="action", objective="Emit prediction record for PSL scoring.", inputs_used=["final_probs", "confidence", "risk_flags"], method="Acting record with full distribution plus top outcome/probability.", decision_rule="Always submit prediction even when order is skipped.", output_summary=f"outcome={outcome}; probability={probability_value}", evidence_refs=[ra], risk_controls=["probabilities normalized", "prediction separate from order"])}, parents=[ra])
        order_action = (order or {}).get("action_type", "skip")
        od = self.add("Acting", "order or skip", {"action_type": order_action, "target_system": "arena", "action_summary": "Place order only if deterministic gates pass; otherwise record skip.", "parameters": order or {}, "dry_run": self.settings.dry_run, "execution_status": "dry_run" if self.settings.dry_run else ("pending" if order_action == "order" else "skipped"), "reasoning_trace": reasoning_trace(step_type="action", objective="Record order attempt or explicit skip.", inputs_used=["edge", "risk", "bet_size", "limit_price"], method="Deterministic action gate.", decision_rule="No order unless edge engine, risk audit, duplicate protection, confidence, and DRY_RUN checks all pass.", output_summary=f"action_type={order_action}; reason={order.get('reason')}", evidence_refs=[pred, ed, ra], risk_controls=risk.get("blocking_risk_flags", []))}, parents=[pred, ed, ra])
        reflection_payload = {"output_payload": reflection or {}, "reasoning_trace": reasoning_trace(step_type="reflection", objective="Explain final decision and audit trace completeness.", inputs_used=["prediction", "order", "risk", "source_contribution"], method="Local self-audit of ledger structure and decision rationale.", decision_rule="Surface weaknesses without mutating aggressive config automatically.", output_summary=f"decision={reflection.get('decision')}; data_complete={reflection.get('data_complete')}", evidence_refs=[pred, od], risk_controls=["post-run review", "trace quality scoring"])}
        provisional = [r.to_wire() for r in self.records]
        reflection_payload["trace_quality"] = evaluate_trace(provisional)
        self.add("Reflecting", "run reflection and trace-quality audit", reflection_payload, parents=[pred, od])
        self.records[-1].payload["trace_quality"] = evaluate_trace([r.to_wire() for r in self.records])
        return [r.to_wire() for r in self.records]

    def validate_dag(self) -> bool:
        seen = set()
        for r in self.records:
            if any(pid not in seen for pid in r.parent_ids):
                return False
            seen.add(r.record_id)
        return True

    def trace_quality(self) -> dict:
        return evaluate_trace([r.to_wire() for r in self.records])

class LedgerAdapter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.urls = [
            f"{settings.arena_base}/api/v1/arena/ledger/records/batch",
            f"{settings.arena_base}/v1/arena/ledger/records",
        ]

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
        for url in self.urls:
            for attempt in range(retries + 1):
                try:
                    r = httpx.post(url, headers=self.settings.headers, json={"records": records}, timeout=30)
                    if r.status_code == 404:
                        last = f"HTTP 404 at {url}"
                        break
                    if r.is_success:
                        return {"submitted": True, "response": r.json(), "records_built": len(records), "endpoint": url}
                    last = f"HTTP {r.status_code}: {r.text[:200]}"
                except Exception as exc:
                    last = repr(exc)
                time.sleep(0.25 * (2 ** attempt))
        self.save_local(session_id, records, ".failed")
        return {"submitted": False, "reason": last, "records_built": len(records)}
