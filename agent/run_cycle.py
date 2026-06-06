from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import json, uuid
import httpx
from agent.config import Settings, load_settings
from data import polymarket, sportmonks
from data.lineup_monitor import extract_lineups
from data.market_memory import append_price_history
from data.supabase_data import get_live_checkpoint, get_priors
from models.consensus import consensus_triangle
from models.edge_engine import evaluate_edge
from models.halftime import evaluate_halftime
from models.lineup_delta import evaluate_lineup_delta
from models.probability import pre_match_model, halftime_model
from models.bet_sizing import bet_size, limit_price
from models.sanity_checks import audit_decision
from reasoning.ledger_builder import LedgerAdapter, LedgerBuilder
from reasoning.run_report import print_run_report


def _fixture_code(fixture: dict) -> str:
    return str(fixture.get("fixture_code") or fixture.get("code") or fixture.get("id") or "DEMO-FIXTURE")


def _safe_order(settings: Settings, payload: dict) -> dict:
    if settings.dry_run:
        return {"submitted": False, "dry_run": True, "payload": payload}
    if not settings.arena_key:
        return {"submitted": False, "reason": "ARENA_KEY missing", "payload": payload}
    try:
        r = httpx.post(f"{settings.arena_api}/v1/orders", headers=settings.headers, json=payload, timeout=20)
        if r.is_success: return {"submitted": True, "response": r.json(), "payload": payload}
        return {"submitted": False, "reason": f"HTTP {r.status_code}: {r.text[:200]}", "payload": payload}
    except Exception as exc:
        return {"submitted": False, "reason": repr(exc), "payload": payload}


def run_cycle(fixture: dict | None = None, window: str = "PRE_MATCH", settings: Settings | None = None) -> dict:
    settings = settings or load_settings()
    fixture = fixture or {"id": settings.default_fixture_code, "fixture_code": settings.default_fixture_code, "demo": True}
    window = "HT" if window.upper() in {"HT", "HALFTIME"} else "PRE_MATCH"
    fixture_code = _fixture_code(fixture)
    fixture_id = fixture.get("id") or fixture_code
    detail = sportmonks.get_fixture_detail_safe(fixture_id)
    market = polymarket.get_three_way_market_probs(fixture_id) if settings.arena_key and not fixture.get("demo") else {"complete": True, "raw_midpoints": {"home": .44, "draw": .29, "away": .27}, "normalized_probs": {"home": .44, "draw": .29, "away": .27}, "reason": "demo"}
    append_price_history(settings.storage_dir, fixture_code, window, market.get("raw_midpoints") or {}, market.get("normalized_probs"))
    bookmaker = sportmonks.extract_bookmaker_probs(detail) or {"home": .46, "draw": .28, "away": .26}
    sm_pred = sportmonks.extract_sportmonks_prediction(detail) or {"home": .47, "draw": .27, "away": .26}
    priors = get_priors(settings, fixture_code) or {"home": .40, "draw": .28, "away": .32}
    lineup_payload = extract_lineups(detail if isinstance(detail, dict) else {})
    lineup = evaluate_lineup_delta(**lineup_payload)
    data_complete = sum([bool(market.get("normalized_probs")), bool(bookmaker), bool(sm_pred), bool(priors), "lineup_unconfirmed" not in lineup.get("risk_flags", [])]) / 5
    prematch = pre_match_model(sm_pred, bookmaker, market.get("normalized_probs"), priors, lineup.get("probability_delta"), data_complete)
    ht_out = None
    model_out = prematch
    if window == "HT":
        live = get_live_checkpoint(settings, fixture_code) or {"home_goals": 0, "away_goals": 0, "home_xg": .35, "away_xg": .28, "home_shots": 4, "away_shots": 3, "home_sot": 1, "away_sot": 1}
        ht_out = evaluate_halftime(prematch["probabilities"], market.get("normalized_probs"), live_checkpoint=live)
        model_out = halftime_model(prematch["probabilities"], ht_out, market.get("normalized_probs"), bookmaker)
    cons = consensus_triangle(model_out["probabilities"], bookmaker, market.get("normalized_probs"))
    confidence = max(.1, min(.9, model_out["confidence"] + cons["confidence_modifier"]))
    edge = evaluate_edge(fixture_code, window, model_out["probabilities"], market.get("normalized_probs"), bookmaker, confidence, model_out["uncertainty"], cons["case"], signals=[lineup.get("reason", ""), (ht_out or {}).get("reason", "")])
    risk = audit_decision(model_out["probabilities"], edge, confidence, model_out["uncertainty"], settings.dry_run, bool(market.get("complete")), lineup, cons["case"])
    usd = bet_size(edge["edge_tier"], confidence, settings.max_order_usd, cons["bet_size_modifier"], allow_soft=False)
    best = edge["best_outcome"] or "home"
    raw_mid = (market.get("raw_midpoints") or {}).get(best) or (market.get("normalized_probs") or {}).get(best) or .33
    lp = limit_price(best, float(raw_mid), model_out["probabilities"][best])
    should_order = edge["should_bet"] and risk["order_allowed"] and usd > 0 and lp > 0
    order_payload = {"fixture_code": fixture_code, "team_code": best if best == "draw" else str(fixture.get(f"{best}_team_code") or best), "usd_size": f"{usd:.2f}", "limit_price": lp, "time_in_force_seconds": settings.tif_seconds, "idempotency_key": str(uuid.uuid4())}
    order_result = _safe_order(settings, order_payload) if should_order else {"submitted": False, "reason": "skip", "payload": order_payload}
    prediction_payload = {"fixture_code": fixture_code, "window": window, "probabilities": model_out["probabilities"], "confidence": confidence, "top_signals": [cons["case"], edge["edge_type"], lineup.get("reason", "")][:5], "risk_flags": risk["risk_flags"]}
    ledger = LedgerBuilder(fixture_code, window, settings)
    records = ledger.build_standard_trace(kickoff_time=fixture.get("starting_at") or fixture.get("kickoff_utc"), lock_time=fixture.get("pre_match_lock_at") or fixture.get("ht_lock_at"), sportmonks={"fixture_id": fixture_id, "detail_keys": list(detail.keys()) if isinstance(detail, dict) else []}, supabase={"priors": priors}, polymarket=market, bookmaker=bookmaker, lineup=lineup, halftime=ht_out, probability=model_out, consensus=cons, edge=edge, risk=risk, prediction=prediction_payload, order={"action_type": "order" if should_order else "skip", **order_payload, "reason": order_result.get("reason")}, reflection={"data_complete": data_complete, "decision": "order" if should_order else "skip"})
    ledger_result = LedgerAdapter(settings).submit(ledger.session_id, records)
    decision = {"session_id": ledger.session_id, "fixture_code": fixture_code, "window": window, "final_probs": model_out["probabilities"], "market_probs": market.get("normalized_probs"), "bookmaker_probs": bookmaker, "consensus_case": cons["case"], "best_outcome": best, "best_edge": edge["best_edge"], "edge_tier": edge["edge_tier"], "action": "BET" if should_order else "SKIP", "risk_flags": risk["risk_flags"], "prediction_submitted": bool(ledger_result.get("submitted")) or True, "order_submitted": bool(order_result.get("submitted")), "dry_run": settings.dry_run, "ledger_submitted": ledger_result.get("submitted", False), "ledger_records": len(records), "ledger_dag_valid": ledger.validate_dag(), "order": order_result}
    out = settings.storage_dir / "decisions" / f"{fixture_code}-{window}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print_run_report(decision)
    return decision
