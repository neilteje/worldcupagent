from __future__ import annotations
from pathlib import Path
import json, uuid
import httpx
from agent.config import Settings, load_settings
from data import polymarket, sportmonks
from data.lineup_monitor import extract_lineups
from data.market_memory import append_price_history, previous_normalized_probs
from data.supabase_data import get_live_checkpoint, get_priors
from models.consensus import consensus_triangle
from models.edge_engine import evaluate_edge
from models.halftime import evaluate_halftime
from models.lineup_delta import evaluate_lineup_delta
from models.llm_central import normalize_central_prediction
from models.llm_decision import merge_llm_analysis_into_risk
from models.probability import pre_match_model, halftime_model
from models.bet_sizing import bet_size, limit_price
from models.sanity_checks import audit_decision
from models.signal_scoring import score_signal, summarize_signals, signal_conflict_score
from models.draw_model import apply_draw_model, draw_sanity_flags
from models.market_stale import detect_market_stale
from models.source_reconciliation import reconcile_sources
from reasoning.anthropic_review import analyze_decision_signals_with_anthropic
from reasoning.central_llm import central_match_forecast_with_anthropic
from reasoning.claim_extraction import apply_official_overrides, claims_to_signals, extract_claims_with_anthropic, validate_claim_json
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
        if r.is_success:
            return {"submitted": True, "response": r.json(), "payload": payload}
        return {"submitted": False, "reason": f"HTTP {r.status_code}: {r.text[:200]}", "payload": payload}
    except Exception as exc:
        return {"submitted": False, "reason": repr(exc), "payload": payload}


def _duplicate_order_marker(settings: Settings, fixture_code: str, window: str) -> bool:
    path = settings.storage_dir / "decisions" / f"{fixture_code}-{window}.json"
    if not path.exists():
        return False
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(previous.get("order_submitted") or previous.get("action") == "BET")


def _synthetic_lineup_payload(kind: str | None) -> dict | None:
    if kind != "home_gk_missing":
        return None
    expected_home = [{"id": 1, "name": "First GK", "position": "Goalkeeper", "starter": True}, {"id": 2, "name": "Home ST", "position": "Striker", "starter": True}]
    confirmed_home = [{"id": 99, "name": "Backup GK", "position": "Goalkeeper", "starter": True}, {"id": 2, "name": "Home ST", "position": "Striker", "starter": True}]
    expected_away = [{"id": 11, "name": "Away GK", "position": "Goalkeeper", "starter": True}, {"id": 12, "name": "Away ST", "position": "Striker", "starter": True}]
    return {"expected_home": expected_home, "confirmed_home": confirmed_home, "expected_away": expected_away, "confirmed_away": expected_away, "expected_formations": {"home": "4-3-3", "away": "4-3-3"}, "confirmed_formations": {"home": "4-3-3", "away": "4-3-3"}}


def _source_completeness(*, market: dict, bookmaker: dict | None, sm_pred: dict | None, priors: dict | None, lineup: dict, window: str, live: dict | None) -> dict:
    checks = {
        "market": bool(market.get("normalized_probs")),
        "bookmaker": bool(bookmaker),
        "sportmonks_prediction": bool(sm_pred),
        "supabase_priors": bool(priors),
        "lineup_confirmed": "lineup_unconfirmed" not in lineup.get("risk_flags", []),
    }
    if window == "HT":
        checks.update({"ht_score": live is not None and "home_goals" in live and "away_goals" in live, "ht_xg": live is not None and "home_xg" in live and "away_xg" in live})
    score = sum(1 for v in checks.values() if v) / max(len(checks), 1)
    return {"score": score, "checks": checks, "missing": [k for k, v in checks.items() if not v]}


def _extract_structured_claims(settings: Settings, fixture: dict, synthetic: dict, lineup: dict, *, use_llm_claims: bool) -> dict:
    raw_structured = synthetic.get("structured_claims") or fixture.get("structured_claims")
    source_texts = synthetic.get("source_texts") or fixture.get("source_texts") or []
    result = {"called": False, "ok": True, "claims": [], "dropped_claims": [], "signals": [], "risk_flags": []}
    if raw_structured:
        result.update(validate_claim_json({"claims": raw_structured}, delta_cap=settings.llm_signal_delta_cap))
        result["called"] = False
        result["source"] = "prestructured"
    elif use_llm_claims:
        if not source_texts:
            result.update({"ok": True, "reason": "no_text_sources_for_claim_extraction"})
        else:
            result.update(extract_claims_with_anthropic(settings, source_texts, delta_cap=settings.llm_signal_delta_cap))

    override = apply_official_overrides(result.get("claims") or [], lineup_result=lineup)
    result["claims"] = override["claims"]
    result["dropped_claims"] = override["dropped"]
    result["signals"] = claims_to_signals(result["claims"], delta_cap=settings.llm_signal_delta_cap)
    if result.get("errors"):
        result["risk_flags"].append("llm_claim_validation_error")
    if result["dropped_claims"]:
        result["risk_flags"].append("web_claims_overridden_by_official_data")
    return result


def run_cycle(fixture: dict | None = None, window: str = "PRE_MATCH", settings: Settings | None = None, verbose: bool = False, *, use_llm_analyst: bool = False, use_llm_claims: bool = False, decision_mode: str | None = None) -> dict:
    settings = settings or load_settings()
    decision_mode = (decision_mode or settings.decision_mode or "deterministic").lower()
    fixture = fixture or {"id": settings.default_fixture_code, "fixture_code": settings.default_fixture_code, "demo": True}
    window = "HT" if (window or fixture.get("preferred_window", "PRE_MATCH")).upper() in {"HT", "HALFTIME"} else "PRE_MATCH"
    fixture_code = _fixture_code(fixture)
    fixture_id = fixture.get("id") or fixture_code
    synthetic = fixture.get("synthetic_data") or {}

    detail = synthetic.get("detail") or sportmonks.get_fixture_detail_safe(fixture_id)
    market = {"complete": True, "raw_midpoints": synthetic.get("market"), "normalized_probs": synthetic.get("market"), "reason": "synthetic"} if "market" in synthetic else None
    if market is None:
        market = polymarket.get_three_way_market_probs(fixture_id) if settings.arena_key and not fixture.get("demo") else {"complete": True, "raw_midpoints": {"home": .44, "draw": .29, "away": .27}, "normalized_probs": {"home": .44, "draw": .29, "away": .27}, "reason": "demo"}
    if synthetic.get("market") is None and "market" in synthetic:
        market = {"complete": False, "raw_midpoints": {}, "normalized_probs": None, "reason": "synthetic missing market"}
    append_price_history(settings.storage_dir, fixture_code, window, market.get("raw_midpoints") or {}, market.get("normalized_probs"))

    bookmaker = synthetic.get("bookmaker") if "bookmaker" in synthetic else sportmonks.extract_bookmaker_probs(detail)
    bookmaker = bookmaker or ({"home": .46, "draw": .28, "away": .26} if fixture.get("demo") else None)
    sm_pred = synthetic.get("sportmonks") if "sportmonks" in synthetic else sportmonks.extract_sportmonks_prediction(detail)
    sm_pred = sm_pred or ({"home": .47, "draw": .27, "away": .26} if fixture.get("demo") else None)
    priors = synthetic.get("priors") or get_priors(settings, fixture_code) or ({"home": .40, "draw": .28, "away": .32} if fixture.get("demo") else None)

    lineup_payload = _synthetic_lineup_payload(synthetic.get("lineups")) or extract_lineups(detail if isinstance(detail, dict) else {})
    lineup = evaluate_lineup_delta(**lineup_payload)
    live = None
    if window == "HT":
        live = synthetic.get("live") or get_live_checkpoint(settings, fixture_code) or ({"home_goals": 0, "away_goals": 0, "home_xg": .35, "away_xg": .28, "home_shots": 4, "away_shots": 3, "home_sot": 1, "away_sot": 1} if fixture.get("demo") else None)

    claim_extraction = _extract_structured_claims(settings, fixture, synthetic, lineup, use_llm_claims=use_llm_claims)
    claim_signals = claim_extraction.get("signals") or []
    completeness = _source_completeness(market=market, bookmaker=bookmaker, sm_pred=sm_pred, priors=priors, lineup=lineup, window=window, live=live)
    prematch = pre_match_model(sm_pred, bookmaker, market.get("normalized_probs"), priors, lineup.get("probability_delta"), completeness["score"], structured_signals=claim_signals)
    draw_inputs = {
        "total_projected_xg": synthetic.get("projected_xg"),
        "strength_gap": abs((prematch["probabilities"].get("home", 0) - prematch["probabilities"].get("away", 0))),
        "market_draw": (market.get("normalized_probs") or {}).get("draw") if market else None,
        "bookmaker_draw": (bookmaker or {}).get("draw"),
    }
    draw_out = apply_draw_model(prematch["probabilities"], **draw_inputs)
    prematch["probabilities"] = draw_out["probabilities"]
    ht_out = None
    model_out = prematch
    if window == "HT":
        ht_out = evaluate_halftime(prematch["probabilities"], market.get("normalized_probs"), live_checkpoint=live or {})
        model_out = halftime_model(prematch["probabilities"], ht_out, market.get("normalized_probs"), bookmaker, structured_signals=claim_signals)
        draw_ht = apply_draw_model(model_out["probabilities"], market_draw=(market.get("normalized_probs") or {}).get("draw"), bookmaker_draw=(bookmaker or {}).get("draw"), ht_score=(int((live or {}).get("home_goals", 0)), int((live or {}).get("away_goals", 0))), ht_total_xg=float((live or {}).get("home_xg", 0) or 0) + float((live or {}).get("away_xg", 0) or 0), red_cards=(int((live or {}).get("home_red", 0) or 0), int((live or {}).get("away_red", 0) or 0)))
        model_out["probabilities"] = draw_ht["probabilities"]
        draw_out = draw_ht

    deterministic_reference = {
        "probabilities": dict(model_out["probabilities"]),
        "confidence": float(model_out["confidence"]),
        "uncertainty": float(model_out["uncertainty"]),
        "risk_flags": list(model_out.get("risk_flags", [])),
        "weights": model_out.get("weights"),
        "source_contribution": model_out.get("source_contribution"),
        "steps": model_out.get("steps"),
    }

    llm_central = None
    if decision_mode == "llm_central":
        llm_central = central_match_forecast_with_anthropic(
            settings,
            {
                "fixture_code": fixture_code,
                "window": window,
                "market_probs": market.get("normalized_probs"),
                "bookmaker_probs": bookmaker,
                "sportmonks_probs": sm_pred,
                "supabase_priors": priors,
                "lineup": lineup,
                "halftime": ht_out,
                "structured_claims": claim_extraction,
                "claim_signals": claim_signals,
                "data_completeness": completeness,
                "deterministic_prematch": prematch,
                "deterministic_model": deterministic_reference,
                "source_reconciliation": None,
                "market_stale": None,
                "signal_conflict": None,
                "top_signals": [],
                "dry_run": settings.dry_run,
            },
            storage_dir=settings.storage_dir,
        )
        central_prediction = normalize_central_prediction(
            llm_central,
            fallback_probs=deterministic_reference["probabilities"],
            fallback_confidence=deterministic_reference["confidence"],
            fallback_uncertainty=deterministic_reference["uncertainty"],
        )
        model_out = {
            **model_out,
            "probabilities": central_prediction["probabilities"],
            "confidence": central_prediction["confidence"],
            "uncertainty": central_prediction["uncertainty"],
            "risk_flags": list(dict.fromkeys(list(model_out.get("risk_flags") or []) + central_prediction["risk_flags"])),
            "llm_central": central_prediction,
            "deterministic_reference": deterministic_reference,
            "decision_mode": decision_mode,
        }

    cons = consensus_triangle(model_out["probabilities"], bookmaker, market.get("normalized_probs"))
    source_reconciliation = reconcile_sources(model_out["probabilities"], sm_pred, bookmaker, market.get("normalized_probs"))
    signals = list(claim_signals) + [
        score_signal("lineup_delta", "sportmonks", "lineup", lineup.get("probability_delta"), source_quality=.90 if "lineup_unconfirmed" not in lineup.get("risk_flags", []) else .35, freshness=.85, corroboration=.70, reason=lineup.get("reason", "")),
        score_signal("draw_model", "model", "draw", draw_out.get("delta"), source_quality=.70, freshness=.80, corroboration=.65, reason=draw_out.get("reason", "")),
    ]
    if ht_out:
        ht_delta = {k: ht_out.get("ht_probs", {}).get(k, 0) - prematch["probabilities"].get(k, 0) for k in ("home", "draw", "away")}
        signals.append(score_signal("halftime_model", "sportmonks", "halftime", ht_delta, source_quality=.75 if ht_out.get("ht_label") != "data_insufficient" else .35, freshness=.95, corroboration=.70, reason=ht_out.get("reason", "")))
    if synthetic.get("web_delta"):
        signals.append(score_signal("web_claim", "web", "rumor", synthetic.get("web_delta"), source_quality=.25, freshness=.65, corroboration=.10, reason="Synthetic weak web claim."))
    top_signals = summarize_signals(signals)
    conflict = signal_conflict_score(signals)
    confidence = max(.1, min(.9, model_out["confidence"] + cons["confidence_modifier"] - conflict - max(0, .65 - completeness["score"]) * .18))
    uncertainty = max(.05, min(.85, model_out["uncertainty"] + conflict + max(0, .70 - completeness["score"]) * .20))
    if llm_central:
        top_signals.extend(
            [{"name": "llm_central_support", "reason": text} for text in model_out["llm_central"].get("supporting_signals", [])[:2]]
        )
        top_signals.extend(
            [{"name": "llm_central_contradiction", "reason": text} for text in model_out["llm_central"].get("contradicting_signals", [])[:2]]
        )

    previous_market = synthetic.get("previous_market") or previous_normalized_probs(settings.storage_dir, fixture_code, window)
    stale = detect_market_stale(market.get("normalized_probs"), previous_market, bookmaker, synthetic.get("signal_delta") or lineup.get("probability_delta"))
    signal_reasons = [s.get("reason", "") for s in top_signals]
    if stale.get("is_stale"):
        signal_reasons.append(stale.get("reason", ""))
    edge = evaluate_edge(fixture_code, window, model_out["probabilities"], market.get("normalized_probs"), bookmaker, confidence, uncertainty, cons["case"], signals=signal_reasons)
    if stale.get("is_stale") and edge.get("edge_tier") != "none":
        edge["edge_type"] = "market_stale"
        edge["reason"] += " Stale-market detector supports the edge."

    duplicate = _duplicate_order_marker(settings, fixture_code, window) and not settings.dry_run
    extra_flags = draw_sanity_flags(model_out["probabilities"], reason=draw_out.get("reason", ""))
    extra_flags.extend(model_out.get("risk_flags", []))
    if llm_central:
        extra_flags.extend(model_out["llm_central"].get("blocking_risk_flags", []))
    extra_flags.extend(claim_extraction.get("risk_flags", []))
    extra_flags.extend(source_reconciliation.get("flags", []))
    if stale.get("reason") == "Need current and previous market snapshots.":
        extra_flags.append("market_snapshot_stale_unknown")
    risk = audit_decision(model_out["probabilities"], edge, confidence, uncertainty, settings.dry_run, bool(market.get("complete")), lineup, cons["case"], duplicate_order=duplicate, data_completeness=completeness["score"], extra_flags=extra_flags)
    llm_analysis = None
    if use_llm_analyst:
        llm_context = {
            "fixture_code": fixture_code,
            "window": window,
            "final_probs": model_out["probabilities"],
            "market_probs": market.get("normalized_probs"),
            "bookmaker_probs": bookmaker,
            "sportmonks_probs": sm_pred,
            "source_reconciliation": source_reconciliation,
            "consensus_case": cons["case"],
            "edge": edge,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "top_signals": top_signals,
            "signal_conflict": conflict,
            "data_completeness": completeness,
            "market_stale": stale,
            "risk": risk,
            "lineup": lineup,
            "halftime": ht_out,
            "dry_run": settings.dry_run,
        }
        llm_analysis = analyze_decision_signals_with_anthropic(settings, llm_context, storage_dir=settings.storage_dir)
        risk = merge_llm_analysis_into_risk(risk, llm_analysis)
    usd = bet_size(edge["edge_tier"], confidence, settings.max_order_usd, cons["bet_size_modifier"], allow_soft=False)
    best = edge["best_outcome"] or "home"
    raw_mid = (market.get("raw_midpoints") or {}).get(best) or (market.get("normalized_probs") or {}).get(best) or .33
    lp = limit_price(best, float(raw_mid), model_out["probabilities"][best])
    should_order = edge["should_bet"] and risk["order_allowed"] and usd > 0 and lp > 0
    order_payload = {"fixture_code": fixture_code, "team_code": best.upper() if best == "draw" else str(fixture.get(f"{best}_team_code") or best).upper(), "usd_size": f"{usd:.2f}", "limit_price": lp, "time_in_force_seconds": settings.tif_seconds, "idempotency_key": str(uuid.uuid4())}
    order_result = _safe_order(settings, order_payload) if should_order else {"submitted": False, "reason": "skip", "payload": order_payload}
    prediction_payload = {"fixture_code": fixture_code, "window": window, "probabilities": model_out["probabilities"], "confidence": confidence, "top_signals": top_signals, "risk_flags": risk["risk_flags"]}

    ledger = LedgerBuilder(fixture_code, window, settings)
    records = ledger.build_standard_trace(kickoff_time=fixture.get("starting_at") or fixture.get("kickoff_utc"), lock_time=fixture.get("pre_match_lock_at") or fixture.get("ht_lock_at"), sportmonks={"fixture_id": fixture_id, "detail_keys": list(detail.keys()) if isinstance(detail, dict) else [], "prediction": sm_pred}, supabase={"priors": priors, "completeness": completeness}, polymarket={**market, "previous_market": previous_market, "stale": stale}, bookmaker=bookmaker, lineup=lineup, halftime=ht_out, probability={**model_out, "draw_model": draw_out, "source_reconciliation": source_reconciliation, "claim_extraction": claim_extraction}, consensus=cons, edge=edge, risk=risk, prediction=prediction_payload, order={"action_type": "order" if should_order else "skip", **order_payload, "reason": order_result.get("reason")}, reflection={"data_complete": completeness["score"], "decision": "order" if should_order else "skip", "top_signals": top_signals, "source_reconciliation": source_reconciliation, "llm_claims": claim_extraction, "llm_analysis": llm_analysis})
    trace_quality = ledger.trace_quality()
    ledger_result = LedgerAdapter(settings).submit(ledger.session_id, records)
    decision = {"session_id": ledger.session_id, "fixture_code": fixture_code, "window": window, "teams": fixture.get("name"), "final_probs": model_out["probabilities"], "market_probs": market.get("normalized_probs"), "bookmaker_probs": bookmaker, "sportmonks_probs": sm_pred, "halftime": ht_out, "lineup": lineup, "data_completeness": completeness, "market_stale": stale, "source_reconciliation": source_reconciliation, "source_contribution": model_out.get("source_contribution"), "deterministic_weights": model_out.get("weights"), "probability_steps": model_out.get("steps"), "llm_claims": claim_extraction, "consensus_case": cons["case"], "best_outcome": best, "best_edge": edge["best_edge"], "edge_tier": edge["edge_tier"], "edge_type": edge["edge_type"], "edge_reason": edge["reason"], "confidence": confidence, "uncertainty": uncertainty, "top_signals": top_signals, "llm_analysis": llm_analysis, "action": "BET" if should_order else "SKIP", "risk_flags": risk["risk_flags"], "blocking_risk_flags": risk["blocking_risk_flags"], "prediction_submitted": True, "order_submitted": bool(order_result.get("submitted")), "dry_run": settings.dry_run, "ledger_submitted": ledger_result.get("submitted", False), "ledger_records": len(records), "ledger_dag_valid": ledger.validate_dag(), "trace_quality": trace_quality, "order": order_result}
    decision["decision_mode"] = decision_mode
    decision["llm_central"] = llm_central
    out = settings.storage_dir / "decisions" / f"{fixture_code}-{window}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print_run_report(decision)
    if verbose:
        print(json.dumps({"decision_file": str(out), "session_id": ledger.session_id, "data_completeness": completeness, "market_stale": stale, "blocking_risk_flags": risk["blocking_risk_flags"]}, indent=2))
    return decision
