from __future__ import annotations
from dataclasses import dataclass
from random import Random
from datetime import datetime, timezone
import json, math
from agent.config import Settings
from backtesting.worldcup_2022 import HistoricalFixture, build_worldcup_2022_history
from models.archetype import classify_match_archetype
from models.consensus import consensus_triangle
from models.draw_model import apply_draw_model
from models.edge_engine import evaluate_edge
from models.llm_central import normalize_central_prediction
from models.probability import pre_match_model
from models.probability_blender import DEFAULT_PREMATCH_WEIGHTS
from models.sanity_checks import audit_decision
from models.bet_sizing import bet_size
from models.source_reliability import dynamic_source_weights
from reasoning.central_llm import central_match_forecast_with_anthropic

@dataclass
class BacktestMatch:
    fixture_code: str
    sportmonks: dict[str, float]
    bookmaker: dict[str, float]
    market: dict[str, float]
    priors: dict[str, float]
    result: str
    home_team: str = "Home"
    away_team: str = "Away"
    kickoff_utc: str = ""
    stage: str = ""
    match_week: int = 0
    lineup: dict | None = None
    pre_state: dict | None = None
    odds: dict | None = None
    odds_quality: dict | None = None
    source: str = "synthetic"


@dataclass
class StrategyResult:
    mode: str
    ending_bankroll: float
    roi: float
    model_brier: float
    market_brier: float
    bets: int
    fallback_count: int
    blocked_by_llm: int
    decisions: list[dict]
    model_log_loss: float = 0.0
    market_log_loss: float = 0.0
    calibration_error: float = 0.0


def synthetic_history(n: int, seed: int = 2026) -> list[BacktestMatch]:
    rng = Random(seed)
    rows = []
    for i in range(n):
        h = rng.uniform(.25, .62); d = rng.uniform(.18, .34); a = max(.08, 1-h-d)
        total = h+d+a; true = {"home": h/total, "draw": d/total, "away": a/total}
        def noisy(scale):
            vals = {k: max(.02, true[k] + rng.gauss(0, scale)) for k in true}; s=sum(vals.values()); return {k: vals[k]/s for k in vals}
        market = noisy(.055); bookmaker = noisy(.035); sm = noisy(.045); priors = noisy(.025)
        r = rng.random(); acc=0; result="away"
        for k,v in true.items():
            acc += v
            if r <= acc: result = k; break
        rows.append(BacktestMatch(f"BT-{i+1:03d}", sm, bookmaker, market, priors, result, source="synthetic"))
    return rows


def brier(probs: dict[str, float], result: str) -> float:
    return sum((probs[k] - (1.0 if k == result else 0.0))**2 for k in ("home","draw","away"))


def log_loss(probs: dict[str, float], result: str) -> float:
    return -math.log(max(1e-9, min(1.0, float(probs.get(result, 0.0) or 0.0))))


def calibration_error(decisions: list[dict], bins: int = 5) -> float:
    bucketed: dict[int, list[tuple[float, float]]] = {}
    for decision in decisions:
        probs = decision.get("probs") or {}
        pick = max(("home", "draw", "away"), key=lambda k: float(probs.get(k, 0.0) or 0.0))
        confidence = float(probs.get(pick, 0.0) or 0.0)
        hit = 1.0 if decision.get("result") == pick else 0.0
        bucket = min(bins - 1, int(confidence * bins))
        bucketed.setdefault(bucket, []).append((confidence, hit))
    if not bucketed:
        return 0.0
    total = sum(len(v) for v in bucketed.values())
    err = 0.0
    for values in bucketed.values():
        avg_conf = sum(v[0] for v in values) / len(values)
        avg_hit = sum(v[1] for v in values) / len(values)
        err += (len(values) / total) * abs(avg_conf - avg_hit)
    return err


def settle_bet(size: float, outcome: str | None, result: str, market_probs: dict[str, float] | None, decimal_odds: dict | None = None) -> float:
    if not outcome or size <= 0:
        return 0.0
    if outcome != result:
        return -size
    if decimal_odds and outcome in decimal_odds and float(decimal_odds[outcome]) > 1.0:
        return size * (float(decimal_odds[outcome]) - 1.0)
    price = max(0.02, min(0.98, float((market_probs or {}).get(outcome, 0.5))))
    return size * ((1 / price) - 1)


def _claude_critique(settings: Settings, summary: dict) -> dict:
    if not settings.anthropic_key:
        return {"used": False, "reason": "ANTHROPIC_KEY missing"}
    try:
        import anthropic
        # Haiku-level critique to preserve the $5 dev wallet/API budget. Hard cap token budget.
        client = anthropic.Anthropic(api_key=settings.anthropic_key)
        prompt = "Review this soccer prediction backtest summary. Return compact JSON with 3 improvements only.\n" + json.dumps(summary)[:3500]
        resp = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=500, messages=[{"role":"user","content":prompt}])
        text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
        return {"used": True, "model": "claude-haiku-4-5-20251001", "critique": text[:3000], "estimated_cost_usd": 0.02}
    except Exception as exc:
        return {"used": False, "reason": repr(exc)}


def _claude_compare_report(settings: Settings, summary: dict) -> dict:
    if not settings.anthropic_key:
        return {"used": False, "reason": "ANTHROPIC_KEY missing"}
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=settings.anthropic_key)
        prompt = (
            "Compare deterministic and llm_central soccer prediction backtest results. "
            "Return compact JSON with keys winner, rationale, risks, recommendations. "
            "Use short arrays only.\n" + json.dumps(summary)[:7000]
        )
        resp = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=700, messages=[{"role": "user", "content": prompt}])
        text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
        try:
            parsed = json.loads(text[text.find("{"): text.rfind("}") + 1])
        except Exception:
            parsed = {"raw": text[:3000]}
        return {"used": True, "model": "claude-haiku-4-5-20251001", "parsed": parsed, "text": text[:3000], "estimated_cost_usd": 0.03}
    except Exception as exc:
        return {"used": False, "reason": repr(exc)}


def run_backtest(settings: Settings, sample_size: int = 50, use_claude: bool = False, dataset: str = "synthetic", mode: str = "deterministic") -> dict:
    rows = load_backtest_rows(settings, dataset=dataset, sample_size=sample_size)
    result = _simulate_strategy(settings, rows, mode=mode)
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "matches": len(rows),
        "starting_bankroll": 5.0,
        **_strategy_summary(result),
        "audit": _dataset_audit(rows),
    }
    if use_claude:
        summary["claude_review"] = _claude_critique(settings, summary)
    path = settings.storage_dir / "backtests" / f"backtest-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": summary, "decisions": result.decisions}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return {"summary": summary, "path": str(path), "decisions": result.decisions}


def run_mode_comparison_backtest(settings: Settings, sample_size: int = 20, *, include_claude_report: bool = True, dataset: str = "synthetic") -> dict:
    rows = load_backtest_rows(settings, dataset=dataset, sample_size=sample_size)
    deterministic = _simulate_strategy(settings, rows, mode="deterministic")
    llm_central = _simulate_strategy(settings, rows, mode="llm_central")
    winner = _pick_winner(deterministic, llm_central)
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "sample_size": sample_size,
        "winner": winner,
        "deterministic": _strategy_summary(deterministic),
        "llm_central": _strategy_summary(llm_central),
        "comparison": {
            "roi_diff": round(llm_central.roi - deterministic.roi, 4),
            "brier_diff": round(llm_central.model_brier - deterministic.model_brier, 4),
            "bet_diff": llm_central.bets - deterministic.bets,
            "llm_fallback_count": llm_central.fallback_count,
            "llm_blocked_by_llm": llm_central.blocked_by_llm,
        },
        "audit": _dataset_audit(rows),
    }
    if include_claude_report:
        summary["claude_comparison"] = _claude_compare_report(settings, summary)
    return _write_comparison_report(settings, summary, deterministic.decisions, llm_central.decisions)


def _simulate_strategy(settings: Settings, rows: list[BacktestMatch], *, mode: str) -> StrategyResult:
    bankroll = 5.0
    decisions = []
    total_brier = market_brier = total_log_loss = market_total_log_loss = 0.0
    fallback_count = 0
    blocked_by_llm = 0
    for row in rows:
        completeness = 1.0 if row.market and row.bookmaker else 0.75
        archetype = classify_match_archetype(
            window="PRE_MATCH",
            sportmonks_probs=row.sportmonks,
            bookmaker_probs=row.bookmaker,
            market_probs=row.market,
            lineup=row.lineup or {},
            data_completeness={"score": completeness},
        )
        reliability = dynamic_source_weights(DEFAULT_PREMATCH_WEIGHTS, archetype=archetype, data_completeness={"score": completeness})
        if mode == "deterministic_v2" and row.pre_state:
            from models.deterministic_v2 import predict_v2, EnsembleConfig
            cfg = EnsembleConfig()
            is_knockout = "group" not in (row.stage or "").lower()
            out = predict_v2(
                row.pre_state["home"], 
                row.pre_state["away"], 
                market_probs=row.market, 
                cfg=cfg,
                is_knockout=is_knockout,
                match_week=row.match_week,
                host_continent="AFC" if "wc2022" in getattr(settings, "dataset", "wc2022").lower() else None
            )
            deterministic_model = {
                "probabilities": out["probabilities"],
                "confidence": out["confidence"],
                "uncertainty": max(0.18, 1.0 - out["confidence"]),
                "expected_goals": out["expected_goals"],
            }
        else:
            deterministic_model = pre_match_model(
                row.sportmonks,
                row.bookmaker,
                row.market,
                row.priors,
                (row.lineup or {}).get("probability_delta"),
                completeness,
                weights=reliability["weights"],
            )
        draw = apply_draw_model(
            deterministic_model["probabilities"],
            strength_gap=abs(deterministic_model["probabilities"]["home"] - deterministic_model["probabilities"]["away"]),
            market_draw=(row.market or {}).get("draw"),
            bookmaker_draw=(row.bookmaker or {}).get("draw"),
        )
        deterministic_model["probabilities"] = draw["probabilities"]
        deterministic_model["archetype"] = archetype
        deterministic_model["source_reliability"] = reliability
        model = deterministic_model
        llm_central = None
        if mode == "llm_central":
            llm_result = central_match_forecast_with_anthropic(
                settings,
                {
                    "fixture_code": row.fixture_code,
                    "window": "PRE_MATCH",
                    "teams": f"{row.home_team} vs {row.away_team}",
                    "market_probs": row.market,
                    "bookmaker_probs": row.bookmaker,
                    "sportmonks_probs": row.sportmonks,
                    "supabase_priors": row.priors,
                    "lineup": row.lineup or {"risk_flags": [], "home_lineup_confirmed": True, "away_lineup_confirmed": True, "reason": "No lineup payload."},
                    "halftime": None,
                    "structured_claims": {},
                    "claim_signals": [],
                    "data_completeness": {"score": 1.0, "missing": []},
                    "deterministic_prematch": deterministic_model,
                    "deterministic_model": deterministic_model,
                    "source_reconciliation": {},
                    "market_stale": {},
                    "signal_conflict": 0.0,
                    "top_signals": [],
                    "dry_run": False,
                },
                storage_dir=settings.storage_dir,
            )
            llm_central = normalize_central_prediction(
                llm_result,
                fallback_probs=deterministic_model["probabilities"],
                fallback_confidence=deterministic_model["confidence"],
                fallback_uncertainty=deterministic_model["uncertainty"],
            )
            if llm_central["used_fallback"]:
                fallback_count += 1
            if llm_central["blocking_risk_flags"]:
                blocked_by_llm += 1
            model = {
                **deterministic_model,
                "probabilities": llm_central["probabilities"],
                "confidence": llm_central["confidence"],
                "uncertainty": llm_central["uncertainty"],
                "risk_flags": list(dict.fromkeys(list(deterministic_model.get("risk_flags") or []) + llm_central["risk_flags"])),
            }
        cons = consensus_triangle(model["probabilities"], row.bookmaker, row.market)
        conf = model["confidence"] + cons["confidence_modifier"]
        edge = evaluate_edge(row.fixture_code, "PRE_MATCH", model["probabilities"], row.market, row.bookmaker, conf, model["uncertainty"], cons["case"])
        odds_quality = row.odds_quality or ((row.pre_state or {}).get("odds_quality") if row.pre_state else None) or {"tradable": True, "flags": []}
        extra_flags = list((llm_central or {}).get("blocking_risk_flags") or [])
        if not odds_quality.get("tradable", True):
            extra_flags.append("market_reference_suspect")
        risk = audit_decision(model["probabilities"], edge, conf, model["uncertainty"], False, True, None, cons["case"], extra_flags=extra_flags)
        max_risk = min(0.75, bankroll * 0.20)
        size = bet_size(edge["edge_tier"], conf, max_risk, cons["bet_size_modifier"], allow_soft=True)
        bet = edge["should_bet"] and risk["order_allowed"] and size > 0
        pnl = 0.0
        if bet:
            outcome = edge["best_outcome"]
            pnl = settle_bet(size, outcome, row.result, row.market, row.odds)
            bankroll += pnl
        total_brier += brier(model["probabilities"], row.result)
        market_brier += brier(row.market, row.result)
        total_log_loss += log_loss(model["probabilities"], row.result)
        market_total_log_loss += log_loss(row.market, row.result)
        decisions.append({
            "fixture_code": row.fixture_code,
            "mode": mode,
            "source": row.source,
            "kickoff_utc": row.kickoff_utc,
            "teams": f"{row.home_team} vs {row.away_team}",
            "stage": row.stage,
            "match_week": row.match_week,
            "result": row.result,
            "probs": model["probabilities"],
            "market": row.market,
            "bookmaker": row.bookmaker,
            "sportmonks": row.sportmonks,
            "priors": row.priors,
            "archetype": deterministic_model.get("archetype"),
            "source_reliability": deterministic_model.get("source_reliability"),
            "lineup": row.lineup,
            "pre_state": row.pre_state,
            "odds": row.odds,
            "odds_quality": odds_quality,
            "edge": edge,
            "risk": risk,
            "bet": bet,
            "size": size,
            "pnl": round(pnl, 4),
            "llm_central": llm_central,
        })
    matches = max(len(rows), 1)
    return StrategyResult(
        mode=mode,
        ending_bankroll=round(bankroll, 4),
        roi=round((bankroll - 5.0) / 5.0, 4),
        model_brier=round(total_brier / matches, 4),
        market_brier=round(market_brier / matches, 4),
        model_log_loss=round(total_log_loss / matches, 4),
        market_log_loss=round(market_total_log_loss / matches, 4),
        calibration_error=round(calibration_error(decisions), 4),
        bets=sum(1 for d in decisions if d["bet"]),
        fallback_count=fallback_count,
        blocked_by_llm=blocked_by_llm,
        decisions=decisions,
    )


def _strategy_summary(result: StrategyResult) -> dict:
    return {
        "mode": result.mode,
        "ending_bankroll": result.ending_bankroll,
        "roi": result.roi,
        "model_brier": result.model_brier,
        "market_brier": result.market_brier,
        "model_log_loss": result.model_log_loss,
        "market_log_loss": result.market_log_loss,
        "calibration_error": result.calibration_error,
        "bets": result.bets,
        "fallback_count": result.fallback_count,
        "blocked_by_llm": result.blocked_by_llm,
    }


def _pick_winner(deterministic: StrategyResult, llm_central: StrategyResult) -> str:
    if llm_central.roi > deterministic.roi and llm_central.model_brier <= deterministic.model_brier:
        return "llm_central"
    if deterministic.roi > llm_central.roi and deterministic.model_brier <= llm_central.model_brier:
        return "deterministic"
    if llm_central.model_brier < deterministic.model_brier:
        return "llm_central_on_calibration"
    if deterministic.model_brier < llm_central.model_brier:
        return "deterministic_on_calibration"
    return "tie"


def _write_comparison_report(settings: Settings, summary: dict, deterministic_decisions: list[dict], llm_decisions: list[dict]) -> dict:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = settings.storage_dir / "backtests" / f"comparison-{stamp}.json"
    md_path = settings.storage_dir / "backtests" / f"comparison-{stamp}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "deterministic_decisions": deterministic_decisions,
        "llm_central_decisions": llm_decisions,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = (
        "# Mode Comparison Backtest\n\n"
        f"- Timestamp: {summary['timestamp']}\n"
        f"- Dataset: {summary.get('dataset')}\n"
        f"- Sample size: {summary['sample_size']}\n"
        f"- Winner: {summary['winner']}\n\n"
        "## Deterministic\n"
        f"- ROI: {summary['deterministic']['roi']}\n"
        f"- Brier: {summary['deterministic']['model_brier']}\n"
        f"- Log loss: {summary['deterministic']['model_log_loss']}\n"
        f"- Bets: {summary['deterministic']['bets']}\n\n"
        "## LLM Central\n"
        f"- ROI: {summary['llm_central']['roi']}\n"
        f"- Brier: {summary['llm_central']['model_brier']}\n"
        f"- Log loss: {summary['llm_central']['model_log_loss']}\n"
        f"- Bets: {summary['llm_central']['bets']}\n"
        f"- Fallbacks: {summary['llm_central']['fallback_count']}\n"
        f"- LLM-blocked decisions: {summary['llm_central']['blocked_by_llm']}\n\n"
        "## Comparison\n"
        f"- ROI diff (llm - det): {summary['comparison']['roi_diff']}\n"
        f"- Brier diff (llm - det): {summary['comparison']['brier_diff']}\n"
        f"- Bet diff (llm - det): {summary['comparison']['bet_diff']}\n"
    )
    claude = summary.get("claude_comparison") or {}
    if claude:
        md += "\n## Claude Synthesis\n"
        if claude.get("used"):
            parsed = claude.get("parsed") or {}
            md += f"- Winner: {parsed.get('winner', 'n/a')}\n"
            md += f"- Rationale: {parsed.get('rationale', 'n/a')}\n"
            md += f"- Risks: {parsed.get('risks', 'n/a')}\n"
            md += f"- Recommendations: {parsed.get('recommendations', 'n/a')}\n"
        else:
            md += f"- Claude report unavailable: {claude.get('reason')}\n"
    md_path.write_text(md, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return {"summary": summary, "json_path": str(path), "markdown_path": str(md_path)}


def load_backtest_rows(settings: Settings, *, dataset: str, sample_size: int) -> list[BacktestMatch]:
    dataset = (dataset or "synthetic").lower()
    if dataset in {"wc2022", "worldcup2022", "historical_2022"}:
        fixtures = build_worldcup_2022_history(settings.storage_dir / "backtests" / "cache" / "wc2022", limit=sample_size or None)
        return [_from_historical_fixture(f) for f in fixtures]
    return synthetic_history(sample_size)


def _from_historical_fixture(fixture: HistoricalFixture) -> BacktestMatch:
    return BacktestMatch(
        fixture_code=fixture.fixture_code,
        sportmonks=fixture.sportmonks,
        bookmaker=fixture.market,
        market=fixture.market,
        priors=fixture.priors,
        result=fixture.result,
        home_team=fixture.home_team,
        away_team=fixture.away_team,
        kickoff_utc=fixture.kickoff_utc,
        stage=fixture.stage,
        match_week=fixture.match_week,
        lineup=fixture.lineup,
        pre_state=fixture.pre_state,
        odds=fixture.odds,
        odds_quality=fixture.odds_quality,
        source="statsbomb_2022_plus_checkbestodds",
    )


def _dataset_audit(rows: list[BacktestMatch]) -> dict:
    if not rows:
        return {"rows": 0}
    odds_rows = sum(1 for r in rows if r.odds)
    tradable_odds_rows = sum(1 for r in rows if r.odds and (r.odds_quality or {}).get("tradable", True))
    suspect_odds_rows = sum(1 for r in rows if r.odds and not (r.odds_quality or {}).get("tradable", True))
    lineup_rows = sum(1 for r in rows if r.lineup and "lineup_unconfirmed" not in (r.lineup.get("risk_flags") or []))
    return {
        "rows": len(rows),
        "source": rows[0].source,
        "real_match_results": all(r.source != "synthetic" for r in rows),
        "real_lineups": lineup_rows,
        "real_odds_rows": odds_rows,
        "tradable_odds_rows": tradable_odds_rows,
        "suspect_odds_rows": suspect_odds_rows,
        "synthetic_rows": sum(1 for r in rows if r.source == "synthetic"),
    }
