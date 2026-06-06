from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from random import Random
from datetime import datetime, timezone
import json, math
from agent.config import Settings
from models.consensus import consensus_triangle
from models.edge_engine import evaluate_edge
from models.llm_central import normalize_central_prediction
from models.probability import pre_match_model
from models.sanity_checks import audit_decision
from models.bet_sizing import bet_size
from reasoning.central_llm import central_match_forecast_with_anthropic

@dataclass
class BacktestMatch:
    fixture_code: str
    sportmonks: dict[str, float]
    bookmaker: dict[str, float]
    market: dict[str, float]
    priors: dict[str, float]
    result: str


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
        rows.append(BacktestMatch(f"BT-{i+1:03d}", sm, bookmaker, market, priors, result))
    return rows


def brier(probs: dict[str, float], result: str) -> float:
    return sum((probs[k] - (1.0 if k == result else 0.0))**2 for k in ("home","draw","away"))


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


def run_backtest(settings: Settings, sample_size: int = 50, use_claude: bool = False) -> dict:
    rows = synthetic_history(sample_size)
    bankroll = 5.0
    decisions = []
    total_brier = market_brier = 0.0
    for row in rows:
        model = pre_match_model(row.sportmonks, row.bookmaker, row.market, row.priors, None, 1.0)
        cons = consensus_triangle(model["probabilities"], row.bookmaker, row.market)
        conf = model["confidence"] + cons["confidence_modifier"]
        edge = evaluate_edge(row.fixture_code, "PRE_MATCH", model["probabilities"], row.market, row.bookmaker, conf, model["uncertainty"], cons["case"])
        risk = audit_decision(model["probabilities"], edge, conf, model["uncertainty"], False, True, None, cons["case"])
        size = bet_size(edge["edge_tier"], conf, min(1.75, bankroll), cons["bet_size_modifier"], allow_soft=True)
        bet = edge["should_bet"] and risk["order_allowed"] and size > 0
        pnl = 0.0
        if bet:
            outcome = edge["best_outcome"]
            price = max(.02, min(.98, row.market[outcome]))
            pnl = size * ((1/price)-1) if outcome == row.result else -size
            bankroll += pnl
        total_brier += brier(model["probabilities"], row.result)
        market_brier += brier(row.market, row.result)
        decisions.append({"fixture_code": row.fixture_code, "result": row.result, "probs": model["probabilities"], "market": row.market, "edge": edge, "bet": bet, "size": size, "pnl": round(pnl, 4)})
    summary = {"matches": len(rows), "starting_bankroll": 5.0, "ending_bankroll": round(bankroll, 4), "roi": round((bankroll-5.0)/5.0, 4), "model_brier": round(total_brier/len(rows), 4), "market_brier": round(market_brier/len(rows), 4), "bets": sum(1 for d in decisions if d["bet"]), "timestamp": datetime.now(timezone.utc).isoformat()}
    if use_claude:
        summary["claude_review"] = _claude_critique(settings, summary)
    path = settings.storage_dir / "backtests" / f"backtest-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": summary, "decisions": decisions}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return {"summary": summary, "path": str(path), "decisions": decisions}


def run_mode_comparison_backtest(settings: Settings, sample_size: int = 20, *, include_claude_report: bool = True) -> dict:
    rows = synthetic_history(sample_size)
    deterministic = _simulate_strategy(settings, rows, mode="deterministic")
    llm_central = _simulate_strategy(settings, rows, mode="llm_central")
    winner = _pick_winner(deterministic, llm_central)
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
    }
    if include_claude_report:
        summary["claude_comparison"] = _claude_compare_report(settings, summary)
    return _write_comparison_report(settings, summary, deterministic.decisions, llm_central.decisions)


def _simulate_strategy(settings: Settings, rows: list[BacktestMatch], *, mode: str) -> StrategyResult:
    bankroll = 5.0
    decisions = []
    total_brier = market_brier = 0.0
    fallback_count = 0
    blocked_by_llm = 0
    for row in rows:
        deterministic_model = pre_match_model(row.sportmonks, row.bookmaker, row.market, row.priors, None, 1.0)
        model = deterministic_model
        llm_central = None
        if mode == "llm_central":
            llm_result = central_match_forecast_with_anthropic(
                settings,
                {
                    "fixture_code": row.fixture_code,
                    "window": "PRE_MATCH",
                    "market_probs": row.market,
                    "bookmaker_probs": row.bookmaker,
                    "sportmonks_probs": row.sportmonks,
                    "supabase_priors": row.priors,
                    "lineup": {"risk_flags": [], "home_lineup_confirmed": True, "away_lineup_confirmed": True, "reason": "Synthetic confirmed lineups for backtest fairness."},
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
        risk = audit_decision(model["probabilities"], edge, conf, model["uncertainty"], False, True, None, cons["case"], extra_flags=(llm_central or {}).get("blocking_risk_flags"))
        size = bet_size(edge["edge_tier"], conf, min(1.75, bankroll), cons["bet_size_modifier"], allow_soft=True)
        bet = edge["should_bet"] and risk["order_allowed"] and size > 0
        pnl = 0.0
        if bet:
            outcome = edge["best_outcome"]
            price = max(.02, min(.98, row.market[outcome]))
            pnl = size * ((1 / price) - 1) if outcome == row.result else -size
            bankroll += pnl
        total_brier += brier(model["probabilities"], row.result)
        market_brier += brier(row.market, row.result)
        decisions.append({
            "fixture_code": row.fixture_code,
            "mode": mode,
            "result": row.result,
            "probs": model["probabilities"],
            "market": row.market,
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
        f"- Sample size: {summary['sample_size']}\n"
        f"- Winner: {summary['winner']}\n\n"
        "## Deterministic\n"
        f"- ROI: {summary['deterministic']['roi']}\n"
        f"- Brier: {summary['deterministic']['model_brier']}\n"
        f"- Bets: {summary['deterministic']['bets']}\n\n"
        "## LLM Central\n"
        f"- ROI: {summary['llm_central']['roi']}\n"
        f"- Brier: {summary['llm_central']['model_brier']}\n"
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
