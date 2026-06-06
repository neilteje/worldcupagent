from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from random import Random
from datetime import datetime, timezone
import json, math
from agent.config import Settings
from models.consensus import consensus_triangle
from models.edge_engine import evaluate_edge
from models.probability import pre_match_model
from models.sanity_checks import audit_decision
from models.bet_sizing import bet_size

@dataclass
class BacktestMatch:
    fixture_code: str
    sportmonks: dict[str, float]
    bookmaker: dict[str, float]
    market: dict[str, float]
    result: str


def synthetic_history(n: int, seed: int = 2026) -> list[BacktestMatch]:
    rng = Random(seed)
    rows = []
    for i in range(n):
        h = rng.uniform(.25, .62); d = rng.uniform(.18, .34); a = max(.08, 1-h-d)
        total = h+d+a; true = {"home": h/total, "draw": d/total, "away": a/total}
        def noisy(scale):
            vals = {k: max(.02, true[k] + rng.gauss(0, scale)) for k in true}; s=sum(vals.values()); return {k: vals[k]/s for k in vals}
        market = noisy(.055); bookmaker = noisy(.035); sm = noisy(.045)
        r = rng.random(); acc=0; result="away"
        for k,v in true.items():
            acc += v
            if r <= acc: result = k; break
        rows.append(BacktestMatch(f"BT-{i+1:03d}", sm, bookmaker, market, result))
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


def run_backtest(settings: Settings, sample_size: int = 50, use_claude: bool = False) -> dict:
    rows = synthetic_history(sample_size)
    bankroll = 5.0
    decisions = []
    total_brier = market_brier = 0.0
    for row in rows:
        model = pre_match_model(row.sportmonks, row.bookmaker, row.market, None, None, .8)
        cons = consensus_triangle(model["probabilities"], row.bookmaker, row.market)
        conf = model["confidence"] + cons["confidence_modifier"]
        edge = evaluate_edge(row.fixture_code, "PRE_MATCH", model["probabilities"], row.market, row.bookmaker, conf, model["uncertainty"], cons["case"])
        risk = audit_decision(model["probabilities"], edge, conf, model["uncertainty"], False, True, None, cons["case"])
        size = bet_size(edge["edge_tier"], conf, min(1.0, bankroll), cons["bet_size_modifier"])
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
