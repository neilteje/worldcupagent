"""
The reasoning council — a four-role deliberation that replaces the single
predict step.

    Scout      (fast model)      → flags from raw external research
    Analyst    (deep model)      → market-blind base probability
    Devil      (DeepSeek-R1)     → strongest counter-case, raw chain-of-thought
    Judge      (deep model)      → final calibrated probability, sees markets

Each role is one LLM call returning an LLMResult, so the agent can emit one
ledger record per role with proper upstream linkage (Scout→Analyst→Devil→Judge),
producing the multi-step DAG the arena rewards. The Analyst never sees market
prices; only the Judge does. Every step degrades gracefully — a failed role
yields an empty parsed dict and the council continues.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

import config
from reasoning import llm
from reasoning.prompts import (
    SCOUT_SYS, scout_input,
    ANALYST_SYS, analyst_input,
    DEVIL_SYS, devil_input,
    JUDGE_SYS, judge_input,
)

# International-football base rates, used only as a last-resort fallback.
_BASE_RATE = {"home": 0.40, "draw": 0.28, "away": 0.32}


@dataclass
class CouncilResult:
    outcome: str
    probability: float
    confidence: str
    council_summary: str
    probabilities: dict                       # {home_code|'draw'|away_code: float}
    scout_flags: list[dict] = field(default_factory=list)
    market_alignment: str = "unknown"
    # Per-role raw results (for the ledger trace)
    scout: Any = None
    analyst: Any = None
    devil: Any = None
    judge: Any = None


def _safe_call(fn, *args, **kwargs):
    """Run an LLM call; on hard failure return an empty LLMResult-like object."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"  [council step failed: {e}]")
        return llm.LLMResult(parsed={}, raw_text="", thinking="",
                             model="", provider="error")


def _pick_outcome(probs: dict, home_code: str, away_code: str) -> tuple[str, float]:
    valid = {k: float(v) for k, v in (probs or {}).items()
             if k in (home_code, "draw", away_code) and isinstance(v, (int, float))}
    if not valid:
        return home_code, _BASE_RATE["home"]
    outcome = max(valid, key=valid.get)
    return outcome, valid[outcome]


def run_council(
    fixture_name: str,
    home_code: str,
    away_code: str,
    sportmonks_digest: dict | None,
    supabase_digest: dict | None,
    polymarket_digest: dict | None,
    kalshi_moneyline: dict | None,
    web_research: dict | None,
    reddit_bundle: dict | None,
) -> CouncilResult:
    # 1 — Scout: triage the unstructured external research.
    scout = _safe_call(
        llm.call_claude,
        SCOUT_SYS,
        scout_input(fixture_name, home_code, away_code,
                    sportmonks_digest, web_research, reddit_bundle),
        model=config.SCOUT_MODEL,
        thinking_budget=config.SCOUT_THINKING_BUDGET,
    )
    scout_flags = (scout.parsed or {}).get("flags") or []

    # 2 — Analyst: market-blind base probability.
    analyst = _safe_call(
        llm.call_claude,
        ANALYST_SYS,
        analyst_input(fixture_name, home_code, away_code,
                      sportmonks_digest, supabase_digest, scout.parsed),
        model=config.ANALYST_MODEL,
        thinking_budget=config.THINKING_BUDGET,
    )

    # 3 — Devil's advocate: strongest counter-case (raw CoT via DeepSeek).
    devil = _safe_call(
        llm.call_deepseek,
        DEVIL_SYS,
        devil_input(fixture_name, home_code, away_code,
                    analyst.parsed, sportmonks_digest, supabase_digest),
        model=config.DEVIL_MODEL,
    )

    # 4 — Judge: synthesize everything, now seeing the markets.
    judge = _safe_call(
        llm.call_claude,
        JUDGE_SYS,
        judge_input(fixture_name, home_code, away_code,
                    analyst.parsed, devil.parsed, polymarket_digest, kalshi_moneyline),
        model=config.JUDGE_MODEL,
        thinking_budget=config.THINKING_BUDGET,
    )

    # Resolve the final view: prefer Judge, fall back to Analyst, then base rate.
    j = judge.parsed or {}
    a = analyst.parsed or {}
    probs = j.get("probabilities") or a.get("probabilities") or {
        home_code: _BASE_RATE["home"], "draw": _BASE_RATE["draw"], away_code: _BASE_RATE["away"]
    }
    outcome = j.get("outcome") or a.get("outcome")
    probability = j.get("probability") or a.get("probability")
    if not outcome or probability is None:
        outcome, probability = _pick_outcome(probs, home_code, away_code)

    return CouncilResult(
        outcome=outcome,
        probability=float(probability),
        confidence=j.get("confidence") or a.get("confidence") or "low",
        council_summary=j.get("council_summary") or a.get("rationale") or "",
        probabilities=probs,
        scout_flags=scout_flags,
        market_alignment=j.get("market_alignment", "unknown"),
        scout=scout,
        analyst=analyst,
        devil=devil,
        judge=judge,
    )
