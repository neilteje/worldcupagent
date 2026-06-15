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
from reasoning import llm, grounding
from models.forecast_layers import build_forecast_layers
from models.market_calibration import normalize_market_probabilities
from reasoning.prompts import (
    SOCIAL_PULSE_SYS, social_pulse_input,
    SCOUT_SYS, scout_input,
    ANALYST_SYS, analyst_input,
    DEVIL_SYS, devil_input,
    JUDGE_SYS, judge_input,
)

# International-football base rates, used only as a last-resort fallback.
_BASE_RATE = {"home": 0.40, "draw": 0.28, "away": 0.32}


def _slot_probs_from_codes(probs: dict | None, home_code: str, away_code: str) -> dict[str, float] | None:
    if not probs:
        return None
    raw = {
        "home": probs.get("home", probs.get(home_code)),
        "draw": probs.get("draw"),
        "away": probs.get("away", probs.get(away_code)),
    }
    if any(not isinstance(v, (int, float)) for v in raw.values()):
        return None
    total = sum(float(v) for v in raw.values())
    if total <= 0:
        return None
    return {k: float(v) / total for k, v in raw.items()}


def _code_probs_from_slots(probs: dict, home_code: str, away_code: str) -> dict:
    return {home_code: probs["home"], "draw": probs["draw"], away_code: probs["away"]}


def _market_probs_from_digest(polymarket_digest: dict | None, home_code: str, away_code: str) -> dict | None:
    digest = polymarket_digest or {}
    return (
        _slot_probs_from_codes(digest.get("implied_win_prob"), home_code, away_code)
        or _slot_probs_from_codes(digest.get("probabilities"), home_code, away_code)
    )


@dataclass
class CouncilResult:
    outcome: str
    probability: float
    confidence: str
    council_summary: str
    probabilities: dict                       # {home_code|'draw'|away_code: float}
    scout_flags: list[dict] = field(default_factory=list)
    market_alignment: str = "unknown"
    social_pulse: dict = field(default_factory=dict)
    grounding: dict = field(default_factory=dict)   # anchor / sanity / shrink audit
    # Per-role raw results (for the ledger trace)
    pulse: Any = None
    scout: Any = None
    analyst: Any = None
    devil: Any = None
    judge: Any = None


def _safe_call(role: str, fn, *args, **kwargs):
    """Run an LLM call; on hard failure return an empty LLMResult-like object.

    Failures degrade gracefully but are logged LOUDLY with the role name so a
    blind council step is never mistaken for a considered one.
    """
    try:
        result = fn(*args, **kwargs)
        if result is not None and not (result.parsed or {}):
            print(f"  [council:{role}] WARNING: returned EMPTY parse "
                  f"(model={getattr(result, 'model', '?')}) — downstream roles "
                  f"will see no {role} output")
        return result
    except Exception as e:
        print(f"  [council:{role}] FAILED: {e!r} — continuing with empty output")
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
    home_name: str,
    away_name: str,
    kickoff: str,
    sportmonks_digest: dict | None,
    supabase_digest: dict | None,
    polymarket_digest: dict | None,
    web_research: dict | None,
    reddit_bundle: dict | None,
    deterministic_context: dict | None = None,
    bz_digest: dict | None = None,
) -> CouncilResult:
    # 0 — Social pulse: Grok reads live X/Twitter + news for breaking signals.
    pulse = _safe_call(
        "pulse",
        llm.call_grok,
        SOCIAL_PULSE_SYS,
        social_pulse_input(fixture_name, home_name, away_name, kickoff),
    )
    social_pulse = pulse.parsed if pulse else {}

    # Structured anchor (bookmaker consensus → Sportmonks ML) for the Analyst to
    # reconcile with and for the post-council sanity/shrink pass.
    anchor = grounding.extract_anchor(sportmonks_digest, home_code, away_code)

    # 1 — Scout: consolidate web + reddit + Grok pulse into severity-tagged flags.
    scout = _safe_call(
        "scout",
        llm.call_claude,
        SCOUT_SYS,
        scout_input(fixture_name, home_code, away_code,
                    sportmonks_digest, web_research, reddit_bundle, social_pulse,
                    deterministic_context=deterministic_context, bz_digest=bz_digest),
        model=config.SCOUT_MODEL,
        thinking_budget=config.SCOUT_THINKING_BUDGET,
    )
    scout_parsed = scout.parsed or {}
    market_blind_evidence = scout_parsed.get("market_blind_evidence")
    if market_blind_evidence is None:
        market_blind_evidence = {"flags": scout_parsed.get("flags") or []}
    scout_flags = market_blind_evidence.get("flags") or []

    # 2 — Analyst: market-blind base probability (TRULY market-blind, no anchor).
    # The raw digests carry bookmaker consensus / odds; scrub those before the
    # Analyst sees anything (spec §12 — Analyst must receive NO market info).
    from reasoning.market_blind import scrub_market_fields
    analyst_sm, _ = scrub_market_fields(sportmonks_digest or {})
    analyst_bz, _ = scrub_market_fields(bz_digest or {})
    analyst_det, _ = scrub_market_fields(deterministic_context or {})
    analyst = _safe_call(
        "analyst",
        llm.call_claude,
        ANALYST_SYS,
        analyst_input(fixture_name, home_code, away_code,
                      analyst_sm, supabase_digest, market_blind_evidence,
                      deterministic_context=analyst_det, bz_digest=analyst_bz),
        model=config.ANALYST_MODEL,
        thinking_budget=config.THINKING_BUDGET,
    )

    # 3 — Devil's advocate: strongest counter-case (raw CoT via DeepSeek).
    devil = _safe_call(
        "devil",
        llm.call_deepseek,
        DEVIL_SYS,
        devil_input(fixture_name, home_code, away_code,
                    analyst.parsed, sportmonks_digest, supabase_digest,
                    deterministic_context=deterministic_context, bz_digest=bz_digest),
        model=config.DEVIL_MODEL,
    )

    # 4 — Judge: synthesize everything, now seeing the markets.
    judge = _safe_call(
        "judge",
        llm.call_claude,
        JUDGE_SYS,
        judge_input(fixture_name, home_code, away_code,
                    analyst.parsed, devil.parsed, polymarket_digest,
                    deterministic_context=deterministic_context),
        model=config.JUDGE_MODEL,
        thinking_budget=config.THINKING_BUDGET,
    )

    j = judge.parsed or {}
    independent = (
        _slot_probs_from_codes((deterministic_context or {}).get("probabilities_hda"), home_code, away_code)
        or _slot_probs_from_codes((deterministic_context or {}).get("probabilities_by_code"), home_code, away_code)
        or _BASE_RATE
    )
    evidence_ids = tuple(
        str(x.get("evidence_id") or x.get("id") or x.get("signal") or "")
        for x in (market_blind_evidence.get("flags") or [])
        if isinstance(x, dict)
    )
    layers = build_forecast_layers(
        independent,
        analyst_output=analyst.parsed,
        devil_output=devil.parsed,
        judge_output=j,
        market_probabilities=_market_probs_from_digest(polymarket_digest, home_code, away_code),
        evidence_ids=tuple(x for x in evidence_ids if x),
        data_coverage_score=max(
            0.0,
            min(
                1.0,
                float((deterministic_context or {}).get("data_coverage_score", 0.0) or 0.0),
            ),
        ),
        confidence={"low": 0.4, "medium": 0.6, "high": 0.8}.get(str(j.get("confidence") or "medium").lower(), 0.6),
    )
    probs = _code_probs_from_slots(layers.scored_probabilities, home_code, away_code)
    confidence = j.get("confidence") or (analyst.parsed or {}).get("confidence") or "low"

    # ── Grounding pass: sanity-check, cap confidence on thin data, and apply
    # the documented low-confidence shrink toward the anchor. Nothing hidden —
    # the full audit rides on CouncilResult.grounding and into the ledger.
    g = grounding.ground_council_output(
        probs, confidence, home_code, away_code,
        sportmonks_digest=sportmonks_digest,
        supabase_digest=supabase_digest,
        devil_parsed=devil.parsed,
    )
    probs = g["probabilities"]
    confidence = g["confidence"]
    for flag in g["sanity_flags"]:
        print(f"  [council:grounding] {flag}")

    # Re-derive the headline pick from the FINAL (grounded) distribution so the
    # outcome/probability always match the probabilities map.
    outcome, probability = _pick_outcome(probs, home_code, away_code)

    return CouncilResult(
        outcome=outcome,
        probability=float(probability),
        confidence=confidence,
        council_summary=j.get("summary") or j.get("council_summary") or (analyst.parsed or {}).get("rationale") or "",
        probabilities=probs,
        scout_flags=scout_flags,
        market_alignment=j.get("market_efficiency_assessment", j.get("market_alignment", "unknown")),
        social_pulse=social_pulse,
        grounding={**g, "forecast_layers": layers.__dict__},
        pulse=pulse,
        scout=scout,
        analyst=analyst,
        devil=devil,
        judge=judge,
    )
