"""
One fixture × window cycle for the 4-agent portfolio.

The expensive part — data fetches + LLM digests + the council — runs ONCE per
window (the agents deliberately share one brain, per docs/STRATEGY.md). Each
agent then runs its own cheap, deterministic tail: wallet → policy →
order(s) → its own full reasoning-ledger session under its own API key.

PRE_MATCH uses the full council; HT uses the Bayesian HT update (and the
runner only calls it when the arena says the HT window is actually open —
release notes 20260610: HT is not enabled yet).
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import config
from data import sportmonks, supabase_client, polymarket as pm
from data import odds2prob
from data import web_search, reddit_sentiment
from data import fixture_bundle
from data.team_codes import fifa_code, normalize_probabilities
from reasoning import llm, council, gates
from reasoning.prompts import ht_predict_input
from ledger.client import LedgerSession
from betting import policy as bet_policy
from betting import recommendation as bet_reco
from betting.portfolio import PortfolioCoordinator, PortfolioLimits
from harness.profiles import confidence_to_num
from models.forecast_contracts import MatchForecast, stable_hash
from live.arena_client import ArenaClient
from live.roster import LiveAgent
from live import metrics

# Per-cycle spend ceiling per agent: min($5 arena rule, wallet − 5¢ buffer).
WALLET_BUFFER_USD = 0.05

from agents.monk import MonkStrategy
from agents.anchor import AnchorStrategy
from agents.hunter import HunterStrategy
from agents.blitz import BlitzStrategy
from agents.contracts import FixtureDataSnapshot, MarketContext

STRATEGIES = {
    "monk": MonkStrategy(),
    "anchor": AnchorStrategy(),
    "hunter": HunterStrategy(),
    "blitz": BlitzStrategy(),
}


class _EmptyResult:
    """Stand-in for a failed/skipped LLM step (keeps ledger building uniform)."""
    parsed: dict = {}
    thinking = ""
    model = ""
    provider = ""
    tokens_in = 0
    tokens_out = 0


@dataclass
class Forecast:
    """Everything one window's shared brain produced."""
    fixture_id: int
    window: str                      # "PRE_MATCH" | "HT"
    fixture_name: str = ""
    kickoff: str = ""
    home_code: str = "HOME"
    away_code: str = "AWAY"
    home_name: str = ""
    away_name: str = ""
    pm_slug: str | None = None
    moneyline: dict | None = None
    market_source: str = "none"
    mids: dict = field(default_factory=dict)
    sm_digest: dict | None = None
    sb_digest: dict | None = None
    bz_digest: dict | None = None
    odds2prob_digest: dict = field(default_factory=dict)
    pm_digest_result: Any = None     # LLM result obj (parsed/provider/model/…)
    web_research: dict = field(default_factory=dict)
    reddit_bundle: dict = field(default_factory=dict)
    social_pulse: dict = field(default_factory=dict)  # Grok live X/news pulse (post-council)
    # PRE_MATCH: council result. HT: ht_pred result + dict.
    cr: Any = None
    ht_pred_result: Any = None
    ht_context: dict = field(default_factory=dict)
    # Unified outputs every consumer reads:
    probabilities: dict = field(default_factory=dict)
    outcome: str = ""
    probability: float = 0.0
    confidence: str = "low"
    scout_flags: list = field(default_factory=list)
    grounding: dict = field(default_factory=dict)
    summary: str = ""
    engine: str = "unknown"
    deterministic_model: dict = field(default_factory=dict)
    independent_forecast: dict = field(default_factory=dict)
    forecast_snapshot_id: str = ""
    market_probabilities: dict = field(default_factory=dict)
    market_adjusted_probabilities: dict = field(default_factory=dict)


def _fetch_market(fixture_id: int) -> tuple[str | None, dict | None, dict]:
    """(pm_slug, moneyline, mids) — all failures degrade to predict-only."""
    slug, ml = None, None
    try:
        slug = pm.get_event_slug(fixture_id)
    except Exception as exc:
        print(f"  [live] Polymarket slug lookup failed: {exc!r}")
    if slug:
        try:
            ml = pm.get_moneyline(fixture_id)
        except Exception as exc:
            print(f"  [live] Polymarket moneyline fetch failed: {exc!r}")
    mids = {}
    if ml:
        mids = {k: (ml["outcomes"].get(k) or {}).get("current_mid_yes")
                for k in ("home", "draw", "away")}
        if not any(isinstance(v, (int, float)) for v in mids.values()):
            ml = None  # market exists but no prices → predict-only
    return slug, ml, mids


def _normalize_hda(probs: dict | None) -> dict | None:
    if not probs:
        return None
    raw = {
        "home": probs.get("home", probs.get("home_win")),
        "draw": probs.get("draw"),
        "away": probs.get("away", probs.get("away_win")),
    }
    vals = {}
    for key, value in raw.items():
        if not isinstance(value, (int, float)):
            return None
        vals[key] = max(0.0, float(value))
    total = sum(vals.values())
    if total <= 0:
        return None
    return {k: vals[k] / total for k in ("home", "draw", "away")}


def _hda_to_code(probs: dict, home_code: str, away_code: str) -> dict:
    hda = _normalize_hda(probs) or {"home": 0.40, "draw": 0.28, "away": 0.32}
    return {
        home_code: round(float(hda["home"]), 4),
        "draw": round(float(hda["draw"]), 4),
        away_code: round(float(hda["away"]), 4),
    }


def _hda_values_to_code(values: dict, home_code: str, away_code: str) -> dict:
    return {
        home_code: round(float(values.get("home", 0.0) or 0.0), 4),
        "draw": round(float(values.get("draw", 0.0) or 0.0), 4),
        away_code: round(float(values.get("away", 0.0) or 0.0), 4),
    }


def _market_prior(fx: Forecast, fixture: dict) -> tuple[dict | None, str]:
    if fx.moneyline:
        mids = {
            slot: (fx.moneyline.get("outcomes", {}).get(slot) or {}).get("current_mid_yes")
            for slot in ("home", "draw", "away")
        }
        prior = _normalize_hda(mids)
        if prior:
            return prior, "polymarket"
    bookmaker_calibrated = _normalize_hda((fx.odds2prob_digest or {}).get("probabilities"))
    if bookmaker_calibrated:
        return bookmaker_calibrated, "odds2prob_bookmaker"
    bookmaker = _normalize_hda(sportmonks.extract_bookmaker_probs(fixture))
    if bookmaker:
        return bookmaker, "sportmonks_bookmaker"
    sm = sportmonks.extract_ml_probabilities(fixture)
    prior = _normalize_hda(sm)
    if prior:
        return prior, "sportmonks_ml"
    return None, "neutral"


def _state_from_prior(prior: dict | None, side: str) -> dict:
    p = prior or {"home": 0.40, "draw": 0.28, "away": 0.32}
    diff = float(p.get("home", 0.40) or 0.40) - float(p.get("away", 0.32) or 0.32)
    rating = diff if side == "home" else -diff
    return {
        "live_rating": rating,
        "matches": 0,
        "xg_for": 0.0,
        "xg_against": 0.0,
        "goals_for": 0.0,
        "goals_against": 0.0,
    }


def _confidence_label(value: float) -> str:
    if value >= 0.72:
        return "high"
    if value >= 0.55:
        return "medium"
    return "low"


def _deterministic_v2_model(fx: Forecast, fixture: dict) -> dict:
    from models.deterministic_v2 import EnsembleConfig, predict_v2

    prior, prior_source = _market_prior(fx, fixture)
    cfg = EnsembleConfig()
    stage = str((fixture.get("stage") or {}).get("name") or fixture.get("stage") or "").lower()
    is_knockout = bool(stage) and "group" not in stage
    home_state = _state_from_prior(prior, "home")
    away_state = _state_from_prior(prior, "away")
    
    bzzoiro_probs = (fx.bz_digest or {}).get("ml_prediction")
    
    out = predict_v2(home_state, away_state, market_probs=prior, bzzoiro_probs=bzzoiro_probs, cfg=cfg, is_knockout=is_knockout)
    confidence = float(out.get("confidence", 0.5) or 0.5)
    return {
        "probabilities": out["probabilities"],
        "confidence": confidence,
        "uncertainty": max(0.12, min(0.70, 1.0 - confidence)),
        "risk_flags": [] if prior else ["deterministic_v2_neutral_cold_start"],
        "steps": [{
            "name": "deterministic_v2",
            "prior_source": prior_source,
            "weights": out["weights"],
            "probabilities": out["probabilities"],
        }],
        "model_version": "deterministic_v2.0",
        "prior_source": prior_source,
        "prior_hda": prior,
        "home_state": home_state,
        "away_state": away_state,
        "expected_goals": out["expected_goals"],
        "components": out["components"],
        "weights": out["weights"],
        "blended_raw": out["blended_raw"],
        "config": out["config"],
    }


def _coverage_score(fx: Forecast) -> float:
    # Diversity across BOTH structured model priors (Sportmonks/Supabase/BZZOIRO)
    # and the independent real-world read (web headlines + Reddit crowd). BZZOIRO
    # is one source among several here, not the spine — keeping it in the count
    # stops a single API from silently dominating the coverage signal.
    parts = [
        bool(fx.sm_digest),
        bool(fx.sb_digest),
        bool(fx.bz_digest),
        bool((fx.odds2prob_digest or {}).get("available")),
        bool((fx.web_research or {}).get("total_results")),
        bool((fx.reddit_bundle or {}).get("threads_found")),
    ]
    return round(sum(1 for p in parts if p) / len(parts), 4)


def _signal_coverage(fx: Forecast) -> dict:
    """Per-window map of which inputs actually carried content into the council.

    Logged so reliance on any single source (notably the BZZOIRO API) is
    auditable after the fact rather than assumed — the council is meant to read
    structured model priors AND the live real-world chatter, not lean on one API.
    """
    pulse = getattr(fx, "social_pulse", None) or {}
    return {
        "sportmonks": bool(fx.sm_digest),
        "supabase": bool(fx.sb_digest),
        "bzzoiro": bool(fx.bz_digest),
        "odds2prob": bool((fx.odds2prob_digest or {}).get("available")),
        "web_search": bool((fx.web_research or {}).get("total_results")),
        "reddit": bool((fx.reddit_bundle or {}).get("threads_found")),
        "grok_pulse": bool(pulse.get("summary") or pulse.get("breaking")
                           or pulse.get("overall_lean")),
        "polymarket": bool(fx.moneyline),
    }


def _evidence_ids(fx: Forecast) -> list[str]:
    ids: list[str] = []
    if fx.sm_digest:
        ids.append("sportmonks_digest")
    if fx.sb_digest:
        ids.append("supabase_digest")
    if fx.bz_digest:
        ids.append("bzzoiro_digest")
    if (fx.odds2prob_digest or {}).get("available"):
        ids.append("odds2prob_calibrated_bookmaker")
    for source in (fx.web_research or {}).get("sources") or []:
        ids.append(f"web:{source}")
    if (fx.reddit_bundle or {}).get("threads_found"):
        ids.append("reddit_sentiment")
    # Grok live X/news pulse: only populated once the council has run, so it is
    # absent from the pre-council snapshot and present when recommendations are
    # built. Counting it keeps the live human read in HUNTER's signal-diversity gate.
    pulse = getattr(fx, "social_pulse", None) or {}
    if pulse.get("summary") or pulse.get("breaking") or pulse.get("overall_lean"):
        ids.append("grok_social_pulse")
    return ids


def _state_from_independent_context(fx: Forecast, side: str) -> dict:
    code = fx.home_code if side == "home" else fx.away_code
    other = fx.away_code if side == "home" else fx.home_code
    state = {
        "live_rating": 0.0,
        "matches": 0,
        "xg_for": 0.0,
        "xg_against": 0.0,
        "goals_for": 0.0,
        "goals_against": 0.0,
        "bzzoiro_xg": 0.0,
        "bzzoiro_momentum": 0.0,
    }

    xg = (fx.sm_digest or {}).get("expected_goals") or {}
    own_xg = xg.get(code)
    opp_xg = xg.get(other)
    if isinstance(own_xg, (int, float)) and isinstance(opp_xg, (int, float)):
        state.update({
            "matches": 1,
            "xg_for": max(0.0, float(own_xg)),
            "xg_against": max(0.0, float(opp_xg)),
            "goals_for": max(0.0, float(own_xg)),
            "goals_against": max(0.0, float(opp_xg)),
        })

    teams = (fx.sb_digest or {}).get("teams") or {}
    team = teams.get(code) or {}
    h2h_wins = team.get("h2h_wins")
    h2h_losses = team.get("h2h_losses")
    try:
        total = float(h2h_wins or 0) + float(h2h_losses or 0)
        if total > 0:
            state["live_rating"] += 0.35 * (float(h2h_wins or 0) - float(h2h_losses or 0)) / total
    except (TypeError, ValueError):
        pass
    try:
        ko_rate = team.get("ko_advancement_rate")
        if ko_rate is not None:
            state["talent_score"] = max(0.0, min(5.0, float(ko_rate) * 5.0))
    except (TypeError, ValueError):
        pass
        
    bz = fx.bz_digest or {}
    bz_stats = bz.get("stats_summary") or {}
    if bz_stats:
        state["bzzoiro_xg"] = float(bz_stats.get(f"{side}_xg", 0.0))
        state["bzzoiro_momentum"] = float(bz_stats.get(f"{side}_momentum", 0.0))
        
    return state


def _bounds_for_probability(probability: float, confidence: float, coverage: float) -> tuple[float, float]:
    width = max(0.04, min(0.34, 0.08 + (1.0 - confidence) * 0.12 + (1.0 - coverage) * 0.14))
    return max(0.0, probability - width / 2.0), min(1.0, probability + width / 2.0)


def _market_probabilities_by_code(fx: Forecast) -> dict:
    hda = _normalize_hda(fx.mids)
    if not hda:
        return {}
    return _hda_to_code(hda, fx.home_code, fx.away_code)


def _build_independent_forecast_snapshot(fx: Forecast, fixture: dict) -> dict:
    from models.deterministic_v2 import EnsembleConfig, predict_v2

    cfg = EnsembleConfig(
        w_elo=0.50,
        w_poisson=0.50,
        w_market=0.0,
        use_market=False,
    )
    stage = str((fixture.get("stage") or {}).get("name") or fixture.get("stage") or "").lower()
    is_knockout = bool(stage) and "group" not in stage
    home_state = _state_from_independent_context(fx, "home")
    away_state = _state_from_independent_context(fx, "away")
    
    bzzoiro_probs = (fx.bz_digest or {}).get("ml_prediction")
    
    out = predict_v2(home_state, away_state, market_probs=None, bzzoiro_probs=bzzoiro_probs, cfg=cfg, is_knockout=is_knockout)
    probabilities = out["probabilities"]
    confidence = max(0.0, min(1.0, float(out.get("confidence", 0.5) or 0.5)))
    coverage = _coverage_score(fx)
    bounds = {slot: _bounds_for_probability(probabilities[slot], confidence, coverage)
              for slot in ("home", "draw", "away")}
    warnings = ["market_inputs_excluded"]
    if coverage < 0.5:
        warnings.append("low_data_coverage_widened_uncertainty")

    feature_payload = {
        "fixture_id": fx.fixture_id,
        "fixture_name": fx.fixture_name,
        "kickoff": fx.kickoff,
        "home_code": fx.home_code,
        "away_code": fx.away_code,
        "sm_digest": fx.sm_digest,
        "sb_digest": fx.sb_digest,
        "odds2prob_digest": fx.odds2prob_digest,
        "web_sources": (fx.web_research or {}).get("sources") or [],
        "reddit_threads": (fx.reddit_bundle or {}).get("threads_found", 0),
        "home_state": home_state,
        "away_state": away_state,
    }
    forecast = MatchForecast(
        fixture_id=str(fx.fixture_id),
        as_of_timestamp=datetime.now(timezone.utc),
        home_probability=probabilities["home"],
        draw_probability=probabilities["draw"],
        away_probability=probabilities["away"],
        home_lower_bound=bounds["home"][0],
        draw_lower_bound=bounds["draw"][0],
        away_lower_bound=bounds["away"][0],
        home_upper_bound=bounds["home"][1],
        draw_upper_bound=bounds["draw"][1],
        away_upper_bound=bounds["away"][1],
        confidence=confidence,
        data_coverage_score=coverage,
        model_version="deterministic_v2_market_blind.1",
        feature_snapshot_hash=stable_hash(feature_payload),
        evidence_ids=_evidence_ids(fx),
        warnings=warnings,
    )
    snapshot = forecast.to_dict()
    snapshot.update({
        "probabilities_by_code": _hda_to_code(probabilities, fx.home_code, fx.away_code),
        "lower_bounds_by_code": _hda_values_to_code(forecast.lower_bounds, fx.home_code, fx.away_code),
        "upper_bounds_by_code": _hda_values_to_code(forecast.upper_bounds, fx.home_code, fx.away_code),
        "active_components": out.get("active_components", []),
        "component_weights": out.get("weights", {}),
        "home_state": home_state,
        "away_state": away_state,
        "expected_goals": out.get("expected_goals"),
        "config": out.get("config"),
    })
    return snapshot


def _deterministic_context_for_council(fx: Forecast, fixture: dict) -> dict:
    det = _deterministic_v2_model(fx, fixture)
    fx.deterministic_model = det
    return {
        "engine": "deterministic_v2",
        "model_version": det["model_version"],
        "prior_source": det["prior_source"],
        "probabilities_hda": det["probabilities"],
        "probabilities_by_code": _hda_to_code(det["probabilities"], fx.home_code, fx.away_code),
        "confidence": det["confidence"],
        "uncertainty": det["uncertainty"],
        "risk_flags": det["risk_flags"],
        "expected_goals": det["expected_goals"],
        "components": det["components"],
        "odds2prob": fx.odds2prob_digest,
        "component_weights": det["weights"],
        "blended_raw": det["blended_raw"],
        "prior_hda": det["prior_hda"],
        "home_state": det["home_state"],
        "away_state": det["away_state"],
        "steps": det["steps"],
        "config": det["config"],
        "instruction": (
            "Use this deterministic_v2 output and component signals as a quantitative "
            "input. You may agree or disagree, but name concrete evidence whenever "
            "you move materially away from it."
        ),
    }


# ── Shared brain: PRE_MATCH ─────────────────────────────────────────────────

def gather_prematch(fixture_id: int) -> Forecast:
    fx = Forecast(fixture_id=fixture_id, window="PRE_MATCH")

    fixture = sportmonks.get_fixture(fixture_id)
    participants = fixture.get("participants") or []
    home = next((p for p in participants if p.get("meta", {}).get("location") == "home"), {})
    away = next((p for p in participants if p.get("meta", {}).get("location") == "away"), {})
    fx.home_code = fifa_code(home.get("short_code"), "HOME")
    fx.away_code = fifa_code(away.get("short_code"), "AWAY")
    fx.home_name = home.get("name", fx.home_code)
    fx.away_name = away.get("name", fx.away_code)
    fx.fixture_name = fixture.get("name", f"{fx.home_name} vs {fx.away_name}")
    fx.kickoff = str(fixture.get("starting_at", ""))

    print(f"  [live] {fx.fixture_name}  kickoff={fx.kickoff}")

    # Market
    fx.pm_slug, fx.moneyline, fx.mids = _fetch_market(fixture_id)
    fx.market_source = "polymarket" if fx.moneyline else "none"
    if fx.moneyline:
        fx.moneyline["market_source"] = "polymarket"
    print(f"  [live] market={fx.market_source}  mids={fx.mids or 'n/a'}")

    # Research (each degrades independently)
    match_date = fx.kickoff[:10]
    have_lineups = any((p.get("meta", {}) or {}).get("position") for p in participants)
    try:
        fx.web_research = web_search.gather_research(
            fx.home_name, fx.away_name, match_date, have_confirmed_lineups=have_lineups)
    except Exception as exc:
        print(f"  [live] web research failed: {exc!r}")
        fx.web_research = {"total_results": 0, "sources": []}
    try:
        fx.reddit_bundle = reddit_sentiment.get_sentiment_bundle(fx.home_name, fx.away_name)
    except Exception as exc:
        print(f"  [live] reddit failed: {exc!r}")
        fx.reddit_bundle = {"threads_found": 0, "top_comments": []}

    # Structured digests (Sportmonks via fixture id, Supabase via names)
    ctx = fixture_bundle.build_context(
        fx.home_name, fx.away_name, fx.home_code, fx.away_code,
        sportmonks_fixture_id=fixture_id, fixture_name=fx.fixture_name,
        match_date=match_date)
    fx.sm_digest = ctx.get("sportmonks_digest")
    fx.sb_digest = ctx.get("supabase_digest")
    fx.bz_digest = ctx.get("bzzoiro_digest")
    fx.odds2prob_digest = odds2prob.from_fixture(fixture)
    if fx.odds2prob_digest.get("available"):
        probs = fx.odds2prob_digest.get("probabilities") or {}
        print(f"  [live] odds2prob: { {k: round(v, 3) for k, v in probs.items()} }")
    else:
        print(f"  [live] odds2prob unavailable: {fx.odds2prob_digest.get('reason')}")

    fx.independent_forecast = _build_independent_forecast_snapshot(fx, fixture)
    fx.forecast_snapshot_id = fx.independent_forecast["forecast_id"]
    fx.market_probabilities = _market_probabilities_by_code(fx)
    metrics.log_event(
        "forecast_snapshot",
        fixture_id=fixture_id, window=fx.window, fixture_name=fx.fixture_name,
        forecast_id=fx.forecast_snapshot_id,
        independent_probabilities=fx.independent_forecast.get("probabilities_by_code"),
        market_probabilities=fx.market_probabilities,
        model_version=fx.independent_forecast.get("model_version"),
        feature_snapshot_hash=fx.independent_forecast.get("feature_snapshot_hash"),
        data_coverage_score=fx.independent_forecast.get("data_coverage_score"),
        warnings=fx.independent_forecast.get("warnings"),
    )

    if fx.moneyline:
        try:
            fx.pm_digest_result = llm.digest_polymarket(json.dumps(fx.moneyline))
        except Exception as exc:
            print(f"  [live] polymarket digest failed: {exc!r}")
    if fx.pm_digest_result is None:
        r = _EmptyResult()
        r.parsed = {"data_availability": "no_market", "implied_win_prob": None,
                    "execution_handles": None, "market_handle": None}
        fx.pm_digest_result = r

    # The council (Scout → Analyst → Devil → Judge + grounding)
    deterministic_context = _deterministic_context_for_council(fx, fixture)
    cr = council.run_council(
        fx.fixture_name, fx.home_code, fx.away_code, fx.home_name, fx.away_name,
        fx.kickoff, fx.sm_digest, fx.sb_digest, fx.pm_digest_result.parsed,
        fx.web_research, fx.reddit_bundle,
        deterministic_context=deterministic_context,
        bz_digest=fx.bz_digest,
    )
    fx.cr = cr
    fx.social_pulse = cr.social_pulse or {}
    fx.probabilities = cr.probabilities
    fx.outcome = cr.outcome
    fx.probability = float(cr.probability)
    fx.confidence = cr.confidence
    fx.scout_flags = cr.scout_flags
    fx.market_adjusted_probabilities = dict(fx.probabilities)
    signal_coverage = _signal_coverage(fx)
    fx.grounding = {"council": cr.grounding, "deterministic_context": deterministic_context,
                    "independent_forecast": fx.independent_forecast,
                    "odds2prob": fx.odds2prob_digest,
                    "market_probabilities": fx.market_probabilities,
                    "market_adjusted_probabilities": fx.market_adjusted_probabilities,
                    "signal_coverage": signal_coverage}
    metrics.log_event(
        "signal_coverage", fixture_id=fixture_id, window=fx.window,
        fixture_name=fx.fixture_name, **signal_coverage)
    active = [k for k, v in signal_coverage.items() if v]
    print(f"  [live] council signals: {', '.join(active) or 'none'}")
    fx.summary = cr.council_summary
    fx.engine = "council_with_deterministic_v2"
    print(f"  [live] council+deterministic_v2: {fx.outcome} @ {fx.probability:.1%} "
          f"({fx.confidence})  probs={ {k: round(v, 3) for k, v in fx.probabilities.items()} }")
    return fx


# ── Shared brain: HT ────────────────────────────────────────────────────────

def gather_halftime(fixture_id: int, prematch_note: dict | None = None) -> Forecast:
    fx = Forecast(fixture_id=fixture_id, window="HT")

    fixture = sportmonks.get_fixture(fixture_id)
    participants = fixture.get("participants") or []
    home = next((p for p in participants if p.get("meta", {}).get("location") == "home"), {})
    away = next((p for p in participants if p.get("meta", {}).get("location") == "away"), {})
    fx.home_code = fifa_code(home.get("short_code"), "HOME")
    fx.away_code = fifa_code(away.get("short_code"), "AWAY")
    fx.home_name = home.get("name", fx.home_code)
    fx.away_name = away.get("name", fx.away_code)
    fx.fixture_name = fixture.get("name", f"{fx.home_name} vs {fx.away_name}")
    fx.kickoff = str(fixture.get("starting_at", ""))

    ht_stats_sm = {}
    try:
        ht_stats_sm = sportmonks.extract_ht_stats(fixture)
    except Exception as exc:
        print(f"  [live] HT stats extract failed: {exc!r}")
    ht_snapshot, ht_score = [], []
    try:
        ht_snapshot = supabase_client.get_ht_snapshot(fixture_id)
        ht_score = supabase_client.get_ht_score(fixture_id)
    except Exception as exc:
        print(f"  [live] HT supabase snapshot failed: {exc!r}")
    fx.ht_context = {"ht_stats_sm": ht_stats_sm, "ht_snapshot": ht_snapshot,
                     "ht_score": ht_score}

    fx.pm_slug, fx.moneyline, fx.mids = _fetch_market(fixture_id)
    fx.market_source = "polymarket" if fx.moneyline else "none"
    if fx.moneyline:
        fx.moneyline["market_source"] = "polymarket"

    if fx.moneyline:
        try:
            fx.pm_digest_result = llm.digest_polymarket(json.dumps(fx.moneyline))
        except Exception:
            pass
    if fx.pm_digest_result is None:
        r = _EmptyResult()
        r.parsed = {"data_availability": "no_market"}
        fx.pm_digest_result = r

    result = llm.ht_predict(ht_predict_input(
        fx.fixture_name, fx.home_code, fx.away_code,
        prematch_note, ht_snapshot, ht_score, ht_stats_sm))
    fx.ht_pred_result = result
    parsed = result.parsed or {}
    fx.outcome = fifa_code(parsed.get("outcome"), fx.home_code)
    fx.probability = float(parsed.get("probability") or 0.34)
    fx.confidence = parsed.get("confidence_level", "low")
    fx.summary = parsed.get("rationale", "")

    # HT predictor emits a single (outcome, p); rebuild a 3-way distribution by
    # giving the residual mass to the other outcomes proportional to the
    # pre-match forecast (uniform if we have none).
    pre_probs = normalize_probabilities((prematch_note or {}).get("probabilities") or {})
    keys = [fx.home_code, "draw", fx.away_code]
    rest = [k for k in keys if k != fx.outcome]
    rest_pre = [max(float(pre_probs.get(k, 1.0)), 1e-6) for k in rest]
    rest_tot = sum(rest_pre)
    residual = max(0.0, 1.0 - fx.probability)
    fx.probabilities = {fx.outcome: fx.probability}
    for k, w in zip(rest, rest_pre):
        fx.probabilities[k] = residual * w / rest_tot
    fx.engine = "ht_bayesian_llm"
    print(f"  [live] HT update: {fx.outcome} @ {fx.probability:.1%} ({fx.confidence})")
    return fx


# ── Per-agent tail: policy → orders → ledger ───────────────────────────────

def act_for_agent(agent: LiveAgent, fx: Forecast, *, dry_run: bool = False,
                  coordinator: PortfolioCoordinator | None = None) -> dict:
    """
    Run one agent's distinct decision pathway using its Strategy class.
    Returns a summary dict for state/metrics. Never raises (logs + degrades).

    MONK/ANCHOR/HUNTER route their picks through the structured recommendation +
    central portfolio allocator (``coordinator``); BLITZ keeps its existing
    direct order path and is never gated by the allocator.
    """
    profile = agent.profile
    client = ArenaClient(agent.api_key, agent.name)

    # ── Ledger session (this agent's own trace, own key) ─────────────────
    session = LedgerSession(fx.fixture_id, fx.fixture_name, fx.window,
                            api_key=agent.api_key, agent_tag=agent.name)
    rec_trigger = session.trigger("live-runner")
    session.planning(
        goal=f"[{profile.name}] {fx.window} cycle for {fx.fixture_name}: calibrated "
             f"prediction + at most {profile.max_bets_per_window} +EV buy-YES order(s)",
        steps=[
            "Ingest shared data bundle (Sportmonks, Polymarket, BZZOIRO, web, Reddit, Supabase)",
            "Run council forecast and ground it against the bookmaker anchor"
            if fx.window == "PRE_MATCH" else
            "Run Bayesian HT update from live score + xG",
            "Submit the scored prediction",
            "At HT, close existing fixture exposure before opening fresh halftime risk"
            if fx.window == "HT" else
            "Carry existing exposure until settlement unless a later window closes it",
            f"Apply the {profile.name} trading policy ({profile.label})",
            "Place and poll any orders; reflect; submit this trace",
        ],
        contingencies=["Degrade to predict-only when market or wallet is unavailable"],
        upstream_ids=[rec_trigger["record_id"]],
    )

    close_results: list[dict] = []
    if fx.window == "HT":
        if dry_run:
            close_results.append({"status": "dry_run", "fixture_id": fx.fixture_id})
        else:
            try:
                close_results = client.close_fixture_orders(fx.fixture_id)
            except Exception as exc:
                close_results = [{"status": "close_error", "error": repr(exc)}]
        if close_results:
            print(f"  [{agent.name}] HT close sweep → {len(close_results)} result(s)")

    # Wallet (a dead wallet ⇒ predict-only, never crash the cycle)
    try:
        wallet = client.wallet()
    except Exception as exc:
        print(f"  [{agent.name}] wallet fetch failed: {exc!r} — predict-only")
        wallet = {"available": 0.0, "locked": 0.0, "address": None}
    available = float(wallet["available"])

    strategy = STRATEGIES.get(agent.name.lower())
    if not strategy:
        print(f"  [{agent.name}] WARNING: unknown strategy, defaulting to blitz")
        strategy = STRATEGIES["blitz"]

    # 1. Build Data Snapshot
    snapshot = FixtureDataSnapshot(
        fixture_id=str(fx.fixture_id),
        fixture_name=fx.fixture_name,
        window=fx.window,
        kickoff=fx.kickoff,
        as_of_timestamp=datetime.now(timezone.utc),
        home_code=fx.home_code,
        away_code=fx.away_code,
        home_name=fx.home_name,
        away_name=fx.away_name,
        sportmonks=None,
        supabase=None,
        bzzoiro=None,
        web=None,
        reddit=None,
        social=None,
        football_context={
            "sportmonks_digest": fx.sm_digest,
            "supabase_digest": fx.sb_digest,
            "bzzoiro_digest": fx.bz_digest,
            "odds2prob_digest": fx.odds2prob_digest,
            "deterministic_model": fx.deterministic_model,
            # Codes + frozen independent snapshot let each agent map outcomes and
            # gate conservative edge against the SAME market-blind belief.
            "home_code": fx.home_code,
            "away_code": fx.away_code,
            "independent_forecast": fx.independent_forecast,
            "forecast_snapshot_id": fx.forecast_snapshot_id,
            "scout_flags": fx.scout_flags,
            # The SHARED goated council belief — all agents bet conviction off
            # this (BZZOIRO + web + Reddit + Grok), differing only by risk.
            "council_forecast": {
                "probabilities": {
                    "home": float(fx.probabilities.get(fx.home_code, 0.0) or 0.0),
                    "draw": float(fx.probabilities.get("draw", 0.0) or 0.0),
                    "away": float(fx.probabilities.get(fx.away_code, 0.0) or 0.0),
                },
                "confidence": confidence_to_num(fx.confidence),
            },
        },
        live_context=fx.ht_context,
        market_context=None,
        snapshot_id=f"snap_{fx.fixture_id}_{fx.window}",
        snapshot_hash="",
    )

    # 2. Build Agent Data View
    view = strategy.build_data_view(snapshot, None)
    
    # Adapter for BlitzLegacyDataView
    if agent.name.lower() == "blitz":
        from models.forecast_contracts import MatchForecast
        from harness.profiles import get_profile
        # Blitz uses the legacy forecast contract
        lf = MatchForecast(
            fixture_id=str(fx.fixture_id),
            as_of_timestamp=datetime.now(timezone.utc),
            home_probability=fx.probabilities.get(fx.home_code, 0.40),
            draw_probability=fx.probabilities.get("draw", 0.28),
            away_probability=fx.probabilities.get(fx.away_code, 0.32),
            home_lower_bound=fx.probabilities.get(fx.home_code, 0.40),
            draw_lower_bound=fx.probabilities.get("draw", 0.28),
            away_lower_bound=fx.probabilities.get(fx.away_code, 0.32),
            home_upper_bound=fx.probabilities.get(fx.home_code, 0.40),
            draw_upper_bound=fx.probabilities.get("draw", 0.28),
            away_upper_bound=fx.probabilities.get(fx.away_code, 0.32),
            confidence=confidence_to_num(fx.confidence),
            data_coverage_score=_coverage_score(fx),
            model_version=fx.engine,
            feature_snapshot_hash="",
            evidence_ids=_evidence_ids(fx),
            warnings=[],
        )
        lf.active_components = []
        lf.component_weights = {}
        view.legacy_forecast = lf

    # 3. Build Forecast
    agent_forecast = strategy.build_forecast(view)

    # 4. Generate Candidates
    market = MarketContext(
        observed_at=datetime.now(timezone.utc),
        polymarket=None,
        kalshi=None,
        bookmaker_consensus=None,
        bookmaker_comparison=None,
        devigged_probabilities=fx.mids,
        best_bid=fx.mids, # stub
        best_ask=fx.mids, # stub
        midpoint=fx.mids,
        expected_fill_price=fx.mids, # stub
        movement={},
        dispersion={},
        overround=None,
    ) if fx.mids else None
    
    candidates = strategy.generate_candidates(agent_forecast, view, market)

    # 5. Generate Recommendations
    recs = strategy.generate_recommendations(candidates, agent_forecast, view, market, available)

    # 6. Coordinate (Portfolio)
    abstentions = sum(1 for r in recs if not getattr(r, "should_trade", True))
    reco_stats = {"recommendations": len(recs), "abstentions": abstentions,
                  "duplicate_recommendations": 0, "rejected": 0,
                  "blitz_draw_candidates_removed": int(getattr(strategy, "draws_removed", 0) or 0)}
    recommendations_payload: list[dict] = []
    rejected_payload: list[dict] = []
    skip_reasons: list[str] = []
    picks = []

    def _slot_for_code(code: str) -> str:
        if str(code).lower() in ("draw", "tie", "x"):
            return "draw"
        return "home" if code == fx.home_code else "away"

    if agent.name.lower() in ["monk", "anchor", "hunter"]:
        # MONK/ANCHOR/HUNTER route through the central allocator. The allocator
        # only accepts should_trade recommendations; abstentions are surfaced as
        # rejections carrying the structured abstain reason.
        alloc = coordinator.allocate(recs) if coordinator else None
        if alloc:
            reco_stats["duplicate_recommendations"] = alloc.duplicate_recommendations
            reco_stats["rejected"] = len(alloc.rejected)
            for rej in alloc.rejected:
                rejected_payload.append(rej)
                skip_reasons.append(f"portfolio {rej.get('reason')}")
            for rec in alloc.accepted:
                recommendations_payload.append(rec.to_dict())
                # Adapt the structured recommendation into the legacy pick shape.
                # ``recommended_stake`` is a fraction of bankroll; convert to USD.
                class PickAdapter:
                    pass
                p = PickAdapter()
                p.code = rec.outcome
                p.slot = _slot_for_code(rec.outcome)
                p.stake_usd = round(float(rec.recommended_stake or 0.0) * max(0.0, available), 2)
                p.limit_price = rec.maximum_acceptable_price or round((rec.expected_fill_price or 0.5) + 0.02, 2)
                p.our_prob = rec.probability_mean
                p.entry_price = rec.expected_fill_price
                p.to_dict = lambda self=p: {"code": self.code, "stake": self.stake_usd}
                picks.append(p)
    elif agent.name.lower() == "blitz":
        # BLITZ keeps its direct order path: it is NEVER routed through the
        # allocator. Stakes come straight from its sized legacy picks.
        legacy_by_slot = {pk.slot: pk for pk in getattr(strategy, "legacy_picks", [])}
        for cand in candidates:
            class PickAdapter:
                pass
            p = PickAdapter()
            slot = cand.outcome.split("_")[0]
            p.code = "draw" if slot == "draw" else (fx.home_code if slot == "home" else fx.away_code)
            p.slot = slot
            sized = legacy_by_slot.get(slot)
            p.stake_usd = round(min(sized.stake_usd if sized else profile.max_bet_usd,
                                    max(0.0, available)), 2)
            p.limit_price = sized.limit_price if sized else round((cand.expected_fill_price or 0.5) + 0.02, 2)
            p.our_prob = cand.probability_mean
            p.entry_price = cand.expected_fill_price
            p.to_dict = lambda self=p: {"code": self.code, "stake": self.stake_usd}
            picks.append(p)

    # Per-cycle wallet cap: min($5, available − 5¢) across ALL orders.
    cycle_cap = min(
        profile.max_bet_usd * max(1, int(profile.max_bets_per_window)),
        max(0.0, available - WALLET_BUFFER_USD),
    )
    spent = 0.0
    capped: list = []
    for p in picks:
        room = round(cycle_cap - spent, 2)
        if room < bet_policy.MIN_ORDER_USD:
            skip_reasons.append(f"{p.slot}: cycle cap ${cycle_cap:.2f} exhausted")
            continue
        p.stake_usd = min(p.stake_usd, room)
        spent += p.stake_usd
        capped.append(p)
    picks = capped
    gate_info = {}

    # Tool-call records for the shared bundle (the calls genuinely happened
    # for this cycle; each agent reports them in its own trace).
    rec_sm = session.tool_call(
        name="sportmonks", endpoint=f"/v3/football/fixtures/{fx.fixture_id}",
        description="Fixture detail with predictions/odds/xG",
        input_payload={"fixture_id": fx.fixture_id},
        output_payload={"fixture": fx.fixture_name, "kickoff": fx.kickoff,
                        "sm_digest": fx.sm_digest,
                        "odds2prob": fx.odds2prob_digest},
        success=fx.sm_digest is not None,
        upstream_ids=[rec_trigger["record_id"]])
    rec_pm = session.tool_call(
        name="polymarket", endpoint="/proxy/polymarket-gamma+clob",
        description="Moneyline market + CLOB mids",
        input_payload={"slug": fx.pm_slug},
        output_payload={"market_source": fx.market_source, "mids": fx.mids},
        success=fx.moneyline is not None,
        upstream_ids=[rec_trigger["record_id"]])
    rec_odds2prob = session.tool_call(
        name="odds2prob", endpoint="/convert",
        description="Calibrated de-vigged bookmaker 1X2 probabilities",
        input_payload=(fx.odds2prob_digest or {}).get("input_odds"),
        output_payload=fx.odds2prob_digest,
        via="external.odds2prob",
        success=bool((fx.odds2prob_digest or {}).get("available")),
        upstream_ids=[rec_sm["record_id"]])
    upstream_data = [rec_sm["record_id"], rec_pm["record_id"], rec_odds2prob["record_id"]]

    if fx.window == "PRE_MATCH" and fx.cr is not None:
        rec_det = session.thinking(
            prompt_system="[DETERMINISTIC_V2] Elo + Poisson + market-prior calibrated ensemble",
            inputs=[{"record_id": rec_sm["record_id"], "payload": fx.sm_digest},
                    {"record_id": rec_odds2prob["record_id"], "payload": fx.odds2prob_digest},
                    {"record_id": rec_pm["record_id"], "payload": fx.pm_digest_result.parsed}],
            output_payload=fx.deterministic_model,
            upstream_ids=[rec_sm["record_id"], rec_odds2prob["record_id"], rec_pm["record_id"]])
        rec_web = session.tool_call(
            name="web_search", endpoint="search",
            description="Injury/lineup/preview research",
            input_payload={"home": fx.home_name, "away": fx.away_name},
            output_payload=fx.web_research, via="external.web",
            success=(fx.web_research or {}).get("total_results", 0) > 0,
            upstream_ids=[rec_trigger["record_id"]])
        rec_reddit = session.tool_call(
            name="reddit", endpoint="r/soccer/search.json",
            description="Crowd sentiment bundle",
            input_payload={"query": f"{fx.home_name} {fx.away_name}"},
            output_payload=fx.reddit_bundle, via="external.reddit",
            success=bool((fx.reddit_bundle or {}).get("top_comments")),
            upstream_ids=[rec_trigger["record_id"]])
        upstream_data += [rec_web["record_id"], rec_reddit["record_id"]]

        cr = fx.cr
        rec_scout = session.thinking(
            prompt_system="[SCOUT_SYS] Severity-ranked triage of news/sentiment/pulse",
            inputs=[{"record_id": rec_web["record_id"], "payload": fx.web_research},
                    {"record_id": rec_reddit["record_id"], "payload": fx.reddit_bundle},
                    {"record_id": rec_det["record_id"], "payload": fx.deterministic_model},
                    {"payload": cr.social_pulse}],
            output_payload=cr.scout.parsed if cr.scout else {},
            provider=cr.scout.provider if cr.scout else "",
            model_name=cr.scout.model if cr.scout else "",
            internal_reasoning=cr.scout.thinking if cr.scout else "",
            upstream_ids=[rec_web["record_id"], rec_reddit["record_id"]])
        rec_analyst = session.thinking(
            prompt_system="[ANALYST_SYS] Market-blind base-rate forecast vs anchor",
            inputs=[{"record_id": rec_sm["record_id"], "payload": fx.sm_digest},
                    {"payload": fx.sb_digest},
                    {"record_id": rec_det["record_id"], "payload": fx.deterministic_model},
                    {"record_id": rec_scout["record_id"],
                     "payload": cr.scout.parsed if cr.scout else {}}],
            output_payload=cr.analyst.parsed if cr.analyst else {},
            provider=cr.analyst.provider if cr.analyst else "",
            model_name=cr.analyst.model if cr.analyst else "",
            internal_reasoning=cr.analyst.thinking if cr.analyst else "",
            upstream_ids=[rec_sm["record_id"], rec_scout["record_id"]])
        rec_devil = session.thinking(
            prompt_system="[DEVIL_SYS] Attack the weakest assumption",
            inputs=[{"record_id": rec_analyst["record_id"],
                     "payload": cr.analyst.parsed if cr.analyst else {}}],
            output_payload=cr.devil.parsed if cr.devil else {},
            provider=cr.devil.provider if cr.devil else "",
            model_name=cr.devil.model if cr.devil else "",
            internal_reasoning=cr.devil.thinking if cr.devil else "",
            upstream_ids=[rec_analyst["record_id"]])
        rec_final = session.thinking(
            prompt_system="[JUDGE_SYS] Calibrated synthesis vs market + grounding pass",
            inputs=[{"record_id": rec_analyst["record_id"],
                     "payload": cr.analyst.parsed if cr.analyst else {}},
                    {"record_id": rec_devil["record_id"],
                     "payload": cr.devil.parsed if cr.devil else {}},
                    {"record_id": rec_pm["record_id"],
                     "payload": fx.pm_digest_result.parsed},
                    {"record_id": rec_det["record_id"], "payload": fx.deterministic_model}],
            output_payload={"probabilities": fx.probabilities, "outcome": fx.outcome,
                            "probability": fx.probability, "confidence": fx.confidence,
                            "grounding": fx.grounding, "summary": fx.summary},
            provider=cr.judge.provider if cr.judge else "",
            model_name=cr.judge.model if cr.judge else "",
            internal_reasoning=cr.judge.thinking if cr.judge else "",
            upstream_ids=[rec_analyst["record_id"], rec_devil["record_id"],
                          rec_pm["record_id"], rec_det["record_id"]])
    else:
        # HT trace: one Thinking record for the Bayesian update.
        r = fx.ht_pred_result or _EmptyResult()
        rec_final = session.thinking(
            prompt_system="[HT_PREDICT_SYS] Bayesian half-time update (score + xG)",
            inputs=[{"record_id": rec_sm["record_id"], "payload": fx.ht_context},
                    {"record_id": rec_pm["record_id"],
                     "payload": fx.pm_digest_result.parsed}],
            output_payload={"probabilities": fx.probabilities, "outcome": fx.outcome,
                            "probability": fx.probability, "confidence": fx.confidence,
                            "summary": fx.summary},
            provider=getattr(r, "provider", ""),
            model_name=getattr(r, "model", ""),
            internal_reasoning=getattr(r, "thinking", ""),
            upstream_ids=upstream_data)

    # ── Prediction (scored even when we don't bet) ────────────────────────
    session.acting_prediction(
        outcome=fx.outcome, probability=fx.probability,
        upstream_ids=[rec_final["record_id"]])

    # ── Decision record ───────────────────────────────────────────────────
    rec_decision = session.thinking(
        prompt_system="[DETERMINISTIC] Profile policy + EV ranking + gates + sizing (no LLM)",
        inputs=[{"record_id": rec_final["record_id"],
                 "payload": {"probabilities": fx.probabilities,
                             "confidence": fx.confidence}},
                {"record_id": rec_pm["record_id"], "payload": fx.mids}],
        output_payload={
            "profile": profile.name,
            "profile_thresholds": {
                "min_edge_vs_fair": profile.min_edge_vs_fair,
                "min_ev_per_dollar": profile.min_ev_per_dollar,
                "min_confidence": profile.min_confidence,
                "max_entry_price": profile.max_entry_price,
                "kelly_fraction": profile.kelly_fraction,
                "max_bet_usd": profile.max_bet_usd,
                "scout_veto": profile.skip_on_high_scout_flag,
            },
            "market_source": fx.market_source,
            "forecast_snapshot_id": fx.forecast_snapshot_id,
            "independent_probabilities": fx.independent_forecast.get("probabilities_by_code"),
            "market_probabilities": fx.market_probabilities,
            "market_adjusted_probabilities": fx.market_adjusted_probabilities or fx.probabilities,
            "wallet_available": available,
            "cycle_cap_usd": cycle_cap,
            "picks": [p.to_dict() for p in picks],
            "recommendations": recommendations_payload,
            "rejected_recommendations": rejected_payload,
            "recommendation_stats": reco_stats,
            "halftime_close_results": close_results,
            "skip_reasons": skip_reasons,
            "gates": gate_info,
            "grounding": fx.grounding,
        },
        upstream_ids=[rec_final["record_id"], rec_pm["record_id"]])

    # ── Orders ────────────────────────────────────────────────────────────
    order_results: list[dict] = []
    for close in close_results:
        status = close.get("status") or close.get("final_status") or "pending"
        exec_status = "confirmed" if str(status).lower() in ("closed", "filled", "success", "ok") else (
            "simulated" if status == "dry_run" else "pending"
        )
        session.acting_order(
            direction="close", outcome=str(close.get("team_code") or close.get("outcome") or "fixture"),
            size_usdc=0.0, limit_price=0.0,
            order_payload={"fixture_id": str(fx.fixture_id), **close},
            execution_status=exec_status,
            execution_id=close.get("order_id"),
            action_type="close_order",
            upstream_ids=[rec_decision["record_id"]])
    for p in picks:
        if dry_run:
            order_results.append({"pick": p.to_dict(), "status": "dry_run",
                                  "execution": {"partial_fill_state": "simulated"}})
            session.acting_order(
                direction="long", outcome=fifa_code(p.code), size_usdc=p.stake_usd,
                limit_price=p.limit_price,
                order_payload={"fixture_id": str(fx.fixture_id), "team_code": fifa_code(p.code),
                               "usd_size": f"{p.stake_usd:.2f}",
                               "limit_price": p.limit_price, "dry_run": True,
                               "skip_reasons": skip_reasons},
                execution_status="simulated",
                upstream_ids=[rec_decision["record_id"]])
            continue
        resp = client.place_order(fx.fixture_id, p.code, p.stake_usd, p.limit_price)
        submitted_ok = isinstance(resp, dict) and "order_id" in resp
        poll = client.poll_order(resp["order_id"]) if submitted_ok else {}
        exec_status = ArenaClient.execution_status_for(
            poll.get("final_status"), poll.get("tx_hash"), submitted_ok)
        print(f"  [{agent.name}] order {fifa_code(p.code)} ${p.stake_usd:.2f} @ ≤{p.limit_price:.2f} "
              f"→ {poll.get('final_status') or resp.get('status')}")
        order_results.append({"pick": p.to_dict(), "order_id": resp.get("order_id"),
                              "status": poll.get("final_status") or resp.get("status"),
                              "reject_reason": poll.get("reject_reason"),
                              "tx_hash": poll.get("tx_hash"),
                              "filled_usdc": poll.get("filled_usdc"),
                              "execution": poll.get("fill_report") or {},
                              "exec_status": exec_status})
        session.acting_order(
            direction="long", outcome=p.code, size_usdc=p.stake_usd,
            limit_price=p.limit_price,
            order_payload={**(resp.get("payload") or {}),
                           "execution": poll.get("fill_report") or {}},
            execution_status=exec_status,
            execution_id=resp.get("order_id") if submitted_ok else None,
            upstream_ids=[rec_decision["record_id"]])

    # ── Reflection ────────────────────────────────────────────────────────
    session.reflecting(
        inputs=[{"record_id": rec_final["record_id"],
                 "payload": {"outcome": fx.outcome, "probability": fx.probability}},
                {"record_id": rec_decision["record_id"],
                 "payload": {"n_orders": len(order_results),
                             "skip_reasons": skip_reasons}}],
        output_payload={
            "fixture": fx.fixture_name, "window": fx.window, "profile": profile.name,
            "prediction": {"outcome": fx.outcome, "probability": fx.probability,
                           "confidence": fx.confidence},
            "traded": bool(order_results),
            "halftime_closed": close_results,
            "orders": [{k: o.get(k) for k in ("order_id", "status", "exec_status")}
                       for o in order_results],
            "grounding_flags": (fx.grounding or {}).get("sanity_flags"),
            "what_to_improve": (
                "Compare this forecast against the de-vigged close and the result "
                "at settlement; revisit the policy bars if skip_reasons dominated."),
        },
        upstream_ids=[rec_final["record_id"], rec_decision["record_id"]])

    # ── Submit ledger ─────────────────────────────────────────────────────
    ledger_result: dict = {}
    if dry_run:
        v = session.validate()
        ledger_result = {"dry_run": True,
                         "validate": (v or {}).get("valid", "endpoint-unavailable"),
                         "records_built": session.record_count()}
    else:
        try:
            resp = session.submit()
            ledger_result = {
                "stored": len(resp.get("records") or []),
                "errors": resp.get("errors") or [],
                "status": resp.get("status"),
            }
            if ledger_result["errors"]:
                print(f"  [{agent.name}] ledger errors: "
                      f"{json.dumps(ledger_result['errors'][:3], default=str)[:300]}")
        except Exception as exc:
            print(f"  [{agent.name}] ledger submit FAILED: {exc!r}")
            ledger_result = {"error": repr(exc),
                             "records_built": session.record_count()}

    summary = {
        "prediction": {"outcome": fx.outcome, "probability": fx.probability,
                       "probabilities": fx.probabilities,
                       "confidence": fx.confidence,
                       "engine": fx.engine},
        "wallet_available": available,
        "n_picks": len(picks),
        "halftime_close_results": close_results,
        "orders": order_results,
        "skip_reasons": skip_reasons,
        "recommendation_stats": reco_stats,
        "ledger": ledger_result,
        "session_id": session.session_id,
    }
    metrics.log_event(
        "agent_window",
        fixture_id=fx.fixture_id, window=fx.window, fixture_name=fx.fixture_name,
        agent=agent.name, profile=profile.name,
        engine=fx.engine,
        probabilities=fx.probabilities, confidence=fx.confidence,
        forecast_snapshot_id=fx.forecast_snapshot_id,
        independent_probabilities=fx.independent_forecast.get("probabilities_by_code"),
        market_probabilities=fx.market_probabilities,
        market_adjusted_probabilities=fx.market_adjusted_probabilities or fx.probabilities,
        market_source=fx.market_source, mids=fx.mids,
        recommendations=len(recommendations_payload),
        abstentions=reco_stats["abstentions"],
        duplicate_recommendations=reco_stats["duplicate_recommendations"],
        rejected_recommendations=reco_stats["rejected"],
        blitz_draw_candidates_removed=reco_stats["blitz_draw_candidates_removed"],
        grounding=fx.grounding, **{k: summary[k] for k in
                                   ("prediction", "wallet_available", "orders",
                                    "skip_reasons", "ledger")})
    return summary


# ── The full window cycle ───────────────────────────────────────────────────

def run_window_cycle(fixture_id: int, window: str, agents: list[LiveAgent],
                     *, prematch_note: dict | None = None,
                     dry_run: bool = False) -> dict:
    """Shared brain once, then every agent's tail. Returns per-agent summaries."""
    if window == "PRE_MATCH":
        fx = gather_prematch(fixture_id)
    else:
        fx = gather_halftime(fixture_id, prematch_note)

    metrics.log_event(
        "forecast",
        fixture_id=fixture_id, window=window, fixture_name=fx.fixture_name,
        kickoff=fx.kickoff, home_code=fx.home_code, away_code=fx.away_code,
        engine=fx.engine,
        probabilities=fx.probabilities, outcome=fx.outcome,
        probability=fx.probability, confidence=fx.confidence,
        forecast_snapshot_id=fx.forecast_snapshot_id,
        independent_probabilities=fx.independent_forecast.get("probabilities_by_code"),
        market_probabilities=fx.market_probabilities,
        market_adjusted_probabilities=fx.market_adjusted_probabilities or fx.probabilities,
        market_source=fx.market_source, mids=fx.mids, grounding=fx.grounding,
        scout_flags=fx.scout_flags, summary=fx.summary)

    # One central allocation book per window, shared by the coordinated agents
    # (MONK/ANCHOR/HUNTER) as they run in sequence. BLITZ ignores it.
    coordinator = PortfolioCoordinator(limits=PortfolioLimits(
        max_fixture_exposure=config.MAX_FIXTURE_EXPOSURE,
        max_outcome_exposure=config.MAX_OUTCOME_EXPOSURE,
        max_ultra_tail_exposure=config.MAX_ULTRA_TAIL_EXPOSURE,
        max_daily_drawdown=config.MAX_DAILY_DRAWDOWN,
    ))

    results: dict[str, dict] = {}
    for agent in agents:
        try:
            results[agent.name] = act_for_agent(agent, fx, dry_run=dry_run,
                                                coordinator=coordinator)
        except Exception as exc:
            print(f"  [{agent.name}] agent tail FAILED: {exc!r}")
            metrics.log_event("error", fixture_id=fixture_id, window=window,
                              agent=agent.name, error=repr(exc))
            results[agent.name] = {"error": repr(exc)}
    return {"fixture_name": fx.fixture_name, "window": window,
            "forecast": {"outcome": fx.outcome, "probability": fx.probability,
                         "probabilities": fx.probabilities,
                         "confidence": fx.confidence,
                         "engine": fx.engine,
                         "forecast_snapshot_id": fx.forecast_snapshot_id,
                         "independent_probabilities": fx.independent_forecast.get("probabilities_by_code"),
                         "market_probabilities": fx.market_probabilities,
                         "market_adjusted_probabilities": (
                             fx.market_adjusted_probabilities or fx.probabilities)},
            "agents": results}
