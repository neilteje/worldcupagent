"""Agent recommendation contracts, conservative-edge gating, and allocation."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from betting import recommendation as bet_reco
from betting.policy import SizedPick, suppress_blitz_draw_picks
from betting.portfolio import (
    PortfolioCoordinator,
    PortfolioLimits,
    allocate_recommendations,
)
from betting.recommendation import (
    REASON_BELOW_MIN_STAKE,
    REASON_CONSERVATIVE_EDGE,
    REASON_EXPECTED_FILL_UNAVAILABLE,
    REASON_INSUFFICIENT_SIGNALS,
    REASON_ULTRA_TAIL_VALIDATION,
    AgentEdgeThresholds,
    build_recommendation,
)
from harness.profiles import get_profile
from models.forecast_contracts import AgentRecommendation

NOW = datetime(2026, 6, 14, tzinfo=timezone.utc)


def _snap(code="AAA", mean=0.60, lower=0.55, upper=0.65,
          coverage=0.75, conf=0.70, evidence=("sportmonks_digest", "supabase_digest")):
    return {
        "probabilities_by_code": {code: mean},
        "lower_bounds_by_code": {code: lower},
        "upper_bounds_by_code": {code: upper},
        "data_coverage_score": coverage,
        "confidence": conf,
        "evidence_ids": list(evidence),
        "forecast_id": "fc-1",
    }


def _pick(code="AAA", entry=0.45, stake=2.0, limit=0.47, our_prob=0.60):
    return SizedPick(
        slot="home", code=code, stake_usd=stake, entry_price=entry,
        limit_price=limit, our_prob=our_prob, fair_prob=0.50,
        edge_vs_fair=0.12, ev_per_dollar=0.30, kelly_usd=stake,
    )


def _build(agent, pick, snap, *, bankroll=100.0, thresholds=None):
    return build_recommendation(
        agent, pick,
        fixture_id="fixture-1", bankroll=bankroll,
        forecast_snapshot=snap, forecast_id=snap.get("forecast_id"),
        evidence_ids=snap.get("evidence_ids"),
        data_coverage_score=snap.get("data_coverage_score"),
        confidence=snap.get("confidence"),
        thresholds=thresholds, now=NOW,
    )


# ── recommendation creation ──────────────────────────────────────────────────

# mean 0.65 / lower 0.60 / entry 0.45 → conservative 0.11, clears MONK's 0.08 too.
_STRONG = dict(mean=0.65, lower=0.60, upper=0.70)


@pytest.mark.parametrize("agent", ["monk", "anchor"])
def test_monk_anchor_recommendation_creation(agent):
    rec = _build(agent, _pick(), _snap(**_STRONG))
    assert isinstance(rec, AgentRecommendation)
    assert rec.should_trade is True
    assert rec.abstain_reason is None
    assert rec.forecast_id == "fc-1"
    assert rec.outcome == "AAA"
    assert rec.probability_lower_bound == pytest.approx(0.60)
    assert rec.probability_upper_bound == pytest.approx(0.70)
    assert rec.expected_fill_price == pytest.approx(0.45)
    assert rec.gross_edge == pytest.approx(0.20)
    assert rec.conservative_edge == pytest.approx(0.11)        # 0.60 - 0.45 - 0.04
    assert rec.expected_value_after_costs == pytest.approx(0.16)
    assert rec.recommended_stake == pytest.approx(0.02)        # $2 / $100 bankroll
    assert rec.maximum_acceptable_price == pytest.approx(0.47)
    assert rec.evidence_ids == ["sportmonks_digest", "supabase_digest"]
    assert rec.correlation_key


def test_hunter_recommendation_creation_skew():
    snap = _snap(mean=0.60, lower=0.55, upper=0.65)
    rec = _build("hunter", _pick(entry=0.35, our_prob=0.60), snap)
    assert rec.should_trade is True
    assert rec.signal_type == "skew_tail"
    assert rec.conservative_edge == pytest.approx(0.16)
    assert rec.expected_value_after_costs == pytest.approx(0.21)


# ── conservative-edge thresholding ───────────────────────────────────────────

def test_conservative_edge_threshold_abstains():
    # lower bound only 0.50 → conservative edge 0.50-0.45-0.04 = 0.01 < anchor 0.05
    rec = _build("anchor", _pick(), _snap(lower=0.50))
    assert rec.should_trade is False
    assert rec.abstain_reason == REASON_CONSERVATIVE_EDGE
    # the edge accounting is still populated for the ledger
    assert rec.conservative_edge == pytest.approx(0.01)


# ── below-minimum stake abstention ───────────────────────────────────────────

def test_below_minimum_stake_abstains():
    rec = _build("anchor", _pick(stake=0.50), _snap())
    assert rec.should_trade is False
    assert rec.abstain_reason == REASON_BELOW_MIN_STAKE


def test_expected_fill_unavailable_abstains():
    rec = _build("anchor", _pick(entry=0.0), _snap())
    assert rec.should_trade is False
    assert rec.abstain_reason == REASON_EXPECTED_FILL_UNAVAILABLE


# ── generic optional gates still available via explicit thresholds ──────────

def test_optional_ultra_tail_gate_via_explicit_thresholds():
    # The generic ultra-tail / signal gates are no longer part of any agent's
    # DEFAULT thresholds (conviction HUNTER dropped them), but the mechanism is
    # still available when a caller passes them explicitly.
    snap = _snap(mean=0.35, lower=0.30, upper=0.40)
    th = AgentEdgeThresholds(
        signal_type="skew_tail", min_conservative_edge=0.08,
        min_ev_after_costs=0.12, min_data_coverage=0.25,
        min_independent_signals=2, ultra_tail_price=0.05,
        ultra_tail_validation_enabled=False,
    )
    rec = _build("hunter", _pick(entry=0.03, our_prob=0.35), snap, thresholds=th)
    assert rec.abstain_reason == REASON_ULTRA_TAIL_VALIDATION


def test_hunter_default_thresholds_require_signal():
    snap = _snap(mean=0.60, lower=0.55, upper=0.65, evidence=())
    rec = _build("hunter", _pick(entry=0.45, our_prob=0.60), snap)
    assert rec.should_trade is False
    assert rec.abstain_reason == REASON_INSUFFICIENT_SIGNALS


# ── portfolio dedup + coordinator threading ──────────────────────────────────

def test_coordinator_prevents_duplicate_positions_across_agents():
    coord = PortfolioCoordinator(limits=PortfolioLimits())
    monk = _build("monk", _pick(), _snap(**_STRONG))
    anchor = _build("anchor", _pick(), _snap(**_STRONG))
    assert monk.correlation_key != anchor.correlation_key

    first = coord.allocate([monk])
    second = coord.allocate([anchor])

    assert [r.agent_name for r in first.accepted] == ["monk"]
    assert [r.agent_name for r in second.accepted] == ["anchor"]
    assert coord.duplicate_positions_prevented == 0


# ── allocator cannot veto BLITZ ──────────────────────────────────────────────

def test_allocator_can_gate_blitz_exposure_like_other_agents():
    blitz = AgentRecommendation(
        agent_name="blitz", fixture_id="fixture-1", outcome="AAA",
        should_trade=True, abstain_reason=None,
        probability_mean=0.6, probability_lower_bound=0.5, probability_upper_bound=0.7,
        market_midpoint=0.45, best_ask=0.46, expected_fill_price=0.46,
        gross_edge=0.15, conservative_edge=0.05, expected_value_after_costs=0.11,
        signal_type="value", evidence_ids=["x"], evidence_summary="",
        confidence=0.5, data_coverage_score=0.5,
        recommended_stake=99.0,              # would blow every exposure limit
        maximum_acceptable_price=0.5,
        signal_created_at=NOW, signal_expires_at=None,
        correlation_key="blitz-key", forecast_id="fc-1",
    )
    result = allocate_recommendations([blitz], limits=PortfolioLimits())
    assert result.accepted == []
    assert result.rejected[0]["reason"] == "fixture_exposure_limit"


# ── BLITZ stays off the coordinated path; non-draw picks are untouched ───────

def test_blitz_is_a_coordinated_agent():
    assert "blitz" in bet_reco.COORDINATED_AGENTS


def test_blitz_non_draw_pick_payload_is_unchanged():
    blitz = get_profile("blitz")
    away = _pick(code="BBB")
    before = away.to_dict()
    reasons: list[str] = []
    kept = suppress_blitz_draw_picks(blitz, [away], reasons)
    assert kept == [away]
    assert kept[0].to_dict() == before     # byte-for-byte identical payload
    assert reasons == []
