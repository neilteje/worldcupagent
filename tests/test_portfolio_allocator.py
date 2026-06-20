"""Joint portfolio coordination (spec §22, acceptance #23/#24/#25/#26)."""
from __future__ import annotations

from datetime import datetime, timezone

from betting.portfolio import (
    PortfolioCoordinator, PortfolioLimits, allocate_jointly, allocate_recommendations,
)
from models.forecast_contracts import AgentRecommendation


def _rec(agent, outcome, *, edge=0.08, stake=0.02, fill=0.40, corr=None, fixture="900",
         should_trade=True):
    return AgentRecommendation(
        agent_name=agent, fixture_id=fixture, outcome=outcome,
        should_trade=should_trade, abstain_reason=None if should_trade else "x",
        probability_mean=0.5, probability_lower_bound=0.45, probability_upper_bound=0.55,
        market_midpoint=fill, best_ask=fill, expected_fill_price=fill,
        gross_edge=edge, conservative_edge=edge, expected_value_after_costs=edge,
        signal_type="coordinated", evidence_ids=[], evidence_summary="",
        confidence=0.7, data_coverage_score=0.8,
        recommended_stake=stake, maximum_acceptable_price=fill + 0.02,
        signal_created_at=datetime.now(timezone.utc), signal_expires_at=None,
        correlation_key=corr or f"{agent}:{fixture}:{outcome}", forecast_id="fc",
    )


def test_joint_allocation_invariant_to_input_order():
    recs = [
        _rec("monk", "AAA", edge=0.12, corr="k1"),
        _rec("anchor", "BBB", edge=0.06, corr="k2"),
        _rec("hunter", "draw", edge=0.09, corr="k3"),
    ]
    a = allocate_jointly(recs)
    b = allocate_jointly(list(reversed(recs)))
    keys_a = sorted(r.correlation_key for r in a.accepted)
    keys_b = sorted(r.correlation_key for r in b.accepted)
    assert keys_a == keys_b, "joint allocation must be invariant to input ordering"


def test_identical_cross_wallet_recommendations_are_independent():
    shared = "fixture:900:home"
    recs = [_rec("hunter", "AAA", edge=0.05, corr=shared),
            _rec("monk", "AAA", edge=0.15, corr=shared)]
    res = allocate_jointly(recs)
    assert {rec.agent_name for rec in res.accepted} == {"monk", "hunter"}
    assert res.duplicate_recommendations == 0


def test_duplicate_signals_rejected():
    recs = [_rec("monk", "AAA", corr="dup"), _rec("monk", "AAA", corr="dup")]
    res = allocate_recommendations(recs)
    assert len(res.accepted) == 1
    assert res.duplicate_recommendations == 1
    assert any(r["reason"] == "duplicate_signal" for r in res.rejected)


def test_fixture_exposure_limit():
    limits = PortfolioLimits(max_fixture_exposure=0.03)
    recs = [_rec("monk", "AAA", stake=0.02, corr="k1"),
            _rec("monk", "BBB", stake=0.02, corr="k2")]
    res = allocate_recommendations(recs, limits=limits)
    assert len(res.accepted) == 1
    assert any(r["reason"] == "fixture_exposure_limit" for r in res.rejected)


def test_outcome_exposure_limit():
    limits = PortfolioLimits(max_fixture_exposure=1.0, max_outcome_exposure=0.03)
    # Same outcome, different forecast ids -> distinct correlation keys, same outcome.
    recs = [_rec("monk", "AAA", stake=0.02, corr="k1"),
            _rec("monk", "AAA", stake=0.02, corr="k2")]
    res = allocate_recommendations(recs, limits=limits)
    assert len(res.accepted) == 1
    assert any(r["reason"] == "outcome_exposure_limit" for r in res.rejected)


def test_ultra_tail_exposure_limit():
    limits = PortfolioLimits(max_fixture_exposure=1.0, max_outcome_exposure=1.0,
                             max_ultra_tail_exposure=0.01)
    recs = [_rec("hunter", "AAA", stake=0.008, fill=0.03, corr="t1"),
            _rec("hunter", "BBB", stake=0.008, fill=0.03, corr="t2")]
    res = allocate_recommendations(recs, limits=limits)
    assert len(res.accepted) == 1
    assert any(r["reason"] == "ultra_tail_exposure_limit" for r in res.rejected)


def test_blitz_common_contract_is_exposure_gated():
    recs = [_rec("blitz", "AAA", stake=0.5, corr="b1")]  # huge stake
    res = allocate_recommendations(recs)
    assert res.accepted == []
    assert len(res.observed_only) == 0
    assert res.rejected[0]["reason"] == "fixture_exposure_limit"


def test_gates_evaluated_per_recommendation():
    # One tradable, one abstaining, one duplicate -> each handled independently.
    recs = [
        _rec("monk", "AAA", corr="g1"),
        _rec("anchor", "BBB", corr="g2", should_trade=False),
        _rec("monk", "AAA", corr="g1"),  # duplicate within the same wallet
    ]
    res = allocate_recommendations(recs)
    assert len(res.accepted) == 1
    reasons = {r["reason"] for r in res.rejected}
    assert "duplicate_signal" in reasons
    assert any(r["reason"] in ("x", "agent_abstained") for r in res.rejected)
