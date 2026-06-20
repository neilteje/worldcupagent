from __future__ import annotations

from datetime import datetime, timezone

import pytest

from betting.conservative import ConservativeEdgeConfig, calculate_conservative_edge
from betting.policy import (
    SizedPick,
    is_draw_outcome,
    suppress_blitz_draw_picks,
)
from betting.portfolio import allocate_recommendations
from harness.profiles import get_profile
from live.arena_client import _fill_report
from live.cycle import (
    Forecast,
    _build_independent_forecast_snapshot,
    _coverage_score,
    _evidence_ids,
    _signal_coverage,
)
from models.forecast_contracts import AgentRecommendation, MatchForecast


def _pick(slot: str, code: str) -> SizedPick:
    return SizedPick(
        slot=slot,
        code=code,
        stake_usd=2.0,
        entry_price=0.25,
        limit_price=0.27,
        our_prob=0.35,
        fair_prob=0.22,
        edge_vs_fair=0.13,
        ev_per_dollar=0.40,
        kelly_usd=2.0,
    )


def test_blitz_draw_filter_passes_all_picks_through():
    blitz = get_profile("blitz")
    draw = _pick("draw", "draw")
    away = _pick("away", "BBB")
    before = [p.to_dict() for p in (draw, away)]
    reasons: list[str] = []

    filtered = suppress_blitz_draw_picks(blitz, [draw, away], reasons)

    assert filtered == [draw, away]
    assert [p.to_dict() for p in filtered] == before
    assert reasons == []


def test_blitz_draw_filter_keeps_draw_only_selection():
    blitz = get_profile("blitz")
    reasons: list[str] = []

    filtered = suppress_blitz_draw_picks(blitz, [_pick("draw", "draw")], reasons)

    assert filtered == [_pick("draw", "draw")]
    assert reasons == []


def test_blitz_draw_filter_allows_draw_code_representations():
    blitz = get_profile("blitz")
    reasons: list[str] = []

    filtered = suppress_blitz_draw_picks(blitz, [_pick("home", "X"), _pick("away", "BBB")], reasons)

    assert [p.code for p in filtered] == ["X", "BBB"]
    assert is_draw_outcome(code="tie")
    assert reasons == []


def test_non_blitz_profiles_are_not_draw_filtered():
    hunter = get_profile("hunter")
    draw = _pick("draw", "draw")
    reasons: list[str] = []

    assert suppress_blitz_draw_picks(hunter, [draw], reasons) == [draw]
    assert reasons == []


def test_blitz_profile_core_configuration_is_unchanged():
    blitz = get_profile("blitz")
    assert blitz.min_edge_vs_fair == pytest.approx(0.02)
    assert blitz.min_confidence == pytest.approx(0.35)
    assert blitz.kelly_fraction == pytest.approx(0.65)
    assert blitz.max_bet_usd == pytest.approx(5.0)
    assert blitz.max_bets_per_window == 2
    assert blitz.skip_on_high_scout_flag is False
    assert blitz.apply_confidence_multiplier is False


def test_match_forecast_validates_and_hashes_snapshot():
    forecast = MatchForecast(
        fixture_id="fixture-1",
        as_of_timestamp=datetime.now(timezone.utc),
        home_probability=0.50,
        draw_probability=0.25,
        away_probability=0.25,
        home_lower_bound=0.42,
        draw_lower_bound=0.18,
        away_lower_bound=0.18,
        home_upper_bound=0.58,
        draw_upper_bound=0.32,
        away_upper_bound=0.32,
        confidence=0.60,
        data_coverage_score=0.75,
        model_version="test",
        feature_snapshot_hash="abc",
    )
    original_id = forecast.forecast_id
    forecast.home_probability = 0.49
    assert forecast.compute_forecast_id() != original_id

    with pytest.raises(ValueError):
        MatchForecast(
            fixture_id="bad",
            as_of_timestamp=datetime.now(timezone.utc),
            home_probability=0.60,
            draw_probability=0.30,
            away_probability=0.30,
            home_lower_bound=0.50,
            draw_lower_bound=0.20,
            away_lower_bound=0.20,
            home_upper_bound=0.70,
            draw_upper_bound=0.40,
            away_upper_bound=0.40,
            confidence=0.50,
            data_coverage_score=0.50,
            model_version="test",
            feature_snapshot_hash="abc",
        )


def test_independent_forecast_snapshot_excludes_market_inputs():
    fx = Forecast(
        fixture_id=123,
        window="PRE_MATCH",
        fixture_name="AAA vs BBB",
        kickoff="2026-06-11 18:00:00",
        home_code="AAA",
        away_code="BBB",
        mids={"home": 0.80, "draw": 0.10, "away": 0.10},
        sm_digest={"expected_goals": {"AAA": 1.6, "BBB": 0.8}},
        sb_digest={"teams": {"AAA": {"h2h_wins": 3, "h2h_losses": 1}}},
        web_research={"total_results": 1, "sources": ["example.com"]},
    )
    first = _build_independent_forecast_snapshot(fx, {"stage": {"name": "Group Stage"}})
    fx.mids = {"home": 0.10, "draw": 0.10, "away": 0.80}
    second = _build_independent_forecast_snapshot(fx, {"stage": {"name": "Group Stage"}})

    assert "market" not in first["active_components"]
    assert first["component_weights"].get("market") is None
    assert first["feature_snapshot_hash"] == second["feature_snapshot_hash"]
    assert sum(first["probabilities_by_code"].values()) == pytest.approx(1.0, abs=0.001)
    assert "market_inputs_excluded" in first["warnings"]


def test_bookmaker_conversion_does_not_affect_independent_snapshot_or_evidence():
    fx = Forecast(
        fixture_id=123,
        window="PRE_MATCH",
        fixture_name="AAA vs BBB",
        kickoff="2026-06-11 18:00:00",
        home_code="AAA",
        away_code="BBB",
        sm_digest={"expected_goals": {"AAA": 1.6, "BBB": 0.8}},
        web_research={"total_results": 1, "sources": ["example.com"]},
    )
    first = _build_independent_forecast_snapshot(fx, {"stage": {"name": "Group Stage"}})
    first_coverage = _coverage_score(fx)
    first_ids = _evidence_ids(fx)

    fx.odds2prob_digest = {
        "available": True,
        "probabilities": {"home": 0.05, "draw": 0.05, "away": 0.90},
    }
    second = _build_independent_forecast_snapshot(fx, {"stage": {"name": "Group Stage"}})

    assert second["feature_snapshot_hash"] == first["feature_snapshot_hash"]
    assert _coverage_score(fx) == first_coverage
    assert _evidence_ids(fx) == first_ids
    assert not any("odds2prob" in evidence_id for evidence_id in second["evidence_ids"])


def test_reddit_fallback_comments_count_as_live_signal_and_evidence():
    fx = Forecast(
        fixture_id=123,
        window="PRE_MATCH",
        fixture_name="AAA vs BBB",
        kickoff="2026-06-11 18:00:00",
        home_code="AAA",
        away_code="BBB",
        sm_digest={"expected_goals": {"AAA": 1.6, "BBB": 0.8}},
        web_research={"total_results": 1, "sources": ["example.com"]},
        reddit_bundle={
            "source": "web_search",
            "threads_found": 0,
            "comments_found": 2,
            "top_comments": ["AAA fan lineup note", "BBB tactical thread"],
        },
    )

    assert _coverage_score(fx) == 0.75
    assert _signal_coverage(fx)["reddit"] is True
    assert "reddit_sentiment" in _evidence_ids(fx)


def test_conservative_edge_includes_fees_slippage_and_model_risk():
    edge = calculate_conservative_edge(
        probability_mean=0.60,
        probability_lower_bound=0.54,
        expected_fill_price=0.48,
        config=ConservativeEdgeConfig(fee_buffer=0.01, slippage_buffer=0.02, model_risk_buffer=0.03),
    )
    assert edge.gross_edge == pytest.approx(0.12)
    assert edge.conservative_edge == pytest.approx(0.00)
    assert edge.expected_value_after_costs == pytest.approx(0.06)


def _recommendation(agent: str, key: str, stake: float = 0.01) -> AgentRecommendation:
    return AgentRecommendation(
        agent_name=agent,
        fixture_id="fixture-1",
        outcome="AAA",
        should_trade=True,
        abstain_reason=None,
        probability_mean=0.60,
        probability_lower_bound=0.55,
        probability_upper_bound=0.65,
        market_midpoint=0.45,
        best_ask=0.46,
        expected_fill_price=0.46,
        gross_edge=0.14,
        conservative_edge=0.05,
        expected_value_after_costs=0.08,
        signal_type="value",
        evidence_ids=["forecast"],
        evidence_summary="test",
        confidence=0.70,
        data_coverage_score=0.80,
        recommended_stake=stake,
        maximum_acceptable_price=0.50,
        signal_created_at=datetime.now(timezone.utc),
        signal_expires_at=None,
        correlation_key=key,
        forecast_id="forecast-1",
    )


def test_portfolio_keeps_identical_signals_from_independent_wallets():
    result = allocate_recommendations([
        _recommendation("anchor", "same-signal"),
        _recommendation("hunter", "same-signal"),
        _recommendation("blitz", "same-signal"),
    ])

    assert [r.agent_name for r in result.accepted] == ["anchor", "hunter", "blitz"]
    assert result.duplicate_recommendations == 0
    assert result.observed_only == []


def test_fill_report_extracts_actual_fill_accounting():
    report = _fill_report({
        "order_id": "ord-1",
        "status": "partially_filled",
        "size_usdc": "5.00",
        "limit_price": "0.55",
        "size_usdc_filled": "2.50",
        "filled_shares": "5",
        "fees_usdc": "0.03",
        "open_fills": [{"tx_hash": "0x1"}],
    })

    assert report["requested_stake_usdc"] == pytest.approx(5.0)
    assert report["filled_notional_usdc"] == pytest.approx(2.5)
    assert report["actual_average_fill_price"] == pytest.approx(0.5)
    assert report["unfilled_usdc"] == pytest.approx(2.5)
    assert report["fees_usdc"] == pytest.approx(0.03)
    assert report["partial_fill_state"] == "partial"
