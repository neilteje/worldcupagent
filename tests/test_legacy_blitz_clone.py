from __future__ import annotations

import json

import pytest

from agents.legacy_blitz import LegacyBlitzStrategy
from betting.portfolio import allocate_recommendations
from harness.profiles import get_profile
from reasoning.prompts import scout_input, analyst_input, devil_input
from conftest import make_football_context, make_market, make_snapshot


AGENTS = ("monk", "anchor", "hunter", "blitz")


def _snapshot():
    context = make_football_context(council={
        "home": 0.50, "draw": 0.30, "away": 0.20, "confidence": 0.80,
    })
    context["event_signals"] = []
    context["evidence_ids"] = ["council"]
    return make_snapshot(context)


def test_all_profiles_equal_legacy_blitz_policy():
    for name in AGENTS:
        profile = get_profile(name)
        assert profile.min_edge_vs_fair == 0.02
        assert profile.min_ev_per_dollar == 0.0
        assert profile.min_confidence == 0.35
        assert profile.kelly_fraction == 0.65
        assert profile.max_bet_usd == 5.0
        assert profile.stake_cap_fraction == 0.15
        assert profile.max_bets_per_window == 2
        assert profile.skip_on_high_scout_flag is False
        assert profile.apply_confidence_multiplier is False
        assert profile.max_entry_price is None


def test_same_snapshot_same_distribution_identity_only():
    forecasts = []
    for name in AGENTS:
        strategy = LegacyBlitzStrategy(name)
        forecasts.append(strategy.build_forecast(strategy.build_data_view(_snapshot())))
    distributions = {
        (f.home_probability, f.draw_probability, f.away_probability) for f in forecasts
    }
    assert distributions == {(0.5, 0.3, 0.2)}
    assert [f.agent_name for f in forecasts] == list(AGENTS)


@pytest.mark.parametrize("name", AGENTS)
def test_no_event_gate_draws_and_two_recommendations(name):
    strategy = LegacyBlitzStrategy(name)
    view = strategy.build_data_view(_snapshot())
    forecast = strategy.build_forecast(view)
    market = make_market(home=0.35, draw=0.15, away=0.50)
    candidates = strategy.generate_candidates(forecast, view, market)
    assert {candidate.outcome for candidate in candidates} == {"draw", "home"}
    recommendations = strategy.generate_recommendations(candidates, forecast, view, market, 100.0)
    assert len(recommendations) == 2
    assert all(rec.agent_name == name for rec in recommendations)
    assert all(rec.signal_type == "legacy_blitz_value" for rec in recommendations)
    assert all(rec.correlation_key.startswith(f"{name}:") for rec in recommendations)


def test_identical_cross_wallet_trades_are_not_deduplicated():
    recs = []
    market = make_market(home=0.35, draw=0.15, away=0.50)
    for name in AGENTS:
        strategy = LegacyBlitzStrategy(name)
        view = strategy.build_data_view(_snapshot())
        forecast = strategy.build_forecast(view)
        candidates = strategy.generate_candidates(forecast, view, market)
        recs.append(strategy.generate_recommendations(candidates, forecast, view, market, 100.0)[0])
    allocation = allocate_recommendations(recs)
    assert {rec.agent_name for rec in allocation.accepted} == set(AGENTS)
    assert allocation.duplicate_recommendations == 0


def test_all_wallets_can_allocate_two_complementary_outcomes():
    recs = []
    market = make_market(home=0.35, draw=0.15, away=0.50)
    for name in AGENTS:
        strategy = LegacyBlitzStrategy(name)
        view = strategy.build_data_view(_snapshot())
        forecast = strategy.build_forecast(view)
        candidates = strategy.generate_candidates(forecast, view, market)
        wallet_recs = strategy.generate_recommendations(
            candidates, forecast, view, market, 100.0
        )
        assert len(wallet_recs) == 2
        recs.extend(wallet_recs)

    allocation = allocate_recommendations(recs)
    assert len(allocation.accepted) == 8
    assert allocation.rejected == []
    for name in AGENTS:
        accepted = [rec for rec in allocation.accepted if rec.agent_name == name]
        assert {rec.outcome for rec in accepted} == {"AAA", "draw"}


def test_bzzoiro_is_absent_from_council_role_inputs():
    bz = {"event_id": 42, "ml_prediction": {"home": 0.9, "draw": 0.05, "away": 0.05}}
    payloads = [
        scout_input("A v B", "AAA", "BBB", {}, {}, {}, {}, {}, bz),
        analyst_input("A v B", "AAA", "BBB", {}, {}, {}, {}, bz),
        devil_input("A v B", "AAA", "BBB", {}, {}, {}, {}, bz),
    ]
    for payload in payloads:
        assert "bzzoiro" not in json.loads(payload)
