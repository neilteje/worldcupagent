from __future__ import annotations

from datetime import datetime, timedelta, timezone

from agents.anchor import AnchorStrategy
from agents.blitz import BlitzStrategy
from agents.hunter import HunterStrategy
from agents.monk import MonkStrategy
from betting.portfolio import allocate_jointly
from live.metrics import runtime_metadata
from models.deterministic_v2 import predict_v2
from models.evidence import normalize_evidence
from models.forecast_layers import (
    apply_evidence_adjustments,
    build_forecast_layers,
    validate_and_aggregate_scenarios,
)
from models.live_state import LiveMatchState
from models.live_update import DeterministicHalftimeModel
from models.market_calibration import information_edge

from conftest import make_football_context, make_market, make_snapshot


def test_polymarket_and_bookmaker_do_not_change_independent_forecast():
    home = {"live_rating": 0.2, "matches": 5, "xg_for": 7, "xg_against": 5}
    away = {"live_rating": 0.0, "matches": 5, "xg_for": 5, "xg_against": 7}
    a = predict_v2(home, away, market_probs={"home": 0.9, "draw": 0.05, "away": 0.05})
    b = predict_v2(home, away, market_probs={"home": 0.05, "draw": 0.05, "away": 0.9})
    assert a["probabilities"] == b["probabilities"]
    assert "market" not in a["active_components"]


def test_market_fields_cannot_reach_analyst():
    view = MonkStrategy().build_data_view(make_snapshot(make_football_context(with_market=True)), None)
    assert view.market_features is None
    assert "polymarket_mid" not in str(view.football_features)
    assert "bookmaker" not in str(view.football_features).lower()


def test_market_enters_only_once_and_edge_uses_pre_market():
    layers = build_forecast_layers(
        {"home": 0.5, "draw": 0.25, "away": 0.25},
        analyst_output={"adjustments": {"home": 0.02, "draw": -0.01, "away": -0.01}},
        devil_output={"scenarios": [
            {"scenario_id": "a", "plausibility": 0.2, "probabilities": {"home": 0.4, "draw": 0.3, "away": 0.3}},
            {"scenario_id": "b", "plausibility": 0.1, "probabilities": {"home": 0.45, "draw": 0.35, "away": 0.2}},
        ]},
        judge_output={"recommended_market_weight": 0.25},
        market_probabilities={"home": 0.2, "draw": 0.3, "away": 0.5},
    )
    once = layers.scored_probabilities
    twice = build_forecast_layers(
        layers.scored_probabilities,
        judge_output={"recommended_market_weight": 0.25},
        market_probabilities={"home": 0.2, "draw": 0.3, "away": 0.5},
    ).scored_probabilities
    assert once != twice
    assert layers.pre_market_probabilities != layers.scored_probabilities
    assert information_edge(layers.pre_market_probabilities["home"], 0.42) == layers.pre_market_probabilities["home"] - 0.42


def test_evidence_deduplicated():
    now = datetime.now(timezone.utc)
    raw = [
        {"evidence_id": "web1", "source": "web", "source_type": "web", "observed_at": now,
         "fixture_id": "1", "team": "AAA", "player": "P", "event_type": "injury",
         "direction": "away", "relevance_score": 0.9, "reliability_score": 0.8,
         "expires_at": now + timedelta(hours=2), "raw_summary": "P ruled out"},
        {"evidence_id": "reddit1", "source": "reddit", "source_type": "reddit", "observed_at": now,
         "fixture_id": "1", "team": "AAA", "player": "P", "event_type": "injury",
         "direction": "away", "relevance_score": 0.7, "reliability_score": 0.6,
         "expires_at": now + timedelta(hours=2), "raw_summary": "P ruled out"},
    ]
    assert len(normalize_evidence(raw, now=now)) == 1


def test_expired_evidence_cannot_trigger_blitz():
    now = datetime.now(timezone.utc)
    ff = make_football_context(council={"home": 0.6, "draw": 0.2, "away": 0.2})
    ff["event_signals"] = [{
        "signal_id": "s1", "event_type": "red_card", "outcome": "home",
        "observed_at": now - timedelta(hours=2), "expires_at": now - timedelta(minutes=1),
    }]
    snap = make_snapshot(ff)
    snap = snap.__class__(**{**snap.__dict__, "as_of_timestamp": now})
    blitz = BlitzStrategy()
    view = blitz.build_data_view(snap, None)
    fc = blitz.build_forecast(view)
    assert blitz.generate_candidates(fc, view, make_market(home=0.45, draw=0.25, away=0.30)) == []


def test_analyst_adjustments_are_capped_and_invalid_zeroes():
    base = {"home": 0.4, "draw": 0.3, "away": 0.3}
    moved, warnings = apply_evidence_adjustments(base, {"adjustments": {"home": 0.5, "draw": -0.4, "away": 0.1}})
    assert abs(moved["home"] - base["home"]) <= 0.09
    zeroed, warnings = apply_evidence_adjustments(base, {"adjustments": "bad"})
    assert zeroed == base
    assert "invalid_analyst_adjustment" in warnings


def test_devil_scenarios_validated_and_deterministic():
    base = {"home": 0.5, "draw": 0.25, "away": 0.25}
    payload = {"scenarios": [
        {"scenario_id": "b", "plausibility": 0.2, "probabilities": {"home": 0.4, "draw": 0.3, "away": 0.3}},
        {"scenario_id": "a", "plausibility": 0.1, "probabilities": {"home": 0.6, "draw": 0.2, "away": 0.2}},
    ]}
    one, _ = validate_and_aggregate_scenarios(base, payload)
    two, _ = validate_and_aggregate_scenarios(base, {"scenarios": list(reversed(payload["scenarios"]))})
    assert one == two
    invalid, warnings = validate_and_aggregate_scenarios(base, {"scenarios": [{"x": 1}]})
    assert invalid == base
    assert warnings


def test_agent_mandates_monk_anchor_hunter_blitz():
    market = make_market(home=0.35, draw=0.30, away=0.25)
    ff = make_football_context(
        council={"home": 0.62, "draw": 0.24, "away": 0.14, "confidence": 0.8}
    )
    ff["evidence_ids"] = ["e1", "e2"]
    snap = make_snapshot(ff)
    monk = MonkStrategy()
    anchor = AnchorStrategy()
    hunter = HunterStrategy()
    assert monk.generate_candidates(monk.build_forecast(monk.build_data_view(snap)), monk.build_data_view(snap), market)
    assert anchor.generate_candidates(anchor.build_forecast(anchor.build_data_view(snap)), anchor.build_data_view(snap), market)
    hunter_view = hunter.build_data_view(snap)
    hunter_fc = hunter.build_forecast(hunter_view)
    assert all(c.outcome != "home" and c.expected_fill_price <= 0.40 for c in hunter.generate_candidates(hunter_fc, hunter_view, market))
    blitz = BlitzStrategy()
    blitz_view = blitz.build_data_view(snap)
    assert blitz.generate_candidates(blitz.build_forecast(blitz_view), blitz_view, market) == []


def test_blitz_uses_common_contract_with_valid_trigger():
    now = datetime.now(timezone.utc)
    ff = make_football_context(council={"home": 0.65, "draw": 0.2, "away": 0.15})
    ff["event_signals"] = [{
        "signal_id": "s1", "event_type": "red_card", "outcome": "home",
        "observed_at": now, "expires_at": now + timedelta(minutes=10),
        "strength": 0.9, "confidence": 0.8,
    }]
    snap = make_snapshot(ff)
    snap = snap.__class__(**{**snap.__dict__, "as_of_timestamp": now})
    blitz = BlitzStrategy()
    view = blitz.build_data_view(snap)
    fc = blitz.build_forecast(view)
    recs = blitz.generate_recommendations(
        blitz.generate_candidates(fc, view, make_market(home=0.35, draw=0.25, away=0.20)),
        fc, view, make_market(home=0.35, draw=0.25, away=0.20), 100.0,
    )
    assert recs and recs[0].signal_type == "event_trigger"


def test_joint_allocation_order_invariant_for_mandates():
    from models.forecast_contracts import AgentRecommendation

    def _rec(agent, outcome, *, edge, corr):
        return AgentRecommendation(
            agent_name=agent, fixture_id="900", outcome=outcome,
            should_trade=True, abstain_reason=None,
            probability_mean=0.5, probability_lower_bound=0.45, probability_upper_bound=0.55,
            market_midpoint=0.35, best_ask=0.35, expected_fill_price=0.35,
            gross_edge=edge, conservative_edge=edge, expected_value_after_costs=edge,
            signal_type="event_trigger" if agent == "blitz" else ("draw_skew" if agent == "hunter" else "value"),
            evidence_ids=["e"], evidence_summary="",
            confidence=0.7, data_coverage_score=0.8,
            recommended_stake=0.01, maximum_acceptable_price=0.37,
            signal_created_at=datetime.now(timezone.utc), signal_expires_at=None,
            correlation_key=corr, forecast_id="fc",
        )

    recs = [
        _rec("anchor", "home", edge=0.08, corr="a"),
        _rec("hunter", "draw", edge=0.07, corr="h"),
        _rec("blitz", "home", edge=0.05, corr="b"),
    ]
    assert [r.correlation_key for r in allocate_jointly(recs).accepted] == [
        r.correlation_key for r in allocate_jointly(list(reversed(recs))).accepted
    ]


def test_halftime_normalized_and_live_conflicts_move_probabilities():
    state = LiveMatchState(
        fixture_id="1", current_minute=45, match_period="HT",
        home_score=0, away_score=1, home_red_cards=0, away_red_cards=1,
        home_live_xg=2.2, away_live_xg=0.2, home_dangerous_attacks=None,
        away_dangerous_attacks=None, home_possession=None, away_possession=None,
        data_coverage={"overall": 0.9},
    )
    neutral = LiveMatchState(**{**state.__dict__, "home_score": 0, "away_score": 0, "away_red_cards": 0, "home_live_xg": 0.5, "away_live_xg": 0.5})
    model = DeterministicHalftimeModel()
    out = model.update_forecast(1.4, 1.2, state)
    base = model.update_forecast(1.4, 1.2, neutral)
    assert abs(sum(out["probabilities"].values()) - 1.0) < 1e-9
    assert out["probabilities"] != base["probabilities"]


def test_reports_include_strategy_and_model_versions():
    meta = runtime_metadata()
    for key in (
        "commit_sha", "strategy_version", "forecast_pipeline_version", "model_version",
        "profile_configuration_hash", "enabled_feature_flags", "active_data_sources", "timestamp",
    ):
        assert key in meta
