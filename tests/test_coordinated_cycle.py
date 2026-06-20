"""Integration: act_for_agent routes all agents through structured recommendations."""
from __future__ import annotations

import pytest

from harness.profiles import get_profile
from live.cycle import Forecast, act_for_agent, _EmptyResult
from live.roster import LiveAgent
from betting.portfolio import PortfolioCoordinator


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Stub the only two network touchpoints act_for_agent hits in dry-run."""
    from live.arena_client import ArenaClient
    from ledger.client import LedgerSession
    # $250 wallet keeps a full-size coordinated bet ($2–4) under the default
    # 3% outcome / 4% fixture portfolio caps so the allocator accepts it.
    monkeypatch.setattr(ArenaClient, "wallet",
                        lambda self: {"available": 250.0, "locked": 0.0, "address": "0x"})
    monkeypatch.setattr(LedgerSession, "validate", lambda self: {"valid": True})
    monkeypatch.setattr(LedgerSession, "submit", lambda self, **k: {"records": [], "status": "ok"})


def _ml(home=0.55, draw=0.26, away=0.25):
    return {
        "market_source": "polymarket",
        "outcomes": {
            "home": {"team_code": "AAA", "current_mid_yes": home},
            "draw": {"team_code": "draw", "current_mid_yes": draw},
            "away": {"team_code": "BBB", "current_mid_yes": away},
        },
    }


def _snapshot(lower_home=0.66):
    return {
        "probabilities_by_code": {"AAA": 0.75, "draw": 0.13, "BBB": 0.12},
        "lower_bounds_by_code": {"AAA": lower_home, "draw": 0.08, "BBB": 0.07},
        "upper_bounds_by_code": {"AAA": 0.84, "draw": 0.20, "BBB": 0.18},
        "data_coverage_score": 0.75,
        "confidence": 0.70,
        "evidence_ids": ["sportmonks_digest", "supabase_digest"],
        "forecast_id": "fc-int-1",
    }


def _forecast(home=0.75, draw=0.13, away=0.12, confidence="high"):
    # Conviction: agents bet off the council belief carried on fx.probabilities,
    # so the strength of the bet is controlled by these probs (the injected
    # independent snapshot is retained only for metrics / fallback).
    fx = Forecast(
        fixture_id=900, window="PRE_MATCH", fixture_name="AAA vs BBB",
        home_code="AAA", away_code="BBB", home_name="Alpha", away_name="Beta",
        moneyline=_ml(), mids={"home": 0.55, "draw": 0.26, "away": 0.25},
        market_source="polymarket",
        probabilities={"AAA": home, "draw": draw, "BBB": away},
        outcome="AAA", probability=home, confidence=confidence,
    )
    fx.independent_forecast = _snapshot()
    fx.forecast_snapshot_id = "fc-int-1"
    fx.sm_digest = {"source": "sportmonks"}
    pm = _EmptyResult()
    pm.parsed = {"data_availability": "no_market"}
    fx.pm_digest_result = pm
    return fx


def _agent(name):
    return LiveAgent(name=name, profile=get_profile(name), api_key="k")


def test_coordinated_agent_emits_recommendation_and_trades():
    fx = _forecast()                       # strong council conviction on AAA (0.75)
    summary = act_for_agent(_agent("anchor"), fx, dry_run=True,
                            coordinator=PortfolioCoordinator())
    stats = summary["recommendation_stats"]
    assert stats["recommendations"] == 1
    assert stats["abstentions"] == 0
    assert summary["n_picks"] == 1
    assert summary["orders"] and summary["orders"][0]["status"] == "dry_run"


def test_coordinated_agent_trades_positive_ev_despite_weak_conservative_edge():
    # Council belief only just above the fill ⇒ conservative edge (lower-band
    # minus fill minus costs) lands under ANCHOR's floor → auditable abstention.
    fx = _forecast(home=0.62, draw=0.20, away=0.18)
    summary = act_for_agent(_agent("anchor"), fx, dry_run=True,
                            coordinator=PortfolioCoordinator())
    stats = summary["recommendation_stats"]
    assert stats["recommendations"] == 1
    assert stats["abstentions"] == 0
    assert summary["n_picks"] == 1
    assert summary["orders"] and summary["orders"][0]["status"] == "dry_run"


def test_roster_order_does_not_give_monk_ordinary_value_ownership():
    coord = PortfolioCoordinator()
    fx = _forecast(home=0.80)
    first = act_for_agent(_agent("monk"), fx, dry_run=True, coordinator=coord)
    second = act_for_agent(_agent("anchor"), fx, dry_run=True, coordinator=coord)

    assert first["n_picks"] in (0, 1)
    assert second["recommendation_stats"]["recommendations"] == 1
    assert coord.exposure


def test_blitz_trades_without_event_trigger():
    fx = _forecast()
    coord = PortfolioCoordinator()
    summary = act_for_agent(_agent("blitz"), fx, dry_run=True, coordinator=coord)
    stats = summary["recommendation_stats"]
    assert stats["recommendations"] == 1
    assert stats["abstentions"] == 0
    assert coord.exposure
    assert summary["n_picks"] == 1
    assert summary["orders"][0]["status"] == "dry_run"
