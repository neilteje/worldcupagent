"""Four-agent architecture: distinct strategies, data views, forecasts (spec §29)."""
from __future__ import annotations

from datetime import datetime, timezone

from agents.monk import MonkStrategy
from agents.anchor import AnchorStrategy
from agents.hunter import HunterStrategy
from agents.blitz import BlitzStrategy
from models.forecast_contracts import MatchForecast

from conftest import make_football_context, make_snapshot


def _all_strategies():
    return [MonkStrategy(), AnchorStrategy(), HunterStrategy(), BlitzStrategy()]


def test_four_distinct_strategy_classes():
    classes = {type(s) for s in _all_strategies()}
    assert len(classes) == 4
    assert {s.name for s in _all_strategies()} == {"monk", "anchor", "hunter", "blitz"}


def _blitz_view(strategy, snapshot):
    view = strategy.build_data_view(snapshot, None)
    view.legacy_forecast = MatchForecast(
        fixture_id="900", as_of_timestamp=datetime.now(timezone.utc),
        home_probability=0.6, draw_probability=0.22, away_probability=0.18,
        home_lower_bound=0.6, draw_lower_bound=0.22, away_lower_bound=0.18,
        home_upper_bound=0.6, draw_upper_bound=0.22, away_upper_bound=0.18,
        confidence=0.7, data_coverage_score=0.8, model_version="legacy",
        feature_snapshot_hash="", evidence_ids=[], warnings=[],
    )
    return view


def test_four_distinct_data_view_hashes():
    snap = make_snapshot(make_football_context())
    hashes = []
    for s in _all_strategies():
        view = s.build_data_view(snap, None)
        hashes.append(view.data_view_hash)
    assert len(set(hashes)) == 4, "each agent must produce a distinct data-view hash"


def test_four_distinct_forecast_ids_and_not_one_shared_probability():
    snap = make_snapshot(make_football_context())
    forecasts = {}
    for s in _all_strategies():
        view = s.build_data_view(snap, None)
        if s.name == "blitz":
            view = _blitz_view(s, snap)
        fc = s.build_forecast(view)
        forecasts[s.name] = fc

    ids = [fc.forecast_id for fc in forecasts.values()]
    assert len(set(ids)) == 4, "four agents must produce four distinct forecast ids"

    # OFFLINE (no shared council forecast) the agents fall back to their own
    # models: MONK/ANCHOR share the independent foundation, HUNTER uses a Poisson
    # fallback, BLITZ uses the legacy stub.
    assert forecasts["monk"].forecast_type == "independent_deterministic"
    assert forecasts["hunter"].forecast_type == "poisson_fallback"


def test_conviction_all_agents_share_the_council_belief():
    # WITH the shared goated council forecast attached, MONK/ANCHOR/HUNTER all
    # adopt the SAME belief (conviction). They differ only by risk downstream.
    ff = make_football_context(council={"home": 0.57, "draw": 0.25, "away": 0.18})
    snap = make_snapshot(ff)

    def vec(fc):
        return (round(fc.home_probability, 3), round(fc.draw_probability, 3),
                round(fc.away_probability, 3))

    for cls in (MonkStrategy, AnchorStrategy, HunterStrategy):
        s = cls()
        fc = s.build_forecast(s.build_data_view(snap, None))
        assert fc.forecast_type == "council_conviction"
        assert vec(fc) == (0.57, 0.25, 0.18)


def test_monk_forecast_sums_to_one_and_has_widened_bounds_on_low_coverage():
    # Full coverage -> tight band; no states (low coverage) -> wider band.
    full = make_football_context()
    snap_full = make_snapshot(full)
    monk = MonkStrategy()
    fc_full = monk.build_forecast(monk.build_data_view(snap_full, None))
    assert abs(fc_full.home_probability + fc_full.draw_probability + fc_full.away_probability - 1.0) < 1e-6

    bare = {"home_code": "AAA", "away_code": "BBB"}  # no deterministic states, no snapshot
    snap_bare = make_snapshot(bare)
    fc_bare = monk.build_forecast(monk.build_data_view(snap_bare, None))
    width_full = fc_full.home_upper_bound - fc_full.home_lower_bound
    width_bare = fc_bare.home_upper_bound - fc_bare.home_lower_bound
    assert width_bare > width_full, "missing football data must widen uncertainty"
