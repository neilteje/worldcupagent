"""HUNTER specialized tail behaviour (spec §10, acceptance #9/#10/#11)."""
from __future__ import annotations

import config
from agents.hunter import HunterStrategy
from models.tails.draw_model import DrawModel
from models.tails.upset_model import UnderdogUpsetModel
from models.tails.signals import SignalAggregator
from models.poisson_model import poisson_1x2

from conftest import make_football_context, make_snapshot, make_market


def _forecast_and_view(ff):
    h = HunterStrategy()
    view = h.build_data_view(make_snapshot(ff), None)
    return h, view, h.build_forecast(view)


def test_draw_and_upset_models_are_distinct_and_poisson_grounded():
    draw = DrawModel()
    upset = UnderdogUpsetModel()
    structural = poisson_1x2(1.6, 0.9)
    # Draw model tracks the structural draw mass (not a flat *1.05 bump).
    assert abs(draw.probability(1.6, 0.9, {}) - structural["draw"]) <= 0.11
    # Upset model returns the underdog's structural win mass under no context.
    assert abs(upset.probability(1.6, 0.9, "away", {}) - structural["away"]) <= 0.11
    # Context tilts move them in the right direction, capped.
    assert draw.probability(1.6, 0.9, {"draw_is_enough": True}) > draw.probability(1.6, 0.9, {})
    assert upset.probability(1.6, 0.9, "away", {"favorite_keeper_absent": True}) > \
        upset.probability(1.6, 0.9, "away", {})


def test_signal_aggregator_collapses_duplicate_source_groups():
    agg = SignalAggregator()
    raw = [
        {"source": "siteA", "source_group": "wire_report", "outcome": "away",
         "direction": "up", "strength": 0.7, "confidence": 0.6, "summary": "x"},
        {"source": "siteB", "source_group": "wire_report", "outcome": "away",
         "direction": "up", "strength": 0.5, "confidence": 0.6, "summary": "copy"},
        {"source": "xg_model", "source_group": "independent_model", "outcome": "away",
         "direction": "up", "strength": 0.8, "confidence": 0.7, "summary": "model"},
    ]
    signals = agg.aggregate(raw)
    # Two copies of one wire report collapse to ONE group; model is a second.
    assert SignalAggregator.independent_count(signals, "away", "up") == 2
    assert len(signals) == 2


def test_hunter_requires_two_independent_signals():
    # Only ONE independent group -> no candidate.
    ff = make_football_context(strong_home=True)
    ff["hunter_evidence"] = [
        {"source": "siteA", "source_group": "wire", "outcome": "away",
         "direction": "up", "strength": 0.7, "confidence": 0.6},
        {"source": "siteB", "source_group": "wire", "outcome": "away",
         "direction": "up", "strength": 0.6, "confidence": 0.6},
    ]
    h, view, fc = _forecast_and_view(ff)
    market = make_market(home=0.78, draw=0.16, away=0.12)  # away underpriced vs model
    cands = h.generate_candidates(fc, view, market)
    assert all(c.outcome != "away" for c in cands) or cands == []

    # Add a second INDEPENDENT group -> away can qualify on signal count.
    ff["hunter_evidence"].append(
        {"source": "xg", "source_group": "independent_model", "outcome": "away",
         "direction": "up", "strength": 0.8, "confidence": 0.7})
    h2, view2, fc2 = _forecast_and_view(ff)
    cands2 = h2.generate_candidates(fc2, view2, market)
    away = [c for c in cands2 if c.outcome == "away"]
    assert away, "two independent signal groups should let the away tail qualify"
    assert len(away[0].signals) >= config.HUNTER_MIN_INDEPENDENT_SIGNALS


def test_hunter_rejects_favorites_by_price():
    ff = make_football_context()
    ff["hunter_evidence"] = [
        {"source": "a", "source_group": "g1", "outcome": "home", "direction": "up",
         "strength": 0.7, "confidence": 0.6},
        {"source": "b", "source_group": "g2", "outcome": "home", "direction": "up",
         "strength": 0.7, "confidence": 0.6},
    ]
    h, view, fc = _forecast_and_view(ff)
    market = make_market(home=0.75, draw=0.15, away=0.20)  # home is the favorite
    cands = h.generate_candidates(fc, view, market)
    assert all(c.outcome != "home" for c in cands), "HUNTER must never buy the favorite"


def test_hunter_rejects_ultra_cheap_by_default():
    ff = make_football_context()
    ff["hunter_evidence"] = [
        {"source": "a", "source_group": "g1", "outcome": "away", "direction": "up",
         "strength": 0.7, "confidence": 0.6},
        {"source": "b", "source_group": "g2", "outcome": "away", "direction": "up",
         "strength": 0.7, "confidence": 0.6},
    ]
    h, view, fc = _forecast_and_view(ff)
    market = make_market(home=0.90, draw=0.10, away=0.03)  # away at 3¢ (ultra tail)
    cands = h.generate_candidates(fc, view, market)
    assert all(c.outcome != "away" for c in cands), "sub-5¢ tails are rejected by default"
