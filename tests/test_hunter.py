"""HUNTER conviction behaviour.

HUNTER is the AGGRESSIVE conviction agent: it bets the shared goated council
forecast's best-EV outcome (favorites included) on the thinnest edges of the
coordinated three, and it still honours the universal probability floor that
kills longshot payout-chasing.
"""
from __future__ import annotations

import config
from agents.hunter import HunterStrategy

from conftest import make_football_context, make_snapshot, make_market


def _forecast_and_view(ff):
    h = HunterStrategy()
    view = h.build_data_view(make_snapshot(ff), None)
    return h, view, h.build_forecast(view)


def test_hunter_uses_shared_council_forecast_when_present():
    ff = make_football_context(council={"home": 0.58, "draw": 0.24, "away": 0.18})
    h, view, fc = _forecast_and_view(ff)
    assert fc.forecast_type == "council_conviction"
    assert round(fc.home_probability, 2) == 0.58
    assert round(fc.away_probability, 2) == 0.18


def test_hunter_allows_the_favorite_when_it_is_value():
    # Council loves the home favorite; market underprices it → HUNTER buys it.
    ff = make_football_context(council={"home": 0.62, "draw": 0.22, "away": 0.16})
    h, view, fc = _forecast_and_view(ff)
    market = make_market(home=0.50, draw=0.27, away=0.23)
    cands = h.generate_candidates(fc, view, market)
    assert any(c.outcome == "home" for c in cands)


def test_hunter_honours_the_conviction_probability_floor():
    # Away is dirt cheap (3¢) but the council gives it < CONVICTION_MIN_PROB —
    # HUNTER must never chase the longshot regardless of payout.
    low = config.CONVICTION_MIN_PROB - 0.04
    ff = make_football_context(
        council={"home": 0.74, "draw": 0.26 - low, "away": low})
    h, view, fc = _forecast_and_view(ff)
    market = make_market(home=0.55, draw=0.30, away=0.03)
    cands = h.generate_candidates(fc, view, market)
    assert all(c.outcome != "away" for c in cands), \
        "sub-floor outcomes are never backed, no matter how cheap"


def test_hunter_requires_positive_edge():
    # Fairly priced favorite (no edge) → no candidate.
    ff = make_football_context(council={"home": 0.50, "draw": 0.27, "away": 0.23})
    h, view, fc = _forecast_and_view(ff)
    market = make_market(home=0.50, draw=0.27, away=0.23)
    cands = h.generate_candidates(fc, view, market)
    assert cands == [] or all(c.gross_edge > 0 for c in cands)
