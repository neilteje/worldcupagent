from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datetime import datetime, timezone


def make_football_context(*, strong_home=True, with_market=False, scout_flags=None,
                          council=None):
    """Build a football_context dict with real deterministic team-states so the
    agents produce genuine (non-stub) forecasts. ``with_market`` injects a
    market-derived field to exercise the scrubber / leakage tests."""
    home_state = {
        "live_rating": 0.45 if strong_home else 0.05,
        "elo_scaled": 0.40 if strong_home else 0.05,
        "matches": 5, "xg_for": 9.0, "xg_against": 4.0,
        "goals_for": 9.0, "goals_against": 4.0, "rest_hours": 96.0,
    }
    away_state = {
        "live_rating": 0.05, "elo_scaled": 0.05,
        "matches": 5, "xg_for": 5.0, "xg_against": 8.0,
        "goals_for": 5.0, "goals_against": 8.0, "rest_hours": 96.0,
    }
    det = {
        "home_state": home_state,
        "away_state": away_state,
        "expected_goals": {"lambda_home": 1.7, "lambda_away": 0.8},
        "components": {"elo": {"home": 0.6, "draw": 0.25, "away": 0.15}},
    }
    ff = {
        "sportmonks_digest": {"ml": {"home_win": 0.6}},
        "supabase_digest": {"h2h": {}},
        "bzzoiro_digest": {"event_id": 42},
        "deterministic_model": det,
        "home_code": "AAA", "away_code": "BBB",
        "independent_forecast": {
            "probabilities_by_code": {"AAA": 0.62, "draw": 0.22, "BBB": 0.16},
            "lower_bounds_by_code": {"AAA": 0.55, "draw": 0.16, "BBB": 0.10},
            "upper_bounds_by_code": {"AAA": 0.70, "draw": 0.28, "BBB": 0.22},
            "data_coverage_score": 0.80,
        },
        "forecast_snapshot_id": "snap-test-1",
        "scout_flags": scout_flags or [],
    }
    if with_market:
        ff["sportmonks_digest"]["bookmaker_consensus_win_prob"] = {"AAA": 0.7}
        ff["polymarket_mid"] = 0.55
        ff["deterministic_model"]["components"]["market"] = {"home": 0.7, "draw": 0.2, "away": 0.1}
    if council is not None:
        # council is {"home":p,"draw":p,"away":p, optional "confidence":num}
        conf = council.get("confidence", 0.6)
        ff["council_forecast"] = {
            "probabilities": {k: float(council.get(k, 0.0)) for k in ("home", "draw", "away")},
            "confidence": conf,
        }
    return ff


def make_snapshot(football_context, *, window="PRE_MATCH"):
    from agents.contracts import FixtureDataSnapshot
    return FixtureDataSnapshot(
        fixture_id="900", fixture_name="AAA vs BBB", window=window,
        kickoff="2026-06-20T18:00:00", as_of_timestamp=datetime.now(timezone.utc),
        home_code="AAA", away_code="BBB", home_name="Alpha", away_name="Beta",
        sportmonks=None, supabase=None, bzzoiro=None, web=None, reddit=None, social=None,
        football_context=football_context, live_context=None, market_context=None,
        snapshot_id="snap_900_PRE_MATCH", snapshot_hash="",
    )


def make_market(*, home=0.50, draw=0.27, away=0.23):
    from agents.contracts import MarketContext
    mids = {"home": home, "draw": draw, "away": away}
    return MarketContext(
        observed_at=datetime.now(timezone.utc), polymarket=None, kalshi=None,
        bookmaker_consensus=None, bookmaker_comparison=None,
        devigged_probabilities=mids, best_bid=mids, best_ask=mids, midpoint=mids,
        expected_fill_price=mids, movement={}, dispersion={}, overround=None,
    )
