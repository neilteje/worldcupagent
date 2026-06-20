from __future__ import annotations
import pytest
from unittest.mock import MagicMock

from data import bzzoiro, bzzoiro_mapper, fixture_bundle
from models.deterministic_v2 import EnsembleConfig, predict_v2
from models.team_strength import StrengthConfig, effective_rating, expected_goals


def test_bzzoiro_api_extract_ml_probabilities():
    # Empty prediction
    assert bzzoiro.extract_ml_probabilities({}) is None

    # Valid prediction
    pred = {
        "match_result": {
            "home_win_probability": 0.45,
            "draw_probability": 0.25,
            "away_win_probability": 0.30
        }
    }
    res = bzzoiro.extract_ml_probabilities(pred)
    assert res == {"home_win": 0.45, "draw": 0.25, "away_win": 0.30}

    # Invalid types
    pred_invalid = {
        "match_result": {
            "home_win_probability": "not_a_float",
            "draw_probability": 0.25,
            "away_win_probability": 0.30
        }
    }
    assert bzzoiro.extract_ml_probabilities(pred_invalid) is None


def test_bzzoiro_api_extract_event_stats_summary():
    # Empty stats
    assert bzzoiro.extract_event_stats_summary({}) == {}

    # Real v2 schema: stats.stats.{home,away}.{xg,...} + momentum time series.
    stats = {
        "event_id": 1,
        "stats": {
            "home": {"xg": 1.75, "possession": 55, "shots": 14},
            "away": {"xg": 1.10, "possession": 45, "shots": 9},
        },
        "momentum": [{"minute": 90, "home": 60.0, "away": 40.0}],
    }
    res = bzzoiro.extract_event_stats_summary(stats)
    assert res["home_xg"] == 1.75
    assert res["away_xg"] == 1.10
    assert res["home_possession"] == 55
    assert res["away_possession"] == 45
    assert res["home_shots"] == 14
    assert res["away_shots"] == 9
    assert res["home_momentum"] == 60.0
    assert res["away_momentum"] == 40.0

    # Upcoming match: only xG present (often null) — must not raise.
    upcoming = {"event_id": 2, "stats": {"home": {"xg": None}, "away": {"xg": None}},
                "momentum": []}
    res2 = bzzoiro.extract_event_stats_summary(upcoming)
    assert res2["home_xg"] == 0.0 and res2["home_momentum"] == 0.0


def test_bzzoiro_api_extract_ml_probabilities_real_schema():
    # Real v2 schema: markets.match_result.prob_* in PERCENT — renormalised to 1.
    pred = {"markets": {"match_result": {"prob_home": 43.14, "prob_draw": 29.03,
                                         "prob_away": 27.83, "predicted": "H"}}}
    res = bzzoiro.extract_ml_probabilities(pred)
    assert res is not None
    assert abs(res["home_win"] + res["draw"] + res["away_win"] - 1.0) < 1e-6
    assert res["home_win"] > res["draw"] > res["away_win"]


def test_bzzoiro_api_extract_prediction_summary():
    pred = {
        "markets": {
            "match_result": {"prob_home": 43.1, "prob_draw": 29.0, "prob_away": 27.8},
            "expected_goals": {"home": 1.85, "away": 1.65},
            "over_under": {"prob_over_25": 63.9},
            "btts": {"prob_yes": 62.0},
            "score": {"most_likely": "1-1"},
        },
        "recommendations": {"favorite": "H", "favorite_prob": 43.1},
        "model": {"confidence": 0.43, "version": "v5.0"},
    }
    summ = bzzoiro.extract_prediction_summary(pred)
    assert summ["expected_goals"] == {"home": 1.85, "away": 1.65}
    assert summ["most_likely_score"] == "1-1"
    assert summ["model_version"] == "v5.0"
    assert summ["model_favorite"] == "H"


def test_bzzoiro_mapper_event_resolution(monkeypatch):
    # Mock bzzoiro.search_events
    mock_search = MagicMock()
    monkeypatch.setattr(bzzoiro, "search_events", mock_search)

    # Case 1: Direct match found. Real v2 schema: home_team/away_team are plain
    # strings and the kickoff field is `event_date`.
    mock_search.return_value = [{"id": 12345, "home_team": "Mexico",
                                 "away_team": "South Africa",
                                 "event_date": "2026-06-15T12:00:00Z"}]
    event_id = bzzoiro_mapper.get_bzzoiro_event_id("Mexico", "South Africa", "2026-06-15")
    assert event_id == 12345
    mock_search.assert_called_with("Mexico", "South Africa", "2026-06-13", "2026-06-17")

    # Case 2: Fallback path when direct match returns empty
    mock_search.reset_mock()
    def search_side_effect(home, away, df, dt):
        if away == "South Africa":
            return []
        elif away == "":
            return [
                {"id": 99999, "home_team": "Mexico", "away_team": "South Africa",
                 "event_date": "2026-06-15T12:00:00Z"},
                {"id": 88888, "home_team": "Mexico", "away_team": "France",
                 "event_date": "2026-06-15T12:00:00Z"},
            ]
        return []
    mock_search.side_effect = search_side_effect
    event_id = bzzoiro_mapper.get_bzzoiro_event_id("Mexico", "South Africa", "2026-06-15")
    assert event_id == 99999


def test_bzzoiro_mapper_resolves_cape_verde_alias():
    from datetime import datetime

    mapping = bzzoiro_mapper.map_event(
        "internal",
        "Spain",
        "Cape Verde Islands",
        datetime.fromisoformat("2026-06-15T16:00:00+00:00"),
        [{
            "id": 19609162,
            "home_team": "Spain",
            "away_team": "Cape Verde",
            "event_date": "2026-06-15T16:00:00Z",
        }],
    )
    assert mapping.bzzoiro_event_id == "19609162"
    assert mapping.confidence >= 0.8


def test_bzzoiro_search_variants_cover_world_cup_api_names():
    assert "South Korea" in bzzoiro_mapper.search_variants("Korea Republic")
    assert "Czechia" in bzzoiro_mapper.search_variants("Czech Republic")
    assert "USA" in bzzoiro_mapper.search_variants("United States")
    assert "Bosnia" in bzzoiro_mapper.search_variants("Bosnia and Herzegovina")
    assert "Cabo Verde" in bzzoiro_mapper.search_variants("Cape Verde Islands")
    assert "DR Congo" in bzzoiro_mapper.search_variants("Congo DR")


def test_bzzoiro_deterministic_v2_predictions():
    home_state = {"live_rating": 0.0, "matches": 0}
    away_state = {"live_rating": 0.0, "matches": 0}

    # Case 1: No Bzzoiro probabilities provided
    out_no_bz = predict_v2(home_state, away_state, bzzoiro_probs=None)
    assert "bzzoiro" not in out_no_bz["components"]
    assert "bzzoiro" not in out_no_bz["active_components"]

    # Compatibility input is ignored: the deterministic output is invariant.
    bz_probs = {"home_win": 0.70, "draw": 0.20, "away_win": 0.10}
    out_with_bz = predict_v2(home_state, away_state, bzzoiro_probs=bz_probs)
    assert out_with_bz == out_no_bz


def test_bzzoiro_team_strength_fields_are_ignored():
    cfg = StrengthConfig()
    
    # Check effective rating with bzzoiro_momentum
    state_no_momentum = {"live_rating": 0.5}
    state_with_momentum = {"live_rating": 0.5, "bzzoiro_momentum": 80.0}
    
    rating_no = effective_rating(state_no_momentum, cfg)
    rating_with = effective_rating(state_with_momentum, cfg)
    
    assert rating_with == rating_no

    # Check expected_goals with bzzoiro_xg
    home_state = {"live_rating": 0.0, "matches": 1, "xg_for": 1.0, "xg_against": 1.0, "goals_for": 1.0, "goals_against": 1.0, "bzzoiro_xg": 2.5}
    away_state = {"live_rating": 0.0, "matches": 1, "xg_for": 1.0, "xg_against": 1.0, "goals_for": 1.0, "goals_against": 1.0}
    
    eg_no_bz = expected_goals(
        {"live_rating": 0.0, "matches": 1, "xg_for": 1.0, "xg_against": 1.0, "goals_for": 1.0, "goals_against": 1.0},
        away_state,
        cfg=cfg
    )
    eg_with_bz = expected_goals(home_state, away_state, cfg=cfg)
    
    assert eg_with_bz == eg_no_bz


def test_bzzoiro_fixture_bundle_build_context(monkeypatch):
    # Mock bzzoiro_mapper and bzzoiro
    monkeypatch.setattr(bzzoiro_mapper, "get_bzzoiro_event_id", lambda h, a, d: 123)
    
    mock_stats = {"event_id": 123,
                  "stats": {"home": {"xg": 1.5}, "away": {"xg": 1.0}},
                  "momentum": []}
    mock_pred = {"markets": {"match_result": {"prob_home": 50.0, "prob_draw": 30.0,
                                              "prob_away": 20.0}}}
    mock_lineups = {"lineups": [{"player": "A"}], "lineup_status": "confirmed",
                    "unavailable_players": {"home": [], "away": []}}

    monkeypatch.setattr(bzzoiro, "get_event_stats", lambda eid: mock_stats)
    monkeypatch.setattr(bzzoiro, "get_event_prediction", lambda eid: mock_pred)
    monkeypatch.setattr(bzzoiro, "get_event_lineups", lambda eid: mock_lineups)

    # Build digest
    digest = fixture_bundle.build_bzzoiro_digest("Mexico", "South Africa", "2026-06-15")
    assert digest is not None
    assert digest["event_id"] == 123
    assert digest["stats_summary"]["home_xg"] == 1.5
    assert digest["ml_prediction"] == {"home_win": 0.5, "draw": 0.3, "away_win": 0.2}
    assert digest["has_lineups"] is True
