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

    # Valid stats
    stats = {
        "teams": {
            "home": {
                "expected_goals": 1.75,
                "possession_time": 55,
                "momentum_score": 60
            },
            "away": {
                "expected_goals": 1.10,
                "possession_time": 45,
                "momentum_score": 40
            }
        }
    }
    res = bzzoiro.extract_event_stats_summary(stats)
    assert res == {
        "home_xg": 1.75,
        "away_xg": 1.10,
        "home_possession": 55,
        "away_possession": 45,
        "home_momentum": 60,
        "away_momentum": 40
    }


def test_bzzoiro_mapper_event_resolution(monkeypatch):
    # Mock bzzoiro.search_events
    mock_search = MagicMock()
    monkeypatch.setattr(bzzoiro, "search_events", mock_search)

    # Case 1: Direct match found
    mock_search.return_value = [{"id": 12345, "away_team": {"name": "South Africa"}, "start_time": "2026-06-15T12:00:00Z"}]
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
                {"id": 99999, "away_team": {"name": "South Africa"}, "start_time": "2026-06-15T12:00:00Z"},
                {"id": 88888, "away_team": {"name": "France"}, "start_time": "2026-06-15T12:00:00Z"}
            ]
        return []
    mock_search.side_effect = search_side_effect
    event_id = bzzoiro_mapper.get_bzzoiro_event_id("Mexico", "South Africa", "2026-06-15")
    assert event_id == 99999


def test_bzzoiro_deterministic_v2_predictions():
    home_state = {"live_rating": 0.0, "matches": 0}
    away_state = {"live_rating": 0.0, "matches": 0}

    # Case 1: No Bzzoiro probabilities provided
    out_no_bz = predict_v2(home_state, away_state, bzzoiro_probs=None)
    assert out_no_bz["components"]["bzzoiro"] is None
    assert "bzzoiro" not in out_no_bz["active_components"]

    # Case 2: Bzzoiro probabilities provided and use_bzzoiro is True
    bz_probs = {"home_win": 0.70, "draw": 0.20, "away_win": 0.10}
    out_with_bz = predict_v2(home_state, away_state, bzzoiro_probs=bz_probs)
    assert out_with_bz["components"]["bzzoiro"]["home"] == pytest.approx(0.70)
    assert out_with_bz["components"]["bzzoiro"]["draw"] == pytest.approx(0.20)
    assert out_with_bz["components"]["bzzoiro"]["away"] == pytest.approx(0.10)
    assert "bzzoiro" in out_with_bz["active_components"]

    # Case 3: Bzzoiro probabilities provided but use_bzzoiro is False
    cfg = EnsembleConfig(use_bzzoiro=False)
    out_disabled = predict_v2(home_state, away_state, bzzoiro_probs=bz_probs, cfg=cfg)
    assert "bzzoiro" not in out_disabled["active_components"]
    
    # Case 4: Check weights influence
    cfg_high_bz = EnsembleConfig(w_elo=0.0, w_poisson=0.0, w_market=0.0, w_bzzoiro=1.0)
    out_only_bz = predict_v2(home_state, away_state, bzzoiro_probs=bz_probs, cfg=cfg_high_bz)
    assert out_only_bz["probabilities"]["home"] > 0.60


def test_bzzoiro_team_strength_influence():
    cfg = StrengthConfig()
    
    # Check effective rating with bzzoiro_momentum
    state_no_momentum = {"live_rating": 0.5}
    state_with_momentum = {"live_rating": 0.5, "bzzoiro_momentum": 80.0}
    
    rating_no = effective_rating(state_no_momentum, cfg)
    rating_with = effective_rating(state_with_momentum, cfg)
    
    expected_diff = 0.8 * cfg.bz_momentum_weight
    assert abs(rating_with - rating_no - expected_diff) < 1e-9

    # Check expected_goals with bzzoiro_xg
    home_state = {"live_rating": 0.0, "matches": 1, "xg_for": 1.0, "xg_against": 1.0, "goals_for": 1.0, "goals_against": 1.0, "bzzoiro_xg": 2.5}
    away_state = {"live_rating": 0.0, "matches": 1, "xg_for": 1.0, "xg_against": 1.0, "goals_for": 1.0, "goals_against": 1.0}
    
    eg_no_bz = expected_goals(
        {"live_rating": 0.0, "matches": 1, "xg_for": 1.0, "xg_against": 1.0, "goals_for": 1.0, "goals_against": 1.0},
        away_state,
        cfg=cfg
    )
    eg_with_bz = expected_goals(home_state, away_state, cfg=cfg)
    
    # Home team should have a higher lambda_home now since their attack form blends in BZZOIRO xG
    assert eg_with_bz["lambda_home"] > eg_no_bz["lambda_home"]


def test_bzzoiro_fixture_bundle_build_context(monkeypatch):
    # Mock bzzoiro_mapper and bzzoiro
    monkeypatch.setattr(bzzoiro_mapper, "get_bzzoiro_event_id", lambda h, a, d: 123)
    
    mock_stats = {"teams": {"home": {"expected_goals": 1.5}, "away": {"expected_goals": 1.0}}}
    mock_pred = {"match_result": {"home_win_probability": 0.5, "draw_probability": 0.3, "away_win_probability": 0.2}}
    mock_lineups = {"lineups": [{"player": "A"}]}
    
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
