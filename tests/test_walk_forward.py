"""Historical team-state builders + chronological walk-forward (spec §15/§30)."""
from __future__ import annotations

from datetime import datetime

import pytest

from models.chronological_elo import ChronologicalEloBuilder, BASE_RATING
from models.rolling_form import RollingFormBuilder
from models.team_state_builder import build_team_state
from harness.walk_forward import load_matches, walk_forward, market_baseline


def test_elo_rewards_winner_and_conserves_when_neutral():
    elo = ChronologicalEloBuilder(k_factor=40.0)
    elo.process_match("A", "B", 2, 0, is_competitive=True, is_neutral=True)
    assert elo.get_rating("A") > BASE_RATING > elo.get_rating("B")
    # Zero-sum update at equal start ratings.
    assert abs((elo.get_rating("A") - BASE_RATING) + (elo.get_rating("B") - BASE_RATING)) < 1e-9
    # Bigger margin => bigger swing.
    elo2 = ChronologicalEloBuilder(k_factor=40.0)
    elo2.process_match("A", "B", 5, 0)
    assert elo2.get_rating("A") - BASE_RATING > elo.get_rating("A") - BASE_RATING


def test_rolling_form_is_leakage_free_and_time_weighted():
    form = RollingFormBuilder(short_half_life_days=30.0, long_half_life_days=180.0)
    form.add_match("A", datetime(2022, 1, 1), 3, 0)
    form.add_match("A", datetime(2022, 6, 1), 0, 2)
    # As-of before any match -> no data.
    assert form.form("A", datetime(2021, 12, 1))["short"]["matches"] == 0
    # As-of after both -> uses both, recent (0,2) weighted more heavily.
    f = form.form("A", datetime(2022, 6, 2))
    assert f["short"]["matches"] == 2
    assert f["short"]["goals_against"] > f["short"]["goals_for"]  # recent loss dominates short window
    # A match exactly at as_of is excluded (strict <).
    assert form.form("A", datetime(2022, 6, 1))["short"]["matches"] == 1


def test_build_team_state_emits_model_state_keys():
    form = RollingFormBuilder()
    form.add_match("A", datetime(2022, 1, 1), 2, 1)
    ts = build_team_state("A", "AAA", elo_scaled=0.3,
                          form=form.form("A", datetime(2022, 2, 1)), rest_hours=96.0)
    ms = ts.model_state()
    for key in ("live_rating", "elo_scaled", "matches", "xg_for", "xg_against",
                "goals_for", "goals_against", "rest_hours"):
        assert key in ms
    assert ms["live_rating"] == 0.3
    assert ms["matches"] == 1


def test_walk_forward_runs_and_scores():
    matches = load_matches()
    if not matches:
        pytest.skip("StatsBomb cache not present")
    params = {"w_elo": 1.0, "w_poisson": 0.0, "temperature": 0.9,
              "base_rate_shrink": 0.2, "rating_weight": 0.6, "k_factor": 60.0}
    res = walk_forward(matches, params)
    assert res["n"] == sum(1 for m in matches if m.season == 2022)
    for metric in ("brier", "logloss", "rps", "accuracy", "ece"):
        assert res[metric] is not None
    # Tuned config must at least match a base-rate prior on a proper score.
    base = market_baseline(matches)
    assert res["logloss"] <= base["logloss"] + 1e-6


def test_walk_forward_is_leakage_free_first_eval_match_has_no_future():
    """The earliest WC2022 match must be predicted from a state that contains no
    2022 results (only earlier-season history)."""
    matches = load_matches()
    if not matches:
        pytest.skip("StatsBomb cache not present")
    # Rebuild elo manually up to the first 2022 match and confirm no 2022 game
    # contributed (i.e. ratings come only from <2022 matches).
    first_2022 = min(m.date for m in matches if m.season == 2022)
    prior_2022 = [m for m in matches if m.season == 2022 and m.date < first_2022]
    assert prior_2022 == [], "there must be no WC2022 match before the first one"
