import math
from dataclasses import replace

from models.deterministic_v2 import EnsembleConfig, ablation_configs, predict_v2
from models.poisson_model import DEFAULT_RHO, poisson_1x2, score_matrix
from models.team_strength import StrengthConfig, effective_rating, elo_1x2, expected_goals

OUTCOMES = ("home", "draw", "away")


def _is_distribution(p):
    assert abs(sum(p[k] for k in OUTCOMES) - 1.0) < 1e-9
    assert all(0.0 < p[k] < 1.0 for k in OUTCOMES)


# ── Poisson / Dixon-Coles ─────────────────────────────────────────────────────

def test_score_matrix_normalizes():
    m = score_matrix(1.5, 1.2)
    assert abs(sum(p for row in m for p in row) - 1.0) < 1e-9


def test_poisson_1x2_is_distribution():
    _is_distribution(poisson_1x2(1.4, 1.1))


def test_equal_lambdas_are_symmetric_home_away():
    p = poisson_1x2(1.3, 1.3)
    assert abs(p["home"] - p["away"]) < 1e-9


def test_higher_lambda_side_is_favored():
    p = poisson_1x2(2.1, 0.9)
    assert p["home"] > p["away"]


def test_dixon_coles_inflates_draws_vs_plain_poisson():
    lam_h, lam_a = 1.0, 1.0
    dc = poisson_1x2(lam_h, lam_a, rho=DEFAULT_RHO)   # rho < 0
    plain = poisson_1x2(lam_h, lam_a, rho=0.0)
    assert dc["draw"] > plain["draw"]


# ── team strength ─────────────────────────────────────────────────────────────

def test_expected_goals_no_history_returns_league_average():
    cfg = StrengthConfig()
    eg = expected_goals({"live_rating": 0.0, "matches": 0}, {"live_rating": 0.0, "matches": 0}, cfg=cfg)
    assert abs(eg["lambda_home"] - cfg.league_avg_goals) < 1e-6
    assert abs(eg["lambda_away"] - cfg.league_avg_goals) < 1e-6


def test_expected_goals_stronger_rating_gets_more_goals():
    strong = {"live_rating": 0.8, "matches": 0}
    weak = {"live_rating": -0.8, "matches": 0}
    eg = expected_goals(strong, weak)
    assert eg["lambda_home"] > eg["lambda_away"]
    assert eg["supremacy"] > 0


def test_expected_goals_uses_xg_form():
    # high-scoring team with leaky opponent -> higher home lambda
    home = {"live_rating": 0.0, "matches": 3, "xg_for": 7.5, "xg_against": 2.0, "goals_for": 7, "goals_against": 2}
    away = {"live_rating": 0.0, "matches": 3, "xg_for": 2.0, "xg_against": 7.5, "goals_for": 2, "goals_against": 7}
    eg = expected_goals(home, away)
    assert eg["lambda_home"] > eg["lambda_away"]


def test_elo_1x2_favors_higher_rating_and_is_distribution():
    p = elo_1x2({"live_rating": 0.6}, {"live_rating": -0.4})
    _is_distribution(p)
    assert p["home"] > p["away"]


# ── ensemble ──────────────────────────────────────────────────────────────────

def test_predict_v2_outputs_valid_distribution():
    out = predict_v2({"live_rating": 0.3, "matches": 2, "xg_for": 3.0, "xg_against": 2.0,
                      "goals_for": 3, "goals_against": 2},
                     {"live_rating": -0.2, "matches": 2, "xg_for": 2.0, "xg_against": 3.0,
                      "goals_for": 2, "goals_against": 3},
                     market_probs={"home": 0.5, "draw": 0.27, "away": 0.23})
    _is_distribution(out["probabilities"])
    assert out["pick"] in OUTCOMES
    assert set(out["components"]) == {"elo", "poisson", "market"}


def test_predict_v2_without_market_drops_market_component():
    cfg = replace(EnsembleConfig(), use_market=False)
    out = predict_v2({"live_rating": 0.0, "matches": 0}, {"live_rating": 0.0, "matches": 0}, cfg=cfg)
    assert "market" not in out["active_components"]
    _is_distribution(out["probabilities"])


def test_predict_v2_is_deterministic():
    args = ({"live_rating": 0.1, "matches": 1, "xg_for": 1.4, "xg_against": 1.0, "goals_for": 1, "goals_against": 0},
            {"live_rating": -0.1, "matches": 1, "xg_for": 1.0, "xg_against": 1.4, "goals_for": 0, "goals_against": 1})
    a = predict_v2(*args, market_probs={"home": 0.45, "draw": 0.28, "away": 0.27})
    b = predict_v2(*args, market_probs={"home": 0.45, "draw": 0.28, "away": 0.27})
    assert a["probabilities"] == b["probabilities"]


def test_market_weight_pulls_toward_market():
    state = {"live_rating": 0.0, "matches": 0}
    market = {"home": 0.7, "draw": 0.2, "away": 0.1}
    high = predict_v2(state, state, market_probs=market, cfg=replace(EnsembleConfig(), w_market=0.95, w_elo=0.03, w_poisson=0.02))
    low = predict_v2(state, state, market_probs=market, cfg=replace(EnsembleConfig(), w_market=0.40, w_elo=0.30, w_poisson=0.30))
    assert high["probabilities"]["home"] > low["probabilities"]["home"]


def test_ablation_configs_isolate_components():
    cfgs = ablation_configs()
    assert cfgs["market_only"].use_market and not cfgs["market_only"].use_elo
    assert cfgs["stats_only_no_market"].use_elo and not cfgs["stats_only_no_market"].use_market
    assert cfgs["no_calibration"].temperature == 1.0 and cfgs["no_calibration"].base_rate_shrink == 0.0


# ── team-strength Elo blend ─────────────────────────────────────────────────

def test_effective_rating_blends_elo_and_form():
    cfg = StrengthConfig(elo_blend=0.5)
    state = {"live_rating": 0.0, "elo_scaled": 0.8}
    assert abs(effective_rating(state, cfg) - 0.4) < 1e-9
    # no elo present -> falls back to live_rating
    assert effective_rating({"live_rating": 0.3}, cfg) == 0.3


def test_elo_blend_changes_prediction():
    home = {"live_rating": 0.0, "elo_scaled": 1.0, "matches": 0}
    away = {"live_rating": 0.0, "elo_scaled": -1.0, "matches": 0}
    no_blend = predict_v2(home, away, cfg=replace(EnsembleConfig(use_market=False),
                                                  strength=StrengthConfig(elo_blend=0.0)))
    with_blend = predict_v2(home, away, cfg=replace(EnsembleConfig(use_market=False),
                                                    strength=StrengthConfig(elo_blend=0.6)))
    assert with_blend["probabilities"]["home"] > no_blend["probabilities"]["home"]


def test_knockout_boost_raises_draw_probability():
    state = {"live_rating": 0.0, "matches": 0}
    market = {"home": 0.45, "draw": 0.25, "away": 0.30}
    group = predict_v2(state, state, market_probs=market, is_knockout=False)
    knockout = predict_v2(state, state, market_probs=market, is_knockout=True)
    assert knockout["probabilities"]["draw"] > group["probabilities"]["draw"]
    _is_distribution(knockout["probabilities"])


def test_knockout_boost_disabled_when_config_zero():
    cfg = replace(EnsembleConfig(), knockout_draw_boost=0.0)
    state = {"live_rating": 0.0, "matches": 0}
    market = {"home": 0.45, "draw": 0.25, "away": 0.30}
    a = predict_v2(state, state, market_probs=market, cfg=cfg, is_knockout=False)
    b = predict_v2(state, state, market_probs=market, cfg=cfg, is_knockout=True)
    assert a["probabilities"] == b["probabilities"]


def test_calibration_prevents_extreme_probabilities():
    # extreme market should be softened, never 0/1
    out = predict_v2({"live_rating": 0.0, "matches": 0}, {"live_rating": 0.0, "matches": 0},
                     market_probs={"home": 0.98, "draw": 0.01, "away": 0.01})
    p = out["probabilities"]
    assert p["home"] <= EnsembleConfig().max_prob + 1e-9
    assert all(v > 0 for v in p.values())
