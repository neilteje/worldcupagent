"""Deterministic halftime model (spec §20, acceptance #21/#22)."""
from __future__ import annotations

from models.live_state import LiveMatchState
from models.live_update import DeterministicHalftimeModel, INSUFFICIENT_LIVE_EVIDENCE


def _state(**kw):
    base = dict(
        fixture_id="900", current_minute=45, match_period="HT",
        home_score=0, away_score=0, home_red_cards=0, away_red_cards=0,
        home_live_xg=None, away_live_xg=None, home_dangerous_attacks=None,
        away_dangerous_attacks=None, home_possession=None, away_possession=None,
        data_coverage={"overall": 0.9},
    )
    base.update(kw)
    return LiveMatchState(**base)


def test_full_1x2_sums_to_one():
    m = DeterministicHalftimeModel()
    out = m.update_forecast(1.5, 1.0, _state())
    p = out["probabilities"]
    assert abs(p["home"] + p["draw"] + p["away"] - 1.0) < 1e-9


def test_remaining_time_reduction():
    m = DeterministicHalftimeModel()
    early = m.update_forecast(1.5, 1.0, _state(current_minute=10))
    late = m.update_forecast(1.5, 1.0, _state(current_minute=80))
    assert late["remaining_minutes"] < early["remaining_minutes"]
    assert late["remaining_lambda_home"] < early["remaining_lambda_home"]


def test_current_score_convolution_favors_leader():
    m = DeterministicHalftimeModel()
    level = m.update_forecast(1.3, 1.3, _state(home_score=0, away_score=0))
    home_up = m.update_forecast(1.3, 1.3, _state(home_score=2, away_score=0))
    assert home_up["probabilities"]["home"] > level["probabilities"]["home"]


def test_red_card_adjustment_hurts_the_down_a_man_side():
    m = DeterministicHalftimeModel()
    neutral = m.update_forecast(1.3, 1.3, _state())
    home_red = m.update_forecast(1.3, 1.3, _state(home_red_cards=1))
    assert home_red["probabilities"]["home"] < neutral["probabilities"]["home"]
    assert home_red["live_multiplier_home"] < 1.0


def test_live_xg_adjustment_boosts_the_attacking_side():
    m = DeterministicHalftimeModel()
    neutral = m.update_forecast(1.3, 1.3, _state())
    hot_home = m.update_forecast(1.3, 1.3, _state(home_live_xg=2.0, away_live_xg=0.1))
    assert hot_home["live_multiplier_home"] >= neutral["live_multiplier_home"]
    assert hot_home["probabilities"]["home"] > neutral["probabilities"]["home"]


def test_missing_live_data_preserves_prematch_and_widens_bounds():
    m = DeterministicHalftimeModel()
    prematch = {"home": 0.5, "draw": 0.3, "away": 0.2}
    out = m.update_forecast(1.5, 1.0, _state(data_coverage={"overall": 0.1}),
                            prematch_probs=prematch)
    assert out["insufficient_evidence"] is True
    assert INSUFFICIENT_LIVE_EVIDENCE in out["warnings"]
    # Pre-match preserved.
    assert abs(out["probabilities"]["home"] - 0.5) < 1e-9
    # Bounds are wide.
    assert (out["upper_bounds"]["home"] - out["lower_bounds"]["home"]) >= 0.30


def test_coordinated_agents_abstain_on_insufficient_evidence_flag():
    """The flag is the signal coordinated agents key on to skip new HT risk."""
    m = DeterministicHalftimeModel()
    out = m.update_forecast(1.5, 1.0, _state(data_coverage={"overall": 0.0}))
    assert out["insufficient_evidence"] is True
    assert INSUFFICIENT_LIVE_EVIDENCE in out["warnings"]
