from models.signal_scoring import score_signal, signal_conflict_score
from models.draw_model import apply_draw_model, draw_sanity_flags
from models.market_stale import detect_market_stale
from models.sanity_checks import audit_decision


def test_weak_web_signal_is_capped():
    s = score_signal("rumor", "web", "rumor", {"home": .08, "draw": -.04, "away": -.04}, source_quality=.2, corroboration=.1)
    assert s["impact"] <= .015
    assert s["final_weight"] < .4


def test_signal_conflict_score_detects_conflicting_directions():
    a = score_signal("a", "sportmonks", "model", {"home": .04, "draw": -.02, "away": -.02})
    b = score_signal("b", "bookmaker", "odds", {"home": -.02, "draw": -.02, "away": .04})
    assert signal_conflict_score([a, b]) > 0


def test_draw_model_boosts_low_xg_even_fixture():
    out = apply_draw_model({"home": .38, "draw": .25, "away": .37}, total_projected_xg=1.8, strength_gap=.01)
    assert out["probabilities"]["draw"] > .25
    assert out["delta"]["draw"] > 0


def test_draw_sanity_flags_unexplained_low_draw():
    assert "draw_probability_requires_reason" in draw_sanity_flags({"home": .78, "draw": .12, "away": .10})


def test_market_stale_detects_bookmaker_signal_without_market_move():
    out = detect_market_stale({"home": .481, "draw": .279, "away": .240}, {"home": .480, "draw": .280, "away": .240}, {"home": .57, "draw": .24, "away": .19}, {"home": .04, "draw": 0, "away": -.04})
    assert out["is_stale"] is True
    assert out["edge_type"] == "market_stale"


def test_duplicate_order_blocks_order_allowed():
    risk = audit_decision({"home": .45, "draw": .28, "away": .27}, {"edge_tier": "medium"}, .70, .25, False, True, duplicate_order=True)
    assert "duplicate_order" in risk["blocking_risk_flags"]
    assert risk["order_allowed"] is False
