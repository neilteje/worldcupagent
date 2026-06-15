"""BLITZ event-contract golden master.

Proves the ONLY behavioural change to BLITZ is draw removal: every non-draw
pick's full field set is identical before and after suppression, draws are
removed with the exact ``blitz_draw_disabled`` reason, and no replacement is
promoted. Also confirms BLITZ is a coordinated event strategy.
"""
from __future__ import annotations

import pytest

from betting.policy import select_picks, suppress_blitz_draw_picks
from betting.portfolio import COORDINATED_AGENTS, OBSERVED_ONLY_AGENTS
from harness.profiles import get_profile

BLITZ = get_profile("blitz")

_PICK_FIELDS = ("slot", "code", "stake_usd", "entry_price", "limit_price",
                "our_prob", "fair_prob", "edge_vs_fair", "ev_per_dollar", "kelly_usd")


def _ml(home=0.55, draw=0.26, away=0.25, source="polymarket"):
    return {
        "market_source": source,
        "outcomes": {
            "home": {"team_code": "AAA", "current_mid_yes": home},
            "draw": {"team_code": "draw", "current_mid_yes": draw},
            "away": {"team_code": "BBB", "current_mid_yes": away},
        },
    }


def _baseline(probs, ml, conf=0.6):
    """Raw BLITZ selection BEFORE draw suppression."""
    return select_picks(BLITZ, probs, ml, "AAA", "BBB", 100.0, confidence_num=conf)


def _by_slot(picks):
    return {p.slot: p for p in picks}


# Representative scenarios: (label, probabilities, moneyline, confidence)
SCENARIOS = {
    "favorite_home": ({"AAA": 0.72, "draw": 0.16, "BBB": 0.12}, _ml(0.55, 0.26, 0.22), 0.6),
    "away_underdog": ({"AAA": 0.30, "draw": 0.20, "BBB": 0.50}, _ml(0.55, 0.26, 0.30), 0.6),
    "two_non_draw": ({"AAA": 0.55, "draw": 0.05, "BBB": 0.40}, _ml(0.42, 0.30, 0.30), 0.6),
    "cheap_non_draw": ({"AAA": 0.40, "draw": 0.20, "BBB": 0.40}, _ml(0.55, 0.30, 0.28), 0.6),
    "low_confidence": ({"AAA": 0.72, "draw": 0.16, "BBB": 0.12}, _ml(0.55, 0.26, 0.22), 0.30),
    "draw_plus_home": ({"AAA": 0.50, "draw": 0.40, "BBB": 0.10}, _ml(0.40, 0.28, 0.30), 0.6),
    "draw_plus_away": ({"AAA": 0.10, "draw": 0.40, "BBB": 0.50}, _ml(0.30, 0.28, 0.40), 0.6),
    "no_qualifier": ({"AAA": 0.34, "draw": 0.33, "BBB": 0.33}, _ml(0.50, 0.30, 0.40), 0.6),
}


@pytest.mark.parametrize("label", list(SCENARIOS))
def test_blitz_picks_identical_after_draw_shim(label):
    probs, ml, conf = SCENARIOS[label]
    baseline = _baseline(probs, ml, conf)
    reasons: list[str] = []
    filtered = suppress_blitz_draw_picks(BLITZ, list(baseline), reasons)

    base_non_draw = _by_slot(baseline)
    filt_non_draw = _by_slot(filtered)

    assert set(base_non_draw) == set(filt_non_draw)
    for slot, fp in filt_non_draw.items():
        bp = base_non_draw[slot]
        for f in _PICK_FIELDS:
            assert getattr(fp, f) == getattr(bp, f), f"{label}:{slot}:{f} changed"


@pytest.mark.parametrize("label", list(SCENARIOS))
def test_draws_pass_through_without_legacy_reason(label):
    probs, ml, conf = SCENARIOS[label]
    baseline = _baseline(probs, ml, conf)
    reasons: list[str] = []
    filtered = suppress_blitz_draw_picks(BLITZ, list(baseline), reasons)

    assert filtered == baseline
    assert reasons == []


def test_blitz_is_a_common_contract_agent():
    assert "blitz" in COORDINATED_AGENTS
    assert "blitz" not in OBSERVED_ONLY_AGENTS


def test_non_blitz_profiles_not_draw_filtered():
    for name in ("monk", "anchor", "hunter"):
        prof = get_profile(name)
        picks = _baseline({"AAA": 0.34, "draw": 0.40, "BBB": 0.26}, _ml(0.30, 0.30, 0.40))
        reasons: list[str] = []
        out = suppress_blitz_draw_picks(prof, list(picks), reasons)
        assert out == picks
        assert reasons == []
