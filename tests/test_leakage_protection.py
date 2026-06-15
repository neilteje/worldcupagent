"""MONK market-blindness + Analyst market-blindness (spec §7/§12, acceptance #5/#6/#15)."""
from __future__ import annotations

from agents.monk import MonkStrategy
from reasoning.market_blind import scrub_market_fields
from reasoning.prompts import analyst_input

from conftest import make_football_context, make_snapshot


def test_monk_view_scrubs_market_fields():
    ff = make_football_context(with_market=True)
    monk = MonkStrategy()
    view = monk.build_data_view(make_snapshot(ff), None)
    assert view.prohibited_fields_removed, "scrubber should report removed market fields"
    blob = str(view.football_features).lower()
    assert "polymarket" not in blob
    assert "bookmaker" not in blob
    assert view.market_features is None


def test_monk_forecast_invariant_to_market_changes():
    monk = MonkStrategy()
    base = make_football_context(with_market=False)
    fc_no_market = monk.build_forecast(monk.build_data_view(make_snapshot(base), None))

    with_market = make_football_context(with_market=True)
    with_market["polymarket_mid"] = 0.99  # extreme market move
    fc_market = monk.build_forecast(monk.build_data_view(make_snapshot(with_market), None))

    assert fc_no_market.forecast_id == fc_market.forecast_id, "market changes must not move MONK's forecast id"
    assert abs(fc_no_market.home_probability - fc_market.home_probability) < 1e-9
    assert abs(fc_no_market.draw_probability - fc_market.draw_probability) < 1e-9
    assert abs(fc_no_market.away_probability - fc_market.away_probability) < 1e-9


def test_scrubber_fails_closed_recursively():
    payload = {
        "keep": 1,
        "nested": {"bookmaker_consensus": {"AAA": 0.7}, "xg": 1.2},
        "rows": [{"odds": 2.1, "team": "AAA"}],
    }
    clean, removed = scrub_market_fields(payload)
    assert "bookmaker_consensus" not in clean["nested"]
    assert clean["nested"]["xg"] == 1.2
    assert "odds" not in clean["rows"][0]
    assert clean["rows"][0]["team"] == "AAA"
    assert removed


def test_analyst_prompt_contains_no_market_fields():
    ff = make_football_context(with_market=True)
    sm_clean, _ = scrub_market_fields(ff["sportmonks_digest"])
    bz_clean, _ = scrub_market_fields(ff["bzzoiro_digest"])
    det_clean, _ = scrub_market_fields(ff["deterministic_model"])
    prompt = analyst_input("AAA vs BBB", "AAA", "BBB", sm_clean, ff["supabase_digest"],
                           {"flags": []}, deterministic_context=det_clean, bz_digest=bz_clean)
    low = prompt.lower()
    for token in ("polymarket", "kalshi", "bookmaker", "midpoint", "best_ask"):
        assert token not in low, f"market token {token!r} leaked into Analyst prompt"
