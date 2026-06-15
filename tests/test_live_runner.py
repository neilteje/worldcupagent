"""Offline tests for the live runner: state resume, policy, ledger schema, report."""
from __future__ import annotations
import json

import pytest

from betting.policy import select_picks
from harness.profiles import get_profile
from ledger.client import LedgerSession
from live.cycle import Forecast, _deterministic_context_for_council
from live.state import LiveState
from live.runner import LiveRunner, flatten_schedule, _parse_kickoff


# ── State: resumability ──────────────────────────────────────────────────────

def test_state_roundtrip_and_resume(tmp_path):
    path = tmp_path / "state.json"
    s = LiveState(path)
    assert not s.window_done(1, "PRE_MATCH")
    s.mark_window(1, "PRE_MATCH", "done", fixture_name="A vs B",
                  agents={"monk": {"n_picks": 0}})
    s.mark_window(2, "PRE_MATCH", "failed", fixture_name="C vs D")
    s.mark_settlement(1, resolved=True, winner_slot="home", winner_code="MEX")

    # Reload from disk — the restart path.
    s2 = LiveState(path)
    assert s2.window_done(1, "PRE_MATCH")
    assert not s2.window_done(2, "PRE_MATCH")        # failed → retryable
    assert s2.window_attempts(2, "PRE_MATCH") == 1
    assert s2.settled(1)
    assert s2.settlement(1)["winner_code"] == "MEX"


def test_state_failed_window_exhausts_after_max_attempts(tmp_path):
    s = LiveState(tmp_path / "state.json")
    for _ in range(3):
        s.mark_window(5, "HT", "failed", fixture_name="X vs Y")
    assert s.window_exhausted(5, "HT")


def test_state_survives_corrupt_file(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not json", encoding="utf-8")
    s = LiveState(path)               # must not raise
    assert s.summary()["windows_total"] == 0


# ── Policy: shared selection ─────────────────────────────────────────────────

def _ml(home=0.55, draw=0.26, away=0.25, source="polymarket"):
    return {
        "market_source": source,
        "outcomes": {
            "home": {"team_code": "AAA", "current_mid_yes": home},
            "draw": {"team_code": "draw", "current_mid_yes": draw},
            "away": {"team_code": "BBB", "current_mid_yes": away},
        },
    }


def test_policy_saw_backs_the_favorite_under_conviction():
    # Conviction: SAW no longer bans favorites — it backs the best-EV pick.
    saw = get_profile("hunter")
    assert saw.max_entry_price is None
    probs = {"AAA": 0.70, "draw": 0.18, "BBB": 0.12}   # strong favorite, underpriced
    picks = select_picks(saw, probs, _ml(), "AAA", "BBB", 100.0,
                         confidence_num=0.6)
    assert picks, "SAW should back the favorite when it carries the edge"
    assert picks[0].code == "AAA"


def test_policy_keel_takes_clear_edge():
    keel = get_profile("anchor")
    probs = {"AAA": 0.70, "draw": 0.18, "BBB": 0.12}
    picks = select_picks(keel, probs, _ml(), "AAA", "BBB", 100.0,
                         confidence_num=0.6)
    assert len(picks) == 1
    assert picks[0].code == "AAA"
    assert 1.0 <= picks[0].stake_usd <= keel.max_bet_usd
    assert picks[0].limit_price == pytest.approx(0.57, abs=0.011)


def test_policy_synthetic_market_blocked_by_default():
    keel = get_profile("anchor")
    probs = {"AAA": 0.70, "draw": 0.18, "BBB": 0.12}
    reasons: list[str] = []
    picks = select_picks(keel, probs, _ml(source="synthetic_demo"),
                         "AAA", "BBB", 100.0, confidence_num=0.6,
                         skip_reasons=reasons)
    assert picks == []
    assert any("synthetic" in r for r in reasons)


def test_policy_floors_sub_dollar_blitz_pick_up_to_minimum():
    # BLITZ retains the existing floor-up behavior.
    surge = get_profile("blitz")
    assert surge.floor_to_min_order
    # Tiny bankroll so stake_cap_fraction drives size under $1.
    probs = {"AAA": 0.30, "draw": 0.50, "BBB": 0.20}
    ml = _ml(home=0.55, draw=0.30, away=0.25)
    reasons: list[str] = []
    picks = select_picks(surge, probs, ml, "AAA", "BBB", 4.0,
                         confidence_num=0.6, skip_reasons=reasons)
    assert picks, "expected a floored +EV draw pick"
    assert picks[0].stake_usd == pytest.approx(1.0)
    assert any("floored up" in r for r in reasons)


def test_policy_coordinated_agent_skips_subminimum_kelly():
    saw = get_profile("hunter")
    assert saw.floor_to_min_order is False
    probs = {"AAA": 0.30, "draw": 0.50, "BBB": 0.20}
    ml = _ml(home=0.55, draw=0.30, away=0.25)
    # Coordinated agents do not round subminimum Kelly stakes up.
    reasons: list[str] = []
    picks = select_picks(saw, probs, ml, "AAA", "BBB", 6.0,
                         confidence_num=0.6, skip_reasons=reasons)
    assert picks == []
    assert any("kelly_below_minimum_order_size" in r for r in reasons)


def test_pnl_tail_profiles_opt_out_of_confidence_multiplier():
    assert get_profile("hunter").apply_confidence_multiplier is False
    assert get_profile("blitz").apply_confidence_multiplier is False
    # The disciplined/score agents keep it on.
    assert get_profile("anchor").apply_confidence_multiplier is True
    assert get_profile("monk").apply_confidence_multiplier is True


def test_policy_confidence_floor():
    monk = get_profile("monk")
    probs = {"AAA": 0.80, "draw": 0.12, "BBB": 0.08}
    picks = select_picks(monk, probs, _ml(), "AAA", "BBB", 100.0,
                         confidence_num=0.40)   # below monk's 0.55 floor
    assert picks == []


def test_live_prematch_builds_deterministic_context_for_council():
    fx = Forecast(
        fixture_id=1,
        window="PRE_MATCH",
        home_code="AAA",
        away_code="BBB",
        moneyline=_ml(home=0.50, draw=0.28, away=0.24),
        sm_digest={"bookmaker_consensus_win_prob": {"AAA": 0.50, "draw": 0.28, "BBB": 0.22}},
    )
    ctx = _deterministic_context_for_council(fx, {"stage": {"name": "Group Stage"}})
    assert ctx["engine"] == "deterministic_v2"
    assert ctx["model_version"] == "deterministic_v2.0"
    assert set(ctx["probabilities_by_code"]) == {"AAA", "draw", "BBB"}
    assert sum(ctx["probabilities_by_code"].values()) == pytest.approx(1.0, abs=0.001)
    assert "components" in ctx and "expected_goals" in ctx
    assert ctx["component_weights"]["market"] == pytest.approx(0.8)


# ── Ledger: schema v0.3 shapes ───────────────────────────────────────────────

def test_prediction_record_uses_fixture_id():
    s = LedgerSession(19609127, "MEX vs RSA", "PRE_MATCH", api_key="k")
    rec = s.acting_prediction("MEX", 0.45)
    assert rec["parameters"]["fixture_id"] == "19609127"
    assert "fixture_code" not in rec["parameters"]
    assert 0.001 <= rec["parameters"]["probability"] <= 0.999


def test_planning_record_has_goal_and_steps():
    s = LedgerSession(1, "A vs B", "PRE_MATCH", api_key="k")
    rec = s.planning(goal="win", steps=["fetch", "think", "act"])
    assert rec["behavior"] == "Planning"
    assert rec["goal"] == "win"
    assert [st["index"] for st in rec["steps"]] == [0, 1, 2]
    assert all(st["description"] for st in rec["steps"])
    assert "output_payload" not in rec and "description" not in rec


def test_reflecting_record_has_inputs_and_string_payload():
    s = LedgerSession(1, "A vs B", "PRE_MATCH", api_key="k")
    rec = s.reflecting(inputs=[{"payload": {"x": 1}}],
                       output_payload={"verdict": "ok"})
    assert rec["behavior"] == "Reflecting"
    assert isinstance(rec["inputs"][0]["input_payload"], str)
    assert "input_record_id" not in rec["inputs"][0]   # null stripped
    assert isinstance(rec["output_payload"], str)


def test_thinking_record_omits_empty_model_invocation():
    s = LedgerSession(1, "A vs B", "PRE_MATCH", api_key="k")
    rec = s.thinking(prompt_system="p", inputs=[{"payload": "x"}],
                     output_payload="y", provider="", model_name="")
    assert "model_invocation" not in rec


def test_session_ids_distinct_per_agent():
    a = LedgerSession(1, "A vs B", "PRE_MATCH", api_key="k1", agent_tag="monk")
    b = LedgerSession(1, "A vs B", "PRE_MATCH", api_key="k2", agent_tag="blitz")
    assert a.session_id != b.session_id
    assert "monk" in a.session_id and "blitz" in b.session_id


# ── Runner helpers ───────────────────────────────────────────────────────────

def test_parse_kickoff_and_flatten():
    ko = _parse_kickoff("2026-06-11 18:00:00")
    assert ko is not None and ko.tzinfo is not None
    schedule = [
        {"rounds": [{"fixtures": [
            {"id": 1, "starting_at": "2026-06-11 18:00:00", "name": "A vs B"},
            {"id": 2, "starting_at": "2026-06-11 15:00:00", "name": "C vs D"},
        ]}]},
        {"id": 2, "participants": [{}], "starting_at": "2026-06-11 15:00:00",
         "name": "C vs D"},  # duplicate of id 2 — must be deduped
    ]
    flat = flatten_schedule(schedule)
    assert [f["id"] for f in flat] == [2, 1]   # sorted by kickoff, deduped


def test_winner_from_settlement_parsing():
    parse = LiveRunner._winner_from_settlement
    slot, code = parse({"outcomes": {
        "home": {"team_code": "MEX", "resolved_price": 1.0},
        "draw": {"team_code": "draw", "resolved_price": 0.0},
        "away": {"team_code": "RSA", "resolved_price": 0.0}}})
    assert (slot, code) == ("home", "MEX")
    slot, code = parse({"markets": [
        {"outcome": "draw", "settlement_price": "1"},
        {"outcome": "ENG", "settlement_price": "0"}]})
    assert code == "draw"
    assert parse({"markets": [{"outcome": "ENG", "price": 0.5}]}) == (None, None)


def test_window_open_infers_ht_from_server_timestamps(monkeypatch):
    runner = LiveRunner.__new__(LiveRunner)

    class Reader:
        def match(self, fixture_id):
            return {
                "fixture_id": str(fixture_id),
                "current_window": None,
                "server_ts_utc": 1_000,
                "ht_open_utc": 900,
                "ht_lock_utc": 1_200,
            }

    runner.reader = Reader()
    assert runner.window_open(123, "HT") is True


def test_window_open_infers_prematch_from_server_timestamps(monkeypatch):
    runner = LiveRunner.__new__(LiveRunner)

    class Reader:
        def match(self, fixture_id):
            return {
                "fixture_id": str(fixture_id),
                "current_window": None,
                "server_ts_utc": 1_000,
                "pre_match_lock_utc": 1_200,
            }

    runner.reader = Reader()
    assert runner.window_open(123, "PRE_MATCH") is True


def test_arena_client_close_fixture_orders_filters_fixture_and_status(monkeypatch):
    from live.arena_client import ArenaClient

    client = ArenaClient(api_key="k")
    closed: list[str] = []

    def fake_orders(status=None):
        return [
            {"order_id": "keep-1", "fixture_id": "123", "status": "filled"},
            {"order_id": "skip-terminal", "fixture_id": "123", "status": "closed"},
            {"order_id": "skip-other", "fixture_id": "999", "status": "filled"},
        ]

    def fake_close(order_id):
        closed.append(order_id)
        return {"order_id": order_id, "status": "closed"}

    monkeypatch.setattr(client, "orders", fake_orders)
    monkeypatch.setattr(client, "close_order", fake_close)

    results = client.close_fixture_orders(123)
    assert closed == ["keep-1"]
    assert results == [{"order_id": "keep-1", "status": "closed", "previous_status": "filled"}]


# ── Report math ──────────────────────────────────────────────────────────────

def test_report_brier_and_devig():
    from live.report import _brier, _devig, _winner_key
    probs = {"MEX": 0.5, "draw": 0.3, "RSA": 0.2}
    assert _brier(probs, "MEX") == pytest.approx(0.25 + 0.09 + 0.04)
    devig = _devig({"home": 0.55, "draw": 0.28, "away": 0.27})
    assert sum(devig.values()) == pytest.approx(1.0)
    assert _winner_key(probs, "home", None, "MEX", "RSA") == "MEX"
    assert _winner_key(probs, None, "draw", "MEX", "RSA") == "draw"
