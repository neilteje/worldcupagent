"""Offline tests for the live runner: state resume, policy, ledger schema, report."""
from __future__ import annotations
import json

import pytest

from betting.policy import select_picks
from harness.profiles import get_profile
from ledger.client import LedgerSession
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


def test_policy_saw_never_buys_above_max_entry_price():
    saw = get_profile("hunter")
    assert saw.max_entry_price == 0.40
    # Big edge on the favourite (priced 0.55) — saw must refuse it.
    probs = {"AAA": 0.70, "draw": 0.18, "BBB": 0.12}
    reasons: list[str] = []
    picks = select_picks(saw, probs, _ml(), "AAA", "BBB", 100.0,
                         confidence_num=0.6, skip_reasons=reasons)
    assert all(p.entry_price <= 0.40 for p in picks)
    assert any("skew filter" in r for r in reasons)


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


def test_policy_confidence_floor():
    monk = get_profile("monk")
    probs = {"AAA": 0.80, "draw": 0.12, "BBB": 0.08}
    picks = select_picks(monk, probs, _ml(), "AAA", "BBB", 100.0,
                         confidence_num=0.40)   # below monk's 0.55 floor
    assert picks == []


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


# ── Report math ──────────────────────────────────────────────────────────────

def test_report_brier_and_devig():
    from live.report import _brier, _devig, _winner_key
    probs = {"MEX": 0.5, "draw": 0.3, "RSA": 0.2}
    assert _brier(probs, "MEX") == pytest.approx(0.25 + 0.09 + 0.04)
    devig = _devig({"home": 0.55, "draw": 0.28, "away": 0.27})
    assert sum(devig.values()) == pytest.approx(1.0)
    assert _winner_key(probs, "home", None, "MEX", "RSA") == "MEX"
    assert _winner_key(probs, None, "draw", "MEX", "RSA") == "draw"
