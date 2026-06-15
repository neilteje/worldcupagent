"""Immutable contracts + stable hashing (spec §6/§18, acceptance #19)."""
from __future__ import annotations

import copy

from agents.contracts import FixtureDataSnapshot
from agents.monk import MonkStrategy
from models.forecast_contracts import stable_hash

from conftest import make_football_context, make_snapshot


def test_snapshot_contract_constructs():
    snap = FixtureDataSnapshot(
        fixture_id="123", fixture_name="test", window="PRE_MATCH", kickoff=None,
        as_of_timestamp=None, home_code="AAA", away_code="BBB", home_name="A",
        away_name="B", sportmonks=None, supabase=None, bzzoiro=None, web=None,
        reddit=None, social=None, football_context={}, live_context={},
        market_context=None, snapshot_id="snap", snapshot_hash="hash",
    )
    assert snap.fixture_id == "123"


def test_stable_hash_is_deterministic_and_sensitive():
    payload = {"a": 1, "b": {"c": 2}}
    assert stable_hash(payload) == stable_hash(copy.deepcopy(payload))
    assert stable_hash(payload) != stable_hash({"a": 1, "b": {"c": 3}})


def test_monk_data_view_hash_changes_when_bzzoiro_input_changes():
    monk = MonkStrategy()
    base = make_football_context()
    h1 = monk.build_data_view(make_snapshot(base), None).data_view_hash

    changed = make_football_context()
    changed["bzzoiro_digest"] = {"event_id": 99, "xg": {"home": 2.4}}
    h2 = monk.build_data_view(make_snapshot(changed), None).data_view_hash
    assert h1 != h2, "data-view hash must change when a forecast-relevant BZZOIRO input changes"


def test_monk_forecast_id_stable_for_identical_inputs():
    monk = MonkStrategy()
    ff = make_football_context()
    a = monk.build_forecast(monk.build_data_view(make_snapshot(ff), None))
    b = monk.build_forecast(monk.build_data_view(make_snapshot(copy.deepcopy(ff)), None))
    assert a.forecast_id == b.forecast_id
