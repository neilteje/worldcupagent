"""Config single-source-of-truth validation (spec §24, acceptance #27)."""
from __future__ import annotations

import pytest

from harness.profiles import AgentProfile, validate_profile_config, LEGACY_BLITZ_POLICY, get_profile


def test_validate_profile_config_passes_at_startup():
    checks = validate_profile_config()
    assert checks  # performed a non-empty set of checks


def test_all_profiles_match_legacy_blitz_source_of_truth():
    for name in ("monk", "anchor", "hunter", "blitz"):
        profile = get_profile(name)
        for field_name, expected in LEGACY_BLITZ_POLICY.items():
            assert getattr(profile, field_name) == expected


def test_blitz_drift_is_detected(monkeypatch):
    import harness.profiles as P
    drifted = AgentProfile(**{**P.DEFAULT_PROFILES["blitz"].to_dict(), "kelly_fraction": 0.99})
    monkeypatch.setitem(P.DEFAULT_PROFILES, "blitz", drifted)
    with pytest.raises(ValueError, match="legacy Blitz policy drift"):
        validate_profile_config()


def test_from_config_resolves_each_agent():
    for name in ("monk", "anchor", "hunter", "blitz"):
        assert AgentProfile.from_config(name).name == name
