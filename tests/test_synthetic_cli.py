from agent.config import load_settings
from agent.scheduler import run_once


def test_synthetic_fixture_mode_generates_decisions(tmp_path):
    settings = load_settings(dry_run_override=True)
    settings = type(settings)(dry_run=True, storage_dir=tmp_path)
    decisions = run_once(settings, use_synthetic_fixtures=True)
    assert len(decisions) == 10
    assert all(d["prediction_submitted"] for d in decisions)
    assert any(d["fixture_code"] == "SYN-MISSING-MARKET" and "market_data_missing_for_order" in d["risk_flags"] for d in decisions)
