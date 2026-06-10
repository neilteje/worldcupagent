import json

from agent.config import load_settings
from agent.run_cycle import run_cycle
from reasoning.ledger_builder import LedgerBuilder
from reasoning.trace_quality import evaluate_trace


def test_ledger_records_have_scoreable_schema_and_trace():
    lb = LedgerBuilder("F", "PRE_MATCH", load_settings(True))
    records = lb.build_standard_trace(
        sportmonks={"fixture_id": "F", "prediction": {"home": 0.45, "draw": 0.3, "away": 0.25}},
        supabase={"priors": {"home": 0.4, "draw": 0.3, "away": 0.3}},
        polymarket={"complete": True, "normalized_probs": {"home": 0.43, "draw": 0.31, "away": 0.26}},
        bookmaker={"home": 0.46, "draw": 0.29, "away": 0.25},
        lineup={"probability_delta": {"home": 0.0, "draw": 0.0, "away": 0.0}, "risk_flags": []},
        probability={
            "probabilities": {"home": 0.44, "draw": 0.3, "away": 0.26},
            "confidence": 0.7,
            "uncertainty": 0.25,
            "weights": {"bookmaker": 0.3, "sportmonks": 0.3, "polymarket": 0.2, "supabase": 0.2},
            "source_contribution": {},
            "risk_flags": [],
        },
        consensus={"case": "all_agree"},
        edge={"edge_tier": "none", "best_outcome": "home", "best_edge": 0.01, "reason": "below threshold"},
        risk={"risk_flags": ["edge_below_threshold"], "blocking_risk_flags": ["edge_below_threshold"], "order_allowed": False},
        prediction={"fixture_code": "F", "window": "PRE_MATCH", "probabilities": {"home": 0.44, "draw": 0.3, "away": 0.26}, "confidence": 0.7},
        order={"action_type": "skip", "reason": "skip"},
        reflection={"decision": "skip", "data_complete": 0.8},
    )
    assert lb.validate_dag()
    assert all(r["schema_version"] == "0.3" for r in records)
    assert all("client_ts_utc" in r for r in records)
    assert all("reasoning_trace" in r for r in records)
    assert any(r["behavior"] == "Acting" and r.get("action_type") == "prediction" for r in records)
    assert any(r["behavior"] == "Thinking" and r.get("model_invocation", {}).get("internal_reasoning") for r in records)
    quality = evaluate_trace(records)
    assert quality["score"] >= 0.9
    assert not quality["gaps"]


def test_run_cycle_persists_trace_quality_and_local_ledger(tmp_path):
    settings = type(load_settings(True))(dry_run=True, storage_dir=tmp_path)
    decision = run_cycle({"id": "TRACE-DEMO", "fixture_code": "TRACE-DEMO", "demo": True}, "PRE_MATCH", settings)
    assert decision["trace_quality"]["score"] >= 0.9
    run_file = tmp_path / "runs" / f"{decision['session_id']}.json"
    assert run_file.exists()
    payload = json.loads(run_file.read_text(encoding="utf-8"))
    assert payload["records"]
    assert any(r["behavior"] == "Reflecting" and "trace_quality" in r for r in payload["records"])
