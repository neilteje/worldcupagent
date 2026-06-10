from models.archetype import classify_match_archetype
from models.council_reconciliation import reconcile_council_output
from models.meta_arbiter import arbitrate_forecast
from models.probability_blender import DEFAULT_PREMATCH_WEIGHTS
from models.source_reliability import dynamic_source_weights
from reasoning.counterfactuals import decision_counterfactuals


def test_archetype_detects_market_against_model_bookmaker():
    out = classify_match_archetype(
        window="PRE_MATCH",
        model_probs={"home": 0.52, "draw": 0.25, "away": 0.23},
        bookmaker_probs={"home": 0.51, "draw": 0.26, "away": 0.23},
        market_probs={"home": 0.30, "draw": 0.27, "away": 0.43},
        lineup={"risk_flags": []},
        data_completeness={"score": 0.8},
    )
    assert out["market_regime"] == "market_against_model_bookmaker"
    assert "model_bookmaker_vs_market" in out["tags"]


def test_dynamic_weights_reduce_stale_market_weight():
    archetype = {"market_regime": "market_stale", "tags": ["stale_market", "rich_data"]}
    out = dynamic_source_weights(DEFAULT_PREMATCH_WEIGHTS, archetype=archetype, data_completeness={"score": 0.9})
    assert out["weights"]["polymarket"] < DEFAULT_PREMATCH_WEIGHTS["polymarket"]
    assert any("Stale-market" in reason for reason in out["reasons"])


def test_council_adjustment_requires_evidence_and_is_bounded():
    council = {
        "probabilities": {"home": 0.70, "draw": 0.18, "away": 0.12},
        "evidence": [{"source_kind": "official", "confidence": 1.0}, {"source_kind": "bookmaker", "confidence": 0.9}],
        "recommendation": "BET",
        "risk_posture": "approve",
    }
    out = reconcile_council_output(
        council,
        deterministic_reference={"probabilities": {"home": 0.50, "draw": 0.27, "away": 0.23}},
        max_delta=0.05,
    )
    assert out["accepted"]
    assert max(abs(v) for v in out["bounded_delta"].values()) <= 0.0500001


def test_meta_arbiter_keeps_base_when_council_weak():
    base = {"probabilities": {"home": 0.45, "draw": 0.30, "away": 0.25}, "confidence": 0.7, "uncertainty": 0.25, "risk_flags": []}
    council = reconcile_council_output(
        {"probabilities": {"home": 0.25, "draw": 0.25, "away": 0.50}, "evidence": [{"source_kind": "rumor", "confidence": 0.2}]},
        deterministic_reference=base,
    )
    out = arbitrate_forecast(base, council_reconciliation=council)
    assert out["probabilities"] == base["probabilities"]
    assert "council_low_evidence" in out["risk_flags"]


def test_counterfactuals_explain_dry_run_and_price_threshold():
    facts = decision_counterfactuals(
        model_probs={"home": 0.50, "draw": 0.28, "away": 0.22},
        market_probs={"home": 0.46, "draw": 0.30, "away": 0.24},
        edge={"best_outcome": "home", "best_edge": 0.04, "edge_tier": "soft"},
        risk={"blocking_risk_flags": ["dry_run_enabled"]},
        confidence=0.7,
        uncertainty=0.3,
    )
    assert any("home_market_price" in fact["condition"] for fact in facts)
    assert any(fact["condition"] == "DRY_RUN=false" for fact in facts)
