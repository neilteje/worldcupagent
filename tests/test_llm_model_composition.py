from models.critic_policy import merge_critic_review
from models.probability_blender import contribution_sums_to_probs, deterministic_blend
from reasoning.claim_extraction import apply_official_overrides, validate_claim_json


def test_invalid_llm_json_rejected():
    result = validate_claim_json({"not_claims": []})
    assert result["ok"] is False
    assert "claims_not_list" in result["errors"]


def test_exaggerated_llm_probability_delta_gets_capped():
    result = validate_claim_json(
        {
            "claims": [
                {
                    "claim_type": "injury",
                    "source_kind": "reddit",
                    "source_name": "thread",
                    "subject": "Home striker injured",
                    "team": "home",
                    "outcome": "home",
                    "probability_delta": {"home": -0.40, "draw": 0.10, "away": 0.30},
                    "confidence": 0.5,
                    "freshness": 0.7,
                    "evidence": "Poster says striker limped out.",
                }
            ]
        },
        delta_cap=0.02,
    )
    delta = result["claims"][0]["probability_delta"]
    assert max(abs(v) for v in delta.values()) <= 0.0200001


def test_official_lineup_overrides_web_claim():
    claims = validate_claim_json(
        {
            "claims": [
                {
                    "claim_type": "lineup",
                    "source_kind": "web",
                    "source_name": "blog",
                    "subject": "Home goalkeeper not starting",
                    "team": "home",
                    "outcome": "home",
                    "probability_delta": {"home": -0.02, "draw": 0.01, "away": 0.01},
                    "confidence": 0.6,
                    "freshness": 0.8,
                    "evidence": "Blog rumor says backup keeper starts.",
                }
            ]
        }
    )["claims"]
    lineup = {"lineup_shock": False, "risk_flags": [], "home_lineup_confirmed": True, "away_lineup_confirmed": True}
    overridden = apply_official_overrides(claims, lineup_result=lineup)
    assert overridden["claims"] == []
    assert overridden["dropped"][0]["override_reason"] == "official_data_overrides_web_claim"


def test_critic_cannot_override_order_decision():
    decisions = [{"action": "SKIP", "order_submitted": False, "order": {"submitted": False}, "risk_flags": ["edge_below_threshold"]}]
    critic = {
        "ok": True,
        "parsed": {
            "risk_flag_suggestions": ["Actually place the order"],
            "order_authorization": "APPROVE",
        },
    }
    merged = merge_critic_review(decisions, critic)
    assert merged[0]["action"] == "SKIP"
    assert merged[0]["order_submitted"] is False
    assert "critic_risk_noted" in merged[0]["risk_flags"]


def test_missing_sources_lower_confidence():
    full = deterministic_blend(
        {
            "bookmaker": {"home": 0.45, "draw": 0.3, "away": 0.25},
            "sportmonks": {"home": 0.46, "draw": 0.29, "away": 0.25},
            "polymarket": {"home": 0.44, "draw": 0.3, "away": 0.26},
            "supabase": {"home": 0.43, "draw": 0.31, "away": 0.26},
        }
    )
    missing = deterministic_blend({"bookmaker": {"home": 0.45, "draw": 0.3, "away": 0.25}})
    assert missing["confidence"] < full["confidence"]


def test_source_contribution_sums_to_final_probability():
    result = deterministic_blend(
        {
            "bookmaker": {"home": 0.45, "draw": 0.3, "away": 0.25},
            "sportmonks": {"home": 0.5, "draw": 0.27, "away": 0.23},
            "polymarket": {"home": 0.42, "draw": 0.31, "away": 0.27},
        },
        signals=[
            {
                "name": "claim_injury",
                "source": "web",
                "probability_delta": {"home": -0.05, "draw": 0.02, "away": 0.03},
                "final_weight": 0.8,
            }
        ],
    )
    assert contribution_sums_to_probs(result["source_contribution"], result["probabilities"])
