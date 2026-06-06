from models.llm_central import normalize_central_prediction


def test_llm_central_uses_model_output_when_valid():
    normalized = normalize_central_prediction(
        {
            "ok": True,
            "parsed": {
                "probabilities": {"home": 0.52, "draw": 0.24, "away": 0.24},
                "confidence": 0.74,
                "uncertainty": 0.22,
                "recommendation": "BET",
                "risk_posture": "approve",
                "supporting_signals": ["lineup edge"],
            },
        },
        fallback_probs={"home": 0.4, "draw": 0.3, "away": 0.3},
        fallback_confidence=0.6,
        fallback_uncertainty=0.3,
    )
    assert normalized["used_fallback"] is False
    assert normalized["probabilities"]["home"] > 0.5
    assert normalized["blocking_risk_flags"] == []


def test_llm_central_missing_result_blocks_and_falls_back():
    normalized = normalize_central_prediction(
        None,
        fallback_probs={"home": 0.4, "draw": 0.3, "away": 0.3},
        fallback_confidence=0.6,
        fallback_uncertainty=0.3,
    )
    assert normalized["used_fallback"] is True
    assert normalized["blocking_risk_flags"] == ["llm_central_missing"]
    assert normalized["probabilities"] == {"home": 0.4, "draw": 0.3, "away": 0.3}


def test_llm_central_watch_blocks_even_with_valid_probs():
    normalized = normalize_central_prediction(
        {
            "ok": True,
            "parsed": {
                "probabilities": {"home": 0.45, "draw": 0.28, "away": 0.27},
                "confidence": 0.66,
                "uncertainty": 0.25,
                "recommendation": "WATCH",
                "risk_posture": "caution",
            },
        },
        fallback_probs={"home": 0.4, "draw": 0.3, "away": 0.3},
        fallback_confidence=0.6,
        fallback_uncertainty=0.3,
    )
    assert "llm_central_recommends_watch" in normalized["blocking_risk_flags"]
    assert "llm_central_not_betting" in normalized["blocking_risk_flags"]
