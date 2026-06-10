from models.llm_decision import llm_analysis_flags, merge_llm_analysis_into_risk


def test_llm_bet_approve_adds_nonblocking_custom_flag_only():
    analysis = {
        "ok": True,
        "parsed": {
            "recommendation": "BET",
            "risk_posture": "approve",
            "additional_risk_flags": ["lineup_needs_watch"],
        },
    }
    assert llm_analysis_flags(analysis) == ["lineup_needs_watch"]


def test_llm_watch_blocks_order():
    risk = {"risk_flags": [], "blocking_risk_flags": [], "order_allowed": True}
    analysis = {"ok": True, "parsed": {"recommendation": "WATCH", "risk_posture": "caution"}}
    merged = merge_llm_analysis_into_risk(risk, analysis)
    assert "llm_recommends_watch" in merged["blocking_risk_flags"]
    assert "llm_not_approving_order" in merged["blocking_risk_flags"]
    assert merged["order_allowed"] is False


def test_missing_llm_analysis_blocks_when_required():
    risk = {"risk_flags": [], "blocking_risk_flags": [], "order_allowed": True}
    merged = merge_llm_analysis_into_risk(risk, None)
    assert merged["blocking_risk_flags"] == ["llm_analysis_missing"]
    assert merged["order_allowed"] is False
