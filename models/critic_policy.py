from __future__ import annotations


def merge_critic_review(decisions: list[dict], critic_review: dict | None) -> list[dict]:
    if not critic_review or not critic_review.get("ok"):
        return decisions
    parsed = critic_review.get("parsed") or {}
    notes = {
        "probability_concerns": parsed.get("probability_concerns") or [],
        "risk_flag_suggestions": parsed.get("risk_flag_suggestions") or [],
        "reporting_improvements": parsed.get("reporting_improvements") or [],
        "next_engineering_steps": parsed.get("next_engineering_steps") or [],
        "order_authorization": parsed.get("order_authorization"),
    }
    for decision in decisions:
        action = decision.get("action")
        order_submitted = decision.get("order_submitted")
        order = decision.get("order")
        risk_flags = list(decision.get("risk_flags") or [])
        if notes["risk_flag_suggestions"]:
            risk_flags.append("critic_risk_noted")
        decision["risk_flags"] = list(dict.fromkeys(risk_flags))
        decision["critic_comments"] = notes
        decision["action"] = action
        decision["order_submitted"] = order_submitted
        decision["order"] = order
    return decisions
