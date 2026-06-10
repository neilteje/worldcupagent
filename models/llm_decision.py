from __future__ import annotations


LLM_BLOCKING_FLAGS = {
    "llm_analysis_missing",
    "llm_analysis_failed",
    "llm_recommends_skip",
    "llm_recommends_watch",
    "llm_veto",
    "llm_not_approving_order",
}


def llm_analysis_flags(llm_analysis: dict | None) -> list[str]:
    if not llm_analysis:
        return ["llm_analysis_missing"]
    if not llm_analysis.get("ok"):
        return ["llm_analysis_failed"]

    parsed = llm_analysis.get("parsed") or {}
    flags: list[str] = []
    recommendation = str(parsed.get("recommendation") or "").strip().upper()
    posture = str(parsed.get("risk_posture") or "").strip().lower()

    if recommendation == "SKIP":
        flags.append("llm_recommends_skip")
    elif recommendation == "WATCH":
        flags.append("llm_recommends_watch")
    elif recommendation != "BET":
        flags.append("llm_not_approving_order")

    if posture == "veto":
        flags.append("llm_veto")
    elif posture != "approve":
        flags.append("llm_not_approving_order")

    for flag in parsed.get("additional_risk_flags") or []:
        normalized = str(flag).strip().lower().replace("-", "_").replace(" ", "_")
        if normalized:
            flags.append(normalized[:80])
    return list(dict.fromkeys(flags))


def merge_llm_analysis_into_risk(risk: dict, llm_analysis: dict | None) -> dict:
    flags = llm_analysis_flags(llm_analysis)
    risk_flags = list(dict.fromkeys(list(risk.get("risk_flags") or []) + flags))
    blocking = list(risk.get("blocking_risk_flags") or [])
    blocking.extend(flag for flag in flags if flag in LLM_BLOCKING_FLAGS)
    blocking = list(dict.fromkeys(blocking))
    return {**risk, "risk_flags": risk_flags, "blocking_risk_flags": blocking, "order_allowed": len(blocking) == 0}
