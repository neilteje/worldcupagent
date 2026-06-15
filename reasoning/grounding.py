"""
Grounding layer for council output — anchors, sanity checks, honest shrink.

The council is an LLM pipeline; this module is the deterministic seatbelt
around it. Three responsibilities:

  1. extract_anchor   — pull the best structured reference distribution from the
                        Sportmonks digest (bookmaker consensus, else Sportmonks
                        ML). The Analyst receives it as an explicit anchor to
                        reconcile with; the sanity check measures divergence
                        against it.
  2. sanity_check     — renormalize the final distribution, clamp evidence-free
                        extremes (<2% / >92%), and flag >15pp divergence from
                        the anchor that the devil's-advocate did not support.
  3. shrink_to_anchor — when council confidence is LOW and an anchor exists,
                        shrink toward it:  p' = (1−λ)·p + λ·anchor, with
                        λ = 0.50 (low) / 0.25 (medium) / 0.0 (high).
                        Nothing is hidden: the lambda, the pre-shrink probs, and
                        every sanity flag are returned and logged to the ledger.

Confidence also gets teeth here: when BOTH structured digests are missing the
label is forced to "low" — "low" means thin data, not "we guessed anyway".
"""
from __future__ import annotations

BASE_RATE = {"home": 0.40, "draw": 0.28, "away": 0.32}

PROB_FLOOR = 0.02      # no outcome below 2% without cited evidence
PROB_CEIL = 0.92       # no outcome above 92% without cited evidence
DIVERGENCE_FLAG_PP = 0.15   # flag >15pp divergence from anchor w/o devil support

_SHRINK_LAMBDA = {"low": 0.50, "medium": 0.25, "high": 0.0}


def _keys(home_code: str, away_code: str) -> tuple[str, str, str]:
    return (home_code, "draw", away_code)


def _normalize(probs: dict | None, home_code: str, away_code: str) -> dict | None:
    if not probs:
        return None
    vals = {}
    for k in _keys(home_code, away_code):
        v = probs.get(k)
        if not isinstance(v, (int, float)) or v < 0:
            return None
        vals[k] = float(v)
    s = sum(vals.values())
    if s <= 0:
        return None
    return {k: round(v / s, 4) for k, v in vals.items()}


def extract_anchor(sportmonks_digest: dict | None, home_code: str,
                   away_code: str) -> dict | None:
    """
    Best structured reference distribution, in preference order:
    bookmaker consensus → Sportmonks ML. None when neither is available.
    """
    if not sportmonks_digest:
        return None
    for field, source in (("bookmaker_consensus_win_prob", "bookmaker_consensus"),
                          ("sportmonks_ml_win_prob", "sportmonks_ml")):
        probs = _normalize(sportmonks_digest.get(field), home_code, away_code)
        if probs:
            return {"source": source, "probabilities": probs}
    return None


def _clamp_simplex(probs: dict, floor: float, ceil: float) -> dict:
    """
    Clamp every probability into [floor, ceil] while keeping the sum at 1, by
    redistributing the clamped mass across unclamped outcomes (a plain
    renormalize would push a clamped favorite straight back over the ceiling).
    """
    fixed: dict[str, float] = {}
    free = dict(probs)
    for _ in range(len(probs)):
        moved = False
        for k, v in list(free.items()):
            if v < floor:
                fixed[k], moved = floor, True
                del free[k]
            elif v > ceil:
                fixed[k], moved = ceil, True
                del free[k]
        if not moved:
            break
        remaining = 1.0 - sum(fixed.values())
        s = sum(free.values())
        if not free or s <= 0 or remaining <= 0:
            break
        free = {k: v * remaining / s for k, v in free.items()}
    out = {**free, **fixed}
    s = sum(out.values())
    return {k: round(v / s, 4) for k, v in out.items()}


def _devil_supports(devil_parsed: dict | None, key: str, direction_up: bool,
                    home_code: str, away_code: str) -> bool:
    """Did the devil's counter-distribution push this outcome the same way?"""
    if not devil_parsed:
        return False
    counter = _normalize(devil_parsed.get("counter_probabilities"),
                         home_code, away_code)
    if not counter:
        # Fall back: devil naming the outcome as its counter pick counts as support.
        return str(devil_parsed.get("counter_outcome", "")) == key
    anchor_free_mid = BASE_RATE["home" if key == home_code else
                                "draw" if key == "draw" else "away"]
    return (counter[key] > anchor_free_mid) == direction_up


def sanity_check(probs: dict, home_code: str, away_code: str,
                 anchor: dict | None, devil_parsed: dict | None) -> dict:
    """
    Deterministic audit of a council distribution. Returns
    {"probabilities": cleaned, "flags": [str], "anchor_divergence_pp": float|None}.
    """
    flags: list[str] = []
    cleaned = _normalize(probs, home_code, away_code)
    if cleaned is None:
        flags.append("invalid_distribution_replaced_with_base_rate")
        cleaned = {home_code: BASE_RATE["home"], "draw": BASE_RATE["draw"],
                   away_code: BASE_RATE["away"]}
    elif any(abs(cleaned[k] - float(probs.get(k, 0))) > 0.01
             for k in _keys(home_code, away_code)):
        flags.append("renormalized_probabilities")

    # Clamp evidence-free extremes (mass redistributed inside the simplex).
    if any(v < PROB_FLOOR or v > PROB_CEIL for v in cleaned.values()):
        flags.append(f"clamped_extreme_probabilities_to_[{PROB_FLOOR},{PROB_CEIL}]")
        cleaned = _clamp_simplex(cleaned, PROB_FLOOR, PROB_CEIL)

    divergence_pp = None
    if anchor:
        a = anchor["probabilities"]
        worst_key = max(cleaned, key=lambda k: abs(cleaned[k] - a[k]))
        divergence_pp = round(abs(cleaned[worst_key] - a[worst_key]), 4)
        if divergence_pp > DIVERGENCE_FLAG_PP:
            up = cleaned[worst_key] > a[worst_key]
            if not _devil_supports(devil_parsed, worst_key, up, home_code, away_code):
                flags.append(
                    f"diverges_{divergence_pp*100:.0f}pp_from_{anchor['source']}"
                    f"_on_{worst_key}_without_devil_support")

    return {"probabilities": cleaned, "flags": flags,
            "anchor_divergence_pp": divergence_pp}


def shrink_to_anchor(probs: dict, anchor: dict | None, confidence: str,
                     home_code: str, away_code: str) -> dict:
    """
    Low-confidence shrink toward the anchor.  p' = (1−λ)·p + λ·anchor.
    Returns {"probabilities", "lambda", "pre_shrink"}; λ=0 → unchanged.
    """
    lam = _SHRINK_LAMBDA.get(str(confidence or "").lower(), 0.25)
    if not anchor or lam <= 0:
        return {"probabilities": dict(probs), "lambda": 0.0, "pre_shrink": None}
    a = anchor["probabilities"]
    mixed = {k: round((1 - lam) * probs[k] + lam * a[k], 4)
             for k in _keys(home_code, away_code)}
    s = sum(mixed.values())
    mixed = {k: round(v / s, 4) for k, v in mixed.items()}
    return {"probabilities": mixed, "lambda": lam, "pre_shrink": dict(probs)}


def ground_council_output(probs: dict, confidence: str, home_code: str,
                          away_code: str, *, sportmonks_digest: dict | None,
                          supabase_digest: dict | None,
                          devil_parsed: dict | None) -> dict:
    """
    Full grounding pass over the council's final distribution.

    Returns:
      probabilities      — cleaned, possibly shrunk distribution
      confidence         — possibly downgraded label
      anchor             — the reference used (or None)
      sanity_flags       — every deterministic audit flag
      shrink_lambda      — λ applied (0 = none)
      pre_shrink         — distribution before shrink (None if unshrunk)
      data_thin          — True when both structured digests were missing
    """
    # Confidence means something: no structured data → at most "low".
    data_thin = sportmonks_digest is None and supabase_digest is None
    conf = str(confidence or "low").lower()
    if data_thin and conf != "low":
        conf = "low"

    anchor = extract_anchor(sportmonks_digest, home_code, away_code)
    audit = sanity_check(probs, home_code, away_code, anchor, devil_parsed)
    shrunk = shrink_to_anchor(audit["probabilities"], anchor, conf,
                              home_code, away_code)

    flags = list(audit["flags"])
    if data_thin:
        flags.append("no_structured_digests_confidence_capped_low")

    return {
        "probabilities": shrunk["probabilities"],
        "confidence": conf,
        "anchor": anchor,
        "sanity_flags": flags,
        "anchor_divergence_pp": audit["anchor_divergence_pp"],
        "shrink_lambda": shrunk["lambda"],
        "pre_shrink": shrunk["pre_shrink"],
        "data_thin": data_thin,
    }
