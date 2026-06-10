from __future__ import annotations

from itertools import combinations

from models.calibration import OUTCOMES, normalize_probs


def reconcile_sources(
    model_probs: dict[str, float],
    sportmonks_probs: dict[str, float] | None,
    bookmaker_probs: dict[str, float] | None,
    market_probs: dict[str, float] | None,
    *,
    divergence_threshold: float = 0.20,
) -> dict:
    sources = {
        "model": normalize_probs(model_probs),
        "sportmonks": normalize_probs(sportmonks_probs) if sportmonks_probs else None,
        "bookmaker": normalize_probs(bookmaker_probs) if bookmaker_probs else None,
        "market": normalize_probs(market_probs) if market_probs else None,
    }
    available = {name: probs for name, probs in sources.items() if probs}
    pairwise = []
    flags: list[str] = []
    for (left_name, left), (right_name, right) in combinations(available.items(), 2):
        outcome_diffs = {k: abs(left[k] - right[k]) for k in OUTCOMES}
        max_outcome = max(OUTCOMES, key=lambda k: outcome_diffs[k])
        max_gap = outcome_diffs[max_outcome]
        pairwise.append(
            {
                "left": left_name,
                "right": right_name,
                "max_outcome": max_outcome,
                "max_gap": max_gap,
                "outcome_diffs": outcome_diffs,
            }
        )
        if max_gap >= divergence_threshold - 1e-9:
            flags.append("source_divergence_high")
    picks = {name: max(OUTCOMES, key=lambda k: probs[k]) for name, probs in available.items()}
    if len(set(picks.values())) >= 3:
        flags.append("all_source_picks_disagree")
    elif len(set(picks.values())) == 2:
        flags.append("source_pick_disagreement")
    top_pair = max(pairwise, key=lambda row: row["max_gap"], default=None)
    return {
        "available_sources": sorted(available),
        "picks": picks,
        "pairwise": pairwise,
        "max_gap": float(top_pair["max_gap"]) if top_pair else 0.0,
        "max_gap_pair": [top_pair["left"], top_pair["right"]] if top_pair else [],
        "flags": list(dict.fromkeys(flags)),
        "reason": _reason(top_pair, flags),
    }


def _reason(top_pair: dict | None, flags: list[str]) -> str:
    if not top_pair:
        return "Insufficient sources for reconciliation."
    base = f"Largest source gap is {top_pair['max_gap']:.3f} on {top_pair['max_outcome']} between {top_pair['left']} and {top_pair['right']}."
    return base + (" Flags: " + ", ".join(dict.fromkeys(flags)) if flags else "")
