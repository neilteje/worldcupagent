from __future__ import annotations
from models.calibration import OUTCOMES, normalize_probs


def detect_market_stale(current_market: dict[str, float] | None, previous_market: dict[str, float] | None = None, bookmaker_probs: dict[str, float] | None = None, signal_delta: dict[str, float] | None = None) -> dict:
    if not current_market or not previous_market:
        return {"is_stale": False, "stale_score": 0.0, "edge_type": "no_clear_edge", "reason": "Need current and previous market snapshots."}
    cur, prev = normalize_probs(current_market), normalize_probs(previous_market)
    move = {k: cur[k] - prev[k] for k in OUTCOMES}
    max_move = max(abs(v) for v in move.values())
    signal_delta = signal_delta or {k: 0.0 for k in OUTCOMES}
    strongest_signal = max(OUTCOMES, key=lambda k: abs(float(signal_delta.get(k, 0.0))))
    signal_strength = abs(float(signal_delta.get(strongest_signal, 0.0)))
    book_gap = 0.0
    if bookmaker_probs:
        book = normalize_probs(bookmaker_probs)
        book_gap = abs(book[strongest_signal] - cur[strongest_signal])
    is_stale = signal_strength >= 0.025 and max_move <= 0.008 and book_gap >= 0.035
    stale_score = min(0.12, max(0.0, signal_strength * 0.7 + book_gap * 0.5 - max_move)) if is_stale else 0.0
    return {"is_stale": is_stale, "stale_score": stale_score, "market_move": move, "edge_type": "market_stale" if is_stale else "no_clear_edge", "reason": "Bookmaker/signal moved while market barely moved." if is_stale else "No stale-market evidence."}
