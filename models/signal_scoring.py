from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Iterable
from models.calibration import OUTCOMES

WEAK_SOURCES = {"reddit", "web", "rumor", "sentiment"}

@dataclass
class SignalScore:
    name: str
    source: str
    category: str
    direction: str
    probability_delta: dict[str, float]
    source_quality: float
    freshness: float
    corroboration: float
    confidence: float
    impact: float
    final_weight: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _direction(delta: dict[str, float]) -> str:
    if not delta:
        return "none"
    best = max(OUTCOMES, key=lambda k: float(delta.get(k, 0.0)))
    return best if float(delta.get(best, 0.0)) > 0 else "none"


def score_signal(name: str, source: str, category: str, probability_delta: dict[str, float] | None = None, *, source_quality: float = 0.5, freshness: float = 0.5, corroboration: float = 0.5, reason: str = "") -> dict:
    delta = {k: float((probability_delta or {}).get(k, 0.0) or 0.0) for k in OUTCOMES}
    source_l = source.lower()
    cap = 0.015 if source_l in WEAK_SOURCES or category.lower() in WEAK_SOURCES else 0.07
    raw_impact = max(abs(v) for v in delta.values()) if delta else 0.0
    if raw_impact > cap:
        scale = cap / raw_impact
        delta = {k: v * scale for k, v in delta.items()}
        raw_impact = cap
    sq, fr, co = map(_clamp01, (source_quality, freshness, corroboration))
    confidence = _clamp01(0.50 * sq + 0.25 * fr + 0.25 * co)
    # weak uncorroborated sources get an extra haircut even after delta capping
    weak_penalty = 0.55 if source_l in WEAK_SOURCES and co < 0.6 else 1.0
    final_weight = _clamp01(confidence * weak_penalty)
    return SignalScore(name, source, category, _direction(delta), delta, sq, fr, co, confidence, raw_impact, final_weight, reason).to_dict()


def summarize_signals(signals: Iterable[dict], limit: int = 5) -> list[dict]:
    return sorted([s for s in signals if s], key=lambda s: (float(s.get("impact", 0.0)) * float(s.get("final_weight", 0.0))), reverse=True)[:limit]


def signal_conflict_score(signals: Iterable[dict]) -> float:
    directions = [s.get("direction") for s in signals if s and s.get("direction") in OUTCOMES and float(s.get("impact", 0.0)) > 0.005]
    return 0.0 if len(set(directions)) <= 1 else min(0.35, 0.10 * (len(set(directions)) - 1) + 0.03 * len(directions))
