"""Typed evidence normalization for forecast and event triggers."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from models.forecast_contracts import stable_hash


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    source: str
    source_type: str
    observed_at: datetime
    published_at: datetime | None
    fixture_id: str
    team: str
    player: str | None
    event_type: str
    confirmation_level: str
    direction: str
    estimated_probability_delta: float | None
    relevance_score: float
    reliability_score: float
    expires_at: datetime | None
    raw_summary: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "relevance_score", _clamp(self.relevance_score))
        object.__setattr__(self, "reliability_score", _clamp(self.reliability_score))
        if self.estimated_probability_delta is not None:
            delta = max(-0.20, min(0.20, float(self.estimated_probability_delta)))
            object.__setattr__(self, "estimated_probability_delta", delta)


def _clamp(value: float | int | None) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def evidence_identity(item: EvidenceItem) -> str:
    """Identity used for deduping the same fact across providers."""
    return stable_hash({
        "fixture_id": item.fixture_id,
        "team": (item.team or "").strip().lower(),
        "player": (item.player or "").strip().lower(),
        "event_type": (item.event_type or "").strip().lower(),
        "direction": (item.direction or "").strip().lower(),
        "summary": " ".join((item.raw_summary or "").lower().split())[:120],
    })


def normalize_evidence(raw_items: list[dict[str, Any]] | tuple[dict[str, Any], ...],
                       *, now: datetime | None = None) -> tuple[EvidenceItem, ...]:
    """Normalize and deduplicate evidence from web/social/API/live providers.

    Coverage and trigger logic consume only returned items. Expired items and
    empty summaries are not usable evidence.
    """
    now = now or datetime.now(timezone.utc)
    out: dict[str, EvidenceItem] = {}
    for raw in raw_items or ():
        observed = _dt(raw.get("observed_at")) or now
        expires = _dt(raw.get("expires_at"))
        summary = str(raw.get("raw_summary") or raw.get("summary") or "").strip()
        if not summary or (expires and expires <= now):
            continue
        item = EvidenceItem(
            evidence_id=str(raw.get("evidence_id") or stable_hash(raw)),
            source=str(raw.get("source") or "unknown"),
            source_type=str(raw.get("source_type") or "unknown"),
            observed_at=observed,
            published_at=_dt(raw.get("published_at")),
            fixture_id=str(raw.get("fixture_id") or ""),
            team=str(raw.get("team") or "neutral"),
            player=raw.get("player"),
            event_type=str(raw.get("event_type") or "context"),
            confirmation_level=str(raw.get("confirmation_level") or "unknown"),
            direction=str(raw.get("direction") or "unclear"),
            estimated_probability_delta=raw.get("estimated_probability_delta"),
            relevance_score=_clamp(raw.get("relevance_score", 0.0)),
            reliability_score=_clamp(raw.get("reliability_score", 0.0)),
            expires_at=expires,
            raw_summary=summary,
        )
        key = evidence_identity(item)
        prev = out.get(key)
        if prev is None or (
            item.relevance_score * item.reliability_score
            > prev.relevance_score * prev.reliability_score
        ):
            out[key] = item
    return tuple(sorted(out.values(), key=lambda e: e.evidence_id))


def usable_evidence_coverage(items: tuple[EvidenceItem, ...], *,
                             required_sources: int = 4) -> float:
    source_groups = {
        f"{i.source_type}:{i.source}"
        for i in items
        if i.relevance_score >= 0.25 and i.reliability_score >= 0.25
    }
    return round(min(1.0, len(source_groups) / max(1, required_sources)), 4)


__all__ = ["EvidenceItem", "normalize_evidence", "usable_evidence_coverage", "evidence_identity"]
