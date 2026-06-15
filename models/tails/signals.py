"""Directional-signal aggregation for HUNTER (spec §10).

Raw evidence (scout flags, web/reddit research, BZZOIRO lineups/unavailable
players, the independent model's directional lean) is converted into
``DirectionalSignal`` objects and **grouped by source_group** so duplicate-source
claims collapse to ONE signal. Two websites copying one wire report = one signal
group; BZZOIRO lineup + official lineup = one lineup group. HUNTER counts DISTINCT
source groups, never raw rows, against its minimum-independent-signals bar.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from agents.contracts import DirectionalSignal


def _stable_hash(payload: dict) -> str:
    import json
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


class SignalAggregator:
    def __init__(self):
        pass

    def aggregate(self, raw_evidence: list[dict]) -> list[DirectionalSignal]:
        """Collapse raw evidence to one DirectionalSignal per (source_group,
        outcome, direction). When a group repeats, keep the strongest by
        ``strength`` (ties broken by ``confidence``)."""
        best: dict[tuple, dict] = {}
        for ev in raw_evidence or []:
            outcome = str(ev.get("outcome") or "").strip().lower()
            direction = str(ev.get("direction") or "up").strip().lower()
            group = str(ev.get("source_group") or ev.get("source") or "unknown").strip().lower()
            if not outcome:
                continue
            key = (group, outcome, direction)
            strength = float(ev.get("strength", 0.0) or 0.0)
            confidence = float(ev.get("confidence", 0.0) or 0.0)
            cur = best.get(key)
            if cur is None or (strength, confidence) > (cur["strength"], cur["confidence"]):
                best[key] = {
                    "source": str(ev.get("source") or group),
                    "source_group": group,
                    "outcome": outcome,
                    "direction": direction,
                    "strength": strength,
                    "confidence": confidence,
                    "summary": str(ev.get("summary") or ""),
                    "observed_at": ev.get("observed_at"),
                    "expires_at": ev.get("expires_at"),
                }

        signals: list[DirectionalSignal] = []
        for (group, outcome, direction), v in best.items():
            observed = v["observed_at"] or datetime.now(timezone.utc)
            evidence_hash = _stable_hash({
                "group": group, "outcome": outcome, "direction": direction,
                "strength": v["strength"],
            })
            signals.append(DirectionalSignal(
                signal_id=f"{group}:{outcome}:{direction}:{evidence_hash}",
                fixture_id="",  # filled by caller if needed
                outcome=outcome,
                source=v["source"],
                source_group=group,
                direction=direction,
                strength=v["strength"],
                confidence=v["confidence"],
                observed_at=observed,
                expires_at=v["expires_at"],
                evidence_hash=evidence_hash,
                summary=v["summary"],
            ))
        return signals

    @staticmethod
    def independent_count(signals: list[DirectionalSignal], outcome: str,
                          direction: str = "up") -> int:
        """Number of DISTINCT source groups supporting ``outcome`` in
        ``direction``. This is the count HUNTER gates on."""
        outcome = str(outcome).strip().lower()
        direction = str(direction).strip().lower()
        groups = {
            s.source_group for s in signals
            if s.outcome == outcome and s.direction == direction
        }
        return len(groups)

    def signals_for(self, raw_evidence: list[dict], fixture_id: str, outcome: str,
                    direction: str = "up") -> tuple[DirectionalSignal, ...]:
        """Convenience: aggregated, fixture-stamped signals supporting one outcome."""
        outcome = str(outcome).strip().lower()
        direction = str(direction).strip().lower()
        out = []
        for s in self.aggregate(raw_evidence):
            if s.outcome == outcome and s.direction == direction:
                out.append(DirectionalSignal(
                    signal_id=s.signal_id, fixture_id=fixture_id, outcome=s.outcome,
                    source=s.source, source_group=s.source_group, direction=s.direction,
                    strength=s.strength, confidence=s.confidence,
                    observed_at=s.observed_at, expires_at=s.expires_at,
                    evidence_hash=s.evidence_hash, summary=s.summary,
                ))
        return tuple(out)
