"""Supplemental odds2prob integration.

The service converts raw 1X2 decimal bookmaker odds into calibrated, de-vigged
home/draw/away probabilities. It is an optional support signal for the
deterministic engine and council, not a hard dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen
import json
import statistics

import config


_SLOTS = ("home", "draw", "away")
_HOME_LABELS = {"home", "1", "home win", "home_win"}
_DRAW_LABELS = {"draw", "x", "tie"}
_AWAY_LABELS = {"away", "2", "away win", "away_win"}


@dataclass(frozen=True)
class OddsRow:
    odds_home: float
    odds_draw: float
    odds_away: float
    bookmaker: str | None = None


def _slot_from_label(value: Any) -> str | None:
    label = str(value or "").strip().lower().replace("_", " ")
    if label in _HOME_LABELS:
        return "home"
    if label in _DRAW_LABELS:
        return "draw"
    if label in _AWAY_LABELS:
        return "away"
    return None


def _as_decimal(value: Any) -> float | None:
    try:
        odd = float(value)
    except (TypeError, ValueError):
        return None
    return odd if odd > 1.01 else None


def _bookmaker_name(row: dict) -> str | None:
    for key in ("bookmaker", "bookmaker_name", "bookmakerName", "name"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, dict):
            nested = val.get("name") or val.get("display_name")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    return None


def extract_decimal_1x2_odds(payload: dict | list | None) -> OddsRow | None:
    """Find a representative home/draw/away decimal-odds row.

    Supports both compact rows (``home/draw/away`` or ``home_odds/...``) and
    Sportmonks-style one-outcome rows with labels. When multiple bookmakers are
    present, use median decimal odds by slot for a stable consensus row.
    """
    compact_rows: list[OddsRow] = []
    grouped: dict[str, dict[str, Any]] = {}

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            lower = {str(k).lower(): v for k, v in obj.items()}
            vals = None
            if all(k in lower for k in _SLOTS):
                vals = {k: lower[k] for k in _SLOTS}
            elif all(f"{k}_odds" in lower for k in _SLOTS):
                vals = {k: lower[f"{k}_odds"] for k in _SLOTS}
            if vals:
                odds = {k: _as_decimal(vals[k]) for k in _SLOTS}
                if all(odds.values()):
                    compact_rows.append(OddsRow(
                        odds_home=float(odds["home"]),
                        odds_draw=float(odds["draw"]),
                        odds_away=float(odds["away"]),
                        bookmaker=_bookmaker_name(obj),
                    ))

            slot = None
            for key in ("label", "name", "outcome", "selection", "participant"):
                slot = _slot_from_label(obj.get(key))
                if slot:
                    break
            odd = None
            for key in ("value", "odd", "odds", "decimal", "price"):
                odd = _as_decimal(obj.get(key))
                if odd:
                    break
            if slot and odd:
                group_key = str(
                    obj.get("bookmaker_id")
                    or obj.get("bookmaker")
                    or obj.get("bookmaker_name")
                    or obj.get("latest_bookmaker_update")
                    or "consensus"
                )
                grouped.setdefault(group_key, {"bookmaker": _bookmaker_name(obj), "odds": {}})
                grouped[group_key]["odds"][slot] = odd

            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)

    walk(payload)
    rows = list(compact_rows)
    for group in grouped.values():
        odds = group.get("odds") or {}
        if all(k in odds for k in _SLOTS):
            rows.append(OddsRow(
                odds_home=float(odds["home"]),
                odds_draw=float(odds["draw"]),
                odds_away=float(odds["away"]),
                bookmaker=group.get("bookmaker"),
            ))
    if not rows:
        return None
    bookmaker = next((r.bookmaker for r in rows if r.bookmaker), None)
    return OddsRow(
        odds_home=statistics.median(r.odds_home for r in rows),
        odds_draw=statistics.median(r.odds_draw for r in rows),
        odds_away=statistics.median(r.odds_away for r in rows),
        bookmaker=bookmaker,
    )


def convert_odds(row: OddsRow) -> dict:
    params = {
        "odds_home": row.odds_home,
        "odds_draw": row.odds_draw,
        "odds_away": row.odds_away,
        "model": config.ODDS2PROB_MODEL,
    }
    if row.bookmaker:
        params["bookmaker"] = row.bookmaker
    base = config.ODDS2PROB_URL.rstrip("/")
    url = f"{base}/convert?{urlencode(params)}"
    with urlopen(url, timeout=config.ODDS2PROB_TIMEOUT_SECONDS) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    probabilities = {
        "home": float(data["p_home"]),
        "draw": float(data["p_draw"]),
        "away": float(data["p_away"]),
    }
    total = sum(probabilities.values())
    if total <= 0:
        raise ValueError("odds2prob returned non-positive probability total")
    data["probabilities"] = {k: probabilities[k] / total for k in _SLOTS}
    data["input_odds"] = {
        "home": row.odds_home,
        "draw": row.odds_draw,
        "away": row.odds_away,
        "bookmaker": row.bookmaker,
    }
    data["source"] = "odds2prob"
    return data


def from_fixture(payload: dict | list | None) -> dict:
    if not config.ODDS2PROB_ENABLED:
        return {"available": False, "source": "odds2prob", "reason": "disabled"}
    row = extract_decimal_1x2_odds(payload)
    if row is None:
        return {"available": False, "source": "odds2prob", "reason": "no_decimal_1x2_odds"}
    try:
        out = convert_odds(row)
        out["available"] = True
        return out
    except Exception as exc:
        return {
            "available": False,
            "source": "odds2prob",
            "reason": "convert_failed",
            "error": repr(exc),
            "input_odds": {
                "home": row.odds_home,
                "draw": row.odds_draw,
                "away": row.odds_away,
                "bookmaker": row.bookmaker,
            },
        }


__all__ = ["OddsRow", "convert_odds", "extract_decimal_1x2_odds", "from_fixture"]
