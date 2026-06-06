from __future__ import annotations
from typing import Any


def _dig(obj: Any, names: tuple[str, ...]) -> Any:
    if isinstance(obj, dict):
        for n in names:
            if n in obj: return obj[n]
        for v in obj.values():
            got = _dig(v, names)
            if got is not None: return got
    if isinstance(obj, list):
        for v in obj:
            got = _dig(v, names)
            if got is not None: return got
    return None


def extract_lineups(fixture_payload: dict) -> dict:
    expected = _dig(fixture_payload, ("expected_lineups", "expectedLineups", "predicted_lineups")) or {}
    confirmed = _dig(fixture_payload, ("lineups", "confirmed_lineups", "starting_xi")) or {}
    def side(src, side):
        if isinstance(src, dict): return src.get(side) or src.get(f"{side}_lineup") or []
        if isinstance(src, list): return [p for p in src if str(p.get("side") or p.get("team_position") or "").lower() == side]
        return []
    formations = _dig(fixture_payload, ("formations", "formation")) or {}
    return {"expected_home": side(expected, "home"), "expected_away": side(expected, "away"), "confirmed_home": side(confirmed, "home"), "confirmed_away": side(confirmed, "away"), "expected_formations": formations.get("expected", {}) if isinstance(formations, dict) else {}, "confirmed_formations": formations.get("confirmed", formations if isinstance(formations, dict) else {})}
