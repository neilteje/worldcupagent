from __future__ import annotations
from models.calibration import normalize_probs


def _pid(p: dict) -> str:
    return str(p.get("player_id") or p.get("id") or p.get("name") or p.get("display_name") or "").lower()


def _name(p: dict) -> str:
    return str(p.get("name") or p.get("display_name") or p.get("player_name") or _pid(p))


def _position(p: dict) -> str:
    return str(p.get("position") or p.get("position_name") or p.get("role") or p.get("type") or "").lower()


def player_importance(player: dict) -> float:
    pos = _position(player)
    key = str(player.get("importance") or player.get("key_role") or "").lower()
    rating = float(player.get("rating") or player.get("average_rating") or 0 or 0)
    star = bool(player.get("star") or player.get("captain") or "star" in key or "primary" in key or rating >= 7.4)
    if "goal" in pos or pos in {"gk", "keeper"}: return 0.05 if star else 0.035
    if any(w in pos for w in ["striker", "forward", "attacker", "centre forward", "cf"]) or any(w in key for w in ["striker", "attacker"]): return 0.04 if star else 0.025
    if any(w in pos for w in ["attacking midfield", "creator", "am"]): return 0.03 if star else 0.02
    if any(w in pos for w in ["center back", "centre back", "central defender", "cb"]): return 0.025
    if any(w in pos for w in ["defensive midfield", "dm"]): return 0.02
    if any(w in pos for w in ["fullback", "full back", "winger", "wing", "left back", "right back", "lb", "rb"]): return 0.012
    return 0.006 if player.get("expected_starter", True) else 0.0


def _starters(lineup: list[dict] | None) -> list[dict]:
    return [p for p in (lineup or []) if p.get("starter", p.get("is_starter", p.get("expected_starter", True)))]


def evaluate_lineup_delta(expected_home: list[dict] | None = None, confirmed_home: list[dict] | None = None, expected_away: list[dict] | None = None, confirmed_away: list[dict] | None = None, expected_formations: dict | None = None, confirmed_formations: dict | None = None, evidence_strength: str = "api") -> dict:
    home_confirmed = confirmed_home is not None and len(confirmed_home) > 0
    away_confirmed = confirmed_away is not None and len(confirmed_away) > 0
    risks: list[str] = []
    if not (home_confirmed and away_confirmed):
        risks.append("lineup_unconfirmed")
        return {"home_lineup_confirmed": home_confirmed, "away_lineup_confirmed": away_confirmed, "lineup_shock": False, "home_missing_expected_starters": [], "away_missing_expected_starters": [], "home_unexpected_starters": [], "away_unexpected_starters": [], "formation_change": {"home": False, "away": False}, "probability_delta": {"home": 0.0, "draw": 0.0, "away": 0.0}, "confidence": 0.35, "risk_flags": risks, "reason": "Confirmed lineup missing for at least one team."}
    eh, ch, ea, ca = map(_starters, [expected_home, confirmed_home, expected_away, confirmed_away])
    ch_ids, ca_ids = {_pid(p) for p in ch}, {_pid(p) for p in ca}
    eh_ids, ea_ids = {_pid(p) for p in eh}, {_pid(p) for p in ea}
    hm = [p for p in eh if _pid(p) not in ch_ids]
    am = [p for p in ea if _pid(p) not in ca_ids]
    hu = [p for p in ch if _pid(p) not in eh_ids]
    au = [p for p in ca if _pid(p) not in ea_ids]
    home_impact, away_impact = sum(player_importance(p) for p in hm), sum(player_importance(p) for p in am)
    cap = 0.10 if any(player_importance(p) >= .045 for p in hm + am) and len(hm + am) >= 2 else 0.07
    if evidence_strength != "api": cap = min(cap, 0.015)
    net = max(-cap, min(cap, away_impact - home_impact))
    # positive net means away has worse absences than home => shift to home
    delta_home = net * 0.70
    delta_away = -net * 0.70
    delta_draw = -(delta_home + delta_away) + abs(net) * 0.15
    if net > 0: delta_away -= abs(net)*0.15
    elif net < 0: delta_home -= abs(net)*0.15
    probability_delta = {"home": delta_home, "draw": delta_draw, "away": delta_away}
    fh = bool(expected_formations and confirmed_formations and expected_formations.get("home") != confirmed_formations.get("home"))
    fa = bool(expected_formations and confirmed_formations and expected_formations.get("away") != confirmed_formations.get("away"))
    shock = bool(abs(net) >= 0.015 or fh or fa)
    return {"home_lineup_confirmed": True, "away_lineup_confirmed": True, "lineup_shock": shock, "home_missing_expected_starters": [_name(p) for p in hm], "away_missing_expected_starters": [_name(p) for p in am], "home_unexpected_starters": [_name(p) for p in hu], "away_unexpected_starters": [_name(p) for p in au], "formation_change": {"home": fh, "away": fa}, "probability_delta": probability_delta, "confidence": 0.72 if not shock else 0.65, "risk_flags": risks, "reason": f"Home missing impact={home_impact:.3f}; away missing impact={away_impact:.3f}; capped net={net:.3f}."}


def apply_lineup_delta(base_probs: dict[str, float], delta: dict[str, float]) -> dict[str, float]:
    p = normalize_probs(base_probs)
    return normalize_probs({k: p[k] + float(delta.get(k, 0.0)) for k in ("home", "draw", "away")})
