from __future__ import annotations


def synthetic_fixtures() -> list[dict]:
    base = {"demo": True, "home_team_code": "HOME", "away_team_code": "AWAY", "synthetic": True}
    cases = [
        ("SYN-MB-VS-PM", "PRE_MATCH", {"sportmonks": {"home": .56, "draw": .24, "away": .20}, "bookmaker": {"home": .54, "draw": .25, "away": .21}, "market": {"home": .45, "draw": .29, "away": .26}, "priors": {"home": .52, "draw": .26, "away": .22}}),
        ("SYN-PM-BK-AGAINST", "PRE_MATCH", {"sportmonks": {"home": .54, "draw": .25, "away": .21}, "bookmaker": {"home": .34, "draw": .28, "away": .38}, "market": {"home": .35, "draw": .28, "away": .37}, "priors": {"home": .49, "draw": .27, "away": .24}}),
        ("SYN-HT-LOW-XG", "HT", {"sportmonks": {"home": .38, "draw": .30, "away": .32}, "bookmaker": {"home": .37, "draw": .31, "away": .32}, "market": {"home": .35, "draw": .34, "away": .31}, "live": {"home_goals": 0, "away_goals": 0, "home_xg": .20, "away_xg": .18, "home_shots": 2, "away_shots": 2, "home_sot": 0, "away_sot": 0}}),
        ("SYN-FAV-LUCKY-TRAIL", "HT", {"sportmonks": {"home": .60, "draw": .23, "away": .17}, "bookmaker": {"home": .58, "draw": .24, "away": .18}, "market": {"home": .30, "draw": .31, "away": .39}, "live": {"home_goals": 0, "away_goals": 1, "home_xg": 1.40, "away_xg": .20, "home_shots": 10, "away_shots": 2, "home_sot": 4, "away_sot": 1}}),
        ("SYN-GK-MISSING", "PRE_MATCH", {"sportmonks": {"home": .45, "draw": .28, "away": .27}, "bookmaker": {"home": .46, "draw": .27, "away": .27}, "market": {"home": .45, "draw": .28, "away": .27}, "lineups": "home_gk_missing"}),
        ("SYN-RED-LEADER", "HT", {"sportmonks": {"home": .48, "draw": .27, "away": .25}, "bookmaker": {"home": .47, "draw": .28, "away": .25}, "market": {"home": .55, "draw": .25, "away": .20}, "live": {"home_goals": 1, "away_goals": 0, "home_xg": .55, "away_xg": .50, "home_red": 1, "away_red": 0, "home_shots": 5, "away_shots": 5}}),
        ("SYN-DRAW-UNDER", "PRE_MATCH", {"sportmonks": {"home": .36, "draw": .32, "away": .32}, "bookmaker": {"home": .35, "draw": .33, "away": .32}, "market": {"home": .40, "draw": .24, "away": .36}, "projected_xg": 1.9}),
        ("SYN-STALE-LINEUP", "PRE_MATCH", {"sportmonks": {"home": .51, "draw": .26, "away": .23}, "bookmaker": {"home": .57, "draw": .24, "away": .19}, "market": {"home": .48, "draw": .28, "away": .24}, "previous_market": {"home": .479, "draw": .281, "away": .240}, "signal_delta": {"home": .04, "draw": .00, "away": -.04}}),
        ("SYN-WEAK-WEB", "PRE_MATCH", {"sportmonks": {"home": .44, "draw": .28, "away": .28}, "bookmaker": {"home": .44, "draw": .28, "away": .28}, "market": {"home": .40, "draw": .30, "away": .30}, "web_delta": {"home": .06, "draw": -.02, "away": -.04}}),
        ("SYN-MISSING-MARKET", "PRE_MATCH", {"sportmonks": {"home": .52, "draw": .26, "away": .22}, "bookmaker": {"home": .51, "draw": .27, "away": .22}, "market": None}),
    ]
    return [{**base, "id": code, "fixture_code": code, "preferred_window": window, "synthetic_data": data, "name": code.replace("SYN-", "Synthetic ")} for code, window, data in cases]
