from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

@dataclass(frozen=True)
class TeamState:
    team_id: str
    team_code: str

    elo: float
    elo_scaled: float

    matches_available: int

    xg_for_short: float
    xg_against_short: float
    xg_for_long: float
    xg_against_long: float

    opponent_adjusted_xg_for: float
    opponent_adjusted_xg_against: float

    goals_for_short: float
    goals_against_short: float

    shots_for: float
    shots_against: float
    shots_on_target_for: float
    shots_on_target_against: float

    clean_sheet_rate: float
    failed_to_score_rate: float
    draw_rate: float

    expected_lineup_strength: float | None
    confirmed_lineup_strength: float | None
    lineup_delta: float | None

    unavailable_player_impact: float
    goalkeeper_strength: float | None
    squad_depth: float | None

    rest_hours: float | None
    travel_distance_km: float | None
    neutral_ground: bool | None

    tournament_incentives: dict
    manager_context: dict

    data_coverage: dict
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def model_state(self) -> dict:
        """Map to the exact dict keys ``deterministic_v2.predict_v2`` /
        ``team_strength`` read. Short-window form drives goals/xG; the long
        window backs the rating via Elo."""
        return {
            "live_rating": self.elo_scaled,
            "elo_scaled": self.elo_scaled,
            "matches": self.matches_available,
            "xg_for": self.opponent_adjusted_xg_for * max(1, self.matches_available),
            "xg_against": self.opponent_adjusted_xg_against * max(1, self.matches_available),
            "goals_for": self.goals_for_short * max(1, self.matches_available),
            "goals_against": self.goals_against_short * max(1, self.matches_available),
            "rest_hours": self.rest_hours if self.rest_hours is not None else 72.0,
            "continent": (self.tournament_incentives or {}).get("continent"),
        }


def build_team_state(
    team_id: str,
    team_code: str,
    *,
    elo_scaled: float,
    form: dict,
    rest_hours: float = 72.0,
    neutral: bool = True,
    opponent_scaled: float = 0.0,
    continent: str | None = None,
) -> TeamState:
    """Assemble a leakage-free TeamState from a chronological-Elo rating and a
    rolling-form dict (``RollingFormBuilder.form`` output). xG is opponent-adjusted
    via a small capped tilt. Coverage reflects how much history backed the form."""
    from models.context_adjustments import ContextAdjustments

    short = form.get("short", {})
    long = form.get("long", {})
    n = int(short.get("matches", 0) or 0)
    gf_s = short.get("goals_for"); ga_s = short.get("goals_against")
    xgf_s = short.get("xg_for"); xga_s = short.get("xg_against")
    xgf_l = long.get("xg_for"); xga_l = long.get("xg_against")

    league_avg = 1.30
    gf_s = league_avg if gf_s is None else gf_s
    ga_s = league_avg if ga_s is None else ga_s
    xgf_s = gf_s if xgf_s is None else xgf_s
    xga_s = ga_s if xga_s is None else xga_s
    xgf_l = xgf_s if xgf_l is None else xgf_l
    xga_l = xga_s if xga_l is None else xga_l

    ctx = ContextAdjustments()
    rest_days = (rest_hours or 72.0) / 24.0
    oadj_for = ctx.apply_adjustments(xgf_s, opponent_quality=opponent_scaled, rest_days=rest_days)
    oadj_against = ctx.apply_adjustments(xga_s, opponent_quality=-opponent_scaled, rest_days=rest_days)

    coverage = min(1.0, n / 8.0)
    elo = 1500.0 + elo_scaled * 400.0
    return TeamState(
        team_id=team_id, team_code=team_code,
        elo=elo, elo_scaled=elo_scaled, matches_available=n,
        xg_for_short=xgf_s, xg_against_short=xga_s, xg_for_long=xgf_l, xg_against_long=xga_l,
        opponent_adjusted_xg_for=oadj_for, opponent_adjusted_xg_against=oadj_against,
        goals_for_short=gf_s, goals_against_short=ga_s,
        shots_for=0.0, shots_against=0.0, shots_on_target_for=0.0, shots_on_target_against=0.0,
        clean_sheet_rate=0.0, failed_to_score_rate=0.0, draw_rate=0.0,
        expected_lineup_strength=None, confirmed_lineup_strength=None, lineup_delta=None,
        unavailable_player_impact=0.0, goalkeeper_strength=None, squad_depth=None,
        rest_hours=rest_hours, travel_distance_km=None, neutral_ground=neutral,
        tournament_incentives={"continent": continent} if continent else {},
        manager_context={},
        data_coverage={"overall": round(coverage, 3), "historical_results": coverage,
                       "historical_xg": 0.0 if xgf_l is None else coverage},
        warnings=tuple() if n else ("no_prior_matches",),
    )


class TeamStateBuilder:
    """Builds TeamState objects based on information available *before* the forecast timestamp."""
    
    def __init__(self, forecast_timestamp: datetime):
        self.forecast_timestamp = forecast_timestamp

    def build_state(self, team_id: str, history: list[dict], lineups: dict | None) -> TeamState:
        """
        Build state for a team.
        Leakage condition: historical_event_end_time < forecast_as_of_timestamp
        """
        # Filter history to prevent leakage
        valid_history = []
        for event in history:
            # Check event end time (approximated as kickoff + 120 mins)
            kickoff_str = event.get("start_time")
            if not kickoff_str:
                continue
            kickoff = datetime.fromisoformat(kickoff_str.replace("Z", "+00:00"))
            event_end_time = kickoff.timestamp() + (120 * 60) # Approx 2 hours
            
            if event_end_time < self.forecast_timestamp.timestamp():
                valid_history.append(event)
                
        # In a real implementation, we would call the other modules here
        # chronological_elo, rolling_form, lineup_strength
        
        return TeamState(
            team_id=team_id,
            team_code="UNK",
            elo=1500.0,
            elo_scaled=0.5,
            matches_available=len(valid_history),
            xg_for_short=1.0,
            xg_against_short=1.0,
            xg_for_long=1.0,
            xg_against_long=1.0,
            opponent_adjusted_xg_for=1.0,
            opponent_adjusted_xg_against=1.0,
            goals_for_short=1.0,
            goals_against_short=1.0,
            shots_for=10.0,
            shots_against=10.0,
            shots_on_target_for=3.0,
            shots_on_target_against=3.0,
            clean_sheet_rate=0.3,
            failed_to_score_rate=0.3,
            draw_rate=0.25,
            expected_lineup_strength=1.0,
            confirmed_lineup_strength=None,
            lineup_delta=0.0,
            unavailable_player_impact=0.0,
            goalkeeper_strength=1.0,
            squad_depth=1.0,
            rest_hours=72.0,
            travel_distance_km=0.0,
            neutral_ground=True,
            tournament_incentives={},
            manager_context={},
            data_coverage={"overall": 0.5},
        )
