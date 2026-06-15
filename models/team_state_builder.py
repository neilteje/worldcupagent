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
