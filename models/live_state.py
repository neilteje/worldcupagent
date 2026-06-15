from dataclasses import dataclass, field

@dataclass(frozen=True)
class LiveMatchState:
    fixture_id: str
    current_minute: int
    match_period: str
    
    home_score: int
    away_score: int
    
    home_red_cards: int
    away_red_cards: int
    
    home_live_xg: float | None
    away_live_xg: float | None
    
    home_dangerous_attacks: int | None
    away_dangerous_attacks: int | None
    
    home_possession: float | None
    away_possession: float | None
    
    data_coverage: dict
    warnings: tuple[str, ...] = field(default_factory=tuple)
