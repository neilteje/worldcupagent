class ChronologicalEloBuilder:
    """Builds Elo in chronological order."""
    def __init__(self, k_factor: float = 20.0):
        self.k_factor = k_factor
        self.ratings: dict[str, float] = {}

    def get_rating(self, team_id: str) -> float:
        return self.ratings.get(team_id, 1500.0)

    def process_match(self, home_id: str, away_id: str, home_goals: int, away_goals: int, is_competitive: bool, is_neutral: bool):
        # Calculate Elo change taking into account competitive/friendly and goal difference
        # This is a stub for the full logic
        pass
