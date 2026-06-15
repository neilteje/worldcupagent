"""Chronological Elo (spec §15).

Ratings are built strictly in match order: ``process_match`` must be called for
games in increasing kickoff time, so a rating queried at forecast time reflects
ONLY prior results (no leakage). World-Football-Elo style: goal-difference margin
multiplier, competitive-vs-friendly K scaling, neutral-venue aware. Never derived
from market prices.
"""
from __future__ import annotations

BASE_RATING = 1500.0


class ChronologicalEloBuilder:
    """Builds Elo in chronological order."""

    def __init__(self, k_factor: float = 40.0, friendly_k: float = 20.0,
                 home_advantage: float = 65.0):
        self.k_factor = k_factor
        self.friendly_k = friendly_k
        self.home_advantage = home_advantage
        self.ratings: dict[str, float] = {}

    def get_rating(self, team_id: str) -> float:
        return self.ratings.get(team_id, BASE_RATING)

    @staticmethod
    def _expected(ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    @staticmethod
    def _margin_multiplier(goal_diff: int) -> float:
        gd = abs(goal_diff)
        if gd <= 1:
            return 1.0
        if gd == 2:
            return 1.5
        if gd == 3:
            return 1.75
        return 1.75 + (gd - 3) / 8.0

    def process_match(self, home_id: str, away_id: str, home_goals: int, away_goals: int,
                      is_competitive: bool = True, is_neutral: bool = True) -> None:
        ra = self.get_rating(home_id)
        rb = self.get_rating(away_id)
        adv = 0.0 if is_neutral else self.home_advantage
        exp_home = self._expected(ra + adv, rb)

        if home_goals > away_goals:
            score_home = 1.0
        elif home_goals == away_goals:
            score_home = 0.5
        else:
            score_home = 0.0

        k = self.k_factor if is_competitive else self.friendly_k
        k *= self._margin_multiplier(home_goals - away_goals)
        delta = k * (score_home - exp_home)
        self.ratings[home_id] = ra + delta
        self.ratings[away_id] = rb - delta

    def scaled(self, team_id: str, *, center: float = BASE_RATING, span: float = 400.0) -> float:
        """Rating on the model's ~[-2.5, 2.5] ``live_rating`` scale."""
        return max(-2.5, min(2.5, (self.get_rating(team_id) - center) / span))
