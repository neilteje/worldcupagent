"""
Deterministic Poisson / Dixon-Coles goal model.

Turns a pair of expected goals (lambda_home, lambda_away) into a 1X2
distribution via an explicit, capped scoreline matrix. A Dixon-Coles
low-score correction (rho) fixes the well-known Poisson under-prediction of
0-0 / 1-1 draws, so draw probability falls out of the score model rather than
a hand-tuned heuristic.

Everything here is pure, deterministic math — no randomness, no I/O.
"""
from __future__ import annotations

import math

from models.calibration import OUTCOMES, normalize_probs

# Dixon-Coles dependence parameter. Negative inflates 0-0 and 1-1, i.e. draws
# for low-scoring games. -0.10 is the conventional football value; it is a
# structural constant, NOT fit to the 2022 results.
DEFAULT_RHO = -0.10
MAX_GOALS = 10


def _poisson_pmf(k: int, lam: float) -> float:
    lam = max(1e-6, float(lam))
    return math.exp(-lam) * lam ** k / math.factorial(k)


def _dc_tau(i: int, j: int, lam_h: float, lam_a: float, rho: float) -> float:
    """Dixon-Coles low-score dependence factor for scoreline (i, j)."""
    if i == 0 and j == 0:
        return 1.0 - lam_h * lam_a * rho
    if i == 0 and j == 1:
        return 1.0 + lam_h * rho
    if i == 1 and j == 0:
        return 1.0 + lam_a * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


def score_matrix(lam_h: float, lam_a: float, *, rho: float = DEFAULT_RHO, max_goals: int = MAX_GOALS) -> list[list[float]]:
    """Joint scoreline probability matrix, Dixon-Coles corrected and normalized.

    Residual mass beyond ``max_goals`` is absorbed by normalization, so the
    matrix always sums to 1 and no probability is ever exactly 0 or 1.
    """
    lam_h = max(0.05, float(lam_h))
    lam_a = max(0.05, float(lam_a))
    home_pmf = [_poisson_pmf(i, lam_h) for i in range(max_goals + 1)]
    away_pmf = [_poisson_pmf(j, lam_a) for j in range(max_goals + 1)]
    matrix = [[0.0] * (max_goals + 1) for _ in range(max_goals + 1)]
    total = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = home_pmf[i] * away_pmf[j] * max(0.0, _dc_tau(i, j, lam_h, lam_a, rho))
            matrix[i][j] = p
            total += p
    if total <= 0:
        total = 1.0
    return [[p / total for p in row] for row in matrix]


def poisson_1x2(lam_h: float, lam_a: float, *, rho: float = DEFAULT_RHO, max_goals: int = MAX_GOALS) -> dict[str, float]:
    """Collapse the scoreline matrix into home/draw/away probabilities."""
    matrix = score_matrix(lam_h, lam_a, rho=rho, max_goals=max_goals)
    home = draw = away = 0.0
    for i, row in enumerate(matrix):
        for j, p in enumerate(row):
            if i > j:
                home += p
            elif i == j:
                draw += p
            else:
                away += p
    return normalize_probs({"home": home, "draw": draw, "away": away})


def expected_total_goals(lam_h: float, lam_a: float) -> float:
    return float(lam_h + lam_a)


__all__ = ["DEFAULT_RHO", "MAX_GOALS", "score_matrix", "poisson_1x2", "expected_total_goals", "OUTCOMES"]
