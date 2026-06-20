"""
Deterministic World Cup predictor — v2.

A clean, inspectable ensemble of independent deterministic signals:

  - **Elo / rating** outcome model       (models.team_strength.elo_1x2)
  - **Poisson + Dixon-Coles** goal model (models.team_strength.expected_goals -> models.poisson_model)
  - **Market** de-vigged odds prior      (optional, configurable weight)

blended with configurable weights, then run through a calibration layer
(temperature scaling + shrinkage toward tournament base rates) so the output is
well-calibrated rather than merely confident. Component toggles drive the
ablation report. No LLM, no randomness, no I/O — given the same inputs it
always returns the same probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from models.calibration import OUTCOMES, clamp_probs, normalize_probs, temperature_scale
from models.poisson_model import DEFAULT_RHO, poisson_1x2
from models.team_strength import StrengthConfig, elo_1x2, expected_goals


@dataclass(frozen=True)
class EnsembleConfig:
    # Market-anchored weights: the de-vigged market is the single strongest
    # forecaster, so it carries the prior; the repo-local statistical layer
    # (Elo + Poisson, built from 2018 + in-tournament data) adds an independent
    # tilt. These are round, a-priori weights — NOT tuned to 2022 outcomes.
    w_elo: float = 0.10
    w_poisson: float = 0.10
    w_market: float = 0.80
    rho: float = DEFAULT_RHO
    # Calibration is done by base-rate shrinkage (temperature held at 1.0); a WC2022
    # parameter sweep with 8-fold CV found this flat optimum (see sweep.md).
    temperature: float = 1.00
    base_rate_shrink: float = 0.10     # shrink toward a generic neutral-site base rate
    base_rate: tuple = (0.40, 0.26, 0.34)  # generic (home/draw/away) WC neutral prior
    # Knockout matches are cagier and level-after-90 counts as a draw -> the market
    # underprices knockout draws. A small prior-justified nudge (not tuned to 2022).
    knockout_draw_boost: float = 0.05
    mw3_temperature_boost: float = 0.15 # widen distribution in Matchweek 3 (rotation/dead rubber variance)
    min_prob: float = 0.02
    max_prob: float = 0.90
    strength: StrengthConfig = field(default_factory=StrengthConfig)
    # ablation toggles
    use_elo: bool = True
    use_poisson: bool = True
    use_market: bool = True


def _shrink_toward(probs: dict[str, float], target: tuple, strength: float) -> dict[str, float]:
    s = max(0.0, min(1.0, strength))
    if s <= 0:
        return normalize_probs(probs)
    p = normalize_probs(probs)
    tgt = normalize_probs({"home": target[0], "draw": target[1], "away": target[2]})
    return normalize_probs({k: p[k] * (1 - s) + tgt[k] * s for k in OUTCOMES})


def _boost_draw(probs: dict[str, float], boost: float) -> dict[str, float]:
    """Nudge draw up by `boost`, taking proportionally from home/away."""
    p = normalize_probs(probs)
    new_draw = min(0.60, p["draw"] + boost)
    scale = (1.0 - new_draw) / max(1e-9, 1.0 - p["draw"])
    return normalize_probs({"home": p["home"] * scale, "draw": new_draw, "away": p["away"] * scale})


def _blend(components: dict[str, dict[str, float]], weights: dict[str, float]) -> dict[str, float]:
    active = {n: normalize_probs(p) for n, p in components.items() if p is not None}
    if not active:
        return normalize_probs(None)
    total_w = sum(max(0.0, weights.get(n, 0.0)) for n in active)
    if total_w <= 0:
        return normalize_probs({k: sum(active[n][k] for n in active) / len(active) for k in OUTCOMES})
    return normalize_probs(
        {k: sum(active[n][k] * max(0.0, weights.get(n, 0.0)) / total_w for n in active) for k in OUTCOMES}
    )


def predict_v2(home_state: dict, away_state: dict, *, market_probs: dict | None = None,
               bzzoiro_probs: dict | None = None,
               cfg: EnsembleConfig | None = None, neutral: bool = True,
               is_knockout: bool = False, match_week: int = 0, host_continent: str | None = None) -> dict:
    """Produce a calibrated 1X2 forecast plus full component breakdown."""
    cfg = cfg or EnsembleConfig()

    eg = expected_goals(home_state, away_state, cfg=cfg.strength, neutral=neutral, is_knockout=is_knockout, host_continent=host_continent)
    poisson = poisson_1x2(eg["lambda_home"], eg["lambda_away"], rho=cfg.rho)
    elo = elo_1x2(home_state, away_state, cfg=cfg.strength, is_knockout=is_knockout)
    market = normalize_probs(market_probs) if market_probs else None
    # Kept as an ignored compatibility argument; it cannot affect this engine.

    components: dict[str, dict[str, float]] = {}
    weights: dict[str, float] = {}
    if cfg.use_elo:
        components["elo"] = elo
        weights["elo"] = cfg.w_elo
    if cfg.use_poisson:
        components["poisson"] = poisson
        weights["poisson"] = cfg.w_poisson
    if cfg.use_market and market is not None:
        components["market"] = market
        weights["market"] = cfg.w_market

    blended = _blend(components, weights)

    # Calibration: temperature softening -> base-rate shrink -> safe clamp.
    temp = cfg.temperature
    if match_week == 3:
        temp += cfg.mw3_temperature_boost
        
    calibrated = temperature_scale(blended, temp)
    calibrated = _shrink_toward(calibrated, cfg.base_rate, cfg.base_rate_shrink)
    if is_knockout and cfg.knockout_draw_boost > 0:
        calibrated = _boost_draw(calibrated, cfg.knockout_draw_boost)
    calibrated = clamp_probs(calibrated, min_prob=cfg.min_prob, max_prob=cfg.max_prob)

    pick = max(OUTCOMES, key=lambda k: calibrated[k])
    return {
        "probabilities": calibrated,
        "pick": pick,
        "confidence": round(calibrated[pick], 4),
        "components": {"elo": elo, "poisson": poisson, "market": market},
        "active_components": list(components.keys()),
        "weights": {n: round(w, 4) for n, w in weights.items()},
        "expected_goals": eg,
        "blended_raw": blended,
        "config": {
            "rho": cfg.rho,
            "temperature": cfg.temperature,
            "base_rate_shrink": cfg.base_rate_shrink,
            "w_elo": cfg.w_elo,
            "w_poisson": cfg.w_poisson,
            "w_market": cfg.w_market,
        },
    }


def ablation_configs(base: EnsembleConfig | None = None) -> dict[str, EnsembleConfig]:
    """Named configs for the ablation report (each isolates a component)."""
    base = base or EnsembleConfig()
    return {
        "full": base,
        "elo_only": replace(base, use_elo=True, use_poisson=False, use_market=False),
        "poisson_only": replace(base, use_elo=False, use_poisson=True, use_market=False),
        "market_only": replace(base, use_elo=False, use_poisson=False, use_market=True, w_market=1.0),
        "stats_only_no_market": replace(base, use_elo=True, use_poisson=True, use_market=False),
        "no_calibration": replace(base, temperature=1.0, base_rate_shrink=0.0, knockout_draw_boost=0.0, mw3_temperature_boost=0.0),
        "no_elo_blend": replace(base, strength=replace(base.strength, elo_blend=0.0)),
        "no_knockout_boost": replace(base, knockout_draw_boost=0.0),
    }


__all__ = ["EnsembleConfig", "predict_v2", "ablation_configs"]
