"""Authoritative deterministic probability engine.

LLM roles may provide structured evidence, feature recommendations, scenarios,
and calibration-policy names.  This module owns every probability map that can
be submitted or traded from the live path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from typing import Any

from models.calibration import OUTCOMES, clamp_probs, normalize_probs
from models.deterministic_v2 import EnsembleConfig, predict_v2
from models.forecast_contracts import stable_hash
from models.market_calibration import apply_market_calibration, normalize_market_probabilities

MODEL_VERSION = "probability_engine_v1"
TOLERANCE = 1e-6
SUPPORTED_FEATURES = {
    "home_attacking_strength",
    "away_attacking_strength",
    "home_defensive_strength",
    "away_defensive_strength",
    "home_lineup_strength",
    "away_lineup_strength",
    "lineup_uncertainty",
    "draw_variance",
    "expected_total_goals",
    "home_rest",
    "away_rest",
}
POLICY_WEIGHTS = {"none": 0.0, "light": 0.15, "moderate": 0.30, "strong": 0.50}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _finite_probability(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return value


@dataclass(frozen=True)
class ProbabilityDistribution:
    home: float
    draw: float
    away: float
    model_version: str
    as_of_timestamp: datetime
    stage: str
    warnings: tuple[str, ...] = field(default_factory=tuple)
    upstream_forecast_ids: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "home", _finite_probability("home", self.home))
        object.__setattr__(self, "draw", _finite_probability("draw", self.draw))
        object.__setattr__(self, "away", _finite_probability("away", self.away))
        total = self.home + self.draw + self.away
        if abs(total - 1.0) > TOLERANCE:
            raise ValueError(f"probability distribution must sum to 1, got {total:.8f}")
        ts = self.as_of_timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
            object.__setattr__(self, "as_of_timestamp", ts)

    @property
    def values(self) -> dict[str, float]:
        return {"home": self.home, "draw": self.draw, "away": self.away}

    @property
    def forecast_id(self) -> str:
        return stable_hash(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["as_of_timestamp"] = self.as_of_timestamp.astimezone(timezone.utc).isoformat()
        data["forecast_id"] = self.forecast_id
        return data


@dataclass(frozen=True)
class EvidenceAdjustment:
    feature: str
    operation: str
    magnitude: float
    evidence_ids: tuple[str, ...]
    confidence: float
    rationale: str = ""


@dataclass(frozen=True)
class StressScenario:
    scenario_id: str
    description: str
    plausibility: float
    feature_overrides: dict[str, float]
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class MarketConsensus:
    observed_at: datetime
    probabilities: dict[str, float] | None
    source_count: int = 0
    liquidity: float | None = None
    disagreement: float | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ForecastLayers:
    base: ProbabilityDistribution
    evidence_adjusted: ProbabilityDistribution
    stressed: ProbabilityDistribution
    market_calibrated: ProbabilityDistribution | None


@dataclass(frozen=True)
class AgentForecast:
    agent_name: str
    submitted: ProbabilityDistribution
    trading_belief: ProbabilityDistribution
    model_policy: str
    expected_goals: dict[str, float]
    components: dict[str, Any]
    uncertainty: dict[str, float]
    data_coverage_score: float
    feature_snapshot_hash: str
    audit: dict[str, Any]

    @property
    def forecast_id(self) -> str:
        return stable_hash({
            "agent_name": self.agent_name,
            "submitted": self.submitted.to_dict(),
            "trading_belief": self.trading_belief.to_dict(),
            "model_policy": self.model_policy,
            "feature_snapshot_hash": self.feature_snapshot_hash,
        })

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["submitted"] = self.submitted.to_dict()
        data["trading_belief"] = self.trading_belief.to_dict()
        data["forecast_id"] = self.forecast_id
        return data


def _dist(values: dict[str, float], *, stage: str, warnings: tuple[str, ...] = (),
          upstream: tuple[str, ...] = ()) -> ProbabilityDistribution:
    p = normalize_probs(values)
    return ProbabilityDistribution(
        home=p["home"], draw=p["draw"], away=p["away"],
        model_version=MODEL_VERSION, as_of_timestamp=_utcnow(), stage=stage,
        warnings=warnings, upstream_forecast_ids=upstream,
    )


def _apply_feature_delta(features: dict[str, float], feature: str, delta: float) -> dict[str, float]:
    out = dict(features)
    out[feature] = float(out.get(feature, 0.0) or 0.0) + delta
    return out


def _feature_adjusted_probs(base: dict[str, float], features: dict[str, float]) -> dict[str, float]:
    p = dict(base)
    p["home"] += 0.050 * features.get("home_attacking_strength", 0.0)
    p["home"] += 0.045 * features.get("home_defensive_strength", 0.0)
    p["home"] += 0.050 * features.get("home_lineup_strength", 0.0)
    p["away"] += 0.050 * features.get("away_attacking_strength", 0.0)
    p["away"] += 0.045 * features.get("away_defensive_strength", 0.0)
    p["away"] += 0.050 * features.get("away_lineup_strength", 0.0)
    p["home"] += 0.020 * features.get("home_rest", 0.0)
    p["away"] += 0.020 * features.get("away_rest", 0.0)
    p["draw"] += 0.045 * features.get("draw_variance", 0.0)
    p["draw"] -= 0.035 * features.get("expected_total_goals", 0.0)
    p["draw"] += 0.030 * features.get("lineup_uncertainty", 0.0)
    return clamp_probs(normalize_probs({k: max(0.001, p[k]) for k in OUTCOMES}), 0.02, 0.90)


def _parse_adjustments(payload: dict | None, evidence_ids: tuple[str, ...]) -> tuple[list[EvidenceAdjustment], list[str]]:
    warnings: list[str] = []
    raw = (payload or {}).get("feature_adjustments")
    if raw is None and isinstance((payload or {}).get("adjustments"), list):
        raw = payload.get("adjustments")
    if raw is None:
        return [], warnings
    if not isinstance(raw, list):
        return [], ["invalid_analyst_adjustment"]
    allowed_evidence = set(evidence_ids)
    out: list[EvidenceAdjustment] = []
    for item in raw:
        if not isinstance(item, dict):
            warnings.append("invalid_analyst_adjustment")
            continue
        feature = str(item.get("feature") or "")
        operation = str(item.get("operation") or item.get("direction") or "")
        ids = tuple(str(x) for x in (item.get("evidence_ids") or ()))
        try:
            magnitude = max(0.0, min(0.20, float(item.get("magnitude", 0.0) or 0.0)))
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0) or 0.0)))
        except (TypeError, ValueError):
            warnings.append("invalid_analyst_adjustment")
            continue
        if feature not in SUPPORTED_FEATURES:
            warnings.append(f"unsupported_feature:{feature}")
            continue
        if ids and not set(ids).issubset(allowed_evidence):
            warnings.append(f"unknown_evidence_reference:{feature}")
            continue
        if operation not in {"increase", "decrease", "set_missing", "increase_uncertainty"}:
            warnings.append(f"unsupported_operation:{operation}")
            continue
        out.append(EvidenceAdjustment(feature, operation, magnitude, ids, confidence,
                                      str(item.get("rationale") or "")))
    return out, warnings


def _parse_scenarios(payload: dict | None, evidence_ids: tuple[str, ...]) -> tuple[list[StressScenario], list[str]]:
    raw = (payload or {}).get("scenarios")
    if raw is None:
        return [], []
    if not isinstance(raw, list):
        return [], ["invalid_devil_scenarios"]
    warnings: list[str] = []
    allowed_evidence = set(evidence_ids)
    out: list[StressScenario] = []
    for item in raw:
        if not isinstance(item, dict):
            warnings.append("invalid_devil_scenarios")
            continue
        overrides = item.get("feature_overrides") or {}
        ids = tuple(str(x) for x in (item.get("evidence_ids") or ()))
        try:
            plaus = max(0.0, min(0.50, float(item.get("plausibility", 0.0) or 0.0)))
            clean = {str(k): max(-0.30, min(0.30, float(v))) for k, v in overrides.items()}
        except (TypeError, ValueError):
            warnings.append("invalid_devil_scenarios")
            continue
        bad = [k for k in clean if k not in SUPPORTED_FEATURES]
        if bad:
            warnings.append(f"unsupported_scenario_feature:{bad[0]}")
            continue
        if ids and not set(ids).issubset(allowed_evidence):
            warnings.append("unknown_scenario_evidence_reference")
            continue
        out.append(StressScenario(
            scenario_id=str(item.get("scenario_id") or stable_hash(item)[:12]),
            description=str(item.get("description") or ""),
            plausibility=plaus,
            feature_overrides=clean,
            evidence_ids=ids,
        ))
    return sorted(out, key=lambda s: s.scenario_id), warnings


def _calibration_weight(judge_output: dict | None) -> tuple[float, str]:
    policy = str((judge_output or {}).get("recommended_calibration_policy") or "none").lower()
    if policy not in POLICY_WEIGHTS:
        return 0.0, "invalid_judge_policy"
    return POLICY_WEIGHTS[policy], ""


def _uncertainty(p: dict[str, float], confidence: float, coverage: float) -> dict[str, float]:
    width = max(0.06, min(0.36, 0.10 + (1.0 - confidence) * 0.10 + (1.0 - coverage) * 0.16))
    out: dict[str, float] = {}
    for slot in OUTCOMES:
        out[f"{slot}_lower"] = max(0.0, p[slot] - width / 2.0)
        out[f"{slot}_upper"] = min(1.0, p[slot] + width / 2.0)
    return out


class DeterministicProbabilityEngine:
    def build_layers(
        self,
        home_state: dict[str, Any],
        away_state: dict[str, Any],
        *,
        analyst_output: dict | None = None,
        devil_output: dict | None = None,
        judge_output: dict | None = None,
        market_consensus: MarketConsensus | None = None,
        evidence_ids: tuple[str, ...] = (),
        data_coverage_score: float = 0.0,
        bzzoiro_probs: dict | None = None,
        bzzoiro_shadow_only: bool = True,
        cfg: EnsembleConfig | None = None,
        is_knockout: bool = False,
    ) -> tuple[ForecastLayers, dict[str, Any]]:
        cfg = cfg or EnsembleConfig(w_elo=0.50, w_poisson=0.50, w_market=0.0, use_market=False)
        if bzzoiro_shadow_only:
            cfg = EnsembleConfig(**{**cfg.__dict__, "use_bzzoiro": False, "w_bzzoiro": 0.0})
        base_out = predict_v2(
            home_state, away_state, market_probs=None, bzzoiro_probs=bzzoiro_probs,
            cfg=cfg, is_knockout=is_knockout,
        )
        base = _dist(base_out["probabilities"], stage="base")
        adjustments, adj_warnings = _parse_adjustments(analyst_output, evidence_ids)
        features: dict[str, float] = {}
        adjustment_audit = []
        for adj in adjustments:
            sign = -1.0 if adj.operation == "decrease" else 1.0
            if adj.operation == "increase_uncertainty":
                feature = "lineup_uncertainty"
                sign = 1.0
            else:
                feature = adj.feature
            delta = sign * adj.magnitude * max(0.0, min(1.0, adj.confidence))
            features = _apply_feature_delta(features, feature, delta)
            adjustment_audit.append(asdict(adj) | {"applied_delta": delta})
        evidence_probs = _feature_adjusted_probs(base.values, features)
        evidence = _dist(evidence_probs, stage="evidence_adjusted",
                         warnings=tuple(adj_warnings), upstream=(base.forecast_id,))

        scenarios, scenario_warnings = _parse_scenarios(devil_output, evidence_ids)
        scenario_runs = []
        if scenarios:
            total = min(0.50, sum(s.plausibility for s in scenarios))
            denom = sum(s.plausibility for s in scenarios) or 1.0
            agg = dict(evidence.values)
            stress_mix = {k: 0.0 for k in OUTCOMES}
            for scenario in scenarios:
                scenario_features = dict(features)
                for feat, delta in scenario.feature_overrides.items():
                    scenario_features = _apply_feature_delta(scenario_features, feat, delta)
                probs = _feature_adjusted_probs(base.values, scenario_features)
                scenario_runs.append({"scenario": asdict(scenario), "probabilities": probs})
                for slot in OUTCOMES:
                    stress_mix[slot] += probs[slot] * scenario.plausibility / denom
            agg = normalize_probs({k: agg[k] * (1.0 - total) + stress_mix[k] * total for k in OUTCOMES})
        else:
            agg = evidence.values
        stressed = _dist(agg, stage="stressed", warnings=tuple(scenario_warnings),
                         upstream=(evidence.forecast_id,))

        market_calibrated = None
        calibration_warning = ""
        weight, calibration_warning = _calibration_weight(judge_output)
        market_probs = normalize_market_probabilities((market_consensus or MarketConsensus(_utcnow(), None)).probabilities)
        if market_probs and weight > 0.0:
            market_calibrated = _dist(
                apply_market_calibration(stressed.values, market_probs, weight),
                stage="market_calibrated",
                warnings=tuple(x for x in (calibration_warning,) if x),
                upstream=(stressed.forecast_id,),
            )
        audit = {
            "base_model": base_out,
            "adjustments": adjustment_audit,
            "feature_deltas": features,
            "scenario_runs": scenario_runs,
            "market_calibration_weight": weight if market_probs else 0.0,
            "market_consensus": asdict(market_consensus) if market_consensus else None,
            "warnings": tuple(adj_warnings + scenario_warnings + ([calibration_warning] if calibration_warning else [])),
            "data_coverage_score": data_coverage_score,
        }
        return ForecastLayers(base, evidence, stressed, market_calibrated), audit

    def build_agent_forecasts(
        self,
        home_state: dict[str, Any],
        away_state: dict[str, Any],
        *,
        analyst_output: dict | None = None,
        devil_output: dict | None = None,
        judge_output: dict | None = None,
        market_consensus: MarketConsensus | None = None,
        evidence_ids: tuple[str, ...] = (),
        data_coverage_score: float = 0.0,
        bzzoiro_probs: dict | None = None,
        bzzoiro_shadow_only: bool = True,
        is_knockout: bool = False,
    ) -> dict[str, AgentForecast]:
        layers, audit = self.build_layers(
            home_state, away_state, analyst_output=analyst_output,
            devil_output=devil_output, judge_output=judge_output,
            market_consensus=market_consensus, evidence_ids=evidence_ids,
            data_coverage_score=data_coverage_score, bzzoiro_probs=bzzoiro_probs,
            bzzoiro_shadow_only=bzzoiro_shadow_only, is_knockout=is_knockout,
        )
        hunter_cfg = EnsembleConfig(
            w_elo=0.20, w_poisson=0.80, w_market=0.0, use_market=False,
            rho=-0.08, base_rate_shrink=0.05, knockout_draw_boost=0.07,
            use_bzzoiro=not bzzoiro_shadow_only, w_bzzoiro=0.0 if bzzoiro_shadow_only else 0.05,
        )
        hunter_layers, hunter_audit = self.build_layers(
            home_state, away_state, analyst_output=analyst_output,
            devil_output=devil_output, judge_output={"recommended_calibration_policy": "none"},
            market_consensus=None, evidence_ids=evidence_ids,
            data_coverage_score=data_coverage_score, bzzoiro_probs=bzzoiro_probs,
            bzzoiro_shadow_only=bzzoiro_shadow_only, cfg=hunter_cfg,
            is_knockout=is_knockout,
        )
        blitz_valid = bool((analyst_output or {}).get("event_signal_valid"))
        blitz_submitted = layers.stressed
        blitz_policy = "event_model"
        blitz_warnings = tuple()
        if not blitz_valid:
            blitz_policy = "abstain_no_valid_event"
            blitz_warnings = ("blitz_no_valid_event_signal",)
            blitz_submitted = _dist(layers.evidence_adjusted.values, stage="blitz_abstain",
                                    warnings=blitz_warnings,
                                    upstream=(layers.evidence_adjusted.forecast_id,))

        def pack(agent: str, submitted: ProbabilityDistribution,
                 trading: ProbabilityDistribution, policy: str,
                 agent_audit: dict[str, Any]) -> AgentForecast:
            probs = submitted.values
            conf = max(probs.values())
            coverage = max(0.0, min(1.0, float(data_coverage_score or 0.0)))
            return AgentForecast(
                agent_name=agent,
                submitted=submitted,
                trading_belief=trading,
                model_policy=policy,
                expected_goals=audit["base_model"].get("expected_goals") or {},
                components=audit["base_model"].get("components") or {},
                uncertainty=_uncertainty(probs, conf, coverage),
                data_coverage_score=coverage,
                feature_snapshot_hash=stable_hash({"home_state": home_state, "away_state": away_state, "agent": agent}),
                audit=agent_audit,
            )

        return {
            "monk": pack("monk", layers.evidence_adjusted, layers.evidence_adjusted,
                         "market_blind_evidence_adjusted", audit),
            "anchor": pack("anchor", layers.market_calibrated or layers.stressed, layers.stressed,
                           "single_market_calibration", audit),
            "hunter": pack("hunter", hunter_layers.stressed, hunter_layers.stressed,
                           "specialized_draw_underdog", hunter_audit),
            "blitz": pack("blitz", blitz_submitted, blitz_submitted,
                          blitz_policy, {**audit, "warnings": tuple(audit.get("warnings", ())) + blitz_warnings}),
        }


__all__ = [
    "AgentForecast",
    "DeterministicProbabilityEngine",
    "EvidenceAdjustment",
    "ForecastLayers",
    "MarketConsensus",
    "MODEL_VERSION",
    "POLICY_WEIGHTS",
    "ProbabilityDistribution",
    "StressScenario",
]
