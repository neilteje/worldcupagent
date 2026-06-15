from typing import Any
import hashlib
from datetime import datetime, timezone
import json
from agents.base import AgentStrategy, ExecutionPlan, AgentRecommendation
from agents.contracts import (
    FixtureDataSnapshot,
    AgentDataView,
    AgentForecast,
    MarketContext,
    TradeCandidate,
    DirectionalSignal,
)
from harness.profiles import get_profile
import config
from agents.monk import _coverage_from_features
from agents._reco import reco_from_candidate
from models.calibration import normalize_probs
from models.poisson_model import poisson_1x2
from models.tails.draw_model import DrawModel
from models.tails.upset_model import UnderdogUpsetModel
from models.tails.ultra_tail_model import UltraTailModel
from models.tails.signals import SignalAggregator

_BAND_HALF = 0.06  # tail forecasts carry a moderate fixed-floor band, widened by coverage


def _hunter_raw_evidence(football_features: dict, forecast) -> list[dict]:
    """Assemble raw directional evidence for HUNTER from the data view.

    A test or the live cycle may inject an explicit ``hunter_evidence`` list;
    otherwise we derive evidence rows from the independent model lean, scout
    flags, BZZOIRO lineups, and public research. Rows are grouped downstream so
    duplicate sources collapse to one signal."""
    ff = football_features or {}
    if isinstance(ff.get("hunter_evidence"), list):
        return ff["hunter_evidence"]

    rows: list[dict] = []
    # 1. Independent model lean toward a tail outcome.
    for slot in ("draw", "away", "home"):
        prob = getattr(forecast, f"{slot}_probability")
        if prob >= 0.30 and slot in ("draw", "away"):
            rows.append({"source": "independent_model", "source_group": "independent_model",
                         "outcome": slot, "direction": "up", "strength": float(prob),
                         "confidence": 0.6, "summary": f"model leans {slot} ({prob:.2f})"})
    # 2. Scout flags (any high/medium severity flag on a side supports its upset).
    for flag in ff.get("scout_flags") or []:
        sev = str(flag.get("severity", "")).lower()
        if sev in ("high", "medium"):
            rows.append({"source": "scout", "source_group": "scout",
                         "outcome": str(flag.get("slot") or flag.get("outcome") or "away"),
                         "direction": "up", "strength": 0.7 if sev == "high" else 0.5,
                         "confidence": 0.6, "summary": flag.get("description", "scout flag")})
    return rows


class HunterStrategy(AgentStrategy):
    name = "hunter"

    def __init__(self):
        self.profile = get_profile(self.name)
        self.draw_model = DrawModel()
        self.upset_model = UnderdogUpsetModel()
        self.ultra_tail = UltraTailModel()
        self.signals = SignalAggregator()

    def build_data_view(
        self,
        snapshot: FixtureDataSnapshot,
        prior_agent_forecast: AgentForecast | None = None,
    ) -> AgentDataView:
        
        coverage = _coverage_from_features(snapshot.football_context)
        view_state = {
            "agent": self.name,
            "fixture_id": snapshot.fixture_id,
            "window": snapshot.window,
        }
        view_hash = hashlib.sha256(json.dumps(view_state, sort_keys=True, default=str).encode()).hexdigest()

        return AgentDataView(
            agent_name=self.name,
            fixture_id=snapshot.fixture_id,
            window=snapshot.window,
            as_of_timestamp=snapshot.as_of_timestamp,
            football_features=snapshot.football_context,
            live_features=snapshot.live_context,
            external_model_predictions={},
            evidence=tuple(),
            market_features=None,
            data_coverage=coverage,
            prohibited_fields_removed=tuple(),
            warnings=tuple(),
            data_view_hash=view_hash,
        )

    def build_forecast(
        self,
        view: AgentDataView,
    ) -> AgentForecast:
        forecast_id = hashlib.sha256(f"{view.data_view_hash}_hunter_v1".encode()).hexdigest()
        coverage = float(view.data_coverage.get("overall", 1.0))

        # HUNTER models the tails SEPARATELY (spec §10): a dedicated draw model
        # and a dedicated underdog-upset model, both grounded in the Poisson
        # score matrix, then renormalized into a full 1X2 distribution. This is
        # a materially different belief from MONK/ANCHOR's symmetric blend.
        ff = view.football_features or {}
        det = ff.get("deterministic_model") or {}
        eg = det.get("expected_goals") or ff.get("expected_goals") or {}
        lam_h = float(eg.get("lambda_home", 1.3) or 1.3)
        lam_a = float(eg.get("lambda_away", 1.3) or 1.3)

        context = {
            "is_knockout": bool(ff.get("is_knockout")),
            "expected_total_goals": lam_h + lam_a,
        }
        underdog = "away" if lam_a <= lam_h else "home"
        draw_p = self.draw_model.probability(lam_h, lam_a, context)
        upset_p = self.upset_model.probability(lam_h, lam_a, underdog, context)
        favorite = "home" if underdog == "away" else "away"
        fav_p = max(0.02, 1.0 - draw_p - upset_p)
        raw = {"draw": draw_p, underdog: upset_p, favorite: fav_p}
        p = normalize_probs(raw)

        half = _BAND_HALF + (1.0 - coverage) * 0.10
        lo = {k: max(0.0, p[k] - half) for k in p}
        hi = {k: min(1.0, p[k] + half) for k in p}
        confidence = round(min(0.99, max(p.values())), 4)

        return AgentForecast(
            agent_name=self.name,
            fixture_id=view.fixture_id,
            window=view.window,
            as_of_timestamp=view.as_of_timestamp,
            home_probability=p["home"],
            draw_probability=p["draw"],
            away_probability=p["away"],
            home_lower_bound=lo["home"],
            draw_lower_bound=lo["draw"],
            away_lower_bound=lo["away"],
            home_upper_bound=hi["home"],
            draw_upper_bound=hi["draw"],
            away_upper_bound=hi["away"],
            confidence=confidence,
            data_coverage_score=coverage,
            forecast_type="tail_focused",
            model_version="hunter_v1",
            components={"draw_model": draw_p, "upset_model": upset_p,
                        "underdog": underdog, "lambda_home": lam_h, "lambda_away": lam_a},
            evidence_ids=tuple(),
            warnings=tuple(),
            data_view_hash=view.data_view_hash,
            forecast_id=forecast_id,
        )

    def generate_candidates(
        self,
        forecast: AgentForecast,
        view: AgentDataView,
        market: MarketContext | None,
    ) -> list[TradeCandidate]:
        if not market:
            return []

        raw_evidence = _hunter_raw_evidence(view.football_features, forecast)

        candidates = []
        for outcome in ("home", "draw", "away"):
            # Skew harvester: only draws and underdogs (priced <= max_entry_price);
            # never favorites — payout asymmetry is the product.
            fill_price = market.expected_fill_price.get(outcome) or market.best_ask.get(outcome) or market.midpoint.get(outcome)

            if not fill_price or fill_price > self.profile.max_entry_price:
                continue

            # Ultra-cheap (<5¢) tails require a dedicated validation pass; rejected
            # by default (spec §10).
            if not self.ultra_tail.is_valid(fill_price):
                continue

            # Require ≥ N INDEPENDENT directional signals supporting THIS exact
            # outcome — distinct source groups, not duplicate copies (spec §10).
            signals = self.signals.signals_for(raw_evidence, forecast.fixture_id, outcome, "up")
            n_independent = SignalAggregator.independent_count(list(signals), outcome, "up")
            if n_independent < config.HUNTER_MIN_INDEPENDENT_SIGNALS:
                continue

            prob = getattr(forecast, f"{outcome}_probability")
            lower = getattr(forecast, f"{outcome}_lower_bound")

            fee_buffer = config.FEE_BUFFER
            slippage_buffer = config.SLIPPAGE_BUFFER
            model_risk_buffer = config.MODEL_RISK_BUFFER

            conservative_edge = lower - fill_price - fee_buffer - slippage_buffer - model_risk_buffer
            ev_after_costs = prob - fill_price - fee_buffer - slippage_buffer

            # Candidate only needs positive raw edge here; the conservative-edge /
            # EV / signal-count gates are re-applied (auditable) in build_recommendation.
            if (prob - fill_price) > 0:
                candidates.append(
                    TradeCandidate(
                        agent_name=self.name,
                        fixture_id=forecast.fixture_id,
                        outcome=outcome,
                        probability_mean=prob,
                        probability_lower_bound=lower,
                        probability_upper_bound=getattr(forecast, f"{outcome}_upper_bound"),
                        market_midpoint=market.midpoint.get(outcome),
                        best_ask=market.best_ask.get(outcome),
                        expected_fill_price=fill_price,
                        gross_edge=prob - fill_price,
                        conservative_edge=conservative_edge,
                        expected_value_after_costs=ev_after_costs,
                        signal_type="tail_value",
                        signals=signals,
                        candidate_created_at=datetime.now(timezone.utc),
                        candidate_expires_at=None,
                        forecast_id=forecast.forecast_id,
                        correlation_key=f"{self.name}_{forecast.fixture_id}_{outcome}",
                    )
                )
        return candidates

    def generate_recommendations(
        self,
        candidates: list[TradeCandidate],
        forecast: AgentForecast,
        view: AgentDataView,
        market: MarketContext | None,
        bankroll: float,
    ) -> list[AgentRecommendation]:
        # At most ONE tail position per fixture (spec §10): the highest-EV tail.
        # Gating (conservative edge, EV-after-costs, signal count, ultra-tail)
        # runs inside build_recommendation, producing auditable abstentions.
        limit = max(1, int(config.HUNTER_MAX_TAIL_POSITIONS_PER_FIXTURE))
        ranked = sorted(candidates, key=lambda c: c.expected_value_after_costs or -1, reverse=True)
        return [
            reco_from_candidate(self.name, cand, view, forecast, bankroll, self.profile)
            for cand in ranked[:limit]
        ]

    def create_execution_plan(
        self,
        recommendations: list[AgentRecommendation],
    ) -> ExecutionPlan:
        return ExecutionPlan()
