from dataclasses import dataclass, field
from typing import Protocol

from agents.contracts import (
    FixtureDataSnapshot,
    AgentDataView,
    AgentForecast,
    MarketContext,
    TradeCandidate,
)

from betting.recommendation import AgentRecommendation

@dataclass(frozen=True)
class ExecutionPlan:
    recommendations: tuple[AgentRecommendation, ...] = field(default_factory=tuple)
    orders: tuple[dict, ...] = field(default_factory=tuple)

class AgentStrategy(Protocol):
    name: str

    def build_data_view(
        self,
        snapshot: FixtureDataSnapshot,
        prior_agent_forecast: AgentForecast | None = None,
    ) -> AgentDataView:
        ...

    def build_forecast(
        self,
        view: AgentDataView,
    ) -> AgentForecast:
        ...

    def generate_candidates(
        self,
        forecast: AgentForecast,
        view: AgentDataView,
        market: MarketContext | None,
    ) -> list[TradeCandidate]:
        ...

    def generate_recommendations(
        self,
        candidates: list[TradeCandidate],
        forecast: AgentForecast,
        view: AgentDataView,
        market: MarketContext | None,
        bankroll: float,
    ) -> list[AgentRecommendation]:
        ...

    def create_execution_plan(
        self,
        recommendations: list[AgentRecommendation],
    ) -> ExecutionPlan:
        ...
