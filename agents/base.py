from typing import Protocol

from agents.contracts import (
    FixtureDataSnapshot,
    AgentDataView,
    AgentForecast,
    MarketContext,
    TradeCandidate,
)

class AgentRecommendation:
    # We will likely use the existing betting.recommendation.AgentRecommendation,
    # but the interface requires this type. For now, we'll just import it or type hint it as Any if needed.
    pass

# We will import the real AgentRecommendation and ExecutionPlan 
from betting.recommendation import AgentRecommendation

class ExecutionPlan:
    # A placeholder if not defined in the current architecture.
    # The requirement says "create_execution_plan(self, recommendations: list['AgentRecommendation']) -> 'ExecutionPlan'"
    pass

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
