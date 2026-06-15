from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

@dataclass(frozen=True)
class ProviderSnapshot:
    provider: str
    resource_type: str
    provider_id: str | None

    retrieved_at: datetime
    provider_updated_at: datetime | None

    payload_hash: str
    payload: dict | list | None

    success: bool
    stale: bool

    error_code: str | None = None
    error_message: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class MarketContext:
    observed_at: datetime

    polymarket: dict | None
    kalshi: dict | None
    bookmaker_consensus: dict | None
    bookmaker_comparison: dict | None

    devigged_probabilities: dict
    best_bid: dict
    best_ask: dict
    midpoint: dict
    expected_fill_price: dict

    movement: dict
    dispersion: dict
    overround: float | None

    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FixtureDataSnapshot:
    fixture_id: str
    fixture_name: str
    window: str
    kickoff: datetime
    as_of_timestamp: datetime

    home_code: str
    away_code: str
    home_name: str
    away_name: str

    sportmonks: ProviderSnapshot | None
    supabase: ProviderSnapshot | None
    bzzoiro: ProviderSnapshot | None
    web: ProviderSnapshot | None
    reddit: ProviderSnapshot | None
    social: ProviderSnapshot | None

    football_context: dict
    live_context: dict | None
    market_context: MarketContext | None

    snapshot_id: str
    snapshot_hash: str
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AgentDataView:
    agent_name: str
    fixture_id: str
    window: str
    as_of_timestamp: datetime

    football_features: dict
    live_features: dict | None
    external_model_predictions: dict
    evidence: tuple[dict, ...]

    market_features: dict | None

    data_coverage: dict
    prohibited_fields_removed: tuple[str, ...]
    warnings: tuple[str, ...]

    data_view_hash: str


@dataclass(frozen=True)
class AgentForecast:
    agent_name: str
    fixture_id: str
    window: str
    as_of_timestamp: datetime

    home_probability: float
    draw_probability: float
    away_probability: float

    home_lower_bound: float
    draw_lower_bound: float
    away_lower_bound: float

    home_upper_bound: float
    draw_upper_bound: float
    away_upper_bound: float

    confidence: float
    data_coverage_score: float

    forecast_type: str
    model_version: str

    components: dict
    evidence_ids: tuple[str, ...]
    warnings: tuple[str, ...]

    data_view_hash: str
    forecast_id: str


@dataclass(frozen=True)
class DirectionalSignal:
    signal_id: str
    fixture_id: str
    outcome: str

    source: str
    source_group: str

    direction: str
    strength: float
    confidence: float

    observed_at: datetime
    expires_at: datetime | None

    evidence_hash: str
    summary: str


@dataclass(frozen=True)
class TradeCandidate:
    agent_name: str
    fixture_id: str
    outcome: str

    probability_mean: float
    probability_lower_bound: float
    probability_upper_bound: float

    market_midpoint: float | None
    best_ask: float | None
    expected_fill_price: float | None

    gross_edge: float | None
    conservative_edge: float | None
    expected_value_after_costs: float | None

    signal_type: str
    signals: tuple[DirectionalSignal, ...]

    candidate_created_at: datetime
    candidate_expires_at: datetime | None

    forecast_id: str
    correlation_key: str
