# Agent Design Report

## MONK (Independent Forecast Specialist)
- **Data Visiblity**: MONK has full access to the football-only deterministic engine, external models (BZZOIRO), rolling form, lineup strength, and historical features.
- **Data Blindness**: MONK is completely blind to Polymarket, Kalshi, and Sportmonks bookmaker probabilities.
- **Forecasting Method**: Forms a full 1X2 distribution based strictly on the deterministic football outputs (Elo + Poisson + adjustments).
- **Candidate Generation**: Exceptionally rare. Requires a structural mispricing where the market deviates significantly from the pure baseline.
- **Risk Mandate**: Minimum edge of 8%. Highly conservative. Primarily serves as the calibration benchmark.
- **Execution Path**: Routed through the Portfolio Coordinator.
- **Ledger Path**: Distinct `AgentDataView` and `AgentForecast` traces omitting market inputs.
- **Primary Metrics**: Brier score, Log loss, Expected Calibration Error.

## ANCHOR (Disciplined Value Agent)
- **Data Visiblity**: Initially shares MONK's blind view to form the foundation. It then introduces executable market prices solely for candidate evaluation.
- **Forecasting Method**: Inherits the MONK baseline forecast to remain unanchored by the market.
- **Candidate Generation**: Evaluates all three outcomes against market prices, applying a deep `conservative_edge` check that deducts fee, slippage, and model risk buffers.
- **Risk Mandate**: Minimum conservative edge of 5%. Seeks pure risk-adjusted value.
- **Execution Path**: Routed through the Portfolio Coordinator.
- **Ledger Path**: Stores the independent probability but overlays it with a `MarketContext` for candidate validation.
- **Primary Metrics**: Return on staked capital, CLV, Realized P&L after fees.

## HUNTER (Specialized Tail Agent)
- **Data Visiblity**: Same as ANCHOR.
- **Forecasting Method**: Replaces the baseline forecast with specialized tail inflation using `DrawModel` and `UnderdogUpsetModel` (found in `models/tails/`). It systematically overestimates these outcomes.
- **Candidate Generation**: Seeks ultra-cheap (<5%) skewed outcomes. Requires at least two independent directional signals.
- **Risk Mandate**: Minimum conservative edge of 8%, minimum EV after costs of 12%.
- **Execution Path**: Routed through the Portfolio Coordinator.
- **Ledger Path**: Forecast hash denotes the injection of the tail models.
- **Primary Metrics**: Tail drawdown, Draw P&L, Underdog P&L.

## BLITZ (Preserved Event-Driven Agent)
- **Data Visiblity**: Full access to the legacy `MatchForecast` (which included market inputs historically).
- **Forecasting Method**: Uses the legacy council output.
- **Candidate Generation**: Existing legacy heuristics. Strictly removes draw picks but preserves the non-draw picks exactly.
- **Risk Mandate**: Fast, aggressive, event-driven.
- **Execution Path**: Bypasses the Portfolio Coordinator entirely. Direct execution.
- **Ledger Path**: Preserves the legacy traces.
- **Primary Metrics**: Legacy forecast performance, Cheap-contract P&L.
