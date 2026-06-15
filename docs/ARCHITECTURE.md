# World Cup Agent Architecture

## Forecast Pipeline

The production forecast is layered and auditable:

1. `independent_probabilities`: market-blind Elo/Poisson/BZZOIRO/statistical forecast. Polymarket, bookmakers, market movement, and market priors are excluded.
2. `evidence_adjusted_probabilities`: Analyst LLM returns bounded deltas only. Python validates, caps, applies, and normalizes.
3. `stressed_probabilities`: Devil's Advocate returns two or three weighted scenarios. Python validates and aggregates deterministically.
4. `pre_market_probabilities`: the final belief before executable prices.
5. `scored_probabilities`: Judge selects one market-calibration weight; Python mixes current market probabilities exactly once.

Arena prediction records use `scored_probabilities`. Trading edge uses `pre_market_probability - expected_fill_price`, so the system never compares a market-adjusted forecast back against the same market.

## Evidence

Evidence is represented by `models.evidence.EvidenceItem`. Inputs from web, Reddit, Grok/social pulse, Sportmonks, BZZOIRO, Supabase, and live match state are normalized, deduplicated by factual identity, filtered for expiry, and scored by relevance and reliability. Coverage measures usable evidence diversity, not non-empty payload count.

## Council

Scout splits output into `market_blind_evidence` and `market_observations`. Only market-blind evidence reaches Analyst. Analyst adjusts an existing independent distribution. Devil returns weighted counterfactual scenarios. Judge returns calibration policy, not a probability map.

## Agents

- `monk` / ORACLE: prediction-score specialist. Predicts every window and trades only exceptional, strongly supported forecast disagreements.
- `anchor` / KEEL: baseline all-outcome EV agent. At most one trade per window, moderate edge and EV-after-cost requirements, scout veto enabled.
- `hunter` / SAW: draw and underdog skew agent. Rejects favorites, rejects entry prices over `0.40`, and defaults to one position per fixture.
- `blitz` / SURGE: event-driven agent. Abstains without a valid unexpired event trigger and uses the same forecast, recommendation, portfolio, and execution contracts as every other agent.

## Allocation

Recommendations are common `AgentRecommendation` records. The allocator is order-invariant and uses mandate fit for ownership: ordinary value to ANCHOR, draw/underdog skew to HUNTER, exceptional forecast disagreement to MONK, and event signals to BLITZ. Deduplication is keyed to identical strategy signals, not merely shared forecasts.

## Halftime

Halftime uses a deterministic full three-way remaining-goals model. It combines prematch strength, score, remaining time, xG, shots/proxy pressure, red cards, substitutions/injuries when available, and match stage. The halftime market can only enter through the same final market-calibration step.

## Reporting

Every metrics event carries commit SHA, strategy version, forecast pipeline version, model version, profile configuration hash, feature flags, active data sources, and timestamp. Reports avoid combining incompatible strategy/model/profile versions in aggregates.
