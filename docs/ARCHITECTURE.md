# World Cup Agent Architecture

The four-agent system is a specialized ensemble designed to navigate low-liquidity and highly variable prediction markets during the World Cup. It leverages a shared reasoning loop while maintaining four strictly independent market strategies.

## The Shared Reasoning Council

The core brain uses a 4-role LLM deliberation loop that runs exactly once per window:
1. **Scout (Claude-fast)**: Aggregates real-time external data (social pulse, Reddit sentiment, web news). Outputs severity-tagged flags.
2. **Analyst (Claude-deep)**: Generates a truly market-blind statistical baseline prediction, incorporating deterministic models (Elo, xG form) without seeing any bookmaker or prediction market odds.
3. **Devil's Advocate (DeepSeek-R1)**: Attacks the weakest assumption of the Analyst's baseline to formulate a counter-prediction.
4. **Judge (Claude-deep)**: Synthesizes the Analyst and Devil's Advocate models and is finally allowed to view the prediction markets, performing calibration and yielding a final probability distribution.

## The Four Independent Agents

Each agent instantiates a concrete strategy protocol (`AgentStrategy`) with a distinct mandate:

### MONK (Value Investor)
- **Data View**: Excludes ALL market data fields from its inputs.
- **Forecast**: Derives an independent, purely statistical forecast utilizing the shared reasoning.
- **Candidate Generation**: Seeks structural mispricings.

### ANCHOR (Disciplined Arbitrage)
- **Data View**: Excludes market inputs for the base forecast but introduces them for execution planning.
- **Forecast**: Same base forecast as MONK.
- **Candidate Generation**: Evaluates conservative edges with deep fee and slippage buffering. Will only trade heavily discounted probabilities.

### HUNTER (Tail Skew)
- **Data View**: Identical to Anchor but heavily modified candidate generation.
- **Forecast**: Employs structural tail inflation using the Upset Model and Draw Model to systematically overestimate underdog and draw probabilities to hunt for >25x payouts.
- **Candidate Generation**: Focuses only on <5% probability "ultra tail" outcomes.

### BLITZ (Legacy Baseline)
- **Data View**: Fully compatible with the pre-refactor, monolithic legacy forecast object.
- **Forecast**: Represents the prior production baseline.
- **Candidate Generation**: Retains legacy heuristics, except it strictly refuses to bet on draws.

## Execution and Ledgers

1. **Portfolio Coordinator**: All trades from MONK, ANCHOR, and HUNTER are routed through a central allocator that suppresses duplicates and manages max exposure limits. BLITZ bypasses the coordinator.
2. **Ledgers**: Every agent executes under its own API key and generates an independent, auditable reasoning trace that captures its full context and reasoning chain.
