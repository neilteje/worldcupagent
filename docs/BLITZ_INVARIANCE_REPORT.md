# BLITZ Invariance Report

The BLITZ legacy pathway was explicitly isolated in this refactoring architecture to ensure strict backward compatibility for the existing event-driven model.

## Validation Method
The `BlitzStrategy` acts as a facade layer over the legacy data structures. It manually constructs the exact `MatchForecast` shape that existed prior to the refactor. 

## Non-Draw Outcome Preservation
1. **Selection:** For all non-draw predictions, BLITZ retains its existing `our_prob`, `entry_price`, and risk selection logic unmodified.
2. **Sizing:** Fractional Kelly and `bet_policy.MIN_ORDER_USD` floors remain identically mapped.
3. **Execution:** BLITZ bypasses the `PortfolioCoordinator` entirely. It continues to dispatch executing calls precisely as before.
4. **Gates & Scout Vetoes:** Risk gates remain evaluated individually using `evaluate_gates`, and Scout high-severity vetoes remain ignored (as specified by `AgentProfile(skip_on_high_scout_flag=False)`).

## Draw Filtering
When BLITZ encounters a draw pick:
- It removes the draw selection from its array of picks.
- It inserts `blitz_draw_disabled` into the `skip_reasons` ledger.
- It **does not** automatically promote the next best pick; it leaves the allocation slot vacant, respecting the exact invariance constraint outlined in the architecture requirements.
