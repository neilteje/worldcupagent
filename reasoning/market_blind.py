"""Shared market-field scrubber (spec §7/§8/§12).

A single recursive scrubber used by both MONK's data view and the council's
market-blind Analyst, so "market-blind" means the same thing everywhere. It
fails closed: any key whose name contains a prohibited market token is dropped,
recursing into nested dicts and lists.
"""
from __future__ import annotations

# Substrings that mark a market-derived field. Matched case-insensitively against
# dict KEYS only (values are preserved unless their key is prohibited).
PROHIBITED_TOKENS = (
    "polymarket", "kalshi", "bookmaker", "odds", "midpoint", "best_bid",
    "best_ask", "market", "implied_win_prob", "payout", "expected_profit",
    "price",
)


def is_prohibited_key(key: str) -> bool:
    k = str(key).lower()
    return any(tok in k for tok in PROHIBITED_TOKENS)


def scrub_market_fields(obj):
    """Return ``(clean, removed_paths)`` with every market-derived key removed,
    recursing through dicts and lists. ``removed_paths`` is a tuple of dotted
    paths for auditing."""
    removed: list[str] = []

    def _walk(node, prefix=""):
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                path = f"{prefix}.{k}" if prefix else str(k)
                if is_prohibited_key(k):
                    removed.append(path)
                    continue
                out[k] = _walk(v, path)
            return out
        if isinstance(node, list):
            return [_walk(v, prefix) for v in node]
        return node

    clean = _walk(obj)
    return clean, tuple(removed)
