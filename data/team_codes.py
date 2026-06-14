"""Team-code normalization for arena order payloads and forecast keys."""
from __future__ import annotations


# Arena now validates these teams by FIFA three-letter codes, not ISO 3166.
_FIFA_CODE_ALIASES = {
    "DZA": "ALG",  # Algeria
    "HTI": "HAI",  # Haiti
    "PRY": "PAR",  # Paraguay
    "ZAF": "RSA",  # South Africa
}


def fifa_code(code: str | None, default: str = "") -> str:
    if not code:
        return default
    raw = str(code).strip()
    if raw.lower() == "draw":
        return "draw"
    upper = raw.upper()
    return _FIFA_CODE_ALIASES.get(upper, upper)


def normalize_probabilities(probs: dict | None) -> dict:
    out = {}
    for key, value in (probs or {}).items():
        out[fifa_code(key)] = value
    return out
