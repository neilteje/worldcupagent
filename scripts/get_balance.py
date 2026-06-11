"""Simple script to fetch and display the Arena API account wallet balance.

Usage:
    python scripts/get_balance.py

The script relies on the same environment configuration as the rest of the
project (see ``worldcupagent/config.py``). It loads the ``ARENA_KEY`` and
``ARENA_API`` values from the ``.env`` file (or environment variables) and
issues a GET request to ``/v1/arena/agents/me``. The JSON response contains a
``wallet_balance_usd`` field which is printed to stdout.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Load environment variables from the project's ``.env`` file located in the
# ``worldcupagent`` package directory. This mirrors the behaviour of
# ``worldcupagent/config.py`` without importing that module.
env_path = Path(__file__).resolve().parents[1] / "worldcupagent" / ".env"
load_dotenv(dotenv_path=env_path, override=True)

def fetch_balance() -> float:
    """Return the current wallet balance in USD.

    Raises ``httpx.HTTPStatusError`` if the request fails.
    """
    # Retrieve required configuration directly from environment variables.
    arena_key = os.getenv("STAIR_API_KEY") or os.getenv("ARENA_KEY", "")
    arena_base = os.getenv("ARENA_BASE", "https://stair-ai.com")
    arena_api = f"{arena_base}/api"

    headers = {"x-api-key": arena_key, "Content-Type": "application/json"}
    url = f"{arena_api}/v1/arena/agents/me"
    resp = httpx.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    return float(resp.json().get("wallet_balance_usd") or 0)


def main() -> None:
    try:
        balance = fetch_balance()
        print(f"Current Arena wallet balance: ${balance:.2f}")
    except Exception as exc:  # pragma: no cover – simple error reporting
        print(f"Failed to fetch balance: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
