from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv(override=True)


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    arena_key: str = os.getenv("ARENA_KEY") or os.getenv("STAIR_API_KEY", "")
    arena_base: str = os.getenv("ARENA_BASE", "https://staging.stair-ai.com")
    supabase_url: str = os.getenv("SUPABASE_URL", "https://ezvbmtvrvzageqixvdak.supabase.co/rest/v1")
    supabase_publishable_key: str = os.getenv("SUPABASE_PUBLISHABLE_KEY") or os.getenv("SUPABASE_KEY", "sb_publishable__m8bOkD05ToFwATpaWST5w_2-3fGS7V")
    agent_name: str = os.getenv("AGENT_NAME", "worldcupagent")
    dry_run: bool = _bool("DRY_RUN", True)
    max_order_usd: float = min(_float("MAX_ORDER_USD", 4.0), 5.0)
    min_edge_to_bet: float = _float("MIN_EDGE_TO_BET", 0.06)
    default_fixture_code: str = os.getenv("DEFAULT_FIXTURE_CODE", "DEMO-FIXTURE")
    storage_dir: Path = Path(os.getenv("STORAGE_DIR", "storage"))
    anthropic_key: str = os.getenv("ANTHROPIC_KEY") or os.getenv("ANTHROPIC_API_KEY", "")
    backtest_llm_budget_usd: float = min(_float("BACKTEST_LLM_BUDGET_USD", 5.0), 5.0)
    tif_seconds: int = int(_float("TIME_IN_FORCE_SECONDS", 30))

    @property
    def arena_api(self) -> str:
        return f"{self.arena_base}/api"

    @property
    def headers(self) -> dict[str, str]:
        return {"x-api-key": self.arena_key, "Content-Type": "application/json"}


def load_settings(dry_run_override: bool | None = None) -> Settings:
    s = Settings()
    if dry_run_override is None:
        return s
    return Settings(dry_run=dry_run_override)
