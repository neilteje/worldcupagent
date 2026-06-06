"""
Central configuration. All secrets come from environment variables.
Copy .env.example → .env and fill in your keys.
"""
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ── Arena ──────────────────────────────────────────────────────────────────
# Env var is STAIR_API_KEY (matches the .env file from the arena team)
ARENA_KEY: str = os.environ.get("STAIR_API_KEY") or os.environ.get("ARENA_KEY", "")
ARENA_BASE: str = "https://staging.stair-ai.com"
ARENA_API: str = f"{ARENA_BASE}/api"

# ── Supabase (shared publishable key — same for all builders, no setup needed) ─
SUPABASE_URL: str = "https://ezvbmtvrvzageqixvdak.supabase.co/rest/v1"
SUPABASE_KEY: str = (
    os.environ.get("SUPABASE_KEY")
    or "sb_publishable__m8bOkD05ToFwATpaWST5w_2-3fGS7V"  # shared staging key
)

# ── LLM keys ───────────────────────────────────────────────────────────────
# Accept both naming conventions (ANTHROPIC_KEY and ANTHROPIC_API_KEY)
ANTHROPIC_KEY: str = os.environ.get("ANTHROPIC_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_KEY: str = os.environ.get("OPENAI_KEY") or os.environ.get("OPENAI_API_KEY", "")
GEMINI_KEY: str = os.environ.get("GEMINI_KEY") or os.environ.get("GOOGLE_API_KEY", "")
OPENROUTER_KEY: str = os.environ.get("OPENROUTER_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL: str = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_SITE_URL: str = os.environ.get("OPENROUTER_SITE_URL", "")
OPENROUTER_APP_NAME: str = os.environ.get("OPENROUTER_APP_NAME", "worldcupagent")

# ── Model choices ──────────────────────────────────────────────────────────
ANTHROPIC_MODEL: str = "claude-sonnet-4-5-20250929"  # extended thinking, confirmed valid
OPENROUTER_MODEL: str = os.environ.get("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
PRIMARY_MODEL: str = os.environ.get("PRIMARY_MODEL", OPENROUTER_MODEL if OPENROUTER_KEY else ANTHROPIC_MODEL)
FALLBACK_MODEL: str = "gemini-2.5-pro"            # ensemble calibration
THINKING_BUDGET: int = 4096                        # tokens for internal reasoning (fits within context)

# ── Betting parameters ─────────────────────────────────────────────────────
MAX_KELLY_FRACTION: float = 0.20   # cap any single bet at 20% of wallet
MIN_EDGE: float = 0.05             # only bet when |model_p - market_p| > 5%
MAX_BET_USD: float = 15.00         # hard USD cap per order
DEFAULT_TIF_SECONDS: int = 30      # time-in-force for limit orders

# ── Tournament ─────────────────────────────────────────────────────────────
SEASON_ID: int = 26618             # FIFA WC 2026 on Sportmonks
