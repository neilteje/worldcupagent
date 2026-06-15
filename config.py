"""
Central configuration. All secrets come from environment variables.
Copy .env.example → .env and fill in your keys.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Always load the .env next to this config file, regardless of cwd
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default



# ── Arena ──────────────────────────────────────────────────────────────────
# Explicit STAIR_API_KEY / ARENA_KEY, else any AGENT_KEY_* (all work for data
# proxies + arena endpoints — you only need STAIR_API_KEY when running a single
# agent without the per-key roster).
_AGENT_KEY_FALLBACK_ORDER = ("ANCHOR", "MONK", "HUNTER", "BLITZ")


def _resolve_arena_key() -> str:
    explicit = (os.environ.get("STAIR_API_KEY") or os.environ.get("ARENA_KEY") or "").strip()
    if explicit:
        return explicit
    for name in _AGENT_KEY_FALLBACK_ORDER:
        k = (os.environ.get(f"AGENT_KEY_{name}") or "").strip()
        if k:
            return k
    return ""


ARENA_KEY: str = _resolve_arena_key()
ARENA_BASE: str = "https://stair-ai.com"
ARENA_API: str = f"{ARENA_BASE}/api"

# ── Supabase (shared publishable key — same for all builders, no setup needed) ─
SUPABASE_URL: str = "https://ezvbmtvrvzageqixvdak.supabase.co/rest/v1"
SUPABASE_KEY: str = (
    os.environ.get("SUPABASE_KEY")
    or "sb_publishable__m8bOkD05ToFwATpaWST5w_2-3fGS7V"  # shared staging key
)

# ── BZZOIRO Football API ───────────────────────────────────────────────────
BZZOIRO_KEY: str = os.environ.get("BZZOIRO_KEY") or os.environ.get("BZZOIRO_API_KEY", "")
BZZOIRO_BASE: str = "https://sports.bzzoiro.com"
BZZOIRO_API: str = f"{BZZOIRO_BASE}/api"


# ── LLM keys ───────────────────────────────────────────────────────────────
# Accept both naming conventions (ANTHROPIC_KEY and ANTHROPIC_API_KEY)
ANTHROPIC_KEY: str = os.environ.get("ANTHROPIC_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_KEY: str = os.environ.get("OPENAI_KEY") or os.environ.get("OPENAI_API_KEY", "")
GEMINI_KEY: str = os.environ.get("GEMINI_KEY") or os.environ.get("GOOGLE_API_KEY", "")
DEEPSEEK_KEY: str = os.environ.get("DEEPSEEK_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL: str = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
XAI_KEY: str = os.environ.get("XAI_KEY") or os.environ.get("XAI_API_KEY", "") or os.environ.get("GROK_API_KEY", "")
XAI_BASE_URL: str = os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1")
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

# ── Reasoning council model assignments ────────────────────────────────────
# Each role uses a distinct model/provider so the ledger trace shows diversity.
SCOUT_MODEL: str = "claude-haiku-4-5-20251001"     # fast triage, cheap
ANALYST_MODEL: str = ANTHROPIC_MODEL               # deep market-blind reasoning
DEVIL_MODEL: str = "deepseek-reasoner"             # raw chain-of-thought contrarian
JUDGE_MODEL: str = ANTHROPIC_MODEL                 # final calibrated synthesis
GROK_MODEL: str = os.environ.get("GROK_MODEL", "grok-4.3")  # live X/social pulse
SCOUT_THINKING_BUDGET: int = 1024
DEVIL_TIMEOUT_SECONDS: int = 120                   # DeepSeek-R1 can be slow
GROK_TIMEOUT_SECONDS: int = 60

# ── External research keys (optional — graceful degradation if absent) ──────
SERPER_API_KEY: str = os.environ.get("SERPER_API_KEY", "")   # serper.dev Google search
KALSHI_API_KEY: str = os.environ.get("KALSHI_API_KEY", "")   # reads work without it
RESEARCH_TIMEOUT_SECONDS: int = 12

# ── Betting parameters ─────────────────────────────────────────────────────
# WHERE EDGE IS GATED (single source of truth — no double-gating):
#   betting/decision.py is the ONLY edge gate: tradability requires EV > 0 at
#   the raw price AND edge ≥ min_edge_vs_fair against the DE-VIGGED fair price.
#   The per-agent bar comes from the profile (harness/profiles.py); the value
#   below is only the fallback when no profile is supplied.
#   reasoning/gates.py is a risk overlay (wallet floor, market existence,
#   consensus/confidence multipliers, scout veto). Its legacy raw-mid edge bar
#   is OPT-IN via the min_edge= parameter and is NOT used by the agent/harness.
MAX_KELLY_FRACTION: float = 0.20   # cap any single bet at 20% of wallet
MIN_EDGE: float = 0.05             # legacy raw-mid bar (kelly.should_bet + opt-in gate)
MIN_EDGE_VS_FAIR: float = 0.03     # fallback edge-vs-fair bar (profiles override)
MAX_BET_USD: float = 5.00          # hard USD cap per order (arena rule: ≤ $5)
DEFAULT_TIF_SECONDS: int = 30      # time-in-force for limit orders
MIN_WALLET_USD: float = 2.00       # never trade below this balance

# Coordinated MONK/ANCHOR/HUNTER redesign defaults. These are not applied to
# BLITZ and should be tuned by backtesting before being used for live gating.
FEE_BUFFER: float = _env_float("FEE_BUFFER", 0.01)
SLIPPAGE_BUFFER: float = _env_float("SLIPPAGE_BUFFER", 0.01)
MODEL_RISK_BUFFER: float = _env_float("MODEL_RISK_BUFFER", 0.02)

MONK_MIN_CONSERVATIVE_EDGE: float = _env_float("MONK_MIN_CONSERVATIVE_EDGE", 0.08)
MONK_MIN_CONFIDENCE: float = _env_float("MONK_MIN_CONFIDENCE", 0.65)
MONK_KELLY_FRACTION: float = _env_float("MONK_KELLY_FRACTION", 0.10)
MONK_MAX_BANKROLL_FRACTION: float = _env_float("MONK_MAX_BANKROLL_FRACTION", 0.015)
MONK_MAX_BETS_PER_FIXTURE: int = _env_int("MONK_MAX_BETS_PER_FIXTURE", 1)

ANCHOR_MIN_CONSERVATIVE_EDGE: float = _env_float("ANCHOR_MIN_CONSERVATIVE_EDGE", 0.05)
ANCHOR_MIN_EV_AFTER_COSTS: float = _env_float("ANCHOR_MIN_EV_AFTER_COSTS", 0.03)
ANCHOR_MIN_ENTRY_PRICE: float = _env_float("ANCHOR_MIN_ENTRY_PRICE", 0.10)
ANCHOR_MAX_ENTRY_PRICE: float = _env_float("ANCHOR_MAX_ENTRY_PRICE", 0.80)
ANCHOR_KELLY_FRACTION: float = _env_float("ANCHOR_KELLY_FRACTION", 0.20)
ANCHOR_MAX_BANKROLL_FRACTION: float = _env_float("ANCHOR_MAX_BANKROLL_FRACTION", 0.025)
ANCHOR_MAX_BETS_PER_FIXTURE: int = _env_int("ANCHOR_MAX_BETS_PER_FIXTURE", 1)

HUNTER_MIN_ENTRY_PRICE: float = _env_float("HUNTER_MIN_ENTRY_PRICE", 0.05)
HUNTER_MAX_ENTRY_PRICE: float = _env_float("HUNTER_MAX_ENTRY_PRICE", 0.35)
HUNTER_MIN_CONSERVATIVE_EDGE: float = _env_float("HUNTER_MIN_CONSERVATIVE_EDGE", 0.08)
HUNTER_MIN_EV_AFTER_COSTS: float = _env_float("HUNTER_MIN_EV_AFTER_COSTS", 0.12)
HUNTER_MIN_INDEPENDENT_SIGNALS: int = _env_int("HUNTER_MIN_INDEPENDENT_SIGNALS", 2)
HUNTER_KELLY_FRACTION: float = _env_float("HUNTER_KELLY_FRACTION", 0.10)
HUNTER_MAX_BANKROLL_FRACTION: float = _env_float("HUNTER_MAX_BANKROLL_FRACTION", 0.015)
HUNTER_MAX_TAIL_POSITIONS_PER_FIXTURE: int = _env_int("HUNTER_MAX_TAIL_POSITIONS_PER_FIXTURE", 1)
# Sub-HUNTER_MIN_ENTRY_PRICE ("ultra tail") buys require a dedicated validation
# pass that is not built yet, so they are rejected unless explicitly enabled.
HUNTER_ULTRA_TAIL_VALIDATION_ENABLED: bool = (
    os.environ.get("HUNTER_ULTRA_TAIL_VALIDATION_ENABLED", "").strip().lower()
    in ("1", "true", "yes", "on")
)

# Minimum independent-forecast data coverage a coordinated recommendation needs.
MIN_DATA_COVERAGE: float = _env_float("MIN_DATA_COVERAGE", 0.25)

# Portfolio exposure caps as a fraction of bankroll. The allocator REJECTS (does
# not shrink) a coordinated bet that would breach a cap, so these must clear a
# full-size single bet on a small (~$100) arena wallet: ANCHOR maxes at $4 (4%)
# and HUNTER at $5 (5%). Caps below those would silently zero those agents; we
# set fixture 10% / outcome 6% so a full bet passes while stacking is still held.
MAX_FIXTURE_EXPOSURE: float = _env_float("MAX_FIXTURE_EXPOSURE", 0.10)
MAX_OUTCOME_EXPOSURE: float = _env_float("MAX_OUTCOME_EXPOSURE", 0.06)
MAX_ULTRA_TAIL_EXPOSURE: float = _env_float("MAX_ULTRA_TAIL_EXPOSURE", 0.02)
MAX_DAILY_DRAWDOWN: float = _env_float("MAX_DAILY_DRAWDOWN", 0.05)

# ── Cross-market (Polymarket vs Kalshi) gate thresholds ─────────────────────
MARKET_CONSENSUS_SPREAD: float = 0.03   # both markets agree within 3pp → boost
MARKET_CONTESTED_SPREAD: float = 0.08   # markets diverge >8pp → size down
CONSENSUS_MULTIPLIER: float = 1.25      # size-up when markets agree with us
CONTESTED_MULTIPLIER: float = 0.50      # size-down when markets disagree
CONFIDENCE_LOW_MULTIPLIER: float = 0.50
CONFIDENCE_HIGH_MULTIPLIER: float = 1.20

# ── Tournament ─────────────────────────────────────────────────────────────
SEASON_ID: int = 26618             # FIFA WC 2026 on Sportmonks

# ── Feature Flags & Rollout ──────────────────────────────────────────────────
def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name, "").strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return default

FOUR_AGENT_ARCHITECTURE_ENABLED: bool = _env_bool("FOUR_AGENT_ARCHITECTURE_ENABLED", False)
MONK_INDEPENDENT_ENABLED: bool = _env_bool("MONK_INDEPENDENT_ENABLED", False)
ANCHOR_INDEPENDENT_VALUE_ENABLED: bool = _env_bool("ANCHOR_INDEPENDENT_VALUE_ENABLED", False)
HUNTER_SPECIALIZED_MODELS_ENABLED: bool = _env_bool("HUNTER_SPECIALIZED_MODELS_ENABLED", False)
JOINT_PORTFOLIO_ENABLED: bool = _env_bool("JOINT_PORTFOLIO_ENABLED", False)
BZZOIRO_ENABLED: bool = _env_bool("BZZOIRO_ENABLED", False)
BZZOIRO_MODEL_SHADOW_ONLY: bool = _env_bool("BZZOIRO_MODEL_SHADOW_ONLY", True)
BZZOIRO_MODEL_WEIGHT: float = _env_float("BZZOIRO_MODEL_WEIGHT", 0.0)
BZZOIRO_TIMEOUT_SECONDS: int = _env_int("BZZOIRO_TIMEOUT_SECONDS", 10)
BZZOIRO_MAX_RETRIES: int = _env_int("BZZOIRO_MAX_RETRIES", 3)
BZZOIRO_CACHE_DIR: str = os.environ.get("BZZOIRO_CACHE_DIR", "storage/bzzoiro_cache")
BZZOIRO_HISTORICAL_CACHE_TTL: int = _env_int("BZZOIRO_HISTORICAL_CACHE_TTL", 86400 * 7)
BZZOIRO_UPCOMING_CACHE_TTL: int = _env_int("BZZOIRO_UPCOMING_CACHE_TTL", 3600)
BZZOIRO_LIVE_CACHE_TTL: int = _env_int("BZZOIRO_LIVE_CACHE_TTL", 60)
DETERMINISTIC_HT_ENABLED: bool = _env_bool("DETERMINISTIC_HT_ENABLED", False)
AGENT_SPECIFIC_LEDGER_ENABLED: bool = _env_bool("AGENT_SPECIFIC_LEDGER_ENABLED", False)
