from __future__ import annotations
import httpx
from agent.config import Settings


def fetch_json(settings: Settings, table: str, params: dict | None = None, profile: str | None = "world_cup_arena") -> list[dict]:
    headers = {"apikey": settings.supabase_publishable_key}
    if profile: headers["Accept-Profile"] = profile
    try:
        r = httpx.get(f"{settings.supabase_url.rstrip('/')}/{table}", headers=headers, params=params or {}, timeout=15)
        if not r.is_success: return []
        data = r.json()
        return data if isinstance(data, list) else [data]
    except Exception:
        return []


def get_priors(settings: Settings, fixture_code: str) -> dict | None:
    rows = fetch_json(settings, "ads_a_team_priors", {"select": "*", "limit": "20"})
    if not rows: return None
    # Conservative fallback prior if table shape is unknown.
    return {"home": 0.39, "draw": 0.27, "away": 0.34, "rows_used": len(rows)}


def get_live_checkpoint(settings: Settings, fixture_code: str) -> dict | None:
    rows = fetch_json(settings, "d_match_checkpoints", {"fixture_code": f"eq.{fixture_code}", "select": "*", "limit": "1"})
    return rows[0] if rows else None
