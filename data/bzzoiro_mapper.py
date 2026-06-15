"""
Maps internal entities to BZZOIRO IDs.
"""
from __future__ import annotations
from typing import Optional
from data import bzzoiro
from datetime import datetime, timedelta

def get_bzzoiro_event_id(home_name: str, away_name: str, match_date: str) -> Optional[int]:
    """
    Search BZZOIRO for an event matching the teams and approximate date.
    match_date should be an ISO string, e.g. '2026-06-15' or '2026-06-15T16:00:00'
    """
    try:
        # Create a +/- 2 day window around the match date
        base_date = datetime.fromisoformat(match_date.replace("Z", "+00:00")).date()
        date_from = (base_date - timedelta(days=2)).isoformat()
        date_to = (base_date + timedelta(days=2)).isoformat()
        
        events = bzzoiro.search_events(home_name, away_name, date_from, date_to)
        if not events:
            # Fallback to searching without away team in the API, then filter manually
            events = bzzoiro.search_events(home_name, "", date_from, date_to)
            away_lower = away_name.lower()
            filtered = []
            for ev in events:
                away_team_data = ev.get("away_team", {}) or {}
                if away_lower in away_team_data.get("name", "").lower():
                    filtered.append(ev)
            events = filtered

        if events:
            # If multiple, try to find the one closest to the kickoff time or just take the first
            return events[0].get("id")
    except Exception as exc:
        print(f"  [bzzoiro_mapper] Failed to resolve event ID for {home_name} vs {away_name}: {exc!r}")
    
    return None
