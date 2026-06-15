from dataclasses import dataclass, field
from datetime import datetime

@dataclass(frozen=True)
class EventMapping:
    internal_fixture_id: str
    bzzoiro_event_id: str | None

    home_match: bool
    away_match: bool
    competition_match: bool

    kickoff_difference_seconds: int | None
    confidence: float

    mapping_method: str
    warnings: tuple[str, ...] = field(default_factory=tuple)

# Aliases for canonical team names
TEAM_ALIASES = {
    "côte d’ivoire": "ivory coast",
    "cote d'ivoire": "ivory coast",
    "korea republic": "south korea",
    "usa": "united states",
}

def canonicalize_team_name(name: str) -> str:
    """Normalize team name for comparison."""
    if not name:
        return ""
    lower_name = name.strip().lower()
    return TEAM_ALIASES.get(lower_name, lower_name)


def _team_name(value) -> str:
    """The v2 events feed returns `home_team`/`away_team` as plain strings, but
    some embeds nest `{name: ...}`. Accept either shape."""
    if isinstance(value, dict):
        return value.get("name", "") or value.get("short_name", "") or ""
    return value or ""


def _event_kickoff(event: dict) -> str | None:
    return event.get("event_date") or event.get("start_time") or event.get("date")

def map_event(
    internal_fixture_id: str,
    home_name: str,
    away_name: str,
    kickoff: datetime,
    bzzoiro_search_results: list[dict]
) -> EventMapping:
    """
    Map an internal fixture to a BZZOIRO event using strict rules.
    1. Canonical home team identity
    2. Canonical away team identity
    3. Home/away orientation
    4. Kickoff timestamp
    """
    if not bzzoiro_search_results:
        return EventMapping(
            internal_fixture_id=internal_fixture_id,
            bzzoiro_event_id=None,
            home_match=False,
            away_match=False,
            competition_match=False,
            kickoff_difference_seconds=None,
            confidence=0.0,
            mapping_method="none",
            warnings=("No search results provided",)
        )

    target_home = canonicalize_team_name(home_name)
    target_away = canonicalize_team_name(away_name)
    
    best_match = None
    best_confidence = 0.0
    best_diff = None
    
    for event in bzzoiro_search_results:
        # Ignore errors if present in pagination stream
        if "error" in event:
            continue
            
        b_home = canonicalize_team_name(_team_name(event.get("home_team")))
        b_away = canonicalize_team_name(_team_name(event.get("away_team")))

        home_match = bool(b_home) and (target_home in b_home or b_home in target_home)
        away_match = bool(b_away) and (target_away in b_away or b_away in target_away)

        # Calculate kickoff difference
        b_kickoff_str = _event_kickoff(event)
        diff_seconds = None
        if b_kickoff_str:
            try:
                b_kickoff = datetime.fromisoformat(b_kickoff_str.replace("Z", "+00:00"))
                diff_seconds = abs(int((kickoff.replace(tzinfo=None) - b_kickoff.replace(tzinfo=None)).total_seconds()))
            except ValueError:
                pass
                
        confidence = 0.0
        if home_match and away_match:
            confidence += 0.6
        elif home_match or away_match:
            confidence += 0.3
            
        if diff_seconds is not None:
            if diff_seconds <= 86400: # Within 1 day
                confidence += 0.3
            if diff_seconds <= 3600: # Within 1 hour
                confidence += 0.1
                
        if confidence > best_confidence:
            best_confidence = confidence
            best_match = event
            best_diff = diff_seconds
            
    if best_match and best_confidence >= 0.8:
        return EventMapping(
            internal_fixture_id=internal_fixture_id,
            bzzoiro_event_id=str(best_match.get("id")),
            home_match=True,
            away_match=True,
            competition_match=True, # Simplified
            kickoff_difference_seconds=best_diff,
            confidence=best_confidence,
            mapping_method="strict_match",
        )
        
    return EventMapping(
        internal_fixture_id=internal_fixture_id,
        bzzoiro_event_id=None,
        home_match=False,
        away_match=False,
        competition_match=False,
        kickoff_difference_seconds=None,
        confidence=best_confidence,
        mapping_method="failed",
        warnings=("Ambiguous or weak mapping",)
    )

def get_bzzoiro_event_id(home_name: str, away_name: str, match_date: str) -> int | None:
    from data import bzzoiro
    from datetime import timedelta
    
    try:
        dt = datetime.fromisoformat(match_date.replace("Z", "+00:00"))
        date_from = (dt - timedelta(days=2)).strftime("%Y-%m-%d")
        date_to = (dt + timedelta(days=2)).strftime("%Y-%m-%d")
    except ValueError:
        dt = datetime.now()
        date_from = ""
        date_to = ""

    results = bzzoiro.search_events(home_name, away_name, date_from, date_to)
    mapping = map_event("internal", home_name, away_name, dt, results)
    if mapping.bzzoiro_event_id is not None:
        return int(mapping.bzzoiro_event_id)
        
    results_fb = bzzoiro.search_events(home_name, "", date_from, date_to)
    mapping_fb = map_event("internal", home_name, away_name, dt, results_fb)
    if mapping_fb.bzzoiro_event_id is not None:
        return int(mapping_fb.bzzoiro_event_id)
        
    return None
