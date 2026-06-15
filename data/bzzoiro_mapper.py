from dataclasses import dataclass, field
from datetime import datetime, timedelta
import re
import unicodedata


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


TEAM_ALIASES = {
    "cote d'ivoire": "ivory coast",
    "cote d ivoire": "ivory coast",
    "cape verde islands": "cape verde",
    "cape verde island": "cape verde",
    "cabo verde": "cape verde",
    "cabo verde islands": "cape verde",
    "korea republic": "south korea",
    "korea dpr": "north korea",
    "usa": "united states",
    "united states of america": "united states",
}


def canonicalize_team_name(name: str) -> str:
    """Normalize team name for comparison."""
    if not name:
        return ""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9']+", " ", text)
    normalized = " ".join(text.split())
    return TEAM_ALIASES.get(normalized, normalized)


def _names_match(target: str, candidate: str) -> bool:
    if not target or not candidate:
        return False
    if target == candidate or target in candidate or candidate in target:
        return True
    target_tokens = set(target.split())
    candidate_tokens = set(candidate.split())
    if not target_tokens or not candidate_tokens:
        return False
    overlap = len(target_tokens & candidate_tokens)
    return overlap / min(len(target_tokens), len(candidate_tokens)) >= 0.75


def _team_name(value) -> str:
    """Accept v2 plain string teams and older nested team embeds."""
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
    bzzoiro_search_results: list[dict],
) -> EventMapping:
    """
    Map an internal fixture to a BZZOIRO event using team identity plus kickoff.
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
            warnings=("No search results provided",),
        )

    target_home = canonicalize_team_name(home_name)
    target_away = canonicalize_team_name(away_name)

    best_match = None
    best_confidence = 0.0
    best_diff = None
    best_home_match = False
    best_away_match = False

    for event in bzzoiro_search_results:
        if "error" in event:
            continue

        b_home = canonicalize_team_name(_team_name(event.get("home_team")))
        b_away = canonicalize_team_name(_team_name(event.get("away_team")))

        home_match = _names_match(target_home, b_home)
        away_match = _names_match(target_away, b_away)

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
            if diff_seconds <= 86400:
                confidence += 0.3
            if diff_seconds <= 3600:
                confidence += 0.1

        if confidence > best_confidence:
            best_confidence = confidence
            best_match = event
            best_diff = diff_seconds
            best_home_match = home_match
            best_away_match = away_match

    if best_match and best_confidence >= 0.8:
        return EventMapping(
            internal_fixture_id=internal_fixture_id,
            bzzoiro_event_id=str(best_match.get("id")),
            home_match=best_home_match,
            away_match=best_away_match,
            competition_match=True,
            kickoff_difference_seconds=best_diff,
            confidence=best_confidence,
            mapping_method="strict_match",
        )

    return EventMapping(
        internal_fixture_id=internal_fixture_id,
        bzzoiro_event_id=None,
        home_match=best_home_match,
        away_match=best_away_match,
        competition_match=False,
        kickoff_difference_seconds=best_diff,
        confidence=best_confidence,
        mapping_method="failed",
        warnings=("Ambiguous or weak mapping",),
    )


def get_bzzoiro_event_id(home_name: str, away_name: str, match_date: str) -> int | None:
    from data import bzzoiro

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
