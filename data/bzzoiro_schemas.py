from typing import Any

def validate_bzzoiro_event(payload: dict) -> list[str]:
    """Validate a BZZOIRO event response, returning a list of validation errors."""
    errors = []
    if not isinstance(payload, dict):
        return ["Payload is not a dictionary"]
        
    if "id" not in payload:
        errors.append("Missing 'id'")
        
    if "home_team" not in payload or not isinstance(payload.get("home_team"), dict):
        errors.append("Missing or invalid 'home_team'")
        
    if "away_team" not in payload or not isinstance(payload.get("away_team"), dict):
        errors.append("Missing or invalid 'away_team'")
        
    if "start_time" not in payload:
        errors.append("Missing 'start_time'")
        
    return errors

def validate_bzzoiro_stats(payload: dict) -> list[str]:
    """Validate a BZZOIRO event stats response."""
    errors = []
    if not isinstance(payload, dict):
        return ["Payload is not a dictionary"]
        
    if "teams" not in payload or not isinstance(payload.get("teams"), dict):
        errors.append("Missing or invalid 'teams' object")
    else:
        teams = payload["teams"]
        if "home" not in teams:
            errors.append("Missing 'teams.home'")
        if "away" not in teams:
            errors.append("Missing 'teams.away'")
            
    return errors

def validate_bzzoiro_lineups(payload: dict) -> list[str]:
    """Validate a BZZOIRO event lineups response."""
    errors = []
    if not isinstance(payload, dict):
        return ["Payload is not a dictionary"]
        
    # Lineups format depends on BZZOIRO's schema
    return errors
