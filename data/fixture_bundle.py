"""
Shared fixture context builder — the council's structured grounding inputs.

Both the arena agent and the harness need the same two digests before the
council convenes:

  sportmonks_digest — ML probabilities, bookmaker consensus, xG (needs a
                      Sportmonks fixture id; WC fixtures have one, friendlies
                      sometimes do)
  supabase_digest   — historical priors (H2H, set-piece, stage records). Keyed
                      by team NAME via the StatsBomb country resolver, so this
                      works for any international fixture, friendlies included.

`agent.py` keeps its own inline flow because every fetch/digest there must emit
a linked ledger record; this module serves the harness and any non-ledger
caller so the council is never fed `None, None` when real data exists.

Every step fails soft but logs loudly — a missing source is a visible warning,
not a silent empty dict.
"""
from __future__ import annotations


def _warn(msg: str) -> None:
    print(f"  [context] WARNING: {msg}")


def build_sportmonks_digest(fixture_id: int | None, home_code: str,
                            away_code: str) -> dict | None:
    """Fetch + digest the Sportmonks fixture record. None if no id or no data."""
    if not fixture_id:
        return None
    try:
        from data import sportmonks
        from reasoning import llm
        from reasoning.prompts import sportmonks_digest_input

        fixture = sportmonks.get_fixture(int(fixture_id))
        if not fixture or fixture.get("demo"):
            _warn(f"Sportmonks fixture {fixture_id} returned no data")
            return None
        result = llm.digest_sportmonks(
            sportmonks_digest_input(fixture, home_code, away_code))
        return result.parsed or None
    except Exception as exc:
        _warn(f"Sportmonks digest failed for fixture {fixture_id}: {exc!r}")
        return None


def build_supabase_digest(home_name: str, away_name: str, home_code: str,
                          away_code: str, fixture_name: str | None = None) -> dict | None:
    """
    Fetch + digest StatsBomb priors for both teams, resolved by NAME (works for
    friendlies — the resolver maps e.g. 'Portugal' → its StatsBomb country_id).
    None when neither team resolves or the fetch fails.
    """
    try:
        from data import supabase_client
        from reasoning import llm
        from reasoning.prompts import supabase_digest_input

        home_id = supabase_client.resolve_country_id(home_name)
        away_id = supabase_client.resolve_country_id(away_name)
        if not home_id or not away_id:
            if (not home_id and supabase_client.known_missing_country(home_name)
                    or not away_id and supabase_client.known_missing_country(away_name)):
                _warn(
                    f"Supabase priors coverage missing for "
                    f"{home_name if not home_id else away_name}; continuing without priors"
                )
            else:
                _warn(f"country resolver missed: {home_name}={home_id}, {away_name}={away_id}")
            return None
        priors = supabase_client.get_all_priors(home_id, away_id)
        if not priors:
            _warn(f"no priors rows for {home_name} vs {away_name}")
            return None
        content = supabase_digest_input(
            fixture_name or f"{home_name} vs {away_name}",
            home_code, away_code, home_id, away_id, home_name, away_name, priors)
        result = llm.digest_supabase(content)
        return result.parsed or None
    except Exception as exc:
        _warn(f"Supabase digest failed for {home_name} vs {away_name}: {exc!r}")
        return None


def build_context(home_name: str, away_name: str, home_code: str, away_code: str,
                  *, sportmonks_fixture_id: int | None = None,
                  fixture_name: str | None = None) -> dict:
    """
    Assemble the council's structured grounding inputs.

    Returns {"sportmonks_digest": dict|None, "supabase_digest": dict|None,
             "sources": {"sportmonks": bool, "supabase": bool}}.
    """
    sm = build_sportmonks_digest(sportmonks_fixture_id, home_code, away_code)
    sb = build_supabase_digest(home_name, away_name, home_code, away_code,
                               fixture_name=fixture_name)
    return {
        "sportmonks_digest": sm,
        "supabase_digest": sb,
        "sources": {"sportmonks": sm is not None, "supabase": sb is not None},
    }
