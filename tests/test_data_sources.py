from __future__ import annotations


def test_fifa_code_aliases():
    from data.team_codes import fifa_code, normalize_probabilities

    assert fifa_code("DZA") == "ALG"
    assert fifa_code("HTI") == "HAI"
    assert fifa_code("PRY") == "PAR"
    assert fifa_code("ZAF") == "RSA"
    assert fifa_code("draw") == "draw"
    assert normalize_probabilities({"ZAF": 0.4, "draw": 0.3, "MEX": 0.3}) == {
        "RSA": 0.4,
        "draw": 0.3,
        "MEX": 0.3,
    }


def test_supabase_known_missing_cote_divoire_is_coverage_gap(monkeypatch):
    from data import supabase_client

    monkeypatch.setattr(supabase_client, "_COUNTRY_ID_MAP", {"ecuador": 65})
    assert supabase_client.resolve_country_id("Côte d'Ivoire") is None
    assert supabase_client.known_missing_country("Côte d'Ivoire")
    assert supabase_client.resolve_country_id("Ecuador") == 65


def test_supabase_resolves_cape_verde_aliases(monkeypatch):
    from data import supabase_client

    monkeypatch.setattr(supabase_client, "_COUNTRY_ID_MAP", {"cape verde": 44})
    assert supabase_client.resolve_country_id("Cape Verde Islands") == 44
    assert supabase_client.resolve_country_id("Cabo Verde") == 44


def test_supabase_resolves_world_cup_name_aliases(monkeypatch):
    from data import supabase_client

    monkeypatch.setattr(supabase_client, "_COUNTRY_ID_MAP", {
        "usa": 241,
        "turkey": 233,
        "korea south": 121,
    })
    assert supabase_client.resolve_country_id("United States") == 241
    assert supabase_client.resolve_country_id("Türkiye") == 233
    assert supabase_client.resolve_country_id("Korea Republic") == 121


def test_supabase_known_missing_world_cup_coverage_gaps(monkeypatch):
    from data import supabase_client

    monkeypatch.setattr(supabase_client, "_COUNTRY_ID_MAP", {})
    for name in ("Bosnia and Herzegovina", "Congo DR", "Iraq", "Algeria",
                 "Jordan", "Uzbekistan", "Curacao"):
        assert supabase_client.resolve_country_id(name) is None
        assert supabase_client.known_missing_country(name)


def test_reddit_fallback_counts_web_snippets(monkeypatch):
    from data import reddit_sentiment

    monkeypatch.setattr(reddit_sentiment, "search_reddit_threads", lambda home, away: [])
    monkeypatch.setattr(reddit_sentiment, "fetch_top_comments", lambda permalink, limit=20: [])
    monkeypatch.setattr(
        reddit_sentiment,
        "_search_fallback",
        lambda home, away: ["Ivory Coast fan preview", "Ecuador tactical note"],
    )

    bundle = reddit_sentiment.get_sentiment_bundle("Côte d'Ivoire", "Ecuador")
    assert bundle["source"] == "web_search"
    assert bundle["threads_found"] == 0
    assert bundle["comments_found"] == 2
    assert bundle["top_comments"]


