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


def test_kalshi_distinguishes_no_clean_market_from_api_failure(monkeypatch):
    from data import kalshi

    monkeypatch.setattr(
        kalshi,
        "_scan_open_markets_result",
        lambda: ([
            {
                "ticker": "PARLAY",
                "title": "yes Ecuador,yes Sweden,yes Spain,yes Ivory Coast",
                "yes_sub_title": "yes Ecuador,yes Sweden,yes Spain,yes Ivory Coast",
            }
        ], "ok"),
    )

    out = kalshi.get_moneyline("Côte d'Ivoire", "Ecuador")
    assert out["status"] == "no_clean_market"
    assert out["markets_found"] == 0
    assert out["markets_scanned"] == 1


def test_kalshi_extracts_clean_fixture_market(monkeypatch):
    from data import kalshi

    monkeypatch.setattr(
        kalshi,
        "_scan_open_markets_result",
        lambda: ([
            {
                "ticker": "H",
                "title": "Ivory Coast vs Ecuador",
                "subtitle": "Ivory Coast",
                "yes_sub_title": "Ivory Coast",
                "yes_bid": 27,
                "yes_ask": 29,
            },
            {
                "ticker": "D",
                "title": "Ivory Coast vs Ecuador",
                "subtitle": "Draw",
                "yes_sub_title": "Draw",
                "yes_bid": 32,
                "yes_ask": 34,
            },
            {
                "ticker": "A",
                "title": "Ivory Coast vs Ecuador",
                "subtitle": "Ecuador",
                "yes_sub_title": "Ecuador",
                "yes_bid": 38,
                "yes_ask": 40,
            },
        ], "ok"),
    )

    out = kalshi.get_moneyline("Côte d'Ivoire", "Ecuador")
    assert out["status"] == "ok"
    assert out["markets_found"] == 3
    assert out["home"] == 0.28
    assert out["draw"] == 0.33
    assert out["away"] == 0.39
