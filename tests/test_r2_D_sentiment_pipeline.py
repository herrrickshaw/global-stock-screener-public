"""Offline tests for sentiment_pipeline.py: fake VADER/feedparser/requests, cache in tmp_path."""

import importlib
import json
import sys
import types
from datetime import datetime, timedelta

import pytest


class FakeAnalyzer:
    def __init__(self):
        self.lexicon = {}

    def polarity_scores(self, text):
        t = text.lower()
        c = 0.0
        if "good" in t:
            c += 0.6
        if "bad" in t:
            c -= 0.6
        return {"compound": c}


class FakeResp:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


def _load(tmp_path, monkeypatch, vader=True, feedparser=None, requests_ok=True):
    """Import a fresh sentiment_pipeline with controlled optional deps; restore afterwards."""
    monkeypatch.chdir(tmp_path)
    for name in (
        "sentiment_pipeline",
        "vaderSentiment",
        "vaderSentiment.vaderSentiment",
        "feedparser",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)
    if vader:
        pkg = types.ModuleType("vaderSentiment")
        sub = types.ModuleType("vaderSentiment.vaderSentiment")
        sub.SentimentIntensityAnalyzer = FakeAnalyzer
        pkg.vaderSentiment = sub
        monkeypatch.setitem(sys.modules, "vaderSentiment", pkg)
        monkeypatch.setitem(sys.modules, "vaderSentiment.vaderSentiment", sub)
    else:
        monkeypatch.setitem(sys.modules, "vaderSentiment", None)  # forces ImportError
    if feedparser is not None:
        monkeypatch.setitem(sys.modules, "feedparser", feedparser)
    else:
        monkeypatch.setitem(sys.modules, "feedparser", None)
    if not requests_ok:
        monkeypatch.setitem(sys.modules, "requests", None)
    mod = importlib.import_module("sentiment_pipeline")
    monkeypatch.setattr(mod, "CACHE_FILE", tmp_path / "cache.json")
    monkeypatch.setattr(mod, "OUT_DIR", tmp_path / "out")
    (tmp_path / "out").mkdir(exist_ok=True)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    for k in ("MARKETAUX_KEY", "ALPHAVANTAGE_KEY", "FINNHUB_KEY", "NEWSDATA_KEY"):
        monkeypatch.delenv(k, raising=False)
    return mod


def _fp(feeds):
    """feedparser stub: feeds maps url -> list of entry dicts (or Exception)."""
    m = types.ModuleType("feedparser")

    def parse(url):
        v = feeds.get(url, [])
        if isinstance(v, Exception):
            raise v
        return types.SimpleNamespace(entries=v)

    m.parse = parse
    m.calls = []
    return m


@pytest.fixture
def sp(tmp_path, monkeypatch):
    return _load(tmp_path, monkeypatch)


def test_score_text_and_lexicon(sp):
    assert sp.score_text("") == 0.0
    assert sp.score_text(None) == 0.0
    assert sp.score_text("a good day") == 0.6
    assert sp._VADER.lexicon["beat"] == 2.5 and sp._VADER.lexicon["fraud"] == -3.0


def test_score_text_without_vader(tmp_path, monkeypatch):
    m = _load(tmp_path, monkeypatch, vader=False)
    assert m._VADER_OK is False and m.score_text("good") == 0.0


@pytest.mark.parametrize(
    "s,exp",
    [
        (0.15, "POSITIVE"),
        (0.1499, "NEUTRAL"),
        (0, "NEUTRAL"),
        (-0.1499, "NEUTRAL"),
        (-0.15, "NEGATIVE"),
        (1, "POSITIVE"),
    ],
)
def test_label_boundaries(sp, s, exp):
    assert sp.label_of(s) == exp


def test_available_requires_key_and_requests(sp, monkeypatch):
    p = sp.FinnhubProvider()
    assert p.available is False
    monkeypatch.setenv("FINNHUB_KEY", "k")
    assert sp.FinnhubProvider().available is True
    monkeypatch.setattr(sp, "_REQ_OK", False)
    assert sp.FinnhubProvider().available is False


def test_requests_missing(tmp_path, monkeypatch):
    m = _load(tmp_path, monkeypatch, requests_ok=False)
    assert m._REQ_OK is False


def test_throttle_sleeps_only_when_needed(sp, monkeypatch):
    slept = []
    monkeypatch.setattr(sp.time, "sleep", lambda s: slept.append(s))
    p = sp.MarketauxProvider()
    now = [1000.0]
    monkeypatch.setattr(sp.time, "time", lambda: now[0])
    p._last_call = 999.5
    p._throttle()
    assert slept == [pytest.approx(0.5)] and p._last_call == 1000.0
    now[0] = 1002.0
    p._throttle()
    assert len(slept) == 1  # elapsed 2.0 >= 1.0
    now[0] = 1003.0  # elapsed exactly == min_interval -> no sleep
    p._throttle()
    assert len(slept) == 1


def _patch_get(sp, monkeypatch, payload=None, exc=None):
    calls = []

    def get(url, params=None, timeout=None):
        calls.append((url, params, timeout))
        if exc:
            raise exc
        return FakeResp(payload)

    monkeypatch.setattr(sp.requests, "get", get)
    return calls


def test_marketaux(sp, monkeypatch):
    monkeypatch.setenv("MARKETAUX_KEY", "k")
    p = sp.MarketauxProvider()
    payload = {
        "data": [
            {
                "title": "T1",
                "published_at": "d",
                "url": "u",
                "description": "s",
                "entities": [{"symbol": "X.NS"}, {"symbol": "TCS.NS", "sentiment_score": 0.4}],
            },
            {"title": "good news", "entities": [{"symbol": "OTHER", "sentiment_score": 0.9}]},
            {"title": "T3", "entities": [{"symbol": "TCS", "sentiment_score": 0.0}]},
        ]
    }
    calls = _patch_get(sp, monkeypatch, payload)
    out = p.fetch_news("TCS", "IN")
    assert calls[0][1]["symbols"] == "TCS.NS" and calls[0][1]["limit"] == 10 and calls[0][2] == 15
    assert [a.sentiment for a in out] == [0.4, 0.6, 0.0]  # 0.0 is a real score, not "missing"
    assert out[0].source == "Marketaux" and out[0].url == "u" and out[0].summary == "s"
    p.fetch_news("AAPL", "US")
    assert calls[1][1]["symbols"] == "AAPL"


def test_marketaux_unavailable_and_error(sp, monkeypatch):
    assert sp.MarketauxProvider().fetch_news("A") == []
    monkeypatch.setenv("MARKETAUX_KEY", "k")
    _patch_get(sp, monkeypatch, exc=RuntimeError("net"))
    assert sp.MarketauxProvider().fetch_news("A") == []


def test_alphavantage(sp, monkeypatch):
    monkeypatch.setenv("ALPHAVANTAGE_KEY", "k")
    p = sp.AlphaVantageProvider()
    payload = {
        "feed": [
            {
                "title": "a",
                "time_published": "t",
                "ticker_sentiment": [
                    {"ticker": "FOO", "ticker_sentiment_score": "0.3"},
                    {"ticker": "TCS.BSE", "ticker_sentiment_score": "-0.2"},
                ],
            },
            {"title": "b", "ticker_sentiment": [], "overall_sentiment_score": "0.5"},
            {"title": "c", "ticker_sentiment": []},
        ]
    }
    calls = _patch_get(sp, monkeypatch, payload)
    out = p.fetch_news("TCS", "IN")
    assert calls[0][1]["tickers"] == "TCS.BSE" and calls[0][1]["function"] == "NEWS_SENTIMENT"
    assert [a.sentiment for a in out] == [-0.2, 0.5, 0.0]
    p.fetch_news("AAPL", "US")
    assert calls[1][1]["tickers"] == "AAPL"
    assert sp.AlphaVantageProvider.min_interval == 13.0


def test_alphavantage_unavailable_error(sp, monkeypatch):
    assert sp.AlphaVantageProvider().fetch_news("A") == []
    monkeypatch.setenv("ALPHAVANTAGE_KEY", "k")
    _patch_get(sp, monkeypatch, exc=RuntimeError)
    assert sp.AlphaVantageProvider().fetch_news("A") == []


def test_finnhub(sp, monkeypatch):
    monkeypatch.setenv("FINNHUB_KEY", "k")
    p = sp.FinnhubProvider()
    data = [
        {"headline": f"good {i}", "summary": "s", "datetime": 17, "url": "u"} for i in range(20)
    ]
    calls = _patch_get(sp, monkeypatch, data)
    out = p.fetch_news("AAPL", "US")
    assert len(out) == 15 and out[0].sentiment == 0.6 and out[0].published == "17"
    prm = calls[0][1]
    assert prm["symbol"] == "AAPL" and prm["token"] == "k"
    frm = datetime.strptime(prm["from"], "%Y-%m-%d")
    to = datetime.strptime(prm["to"], "%Y-%m-%d")
    assert (to - frm) == timedelta(days=7)
    p.fetch_news("TCS", "IN")
    assert calls[1][1]["symbol"] == "TCS.NS"
    _patch_get(sp, monkeypatch, {"error": "limit"})  # non-list payload
    assert p.fetch_news("AAPL", "US") == []
    _patch_get(sp, monkeypatch, exc=RuntimeError)
    assert p.fetch_news("AAPL", "US") == []


def test_finnhub_unavailable(sp):
    assert sp.FinnhubProvider().fetch_news("A") == []


def test_newsdata(sp, monkeypatch):
    monkeypatch.setenv("NEWSDATA_KEY", "k")
    p = sp.NewsDataProvider()
    res = [{"title": "bad", "description": None, "pubDate": "p", "link": "l"}] + [
        {"title": "good", "description": "d"}
    ] * 20
    calls = _patch_get(sp, monkeypatch, {"results": res})
    out = p.fetch_news("TCS", "IN")
    assert (
        len(out) == 15 and out[0].sentiment == -0.6 and out[0].summary == "" and out[0].url == "l"
    )
    assert calls[0][1]["country"] == "in" and calls[0][1]["q"] == "TCS"
    p.fetch_news("AAPL", "US")
    assert calls[1][1]["country"] == "us"
    _patch_get(sp, monkeypatch, exc=RuntimeError)
    assert p.fetch_news("A") == []


def test_newsdata_unavailable(tmp_path, monkeypatch):
    m = _load(tmp_path, monkeypatch)
    assert m.NewsDataProvider().fetch_news("A") == []


# ---------------- RSS providers ----------------

IN_URL = "https://www.moneycontrol.com/rss/business.xml"
ET_URL = "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"


def _rss(tmp_path, monkeypatch, feeds):
    return _load(tmp_path, monkeypatch, feedparser=_fp(feeds))


def test_rss_available_flag(tmp_path, monkeypatch):
    m = _load(tmp_path, monkeypatch)
    assert m.IndianRSSProvider().available is False
    assert m.IndianRSSProvider().fetch_news("RELIANCE") == []
    assert m.IndianRSSProvider().fetch_market_mood() == {}
    m2 = _rss(tmp_path, monkeypatch, {})
    assert m2.IndianRSSProvider().available is True


def test_all_entries_cached_and_error_isolated(tmp_path, monkeypatch):
    feeds = {
        IN_URL: [
            {
                "title": "Reliance Industries profit good",
                "summary": "x",
                "link": "l",
                "published": "p",
            },
            {"title": "no summary", "description": "desc"},
        ],
        ET_URL: RuntimeError("down"),
    }
    m = _rss(tmp_path, monkeypatch, feeds)
    p = m.IndianRSSProvider()
    ents = p._all_entries()
    assert len(ents) == 2 and ents[1]["summary"] == "desc" and ents[0]["outlet"] == "Moneycontrol"
    assert p._feed_cache[ET_URL] == []
    # second call served from cache: mutate the stub so a re-parse would differ
    sys.modules["feedparser"].parse = lambda url: (_ for _ in ()).throw(AssertionError("reparsed"))
    assert len(p._all_entries()) == 2


def test_rss_fetch_news_by_company_name(tmp_path, monkeypatch):
    feeds = {
        IN_URL: [
            {
                "title": "Adani Enterprises Q4 good",
                "summary": "s" * 300,
                "link": "L",
                "published": "P",
            },
            {"title": "Adani Ports rises", "summary": ""},
            {"title": "Reliance bad", "summary": ""},
        ]
    }
    m = _rss(tmp_path, monkeypatch, feeds)
    p = m.IndianRSSProvider()
    out = p.fetch_news("ADANIENT", "IN", company_name="Adani Enterprises Ltd")
    assert len(out) == 1
    a = out[0]
    assert (
        a.source == "IndianRSS:Moneycontrol"
        and a.sentiment == 0.6
        and len(a.summary) == 200
        and a.url == "L"
    )
    # single-token name; short tokens (<3) dropped
    out = p.fetch_news("REL", "IN", company_name="Reliance")
    assert [x.title for x in out] == ["Reliance bad"]
    out = p.fetch_news("X", "IN", company_name="AB Reliance")  # "AB" dropped -> single token
    assert len(out) == 1
    assert p.fetch_news("ADANIENT", "US", company_name="Adani Enterprises") == []  # wrong market


def test_rss_name_lookup_and_fallbacks(tmp_path, monkeypatch):
    feeds = {
        IN_URL: [
            {"title": "INFY posts growth", "summary": ""},
            {"title": "TCS posts good growth", "summary": ""},
            {"title": "GOLD rally", "summary": ""},
        ]
    }
    m = _rss(tmp_path, monkeypatch, feeds)
    p = m.IndianRSSProvider()
    sm = types.ModuleType("symbol_master")
    names = {"INFY": "Infosys Limited"}
    sm.clean_name_for = lambda t: names.get(t, "")
    monkeypatch.setitem(sys.modules, "symbol_master", sm)
    # name found but absent in feeds -> nothing (no ticker fallback when a name pattern exists)
    assert p.fetch_news("INFY", "IN") == []
    # no name; ticker length 4 not common -> word boundary fallback
    sm.clean_name_for = lambda t: ""
    assert [a.title for a in p.fetch_news("INFY", "IN")] == ["INFY posts growth"]
    assert p.fetch_news("TCS", "IN") == []  # len < 4 skipped
    assert p.fetch_news("GOLD", "IN") == []  # common-word ticker skipped

    # lookup raising -> treated as no name
    def boom(t):
        raise RuntimeError

    sm.clean_name_for = boom
    assert len(p.fetch_news("INFY", "IN")) == 1
    # name made only of short tokens -> pattern None -> ticker fallback
    sm.clean_name_for = lambda t: "AB CD"
    assert len(p.fetch_news("INFY", "IN")) == 1


def test_rss_fetch_news_caps_at_15(tmp_path, monkeypatch):
    feeds = {IN_URL: [{"title": f"Acme Corp news {i}", "summary": ""} for i in range(20)]}
    m = _rss(tmp_path, monkeypatch, feeds)
    assert len(m.IndianRSSProvider().fetch_news("ACME", "IN", company_name="Acme Corp")) == 15


def test_rss_market_mood(tmp_path, monkeypatch):
    m = _rss(tmp_path, monkeypatch, {})
    assert m.IndianRSSProvider().fetch_market_mood() == {}  # no entries
    m = _rss(
        tmp_path,
        monkeypatch,
        {IN_URL: [{"title": "neutral", "summary": ""}, {"title": "meh", "summary": ""}]},
    )
    r = m.IndianRSSProvider().fetch_market_mood()
    assert r == {"mood": "NEUTRAL", "score": 0.0, "n": 2}
    m = _rss(
        tmp_path,
        monkeypatch,
        {
            IN_URL: [
                {"title": "good", "summary": ""},
                {"title": "good bad good", "summary": ""},
                {"title": "other", "summary": ""},
            ]
        },
    )
    r = m.IndianRSSProvider().fetch_market_mood()
    assert r["mood"] == "POSITIVE" and r["n_articles"] == 3 and r["score"] == 0.6
    assert r["by_outlet"]["Moneycontrol"] == 3 and r["by_outlet"]["LiveMint"] == 1


def test_rss_mood_scores_exclude_zero(tmp_path, monkeypatch):
    m = _rss(
        tmp_path,
        monkeypatch,
        {IN_URL: [{"title": "good", "summary": ""}, {"title": "nothing", "summary": ""}]},
    )
    r = m.IndianRSSProvider().fetch_market_mood()
    assert (
        r["score"] == 0.6 and r["mood"] == "POSITIVE" and r["n_articles"] == 2
    )  # zero score ignored in avg
    m = _rss(tmp_path, monkeypatch, {IN_URL: [{"title": "bad", "summary": ""}]})
    assert m.IndianRSSProvider().fetch_market_mood()["mood"] == "NEGATIVE"


def test_us_and_eu_providers(tmp_path, monkeypatch):
    us_url = "https://feeds.content.dowjones.io/public/rss/mw_topstories"
    eu_url = "https://www.cityam.com/feed/"
    feeds = {
        us_url: [{"title": "Apple Inc good", "summary": ""}],
        eu_url: [{"title": "Adidas AG good", "summary": ""}],
    }
    m = _rss(tmp_path, monkeypatch, feeds)
    us = m.USRSSProvider()
    out = us.fetch_news("AAPL", "US", company_name="Apple Inc")
    assert len(out) == 1 and out[0].source == "USRSS:MarketWatch"
    assert us.fetch_news("AAPL", "IN", company_name="Apple Inc") == []
    mood = us.fetch_market_mood()
    assert mood["market"] == "US" and mood["n_articles"] == 1
    eu = m.EuropeanRSSProvider()
    out = eu.fetch_news("ADS", "EU", company_name="Adidas AG")
    assert len(out) == 1 and out[0].source == "EuropeanRSS:CityAM"
    assert eu.fetch_news("ADS", "IN", company_name="Adidas AG") == []
    assert eu.fetch_market_mood()["market"] == "EU"
    # empty feeds -> mood stays {} (no market key added)
    m2 = _rss(tmp_path, monkeypatch, {})
    assert m2.USRSSProvider().fetch_market_mood() == {}
    assert m2.EuropeanRSSProvider().fetch_market_mood() == {}


# ---------------- pipeline ----------------


class StubProvider:
    def __init__(self, name, arts, record=None):
        self.name = name
        self.arts = arts
        self.record = record if record is not None else []

    def fetch_news(self, ticker, market="IN", **kw):
        self.record.append((ticker, market, kw))
        return list(self.arts)


def _pipe(m, monkeypatch, providers):
    p = m.SentimentPipeline()
    p.active = providers
    return p


def test_pipeline_init_and_status(tmp_path, monkeypatch):
    m = _rss(tmp_path, monkeypatch, {})
    p = m.SentimentPipeline()
    assert [type(x).__name__ for x in p.active] == [
        "IndianRSSProvider",
        "USRSSProvider",
        "EuropeanRSSProvider",
    ]
    assert p.status() == "Active providers: IndianRSS, USRSS, EuropeanRSS"
    assert (
        p._rss_in.name == "IndianRSS"
        and p._rss_us.name == "USRSS"
        and p._rss_eu.name == "EuropeanRSS"
    )
    monkeypatch.setenv("FINNHUB_KEY", "k")
    assert "Finnhub" in m.SentimentPipeline().status()
    m2 = _load(tmp_path, monkeypatch)
    p2 = m2.SentimentPipeline()
    assert p2.active == [] and p2._rss_in is None
    assert p2.status().startswith("No providers active")
    assert p2.get_market_mood("IN") == {} and p2.get_market_mood("US") == {}


def test_market_mood_routing(tmp_path, monkeypatch):
    m = _rss(tmp_path, monkeypatch, {})
    p = m.SentimentPipeline()
    p._rss_in = types.SimpleNamespace(fetch_market_mood=lambda: "IN")
    p._rss_us = types.SimpleNamespace(fetch_market_mood=lambda: "US")
    p._rss_eu = types.SimpleNamespace(fetch_market_mood=lambda: "EU")
    assert p.get_market_mood() == "IN" and p.get_market_mood("US") == "US"
    assert p.get_market_mood("EU") == "EU" and p.get_market_mood("JP") == "IN"


def test_cache_load_and_save(tmp_path, monkeypatch):
    m = _load(tmp_path, monkeypatch)
    assert m.SentimentPipeline()._cache == {}
    (tmp_path / "cache.json").write_text("{bad")
    assert m.SentimentPipeline()._cache == {}
    (tmp_path / "cache.json").write_text('{"k": {"a": 1}}')
    p = m.SentimentPipeline()
    assert p._cache == {"k": {"a": 1}}
    p._cache["z"] = {"when": datetime(2020, 1, 1)}
    p._save_cache()
    assert json.loads((tmp_path / "cache.json").read_text())["z"]["when"].startswith("2020-01-01")


def test_is_fresh(tmp_path, monkeypatch):
    m = _load(tmp_path, monkeypatch)
    p = m.SentimentPipeline()
    assert p._is_fresh("nope") is False
    now = datetime.now()
    p._cache["a"] = {"fetched_at": (now - timedelta(hours=5, minutes=59)).isoformat()}
    p._cache["b"] = {"fetched_at": (now - timedelta(hours=6, minutes=1)).isoformat()}
    p._cache["c"] = {"fetched_at": "garbage"}
    p._cache["d"] = {}
    assert p._is_fresh("a") is True and p._is_fresh("b") is False
    assert p._is_fresh("c") is False and p._is_fresh("d") is False


def test_get_ticker_sentiment_aggregates(tmp_path, monkeypatch):
    m = _load(tmp_path, monkeypatch)
    A = m.Article
    rec1, rec2 = [], []
    p1 = StubProvider(
        "Marketaux", [A("t" * 120, "Marketaux", "", 2.0), A("b", "Marketaux", "", -0.4)], rec1
    )  # clamped to 1.0
    p2 = StubProvider("NewsData", [A("c", "NewsData", "", 0.1)], rec2)
    p3 = StubProvider("Weird", [], [])
    p = _pipe(m, monkeypatch, [p1, p2, p3])
    ts = p.get_ticker_sentiment("TCS", "US")
    # weights: Marketaux 1.2 (x2), NewsData 0.8: (1.0*1.2 + -0.4*1.2 + 0.1*0.8)/(1.2+1.2+0.8)
    assert ts.score == round((1.2 - 0.48 + 0.08) / 3.2, 4)
    assert (
        ts.label == "POSITIVE" and ts.n_articles == 3 and ts.providers == ["Marketaux", "NewsData"]
    )
    assert len(ts.top_headlines[0]["title"]) == 90 and ts.top_headlines[0]["sentiment"] == 1.0
    assert [h["sentiment"] for h in ts.top_headlines] == [1.0, -0.4, 0.1]  # by |sentiment|
    assert rec1 == [("TCS", "US", {})]
    # cached now
    assert "US:TCS" in json.loads((tmp_path / "cache.json").read_text())
    ts2 = p.get_ticker_sentiment("TCS", "US")
    assert ts2.score == ts.score and len(rec1) == 1  # no refetch


def test_get_ticker_sentiment_company_name_routing(tmp_path, monkeypatch):
    m = _rss(tmp_path, monkeypatch, {})
    seen = {}

    class R(m.IndianRSSProvider):
        def fetch_news(self, ticker, market="IN", company_name=""):
            seen["r"] = company_name
            return []

    api = StubProvider("Finnhub", [])
    p = _pipe(m, monkeypatch, [R(), api])
    ts = p.get_ticker_sentiment("X", "IN", company_name="Foo Inc")
    assert seen["r"] == "Foo Inc" and api.record == [("X", "IN", {})]
    assert ts.label == "NO_DATA" and ts.score == 0.0 and ts.n_articles == 0 and ts.providers == []
    assert "IN:X" not in p._cache  # NO_DATA not cached
    p.get_ticker_sentiment("X", "IN")
    assert seen["r"] == ""


def test_negative_label_and_unknown_weight(tmp_path, monkeypatch):
    m = _load(tmp_path, monkeypatch)
    p = _pipe(m, monkeypatch, [StubProvider("Mystery", [m.Article("a", "M", "", -0.5)])])
    ts = p.get_ticker_sentiment("Z")
    assert ts.score == -0.5 and ts.label == "NEGATIVE"  # weight defaults to 1.0


def test_fresh_cache_returns_stored(tmp_path, monkeypatch):
    m = _load(tmp_path, monkeypatch)
    p = _pipe(m, monkeypatch, [])
    p._cache["IN:Q"] = {
        "score": 0.3,
        "label": "POSITIVE",
        "n_articles": 2,
        "providers": ["A"],
        "top_headlines": [],
        "fetched_at": datetime.now().isoformat(),
    }
    ts = p.get_ticker_sentiment("Q")
    assert (ts.ticker, ts.score, ts.n_articles, ts.providers) == ("Q", 0.3, 2, ["A"])


def test_get_batch_progress(tmp_path, monkeypatch, capsys):
    m = _load(tmp_path, monkeypatch)
    p = _pipe(m, monkeypatch, [])
    scores = {"a": ("POSITIVE"), "b": "NEGATIVE", "c": "NEUTRAL"}
    p.get_ticker_sentiment = lambda t, mk: m.TickerSentiment(t, 0, scores.get(t, "POSITIVE"), 1)
    res = p.get_batch(["a", "b", "c"], "IN")
    assert list(res) == ["a", "b", "c"]
    out = capsys.readouterr().out
    assert out.strip() == "3/3 | POSITIVE:1 NEGATIVE:1"
    names = [f"t{i}" for i in range(25)]
    p.get_batch(names)
    lines = capsys.readouterr().out.strip().splitlines()
    assert [l.split("|")[0].strip() for l in lines] == ["10/25", "20/25", "25/25"]


# ---------------- CLI ----------------


def test_main_demo_mode(tmp_path, monkeypatch, capsys):
    m = _load(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["sp", "--tickers", "A"])
    m.main()
    out = capsys.readouterr().out
    assert "VADER scoring: available" in out and "Demo mode" in out and "No providers active" in out
    assert "RELIANCE" in out and "Set the API keys" in out
    m2 = _load(tmp_path, monkeypatch, vader=False)
    m2.main()
    assert "MISSING" in capsys.readouterr().out


def test_main_live(tmp_path, monkeypatch, capsys):
    m = _load(tmp_path, monkeypatch)
    ts = m.TickerSentiment(
        "AAA",
        0.5,
        "POSITIVE",
        2,
        ["P1"],
        [
            {"sentiment": 0.9, "source": "S", "title": "hello"},
            {"sentiment": -0.1, "source": "S2", "title": "w"},
            {"sentiment": 0.0, "source": "S3", "title": "third"},
        ],
        "t",
    )

    class FakeSP:
        active = [1]

        def status(self):
            return "ok"

        def get_batch(self, tickers, market):
            assert (tickers, market) == (["AAA"], "US")
            return {"AAA": ts}

    monkeypatch.setattr(m, "SentimentPipeline", FakeSP)
    monkeypatch.setattr(sys, "argv", ["sp", "--tickers", "AAA", "--market", "US"])
    m.main()
    out = capsys.readouterr().out
    assert (
        "+0.500" in out
        and "[+0.90 S] hello" in out
        and "[-0.10 S2] w" in out
        and "third" not in out
    )
    files = list((tmp_path / "out").glob("sentiment_US_*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text())["AAA"]["label"] == "POSITIVE"


def test_main_guard(tmp_path, monkeypatch, capsys):
    import runpy

    m = _load(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["sp", "--tickers", "A"])
    runpy.run_path(m.__file__, run_name="__main__")
    assert "NEWS SENTIMENT" in capsys.readouterr().out
