import sys
import types

import pytest

import trendlyne_auth as tl

FAKE_EMAIL = "user@example.invalid"
FAKE_PW = "not-a-real-password"  # placeholder for tests only


class Resp:
    def __init__(self, url="https://x/home/", text="", js=None, exc=None):
        self.url, self.text, self._js, self._exc = url, text, js, exc

    def json(self):
        if self._exc:
            raise self._exc
        return self._js


class FakeSession:
    def __init__(self, routes=None, post_url="https://trendlyne.com/home/"):
        self.routes = routes or {}
        self.cookies = {"csrftoken": "TOK"}
        self.calls = []
        self.post_url = post_url

    def get(self, url, headers=None, timeout=None):
        self.calls.append(("get", url, headers, timeout))
        for frag, r in self.routes.items():
            if frag in url:
                if isinstance(r, Exception):
                    raise r
                return r
        raise AssertionError(url)

    def post(self, url, data=None, headers=None, timeout=None):
        self.calls.append(("post", url, data, headers, timeout))
        return Resp(url=self.post_url)


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    monkeypatch.setattr(tl, "__file__", str(tmp_path / "trendlyne_auth.py"))
    for k in ("TRENDLYNE_EMAIL", "TRENDLYNE_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    return tmp_path


def test_load_dotenv(clean_env, monkeypatch):
    # no file -> nothing
    tl._load_dotenv()
    import os

    assert "TRENDLYNE_EMAIL" not in os.environ
    (clean_env / ".env").write_text(
        "OTHER=1\n"
        f"TRENDLYNE_EMAIL = {FAKE_EMAIL}\n"
        "# TRENDLYNE_SKIP=1\n"
        "TRENDLYNE_NOEQUALS\n"
        "TRENDLYNE_PASSWORD=a=b\n"
        "  # TRENDLYNE_INDENTED=2\n"
    )
    monkeypatch.setenv("TRENDLYNE_PASSWORD", "preset")
    try:
        tl._load_dotenv()
        assert os.environ["TRENDLYNE_EMAIL"] == FAKE_EMAIL
        assert os.environ["TRENDLYNE_PASSWORD"] == "preset"  # setdefault keeps existing
        assert "OTHER" not in os.environ
        assert "TRENDLYNE_SKIP" not in os.environ
    finally:
        os.environ.pop("TRENDLYNE_EMAIL", None)


def test_session_requires_creds(clean_env):
    with pytest.raises(RuntimeError, match="set TRENDLYNE_EMAIL"):
        tl.session()


def test_session_requires_both(clean_env, monkeypatch):
    monkeypatch.setenv("TRENDLYNE_EMAIL", FAKE_EMAIL)
    with pytest.raises(RuntimeError):
        tl.session()


def _fake_requests(monkeypatch, sess):
    m = types.ModuleType("requests")
    m.Session = lambda: sess
    monkeypatch.setitem(sys.modules, "requests", m)


def test_session_login_ok(clean_env, monkeypatch):
    monkeypatch.setenv("TRENDLYNE_EMAIL", FAKE_EMAIL)
    monkeypatch.setenv("TRENDLYNE_PASSWORD", FAKE_PW)
    s = FakeSession({"accounts/login": Resp()})
    _fake_requests(monkeypatch, s)
    assert tl.session() is s
    kinds = [c[0] for c in s.calls]
    assert kinds == ["get", "post"]
    post = s.calls[1]
    assert post[1] == tl._LOGIN
    assert post[2]["csrfmiddlewaretoken"] == "TOK"
    assert post[2]["login"] == FAKE_EMAIL
    assert post[3]["Referer"] == tl._LOGIN
    assert post[4] == 20 and s.calls[0][3] == 20


def test_session_login_failed_redirect(clean_env, monkeypatch):
    monkeypatch.setenv("TRENDLYNE_EMAIL", FAKE_EMAIL)
    monkeypatch.setenv("TRENDLYNE_PASSWORD", FAKE_PW)
    s = FakeSession({"accounts/login": Resp()}, post_url="https://trendlyne.com/accounts/LOGIN/")
    _fake_requests(monkeypatch, s)
    with pytest.raises(RuntimeError, match="login failed"):
        tl.session()


def test_lookup_exact_match_and_fallback():
    rows = [
        {"k": 1, "stock_code": "RELIANCEX", "slugname": "a"},
        {"k": 2, "stock_code": "reliance", "slugname": "b"},
    ]
    s = FakeSession({"ac_snames": Resp(js=rows)})
    assert tl._lookup(s, "RELIANCE") == (2, "reliance", "b")
    assert "term=RELIANCE" in s.calls[0][1]
    assert s.calls[0][2]["X-Requested-With"] == "XMLHttpRequest"
    assert s.calls[0][3] == 15
    s = FakeSession({"ac_snames": Resp(js=rows)})
    assert tl._lookup(s, "ZZZ") == (1, "RELIANCEX", "a")  # first row fallback


def test_lookup_empty_and_errors():
    assert tl._lookup(FakeSession({"ac_snames": Resp(js=[])}), "A") is None
    assert tl._lookup(FakeSession({"ac_snames": Resp(js=None)}), "A") is None
    assert tl._lookup(FakeSession({"ac_snames": Resp(exc=ValueError())}), "A") is None
    assert tl._lookup(FakeSession({"ac_snames": OSError()}), "A") is None


def _routes(text="PE TTM", lookup=True):
    rows = [{"k": 7, "stock_code": "ABC", "slugname": "abc-ltd"}] if lookup else []
    return {"ac_snames": Resp(js=rows), "/equity/": Resp(text=text)}


def test_page_unescapes_and_url():
    s = FakeSession(_routes("a &amp; b"))
    assert tl._page(s, "ABC") == "a & b"
    assert s.calls[1][1] == "https://trendlyne.com/equity/7/ABC/abc-ltd/"
    assert s.calls[1][3] == 20


def test_page_none_cases():
    assert tl._page(FakeSession(_routes(lookup=False)), "ABC") is None
    r = _routes()
    r["/equity/"] = OSError()
    assert tl._page(FakeSession(r), "ABC") is None


BLOB = '{"title": "PE TTM", "value": 12.5} {"title": "Piotroski Score", "value": "8"} '
TOOLTIP = "Durability Score: 61 Valuation Score : 42 Momentum Score:9"


def test_parse_blobs_extras():
    out = tl._parse_blobs('{"title": " A ", "value": "-3.5"}{"title": "A", "value": 9}')
    assert out == {"A": -3.5}  # stripped title, setdefault keeps first
    assert tl._parse_blobs('{"title": "B", "value": "."}') == {}  # float('.') swallowed
    assert tl._parse_blobs('{"title":"C","value":"5"}') == {"C": 5.0}


def test_to_canonical_first_match_and_dedup():
    out = tl.to_canonical(
        {
            "Return on Equity": 1.0,
            "ROE Annual %": 2.0,  # key already present -> ignored
            "Price to Earning Ratio": 3.0,
            "Durability Score": 70.0,
            "Unknown": 4.0,
            "Promoter Pledge %": 5.0,
            "Sales Growth": 6.0,
        }
    )
    assert out == {
        "roe": 1.0,
        "pe": 3.0,
        "dvm_durability": 70.0,
        "promoter_pledge": 5.0,
        "revenue_growth": 6.0,
    }


def test_title_map_keys_are_reachable():
    for frag, key in tl._TITLE_MAP:
        assert tl.to_canonical({frag.upper(): 1.0}).get(key) == 1.0


def test_snapshot_and_fundamentals():
    s = FakeSession(_routes(BLOB))
    assert tl.snapshot("ABC", s) == {"PE TTM": 12.5, "Piotroski Score": 8.0}
    s = FakeSession(_routes(BLOB))
    f = tl.fundamentals("ABC", s)
    assert f == {"source": "trendlyne", "pe": 12.5, "piotroski": 8.0}
    assert tl.fundamentals("ABC", FakeSession(_routes(lookup=False))) == {}
    assert tl.fundamentals("ABC", FakeSession(_routes("nothing"))) == {}
    assert tl.snapshot("ABC", FakeSession(_routes(lookup=False))) == {}


def test_snapshot_creates_session_when_missing(monkeypatch):
    s = FakeSession(_routes(BLOB))
    monkeypatch.setattr(tl, "session", lambda: s)
    assert tl.snapshot("ABC") == {"PE TTM": 12.5, "Piotroski Score": 8.0}
    s2 = FakeSession(_routes(BLOB))
    monkeypatch.setattr(tl, "session", lambda: s2)
    assert tl.scores("ABC")["piotroski"] == 8


def test_scores_parsing():
    s = FakeSession(_routes(BLOB + TOOLTIP))
    assert tl.scores("ABC", s) == {
        "piotroski": 8,
        "dvm_durability": 61,
        "dvm_valuation": 42,
        "dvm_momentum": 9,
    }
    s = FakeSession(_routes("Momentum Score: 10"))
    assert tl.scores("ABC", s) == {"dvm_momentum": 10}
    assert tl.scores("ABC", FakeSession(_routes("zip"))) == {}
    assert tl.scores("ABC", FakeSession(_routes(lookup=False))) == {}


def test_main(monkeypatch, capsys):
    monkeypatch.setattr(tl, "session", lambda: "S")
    monkeypatch.setattr(tl.time, "sleep", lambda t: None)
    got = []
    monkeypatch.setattr(tl, "fundamentals", lambda sym, s: got.append(sym) or {"a": 1})
    monkeypatch.setattr(tl, "scores", lambda sym, s: {"b": 2})
    monkeypatch.setattr(sys, "argv", ["tl"])
    assert tl.main() == 0
    assert got == ["RELIANCE", "TCS", "INFY"]
    out = capsys.readouterr().out
    assert "login ok" in out and "fundamentals={'a': 1}  scores={'b': 2}" in out
    got.clear()
    monkeypatch.setattr(sys, "argv", ["tl", "X"])
    tl.main()
    assert got == ["X"]
