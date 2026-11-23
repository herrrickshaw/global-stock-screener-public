import io

import pandas as pd
import pytest

import screener_in_auth as sa

FAKE_EMAIL = "user@example.invalid"
FAKE_PW = "placeholder-not-real"
_ORIG_SESSION = sa.session


class Resp:
    def __init__(self, text="", status=200, content=b""):
        self.text, self.status_code, self.content = text, status, content


class FakeSess:
    def __init__(self, routes=None, post_routes=None):
        self.headers = {}
        self.cookies = {"csrftoken": "COOKIETOK"}
        self.routes = routes or {}
        self.post_routes = post_routes or {}
        self.calls = []

    def _find(self, table, url):
        for frag, r in table.items():
            if frag == "*" or url.endswith(frag) or frag in url:
                if isinstance(r, Exception):
                    raise r
                return r
        raise AssertionError(url)

    def get(self, url, headers=None, timeout=None):
        self.calls.append(("get", url, headers, timeout))
        return self._find(self.routes, url)

    def post(self, url, data=None, headers=None, timeout=None):
        self.calls.append(("post", url, data, headers, timeout))
        return self._find(self.post_routes, url)


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setattr(sa, "__file__", str(tmp_path / "screener_in_auth.py"))
    for k in ("SCREENER_EMAIL", "SCREENER_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    _ORIG_SESSION.cache_clear()
    yield
    _ORIG_SESSION.cache_clear()


# ── dotenv ───────────────────────────────────────────────────────────────────────
def test_dotenv_missing_file_is_noop(tmp_path):
    sa._load_dotenv()


def test_dotenv_parsing(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "\n# comment\nNOEQ\n"
        f"SCREENER_EMAIL = '{FAKE_EMAIL}'\n"
        'R2H_QUOTED="va=lue"\n'
        "R2H_EXISTING=new\n"
        "=novalue\n"
    )
    monkeypatch.setenv("R2H_EXISTING", "old")
    import os

    try:
        sa._load_dotenv()
        assert os.environ["SCREENER_EMAIL"] == FAKE_EMAIL
        assert os.environ["R2H_QUOTED"] == "va=lue"
        assert os.environ["R2H_EXISTING"] == "old"
        assert "NOEQ" not in os.environ and "" not in os.environ
    finally:
        os.environ.pop("SCREENER_EMAIL", None)
        os.environ.pop("R2H_QUOTED", None)


# ── session ──────────────────────────────────────────────────────────────────────
def test_session_needs_credentials():
    with pytest.raises(sa.AuthError, match="SCREENER_EMAIL"):
        sa.session()


def _creds(monkeypatch):
    monkeypatch.setenv("SCREENER_EMAIL", FAKE_EMAIL)
    monkeypatch.setenv("SCREENER_PASSWORD", FAKE_PW)


def test_session_email_only_fails(monkeypatch):
    monkeypatch.setenv("SCREENER_EMAIL", FAKE_EMAIL)
    with pytest.raises(sa.AuthError):
        sa.session()


def test_session_csrf_missing(monkeypatch):
    _creds(monkeypatch)
    fs = FakeSess({"/login/": Resp("<html>no token</html>")})
    monkeypatch.setattr(sa.requests, "Session", lambda: fs)
    with pytest.raises(sa.AuthError, match="CSRF"):
        sa.session()


@pytest.mark.parametrize("quote", ['"', "'"])
def test_session_login_success_and_cached(monkeypatch, quote):
    _creds(monkeypatch)
    login = Resp(f"<input name={quote}csrfmiddlewaretoken{quote} value={quote}TOK123{quote}>")
    fs = FakeSess(
        {"/login/": login, "screener.in/": Resp('<a href="/logout/">out</a>')},
        {"/login/": Resp()},
    )
    created = []
    monkeypatch.setattr(sa.requests, "Session", lambda: created.append(1) or fs)
    s = sa.session()
    assert s is fs and sa.session() is fs and created == [1]
    assert fs.headers["User-Agent"] == sa._UA["User-Agent"]
    post = [c for c in fs.calls if c[0] == "post"][0]
    assert post[2] == {
        "csrfmiddlewaretoken": "TOK123",
        "username": FAKE_EMAIL,
        "password": FAKE_PW,
        "next": "/",
    }
    assert post[3] == {"Referer": f"{sa.BASE}/login/"}
    assert all(c[-1] == 25 for c in fs.calls)


def test_session_login_failed(monkeypatch):
    _creds(monkeypatch)
    login = Resp("<input name='csrfmiddlewaretoken' value='T'>")
    fs = FakeSess({"/login/": login, "screener.in/": Resp("welcome guest")}, {"/login/": Resp()})
    monkeypatch.setattr(sa.requests, "Session", lambda: fs)
    with pytest.raises(sa.AuthError, match="login failed"):
        sa.session()


def test_session_logout_match_is_case_insensitive(monkeypatch):
    _creds(monkeypatch)
    login = Resp("<input name='csrfmiddlewaretoken' value='T'>")
    fs = FakeSess({"/login/": login, "screener.in/": Resp("click LOGOUT")}, {"/login/": Resp()})
    monkeypatch.setattr(sa.requests, "Session", lambda: fs)
    assert sa.session() is fs


# ── _num ─────────────────────────────────────────────────────────────────────────
def test_num():
    assert sa._num("1,234.5") == 1234.5
    assert sa._num(" 12 % ") == 12.0
    assert sa._num(None) is None
    assert sa._num("abc") is None
    assert sa._num("") is None
    assert sa._num(7) == 7.0


# ── company_export_bytes ─────────────────────────────────────────────────────────
def _install(monkeypatch, fs):
    monkeypatch.setattr(sa, "session", lambda: fs)
    return fs


PAGE = (
    '<form><input name="csrfmiddlewaretoken" value="PAGETOK">'
    '<button formaction="/user/company/export/123/">x</button></form>'
)
XLSX = b"PK\x03\x04data"


def test_export_no_link(monkeypatch):
    fs = _install(monkeypatch, FakeSess({"/company/ABC/": Resp("nothing")}))
    assert sa.company_export_bytes("ABC") is None


def test_export_post_success(monkeypatch):
    fs = _install(
        monkeypatch,
        FakeSess({"/company/ABC/": Resp(PAGE)}, {"/export/123/": Resp(content=XLSX)}),
    )
    assert sa.company_export_bytes("ABC") == XLSX
    post = [c for c in fs.calls if c[0] == "post"][0]
    assert post[1] == sa.BASE + "/user/company/export/123/"
    assert post[2] == {"csrfmiddlewaretoken": "PAGETOK"}
    assert post[3] == {"Referer": f"{sa.BASE}/company/ABC/"}
    assert post[4] == 40


def test_export_falls_back_to_get_and_cookie_token(monkeypatch):
    page = '<a href="https://www.screener.in/company/export/9/">x</a>'
    fs = _install(
        monkeypatch,
        FakeSess(
            {"/company/ABC/": Resp(page), "/export/9/": Resp(content=XLSX)},
            {"/export/9/": Resp(status=500, content=b"err")},
        ),
    )
    assert sa.company_export_bytes("ABC") == XLSX
    post = [c for c in fs.calls if c[0] == "post"][0]
    assert post[1] == "https://www.screener.in/company/export/9/"  # absolute kept
    assert post[2] == {"csrfmiddlewaretoken": "COOKIETOK"}
    assert [c[0] for c in fs.calls] == ["get", "post", "get"]


def test_export_rejects_non_zip_and_non_200(monkeypatch):
    fs = _install(
        monkeypatch,
        FakeSess(
            {"/company/ABC/": Resp(PAGE), "/export/123/": Resp(status=200, content=b"<html>")},
            {"/export/123/": Resp(status=200, content=b"PX")},
        ),
    )
    assert sa.company_export_bytes("ABC") is None
    fs = _install(
        monkeypatch,
        FakeSess(
            {"/company/ABC/": Resp(PAGE), "/export/123/": Resp(status=403, content=XLSX)},
            {"/export/123/": Resp(status=404, content=XLSX)},
        ),
    )
    assert sa.company_export_bytes("ABC") is None


def test_export_exceptions_swallowed(monkeypatch):
    fs = _install(
        monkeypatch,
        FakeSess(
            {"/company/ABC/": Resp(PAGE), "/export/123/": OSError()},
            {"/export/123/": OSError()},
        ),
    )
    assert sa.company_export_bytes("ABC") is None
    # post raises but get works
    fs = _install(
        monkeypatch,
        FakeSess(
            {"/company/ABC/": Resp(PAGE), "/export/123/": Resp(content=XLSX)},
            {"/export/123/": OSError()},
        ),
    )
    assert sa.company_export_bytes("ABC") == XLSX


# ── page_ratios ──────────────────────────────────────────────────────────────────
RATIOS = """
<li><span class="name"> Market Cap </span> <span class="nowrap value">&#8377; <span class="number">1,23,456</span> Cr.</span></li>
<li><span class="name">Stock P/E</span><span class="value"><span class="number">21.5</span></span></li>
<li><span class="name">ROE</span><span class="value"><span class="number">9.5</span> %</span></li>
<li><span class="name">ROCE</span><span class="value"><span class="number">12</span> %</span></li>
<li><span class="name">Dividend Yield</span><span class="value">0.5 %</span></li>
<li><span class="name">Current Price</span><span class="value">&#8377; 100</span></li>
<li><span class="name">Book Value</span><span class="value">n/a</span></li>
<li><span class="name">Unknown</span><span class="value">5</span></li>
<li><span class="name">Face Value</span><span class="value">1</span></li>
"""


def test_page_ratios(monkeypatch):
    _install(monkeypatch, FakeSess({"/company/ABC/": Resp(RATIOS)}))
    out = sa.page_ratios("ABC")
    assert out["pe"] == 21.5
    assert out["roe_page"] == 9.5
    assert out["roce_page"] == 12.0
    assert out["dividend_yield"] == 0.5
    assert out["face_value"] == 1.0
    assert "book_value" not in out  # unparseable number skipped
    assert "Unknown" not in out and len(out) >= 5


def test_page_ratios_market_cap_and_price_currency_symbols(monkeypatch):
    html = (
        '<span class="name">Market Cap</span><span class="nowrap value">'
        '<span class="number">1,000</span></span>'
        '<span class="name">Current Price</span><span class="value"><span class="number">55.5</span></span>'
    )
    _install(monkeypatch, FakeSess({"/company/ABC/": Resp(html)}))
    assert sa.page_ratios("ABC") == {"market_cap_cr": 1000.0, "current_price": 55.5}


def test_page_ratios_error(monkeypatch):
    _install(monkeypatch, FakeSess({"/company/ABC/": OSError()}))
    assert sa.page_ratios("ABC") == {}
    monkeypatch.setattr(sa, "session", lambda: (_ for _ in ()).throw(sa.AuthError("x")))
    assert sa.page_ratios("ABC") == {}


# ── export parsing ───────────────────────────────────────────────────────────────
def _xlsx(rows, sheet="Data Sheet", extra=None):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as w:
        if extra:
            pd.DataFrame([["x"]]).to_excel(w, sheet_name=extra, header=False, index=False)
        pd.DataFrame(rows).to_excel(w, sheet_name=sheet, header=False, index=False)
    return buf.getvalue()


def test_read_export_df_picks_data_sheet_or_first():
    raw = _xlsx([["Sales", 1, 2]], sheet="My DATA", extra="Cover")
    df = sa._read_export_df(raw)
    assert df.iloc[0, 0] == "Sales"
    raw = _xlsx([["Sales", 1, 2]], sheet="Other")
    assert sa._read_export_df(raw).iloc[0, 0] == "Sales"
    assert sa._read_export_df(b"garbage") is None


def test_row_reader():
    df = pd.DataFrame(
        [
            ["Report Date", "Mar-24", "Mar-23"],
            [" Net Profit ", "1,000", "800"],
            ["Sales", "n/a", "5"],
            ["Empty", "n/a", "x"],
        ]
    )
    row = sa._make_row_reader(df)
    assert row("net profit") == [1000.0, 800.0]
    assert row("Nope", "Sales") == [5.0]  # fallback label + non-numeric dropped
    assert row("Empty") == []
    assert row("Missing") == []


def test_g_and_pct():
    assert sa._g([1, 2], 1) == 2
    assert sa._g([1], 1) is None
    assert sa._g([]) is None
    assert sa._pct(110, 100) == pytest.approx(10.0)
    assert sa._pct(90, -100) == pytest.approx(190.0)  # |b| denominator
    assert sa._pct(1, 0) is None
    assert sa._pct(None, 1) is None
    assert sa._pct(1, None) is None


def test_derived_series():
    sr = {
        "ni": [10.0, 5.0],
        "capital": [10.0, 10.0, 10.0],
        "reserves": [90.0, 40.0],
        "debt": [50.0],
        "other_liab": [],
        "pbt": [12.0],
        "interest": [3.0],
    }
    d = sa._derived_series(sr)
    assert d["eq"] == [100.0, 50.0, 10.0]
    assert d["assets_tot"] == [150.0, 50.0, 10.0]
    assert d["roe_hist"] == [10.0, 10.0]
    assert d["ebit"] == 15.0
    empty = {k: [] for k in sr}
    d = sa._derived_series(empty)
    assert d == {"eq": [], "assets_tot": [], "roe_hist": [], "ebit": None}
    # only interest present
    d = sa._derived_series({**empty, "interest": [4.0]})
    assert d["ebit"] == 4.0
    # zero equity excluded from roe_hist
    d = sa._derived_series({**empty, "ni": [1.0], "capital": [0.0]})
    assert d["roe_hist"] == []


def test_fundamentals_dict_values():
    sr = {k: [] for k in (
        "ni rev interest pbt eps debt reserves capital other_liab net_block inv recv cash cfo shares div".split()
    )}  # fmt: skip
    sr.update(
        ni=[120.0, 100.0],
        rev=[1000.0, 900.0],
        eps=[12.0, 10.0],
        debt=[50.0, 40.0],
        cfo=[7.0],
        shares=[10.0, 11.0],
        inv=[3.0],
        recv=[4.0],
        cash=[20.0],
        div=[1.0, 0.0, 2.0],
        net_block=[5.0],
    )
    dv = {"eq": [200.0, 180.0], "assets_tot": [400.0, 350.0], "roe_hist": [60.0], "ebit": 30.0}
    r = {"market_cap_cr": 1000.0, "pe": 20.0, "roe_page": 15.0, "roce_page": 18.0}
    out = sa._fundamentals_dict(sr, dv, r)
    assert out["source"] == "screener.in"
    assert out["net_income"] == 120.0 and out["net_income_prev"] == 100.0
    assert out["profit_growth"] == pytest.approx(20.0)
    assert out["eps_growth"] == pytest.approx(20.0)
    assert out["revenue_5y_ago"] == 900.0  # fallback to last when <6 entries
    assert out["roe"] == 15.0  # page value wins
    assert out["roa"] == 30.0 and out["roa_prev"] == round(100 / 350 * 100, 1)
    assert out["debt_to_equity"] == 0.25
    assert out["debt_to_assets"] == 0.12 and out["debt_to_assets_prev"] == round(40 / 350, 2)
    assert out["asset_turnover"] == 2.5 and out["asset_turnover_prev"] == round(900 / 350, 2)
    assert out["enterprise_value"] == 1000.0 + 50 - 20
    assert out["earnings_yield"] == round(30 / 1030 * 100, 2)
    assert out["dividend_pay_years"] == 2
    assert out["dividend_history"] == [1.0, 0.0, 2.0]
    assert out["quarterly_net_income"] == [120.0, 100.0]
    assert out["roc"] == 18.0 and out["roe_history"] == [60.0]
    assert out["capex_history"] == [5.0] and out["debt_history"] == [50.0, 40.0]
    assert out["inventory"] == 3.0 and out["receivables"] == 4.0 and out["cash"] == 20.0
    assert out["shares"] == 10.0 and out["shares_prev"] == 11.0 and out["cfo"] == 7.0


def test_fundamentals_dict_revenue_5y_ago_and_roe_fallback():
    sr = {k: [] for k in (
        "ni rev interest pbt eps debt reserves capital other_liab net_block inv recv cash cfo shares div".split()
    )}  # fmt: skip
    sr.update(ni=[10.0], rev=[6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5], debt=[20.0])
    dv = {"eq": [100.0], "assets_tot": [], "roe_hist": [], "ebit": None}
    out = sa._fundamentals_dict(sr, dv, {})
    assert out["revenue_5y_ago"] == 1.0  # index 5
    assert out["roe"] == 10.0  # computed from NI / equity
    for k in ("roa", "enterprise_value", "earnings_yield", "roe_history", "dividend_history"):
        assert k not in out
    out = sa._fundamentals_dict(
        {k: [] for k in sr}, {"eq": [], "assets_tot": [], "roe_hist": [], "ebit": None}, {}
    )
    assert out == {"source": "screener.in"}


def test_fundamentals_dict_no_ebit_means_no_earnings_yield():
    sr = {k: [] for k in (
        "ni rev interest pbt eps debt reserves capital other_liab net_block inv recv cash cfo shares div".split()
    )}  # fmt: skip
    dv = {"eq": [], "assets_tot": [], "roe_hist": [], "ebit": 0}
    out = sa._fundamentals_dict(sr, dv, {"market_cap_cr": 100.0})
    assert out["enterprise_value"] == 100.0
    assert "earnings_yield" not in out


# ── company_financials / batch / main ────────────────────────────────────────────
def test_company_financials_paths(monkeypatch):
    monkeypatch.setattr(sa, "company_export_bytes", lambda s: None)
    assert sa.company_financials("A") == {}
    monkeypatch.setattr(sa, "company_export_bytes", lambda s: b"garbage")
    assert sa.company_financials("A") == {}
    raw = _xlsx(
        [
            ["Net profit", 10, 5],
            ["Equity Capital", 50, 50],
            ["Reserves", 50, 0],
            ["Borrowings", 20, 10],
            ["Sales", 200, 100],
        ]
    )
    monkeypatch.setattr(sa, "company_export_bytes", lambda s: raw)
    monkeypatch.setattr(sa, "page_ratios", lambda s: {"pe": 11.0})
    out = sa.company_financials("A")
    assert out["net_income"] == 10.0 and out["pe"] == 11.0 and out["roe"] == 10.0


def test_batch(monkeypatch, capsys):
    sleeps = []
    monkeypatch.setattr(sa.time, "sleep", lambda t: sleeps.append(t))

    def cf(sym):
        if sym == "bad":
            raise ValueError
        return {} if sym == "empty" else {"s": sym}

    monkeypatch.setattr(sa, "company_financials", cf)
    out = sa.fundamentals_batch(["a", "bad", "empty", "b"], pause=0.3, verbose=False)
    assert out == {"a": {"s": "a"}, "b": {"s": "b"}}
    assert sleeps == [0.3] * 4
    assert capsys.readouterr().out == ""
    syms = [f"s{i}" for i in range(25)]
    out = sa.fundamentals_batch(syms)
    assert len(out) == 25
    assert "25/25 fetched (25 ok)" in capsys.readouterr().out


def test_batch_auth_error_propagates(monkeypatch):
    def cf(sym):
        raise sa.AuthError("no")

    monkeypatch.setattr(sa, "company_financials", cf)
    with pytest.raises(sa.AuthError):
        sa.fundamentals_batch(["a"])


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="screener_in_auth.py:297 debt_to_equity divides g(debt)=None by equity when the "
    "export has equity rows but no Borrowings row",
)
def test_fundamentals_dict_no_debt_row_should_not_crash():
    sr = {k: [] for k in (
        "ni rev interest pbt eps debt reserves capital other_liab net_block inv recv cash cfo shares div".split()
    )}  # fmt: skip
    sr["ni"] = [10.0]
    dv = {"eq": [100.0], "assets_tot": [], "roe_hist": [], "ebit": None}
    out = sa._fundamentals_dict(sr, dv, {})
    assert "debt_to_equity" not in out and out["roe"] == 10.0


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="screener_in_auth.py:291 roe divides g(ni)=None by equity when no Net profit row "
    "and no page ROE",
)
def test_fundamentals_dict_no_net_profit_row_should_not_crash():
    sr = {k: [] for k in (
        "ni rev interest pbt eps debt reserves capital other_liab net_block inv recv cash cfo shares div".split()
    )}  # fmt: skip
    dv = {"eq": [100.0], "assets_tot": [], "roe_hist": [], "ebit": None}
    out = sa._fundamentals_dict(sr, dv, {})
    assert "roe" not in out
