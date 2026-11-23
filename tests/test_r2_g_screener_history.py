import io
import sys
import types

import numpy as np
import pandas as pd
import pytest

import screener_history_collector as sh

T = pd.Timestamp


# ---------------------------------------------------------------- tiny helpers
def test_zero():
    assert sh._zero(None) == 0.0
    assert sh._zero(float("nan")) == 0.0
    assert sh._zero(np.nan) == 0.0
    assert sh._zero(5) == 5.0 and isinstance(sh._zero(5), float)
    assert sh._zero(0) == 0.0 and sh._zero(-2.5) == -2.5


@pytest.mark.parametrize(
    "sym,expected",
    [
        ("1010EFSL29", True),
        ("805HDFCB29", True),
        ("360ONE", False),  # no maturity digits
        ("RELIANCE", False),
        ("1018GS2026", True),
        ("SDL2030", True),
        ("SDLKA", True),
        ("MYSDL5", True),
        ("GOI2030", True),
        ("91D0123", True),
        ("182DTB", True),
        ("364D1", True),
        ("TB26", True),
        ("  1010efsl29 ", True),  # stripped, upper-cased
        ("", False),
        (None, False),
        ("123A45", True),  # 3-digit coupon, 2-digit year
        ("12AB34", False),  # only 2 leading digits
        ("12345AB34", False),  # 5 leading digits
        ("123A4", False),  # 1-digit tail
        ("123A12345", False),  # 5-digit tail
        ("HDFCBANK", False),
    ],
)
def test_is_non_equity(sym, expected):
    assert sh.is_non_equity(sym) is expected


def test_num():
    assert sh._num("1,234.5") == 1234.5
    assert sh._num(" 12% ") == 12.0
    assert sh._num(3) == 3.0
    assert sh._num("abc") is None and sh._num(None) is None
    assert sh._num("") is None
    assert np.isnan(sh._num(np.nan))  # documented: NaN survives as NaN


def test_is_annual_axis():
    mar = [T("2023-03-31"), T("2024-03-31")]
    assert sh._is_annual_axis(mar) == mar
    assert sh._is_annual_axis([T("2023-03-31 10:00")]) == [T("2023-03-31")]  # normalised
    assert sh._is_annual_axis([T("2023-03-31"), T("2023-06-30")]) is None
    assert sh._is_annual_axis([T("2023-12-31")]) is None
    assert sh._is_annual_axis(["x", None, np.nan]) is None
    assert sh._is_annual_axis([]) is None
    assert sh._is_annual_axis(["x", T("2023-03-31")]) == [T("2023-03-31")]  # non-dates ignored


# ---------------------------------------------------------------- industry
class Sess:
    def __init__(self, html=None, exc=None):
        self.html, self.exc, self.urls = html, exc, []

    def session(self):
        outer = self

        class S:
            def get(self, url, timeout):
                outer.urls.append((url, timeout))
                if outer.exc:
                    raise outer.exc
                return types.SimpleNamespace(text=outer.html)

        return S()


@pytest.fixture(autouse=True)
def reset_regex(monkeypatch):
    monkeypatch.setattr(sh, "_INDUSTRY_RE", None)


def test_industry_of_parses_and_caches_regex():
    s = Sess("<td>Industry</td> <a>  Banks </a>")
    assert sh.industry_of(s, "HDFCBANK") == "Banks"
    assert s.urls == [("https://www.screener.in/company/HDFCBANK/", 25)]
    assert sh._INDUSTRY_RE is not None
    assert (
        sh.industry_of(Sess("<b>INDUSTRY:</b>\n<span>IT - Software</span>"), "TCS")
        == "IT - Software"
    )


def test_industry_of_none_cases():
    assert sh.industry_of(Sess("<p>nothing</p>"), "X") is None
    assert sh.industry_of(Sess(exc=RuntimeError("down")), "X") is None
    assert sh.industry_of(Sess("<td>Industry</td><a>A</a>"), "X") is None  # < 2 chars? 'A' is 1


def test_is_financial():
    assert sh.is_financial(None) is False and sh.is_financial("") is False
    assert sh.is_financial("Banks") is True
    assert sh.is_financial("  Capital Markets ") is True
    assert sh.is_financial("Housing Finance") is True
    assert sh.is_financial("INSURANCE") is True
    assert sh.is_financial("Power") is False and sh.is_financial("IT - Software") is False
    assert sh.is_financial("Diversified") is False


# ---------------------------------------------------------------- _extract_annual
def sheet(rows):
    width = max(len(r) for r in rows)
    return pd.DataFrame([r + [np.nan] * (width - len(r)) for r in rows])


def test_extract_annual_basic_alignment():
    d = [T("2023-03-31"), T("2024-03-31")]
    df = sheet(
        [
            ["Report Date", *d],
            ["Sales", "1,000", 2000],
            ["Net profit", 100, "bad"],
            ["  Reserves ", 5, 6],
            ["Unmapped", 1, 1],
        ]
    )
    out = sh._extract_annual(df)
    assert out == {
        T("2023-03-31"): {"revenue": 1000.0, "net_income": 100.0, "reserves": 5.0},
        T("2024-03-31"): {"revenue": 2000.0, "reserves": 6.0},
    }


def test_extract_annual_quarterly_block_ignored_and_first_wins():
    ann = [T("2023-03-31"), T("2024-03-31")]
    q = [T("2023-06-30"), T("2023-09-30")]
    df = sheet(
        [
            ["Sales", 9, 9],  # before any axis -> skipped
            ["Report Date", *ann],
            ["Sales", 10, 20],
            ["Sales+", 99, 99],  # same field already seen
            ["Report Date", *q],
            ["Net profit", 1, 2],  # quarterly axis -> skipped
            ["Report Date", *ann],
            ["Net profit", 7, 8],  # annual again -> captured
        ]
    )
    out = sh._extract_annual(df)
    assert out[T("2023-03-31")] == {"revenue": 10.0, "net_income": 7.0}
    assert out[T("2024-03-31")] == {"revenue": 20.0, "net_income": 8.0}
    assert T("2023-06-30") not in out


def test_extract_annual_label_case_and_alternates_and_nan():
    d = [T("2023-03-31"), T("2024-03-31")]
    df = sheet(
        [
            ["REPORT DATE", *d],
            ["profit before tax", np.nan, 5],
            ["Finance Costs", 1, 2],
            ["No. of Equity Shares", 3, 4],
        ]
    )
    out = sh._extract_annual(df)
    assert out[T("2023-03-31")]["interest"] == 1.0
    assert np.isnan(out[T("2023-03-31")]["profit_before_tax"])  # NaN kept (documented quirk)
    assert out[T("2024-03-31")]["profit_before_tax"] == 5.0
    assert out[T("2024-03-31")]["shares"] == 4.0
    assert sh._extract_annual(pd.DataFrame([["Sales", 1]])) == {}


# ---------------------------------------------------------------- derived fields
def test_add_ebit_and_industry():
    row = {"profit_before_tax": 10.0, "interest": 4.0, "revenue": -100.0}
    sh._add_ebit_and_industry(row, "Power", False)
    assert row["ebit"] == 14.0 and row["interest_share_of_revenue"] == 0.04  # abs()/abs()
    assert row["industry"] == "Power" and row["is_financial"] is False
    lender = {"profit_before_tax": 10.0, "interest": 4.0, "revenue": 100.0}
    sh._add_ebit_and_industry(lender, "Banks", True)
    assert "ebit" not in lender and lender["is_financial"] is True
    assert lender["interest_share_of_revenue"] == 0.04  # still emitted
    r = {"interest": 4.0, "revenue": 0}
    sh._add_ebit_and_industry(r, None, False)
    assert "interest_share_of_revenue" not in r and "ebit" not in r and r["industry"] is None
    r = {"interest": 4.0}
    sh._add_ebit_and_industry(r, None, False)
    assert "interest_share_of_revenue" not in r
    r = {"profit_before_tax": 0.0, "interest": 0.0}
    sh._add_ebit_and_industry(r, None, False)
    assert r["ebit"] == 0.0  # zeros are values


def test_add_gross_margin_threshold_and_nan_safety():
    row = {"revenue": 100.0, "raw_material_cost": 30.0}
    sh._add_gross_margin(row)
    assert row["rm_share"] == 0.3 and row["cogs"] == 30.0  # >= 0.30 inclusive
    assert row["gross_profit"] == 70.0 and row["gross_margin"] == pytest.approx(0.7)
    row = {"revenue": 100.0, "raw_material_cost": 29.99}
    sh._add_gross_margin(row)
    assert (
        row["rm_share"] == pytest.approx(0.2999) and "gross_margin" not in row and "cogs" not in row
    )
    full = {
        "revenue": 200.0,
        "raw_material_cost": 100.0,
        "change_in_inventory": 10.0,  # credit -> reduces cost
        "power_and_fuel": 6.0,
        "other_mfr_exp": 4.0,
    }
    sh._add_gross_margin(full)
    assert full["cogs"] == 100.0 and full["gross_profit"] == 100.0 and full["gross_margin"] == 0.5
    nan = {
        "revenue": 100.0,
        "raw_material_cost": 50.0,
        "change_in_inventory": np.nan,
        "power_and_fuel": np.nan,
    }
    sh._add_gross_margin(nan)
    assert nan["cogs"] == 50.0  # NaN components are zero, not poison
    for bad in (
        {"raw_material_cost": 5.0},
        {"revenue": 0, "raw_material_cost": 5.0},
        {"revenue": 10.0},
    ):
        r = dict(bad)
        sh._add_gross_margin(r)
        assert "rm_share" not in r


def test_add_capital_employed_and_roce():
    row = {"equity_capital": 10.0, "reserves": 40.0, "borrowings": 50.0, "ebit": 20.0}
    sh._add_capital_employed_and_roce(row)
    assert row["capital_employed"] == 100.0 and row["roce"] == 0.2
    # negative net worth rescued by debt -> declined
    neg = {"equity_capital": 10.0, "reserves": -20.0, "borrowings": 500.0, "ebit": 20.0}
    sh._add_capital_employed_and_roce(neg)
    assert neg["capital_employed"] is None and "roce" not in neg
    zero_nw = {"equity_capital": 10.0, "reserves": -10.0, "borrowings": 5.0, "ebit": 1.0}
    sh._add_capital_employed_and_roce(zero_nw)
    assert zero_nw["capital_employed"] is None  # (eq+res) > 0 strict
    nonpos = {"equity_capital": 1.0, "reserves": 1.0, "borrowings": -5.0, "ebit": 1.0}
    sh._add_capital_employed_and_roce(nonpos)
    assert nonpos["capital_employed"] is None  # ce > 0 strict
    missing = {"equity_capital": 1.0, "reserves": 1.0, "ebit": 3.0}
    sh._add_capital_employed_and_roce(missing)
    assert "capital_employed" not in missing and "roce" not in missing
    noebit = {"equity_capital": 1.0, "reserves": 1.0, "borrowings": 0.0}
    sh._add_capital_employed_and_roce(noebit)
    assert noebit["capital_employed"] == 2.0 and "roce" not in noebit
    # zero borrowings is a value (not missing)
    z = {"equity_capital": 1.0, "reserves": 1.0, "borrowings": 0.0, "ebit": 1.0}
    sh._add_capital_employed_and_roce(z)
    assert z["roce"] == 0.5


def test_derive_annual_row():
    fields = {
        "revenue": 100.0,
        "raw_material_cost": 50.0,
        "profit_before_tax": 10.0,
        "interest": 5.0,
        "equity_capital": 10.0,
        "reserves": 40.0,
        "borrowings": 50.0,
    }
    row = sh._derive_annual_row("ABC", T("2024-03-31"), fields, "Steel", False)
    assert row["ticker"] == "ABC" and row["fy_end"] == "2024-03-31"
    assert row["ebit"] == 15.0 and row["roce"] == 0.15 and row["gross_margin"] == 0.5
    assert row["industry"] == "Steel" and row["is_financial"] is False
    assert "ebit" not in sh._derive_annual_row("B", T("2024-03-31"), fields, "Banks", True)
    assert fields.get("ebit") is None  # input dict not mutated


# ---------------------------------------------------------------- collect_ticker
def xlsx_bytes(sheets):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name, header=False, index=False)
    return buf.getvalue()


class Mod(Sess):
    def __init__(self, raw=None, exc=None, html="<td>Industry</td><a>Steel</a>"):
        super().__init__(html)
        self.raw, self.exc2 = raw, exc

    def company_export_bytes(self, t):
        if self.exc2:
            raise self.exc2
        return self.raw


def good_sheet():
    d = [T("2023-03-31"), T("2024-03-31")]
    return sheet([["Report Date", *d], ["Sales", 100, 200], ["Raw Material Cost", 50, 100]])


def test_collect_ticker_happy_and_sheet_choice():
    raw = xlsx_bytes({"Profit & Loss": pd.DataFrame([["x", 1]]), "Data Sheet": good_sheet()})
    rows = sh.collect_ticker(Mod(raw), "ABC")
    assert [r["fy_end"] for r in rows] == ["2023-03-31", "2024-03-31"]
    assert (
        rows[0]["revenue"] == 100.0
        and rows[0]["industry"] == "Steel"
        and rows[0]["gross_margin"] == 0.5
    )
    # no sheet with "data" -> first sheet
    rows = sh.collect_ticker(Mod(xlsx_bytes({"Anything": good_sheet()})), "ABC")
    assert len(rows) == 2
    fin = sh.collect_ticker(Mod(raw, html="<td>Industry</td><a>Banks</a>"), "ABC")
    assert fin[0]["is_financial"] is True


def test_collect_ticker_failures():
    assert sh.collect_ticker(Mod(None), "A") == []
    assert sh.collect_ticker(Mod(b""), "A") == []
    assert sh.collect_ticker(Mod(b"not an excel file"), "A") == []
    assert sh.collect_ticker(Mod(exc=ValueError("404")), "A") == []
    with pytest.raises(sh.Blocked, match="host refused connection at A"):
        sh.collect_ticker(Mod(exc=RuntimeError("Max retries exceeded")), "A")

    class ConnectionError_(Exception):
        pass

    ConnectionError_.__name__ = "ConnectionError"
    with pytest.raises(sh.Blocked):
        sh.collect_ticker(Mod(exc=ConnectionError_("x")), "A")


# ---------------------------------------------------------------- _universe
def args(**kw):
    d = dict(tickers=None, liquid_only=False, limit=0)
    d.update(kw)
    return types.SimpleNamespace(**d)


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(sh, "UNIVERSE", str(tmp_path / "IN.csv"))
    monkeypatch.setattr(sh, "LTM", str(tmp_path / "ltm.parquet"))
    monkeypatch.setattr(sh, "BASE", str(tmp_path))
    monkeypatch.setattr(sh, "OUT_DIR", str(tmp_path / "fh"))
    monkeypatch.setattr(sh, "OUT", str(tmp_path / "fh" / "IN.parquet"))
    return tmp_path


def test_universe_explicit_tickers(paths):
    assert sh._universe(args(tickers=["tcs", "Infy"])) == ["TCS", "INFY"]


def test_universe_from_csv(paths):
    pd.DataFrame({"yf_ticker": ["tcs.NS", "TCS.BO", "INFY.NS", "ABC"]}).to_csv(
        sh.UNIVERSE, index=False
    )
    assert sh._universe(args()) == ["ABC", "INFY", "TCS"]
    # liquid-only without LTM file -> unfiltered
    assert sh._universe(args(liquid_only=True)) == ["ABC", "INFY", "TCS"]
    pd.DataFrame({"Symbol": ["tcs", "abc"]}).to_parquet(sh.LTM)
    assert sh._universe(args(liquid_only=True)) == ["ABC", "TCS"]
    assert sh._universe(args()) == ["ABC", "INFY", "TCS"]


def test_universe_missing_everything_exits(paths):
    with pytest.raises(SystemExit) as e:
        sh._universe(args())
    assert "no universe" in str(e.value)


def ltm_frame():
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    recs = []
    for s, c, v in [
        ("liq", 100, 100_000),  # 1e7 -> included
        ("under", 100, 99_999),
        ("1010EFSL29", 1000, 1_000_000),  # bond passes value gate, dropped by structure
        ("RELIANCE", 1000, 100_000),
    ]:
        for d in dates:
            recs.append((s, c, v, d))
    for d in dates[:20]:  # only old sessions -> outside the 120d window
        recs.append(("OLD", 1000, 10_000_000, d))
    return pd.DataFrame(recs, columns=["Symbol", "Close", "Volume", "Date"])


def test_universe_ltm_fallback(paths, capsys):
    ltm_frame().to_parquet(sh.LTM)
    assert sh._universe(args()) == ["1010EFSL29", "LIQ", "OLD", "RELIANCE", "UNDER"]
    assert capsys.readouterr().out == ""
    assert sh._universe(args(liquid_only=True)) == ["LIQ", "RELIANCE"]
    assert "dropped 1 non-equity instruments" in capsys.readouterr().out


def test_universe_ltm_liquid_nothing_dropped_prints_nothing(paths, capsys):
    f = ltm_frame()
    f = f[f["Symbol"] != "1010EFSL29"]
    f.to_parquet(sh.LTM)
    assert sh._universe(args(liquid_only=True)) == ["LIQ", "RELIANCE"]
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------- main
class AuthMod:
    def __init__(self):
        self.logins = 0
        self.script = {}
        self.calls = []

    def session(self):
        self.logins += 1


@pytest.fixture
def run(paths, monkeypatch):
    auth = AuthMod()
    monkeypatch.setitem(sys.modules, "screener_in_auth", auth)
    monkeypatch.setattr(sh.time, "sleep", lambda s: None)
    calls = []

    def ct(mod, tk):
        calls.append(tk)
        beh = auth.script.get(tk, "ok")
        if isinstance(beh, list):
            beh = beh.pop(0)
        if beh == "block":
            raise sh.Blocked("x")
        if beh == "err":
            raise RuntimeError("x")
        if beh == "empty":
            return []
        return [{"ticker": tk, "fy_end": "2024-03-31", "revenue": 1.0}]

    monkeypatch.setattr(sh, "collect_ticker", ct)

    def go(*argv):
        monkeypatch.setattr(sys, "argv", ["x", *argv])
        sh.main()

    go.auth, go.calls = auth, calls
    return go


def test_main_basic_resume_and_limit(run, capsys):
    run("--tickers", "a", "b", "c", "--limit", "2")
    out = capsys.readouterr().out
    assert run.auth.logins == 1
    assert "2 tickers, 0 already done, 2 to fetch (~0 min at 0.7s/req)" in out
    assert "2/2 | 2 with data | 2 rows" in out and "done → " in out
    assert run.calls == ["A", "B"]
    run.calls.clear()
    run("--tickers", "a", "b", "c")
    assert run.calls == ["C"] and "2 already done, 1 to fetch" in capsys.readouterr().out
    assert set(pd.read_parquet(sh.OUT)["ticker"]) == {"A", "B", "C"}


def test_main_retry_absorbs_single_flake(run, capsys):
    run.auth.script = {"A": ["empty", "ok"], "B": ["err", "ok"], "C": ["empty", "err"]}
    run("--tickers", "a", "b", "c")
    assert run.calls == ["A", "A", "B", "B", "C", "C"]
    assert "3/3 | 2 with data" in capsys.readouterr().out  # C failed twice -> no data


def test_main_block_first_try_aborts_and_saves(run, capsys):
    run.auth.script = {"B": "block"}
    run("--tickers", "a", "b", "c")
    out = capsys.readouterr().out
    assert (
        "ABORT: host refused the connection at B — hard block, not retrying. Collected 1 before this."
        in out
    )
    assert run.calls == ["A", "B"]  # no retry, C never tried
    assert list(pd.read_parquet(sh.OUT)["ticker"]) == ["A"]


def test_main_block_on_retry_aborts(run, capsys):
    run.auth.script = {"B": ["empty", "block"]}
    run("--tickers", "a", "b", "c")
    out = capsys.readouterr().out
    assert (
        "ABORT: host refused the connection at B (on retry)" in out and "Collected 1 before" in out
    )
    assert run.calls == ["A", "B", "B"]


def test_main_circuit_breaker_30(run, capsys):
    names = [f"t{i:02d}" for i in range(40)]
    run.auth.script = {n.upper(): "empty" for n in names[1:]}
    run("--tickers", *names)
    out = capsys.readouterr().out
    assert "ABORT: 30 consecutive tickers returned nothing (each retried once)" in out
    # T00 ok + 30 empties (each fetched twice)
    assert len(run.calls) == 1 + 30 * 2
    assert "Collected 1 before" in out


def test_main_29_empties_do_not_abort_and_success_resets(run, capsys):
    names = [f"t{i:02d}" for i in range(60)]
    run.auth.script = {n.upper(): "empty" for n in names[:29] + names[30:59]}
    run("--tickers", *names)
    out = capsys.readouterr().out
    assert "ABORT" not in out and "60/60 | 2 with data" in out


def test_main_checkpoint_100_and_nothing_written(run, capsys):
    run("--tickers", *[f"t{i:03d}" for i in range(101)])
    out = capsys.readouterr().out
    assert (
        "100/101 | 100 with data | 100 rows" in out and "101/101 | 101 with data | 101 rows" in out
    )
    import os

    os.remove(sh.OUT)
    run.auth.script = {"Z": "empty"}
    run("--tickers", "z")
    assert not os.path.exists(sh.OUT)
    assert "1/1 | 0 with data | 0 rows" in capsys.readouterr().out


def test_main_uses_universe_when_no_tickers(run, paths, capsys):
    pd.DataFrame({"yf_ticker": ["aaa.NS", "bbb.NS"]}).to_csv(sh.UNIVERSE, index=False)
    run()
    assert run.calls == ["AAA", "BBB"]
    run.calls.clear()
    pd.DataFrame(
        {"Symbol": ["AAA"], "Close": [1], "Volume": [1], "Date": [T("2024-01-01")]}
    ).to_parquet(sh.LTM)
    import os

    os.remove(sh.OUT)
    run("--liquid-only")
    assert run.calls == ["AAA"]
