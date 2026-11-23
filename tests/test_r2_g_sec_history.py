import sys

import pandas as pd
import pytest

import sec_history_collector as sh


def f(val, end, filed, start=None, form="10-K", fp="FY"):
    r = {"val": val, "end": end, "filed": filed, "form": form, "fp": fp}
    if start:
        r["start"] = start
    return r


def node(*recs, unit="USD"):
    return {"units": {unit: list(recs)}}


# ------------------------------------------------------------ _is_annual_period
def test_is_annual_period_basic():
    assert sh._is_annual_period({"end": "2023-12-31"}) is True  # instant fact
    assert sh._is_annual_period({"start": "", "end": "2023-12-31"}) is True
    assert sh._is_annual_period({"start": "2023-01-01", "end": "2023-12-31"}) is True
    assert sh._is_annual_period({"start": "2023-01-01", "end": "2023-03-31"}) is False


def test_is_annual_period_bounds():
    # exactly 300 and 400 days are included; 299 and 401 are not
    from datetime import date, timedelta

    s = date(2020, 1, 1)
    for days, ok in [(299, False), (300, True), (400, True), (401, False)]:
        assert (
            sh._is_annual_period(
                {"start": s.isoformat(), "end": (s + timedelta(days=days)).isoformat()}
            )
            is ok
        )
    assert sh._is_annual_period({"start": "bad", "end": "2020-01-01"}) is False
    assert sh._is_annual_period({"start": "2020-01-01"}) is False  # missing end


# ------------------------------------------------------------ _series
def test_series_filters_and_earliest_filing_wins():
    facts = {
        "us-gaap": {
            "A": node(
                f(1, "2022-12-31", "2023-02-01", "2022-01-01"),
                f(2, "2022-12-31", "2023-01-15", "2022-01-01"),  # earlier filing wins
                f(3, "2022-12-31", "2024-02-01", "2022-01-01"),
                f(9, "2021-12-31", "2022-02-01", "2021-01-01", form="10-Q"),
                f(9, "2021-12-31", "2022-02-01", "2021-01-01", fp="Q1"),
                f(None, "2021-12-31", "2022-02-01", "2021-01-01"),
                f(9, "2021-12-31", "2022-02-01", "2021-10-01"),  # quarter label lie
                f(5, "2020-12-31", None, "2020-01-01"),  # no filed -> sentinel
            )
        }
    }
    s = sh._series(facts, ["A"])
    assert s == {"2022-12-31": (2, "2023-01-15"), "2020-12-31": (5, "9999-99-99")}
    # a real filing beats the sentinel
    facts["us-gaap"]["A"]["units"]["USD"].append(f(7, "2020-12-31", "2021-03-01", "2020-01-01"))
    assert sh._series(facts, ["A"])["2020-12-31"] == (7, "2021-03-01")
    # equal filing date keeps the first seen (strict <)
    facts2 = {
        "us-gaap": {
            "A": node(f(1, "2022-12-31", "d", "2022-01-01"), f(2, "2022-12-31", "d", "2022-01-01"))
        }
    }
    assert sh._series(facts2, ["A"])["2022-12-31"][0] == 1


def test_series_concept_fallback_and_empty():
    facts = {
        "us-gaap": {
            "A": node(f(1, "2022-12-31", "x", "2022-01-01", form="10-Q")),  # present but unusable
            "B": node(f(2, "2022-12-31", "x", "2022-01-01")),
            "C": node(f(3, "2022-12-31", "x", "2022-01-01")),
        }
    }
    assert sh._series(facts, ["Z", "A", "B", "C"]) == {"2022-12-31": (2, "x")}
    assert sh._series(facts, ["Z"]) == {}
    assert sh._series({}, ["A"]) == {}
    assert sh._series({"us-gaap": {"A": {}}}, ["A"]) == {}


def test_series_multiple_units_merge():
    facts = {
        "us-gaap": {
            "A": {
                "units": {
                    "USD": [f(1, "2022-12-31", "b", "2022-01-01")],
                    "shares": [f(2, "2022-12-31", "a", "2022-01-01")],
                }
            }
        }
    }
    assert sh._series(facts, ["A"]) == {"2022-12-31": (2, "a")}


# ------------------------------------------------------------ collect_ticker
class Resp:
    def __init__(self, status=200, payload=None):
        self.status_code, self._p = status, payload

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("http")


class Sess:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, headers=None, timeout=None):
        self.calls.append((url, timeout))
        for k, v in self.routes.items():
            if k in url:
                if isinstance(v, Exception):
                    raise v
                return v
        raise AssertionError(url)


def ann(val, year, filed):
    return f(val, f"{year}-12-31", filed, f"{year}-01-01")


def inst(val, year, filed):
    return f(val, f"{year}-12-31", filed)


def test_collect_ticker_non_200():
    s = Sess({"CIK1": Resp(404)})
    assert sh.collect_ticker(s, "A", "1") == []
    assert s.calls[0] == ("https://data.sec.gov/api/xbrl/companyfacts/CIK1.json", 30)


def test_collect_ticker_rows_and_derivations():
    facts = {
        "us-gaap": {
            "Revenues": node(ann(100, 2022, "2023-02-01"), ann(200, 2023, "2024-02-01")),
            "CostOfRevenue": node(
                ann(60, 2022, "2023-02-01"), ann(250, 2023, "2024-02-01")
            ),  # > revenue
            "Assets": node(inst(500, 2022, "2023-02-02"), inst(600, 2023, "2024-02-02")),
            "NetIncomeLoss": node(ann(10, 2022, "2023-02-03")),
        }
    }
    rows = sh.collect_ticker(Sess({"CIK1": Resp(200, {"facts": facts})}), "AAA", "1")
    assert [r["fy_end"] for r in rows] == ["2022-12-31", "2023-12-31"]
    a, b = rows
    assert (
        a["ticker"] == "AAA"
        and a["revenue"] == 100
        and a["total_assets"] == 500
        and a["net_income"] == 10
    )
    assert a["revenue_suspect"] is False and b["revenue_suspect"] is True
    assert a["gross_profit"] == 40 and a["gross_profit_derived"] is True
    assert b["gross_profit"] == -50
    assert "net_income" not in b
    # `filed` is the first present field in CONCEPTS order (net_income precedes revenue)
    assert a["filed"] == "2023-02-03" and b["filed"] == "2024-02-02"


def test_collect_ticker_direct_gross_profit_wins_and_no_suspect_when_rev_zero():
    facts = {
        "us-gaap": {
            "Revenues": node(ann(0, 2022, "d")),
            "CostOfRevenue": node(ann(5, 2022, "d")),
            "GrossProfit": node(ann(0, 2022, "d")),
            "Liabilities": node(inst(7, 2022, "d")),
        }
    }
    (r,) = sh.collect_ticker(Sess({"CIK1": Resp(200, {"facts": facts})}), "A", "1")
    assert "revenue_suspect" not in r  # revenue must be > 0
    assert r["gross_profit"] == 0 and "gross_profit_derived" not in r  # 0 is a value, not missing
    assert r["total_liabilities"] == 7
    # equal revenue and cost is not suspect (strict >)
    facts["us-gaap"]["CostOfRevenue"] = node(ann(0, 2022, "d"))
    facts["us-gaap"]["Revenues"] = node(ann(5, 2022, "d"))
    facts["us-gaap"]["CostOfRevenue"] = node(ann(5, 2022, "d"))
    (r,) = sh.collect_ticker(Sess({"CIK1": Resp(200, {"facts": facts})}), "A", "1")
    assert r["revenue_suspect"] is False


def test_collect_ticker_no_facts():
    assert sh.collect_ticker(Sess({"CIK1": Resp(200, {})}), "A", "1") == []
    assert sh.collect_ticker(Sess({"CIK1": Resp(200, {"facts": {}})}), "A", "1") == []


def test_collect_ticker_fallback_concept_and_quarter_trap():
    facts = {
        "us-gaap": {
            "Revenues": node(f(24, "2023-03-31", "d", "2023-01-01")),  # quarter only -> unusable
            "SalesRevenueNet": node(ann(93, 2023, "d")),
        }
    }
    (r,) = sh.collect_ticker(Sess({"CIK1": Resp(200, {"facts": facts})}), "A", "1")
    assert r["revenue"] == 93


# ------------------------------------------------------------ _ticker_cik
def test_ticker_cik():
    s = Sess({"company_tickers": Resp(200, {"0": {"ticker": "aapl", "cik_str": 320193}})})
    assert sh._ticker_cik(s) == {"AAPL": "0000320193"}
    assert s.calls[0][1] == 30
    with pytest.raises(RuntimeError):
        sh._ticker_cik(Sess({"company_tickers": Resp(500)}))


# ------------------------------------------------------------ main
@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(sh, "BASE", str(tmp_path))
    monkeypatch.setattr(sh, "OUT_DIR", str(tmp_path / "fh"))
    monkeypatch.setattr(sh, "OUT", str(tmp_path / "fh" / "US.parquet"))
    monkeypatch.setattr(sh.time, "sleep", lambda s: None)
    state = {"cik": {"AAA": "1", "BBB": "2", "CCC": "3"}, "fail": set(), "empty": set(), "seen": []}
    monkeypatch.setattr(sh.requests, "Session", lambda: object())
    monkeypatch.setattr(sh, "_ticker_cik", lambda s: state["cik"])

    def ct(sess, tk, cik):
        state["seen"].append(tk)
        if tk in state["fail"]:
            raise RuntimeError("x")
        if tk in state["empty"]:
            return []
        return [{"ticker": tk, "fy_end": "2022-12-31", **{c: 1.0 for c in sh.CONCEPTS}}]

    monkeypatch.setattr(sh, "collect_ticker", ct)
    state["tmp"] = tmp_path
    return state


def run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["x", *argv])
    sh.main()


def test_main_tickers_resume_and_unknown(env, monkeypatch, capsys):
    env["fail"].add("BBB")
    run(monkeypatch, "--tickers", "aaa", "bbb", "zzz")
    out = capsys.readouterr().out
    assert "2 tickers in universe, 0 already done, 2 to fetch" in out
    assert "2/2 | 1 with data | 1 rows" in out
    assert env["seen"] == ["AAA", "BBB"]  # ZZZ not in the CIK map -> dropped
    env["seen"].clear()
    env["fail"].clear()
    run(monkeypatch, "--tickers", "aaa", "bbb")
    assert env["seen"] == ["BBB"]
    assert set(pd.read_parquet(sh.OUT)["ticker"]) == {"AAA", "BBB"}


def test_main_limit_and_cik_universe(env, monkeypatch, capsys):
    run(monkeypatch, "--limit", "2")
    out = capsys.readouterr().out
    assert "universe from SEC ticker->CIK map: 3 filers" in out
    assert env["seen"] == ["AAA", "BBB"]


def test_main_universe_csv_then_ltm(env, monkeypatch, capsys):
    tmp = env["tmp"]
    ltm = tmp / "cache_seed" / "ltm"
    ltm.mkdir(parents=True)
    pd.DataFrame({"Symbol": ["ccc", "AAA", "NOPE"]}).to_parquet(ltm / "US.parquet")
    run(monkeypatch)
    assert "universe from LTM price panel: 3 symbols" in capsys.readouterr().out
    assert env["seen"] == ["AAA", "CCC"]
    # CSV takes precedence over LTM
    env["seen"].clear()
    (tmp / "fh" / "US.parquet").unlink()
    uni = tmp / "data" / "global_universe"
    uni.mkdir(parents=True)
    pd.DataFrame({"yf_ticker": ["bbb", "BBB"]}).to_csv(uni / "US.csv", index=False)
    run(monkeypatch)
    assert env["seen"] == ["BBB"] and "LTM" not in capsys.readouterr().out


def test_main_old_schema_triggers_refetch(env, monkeypatch, capsys):
    run(monkeypatch, "--tickers", "AAA")
    capsys.readouterr()
    prev = pd.read_parquet(sh.OUT).drop(columns=["capex"])
    prev.to_parquet(sh.OUT, index=False)
    env["seen"].clear()
    run(monkeypatch, "--tickers", "AAA")
    out = capsys.readouterr().out
    assert "concepts added since last collection (['capex'])" in out
    assert env["seen"] == ["AAA"] and "0 already done" in out
    # existing frame is not reused on re-fetch -> exactly one AAA row
    assert len(pd.read_parquet(sh.OUT)) == 1


def test_main_checkpoint_250_and_empty(env, monkeypatch, capsys):
    env["cik"] = {f"T{i:03d}": str(i) for i in range(251)}
    run(monkeypatch)
    out = capsys.readouterr().out
    assert (
        "250/251 | 250 with data | 250 rows" in out and "251/251 | 251 with data | 251 rows" in out
    )
    env["empty"].update(["AAA"])
    env["cik"] = {"AAA": "1"}
    import os

    os.remove(sh.OUT)
    run(monkeypatch)
    assert not os.path.exists(sh.OUT)
    assert "1/1 | 0 with data | 0 rows" in capsys.readouterr().out
