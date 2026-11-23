import sys

import pandas as pd
import pytest
import requests

import collect_nse_results as cn


class Resp:
    def __init__(self, status=200, payload=None, bad_json=False):
        self.status_code, self._p, self._bad = status, payload, bad_json

    def json(self):
        if self._bad:
            raise ValueError("no json")
        return self._p


class Sess:
    def __init__(self, resp=None):
        self.resp = resp
        self.urls = []

    def get(self, url, headers=None, timeout=None):
        self.urls.append((url, timeout))
        if isinstance(self.resp, Exception):
            raise self.resp
        return self.resp


def row(**kw):
    base = {
        "re_create_dt": "16-Jan-2025",
        "re_to_dt": "31-Dec-2024",
        "re_from_dt": "01-Oct-2024",
        "re_seq_num": "7",
        "re_total_inc": "10",
        "re_net_profit": "2.5",
        "re_basic_eps_for_cont_dic_opr": "3.5",
        "re_debt_eqt_rat": "0.4",
        "re_res_type": "U",
        "re_tax": "-",
        "re_int_new": "",
        "re_staff_cost": None,
        "re_depr_und_exp": "abc",
    }
    base.update(kw)
    return base


def test_session_warms_up_twice(monkeypatch):
    s = Sess(Resp())
    monkeypatch.setattr(cn.requests, "Session", lambda: s)
    assert cn._session() is s
    assert [u for u, _ in s.urls] == [
        "https://www.nseindia.com",
        "https://www.nseindia.com/option-chain",
    ]
    assert all(t == 10 for _, t in s.urls)


def test_parse_date():
    assert cn._parse_ddmonyyyy(None) is None and cn._parse_ddmonyyyy("") is None
    assert cn._parse_ddmonyyyy("16-Jan-2025") == pd.Timestamp("2025-01-16")
    assert pd.isna(cn._parse_ddmonyyyy("garbage"))
    assert pd.isna(cn._parse_ddmonyyyy("2025-01-16"))  # wrong format -> coerced


def test_parse_date_swallows_parser_errors(monkeypatch):
    monkeypatch.setattr(cn.pd, "to_datetime", lambda *a, **k: (_ for _ in ()).throw(TypeError("x")))
    assert cn._parse_ddmonyyyy("16-Jan-2025") is None


def test_fetch_ticker_errors_and_empties():
    assert cn.fetch_ticker(Sess(requests.ConnectionError()), "X") == []
    for code in (401, 403):
        with pytest.raises(cn.Blocked, match=str(code)):
            cn.fetch_ticker(Sess(Resp(code)), "X")
    for code in (404, 429, 500, 204):
        assert cn.fetch_ticker(Sess(Resp(code)), "X") == []
    with pytest.raises(cn.Blocked, match="non-JSON"):
        cn.fetch_ticker(Sess(Resp(200, bad_json=True)), "X")
    for p in (None, [], {}, {"resCmpData": []}, {"resCmpData": None}, ["x"]):
        assert cn.fetch_ticker(Sess(Resp(200, p)), "X") == []


def test_fetch_ticker_url_and_values():
    s = Sess(Resp(200, {"resCmpData": [row()]}))
    out = cn.fetch_ticker(s, "RELIANCE")
    assert s.urls[0] == ("https://www.nseindia.com/api/results-comparision?symbol=RELIANCE", 10)
    (r,) = out
    assert r["ticker"] == "RELIANCE" and r["seq_num"] == "7"
    assert r["filed"] == pd.Timestamp("2025-01-16") and r["fy_end"] == pd.Timestamp("2024-12-31")
    assert r["fy_start"] == pd.Timestamp("2024-10-01")
    assert r["revenue"] == 10 * 100_000 and r["net_income"] == 2.5 * 100_000  # lakh -> units
    assert r["eps"] == 3.5 and r["debt_to_equity"] == 0.4  # ratios not scaled
    assert r["result_type"] == "U"  # passthrough string
    assert r["tax"] is None and r["interest_expense"] is None and r["staff_cost"] is None
    assert r["depreciation"] == "abc"  # unparseable kept raw
    assert r["pretax_income"] is None  # absent key
    assert set(cn.FIELD_MAP.values()) <= set(r)


def test_fetch_ticker_ratio_fields_unscaled_and_zero_kept():
    p = {
        "resCmpData": [
            row(re_face_val="10", re_int_ser_cov="2", re_debt_ser_cov="1.5", re_net_sale="0")
        ]
    }
    (r,) = cn.fetch_ticker(Sess(Resp(200, p)), "X")
    assert (
        r["face_value"] == 10.0
        and r["interest_coverage"] == 2.0
        and r["debt_service_coverage"] == 1.5
    )
    assert r["net_sale"] == 0.0  # "0" is a value, not missing


def test_fetch_ticker_skips_rows_without_dates():
    p = {"resCmpData": [row(re_create_dt=None), row(re_to_dt=""), row(re_seq_num="ok")]}
    out = cn.fetch_ticker(Sess(Resp(200, p)), "X")
    assert [r["seq_num"] for r in out] == ["ok"]


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="collect_nse_results.py:~147: _parse_ddmonyyyy returns NaT (not None) for an unparseable "
    "date, so `filed is None or fy_end is None` never skips the row and a NaT filing date leaks in",
)
def test_fetch_ticker_skips_unparseable_dates():
    p = {"resCmpData": [row(re_create_dt="garbage"), row(re_seq_num="ok")]}
    out = cn.fetch_ticker(Sess(Resp(200, p)), "X")
    assert [r["seq_num"] for r in out] == ["ok"]


# ---------------------------------------------------------------- _universe
def test_universe_liquidity_and_filters(tmp_path, monkeypatch):
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    recs = []
    spec = {  # symbol: (close, volume)
        "liquid": (100, 100_000),  # 1e7 exactly -> included (>=)
        "justunder": (100, 99_999),
        "GOLDBEES": (1000, 10_000_000),  # ETF hint
        "NIFTYETF": (1000, 10_000_000),
        "abc": (50, 1_000_000),  # lower-case normalised
        "07AGG": (1000, 10_000_000),
    }
    for s, (c, v) in spec.items():
        for d in dates:
            recs.append((s, c, v, d))
    # liquid only in the old window (before cutoff) -> excluded
    for d in dates[:50]:
        recs.append(("stale", 1000, 10_000_000, d))
    pd.DataFrame(recs, columns=["Symbol", "Close", "Volume", "Date"]).to_parquet(
        tmp_path / "ltm.parquet"
    )
    monkeypatch.setattr(cn, "LTM", str(tmp_path / "ltm.parquet"))
    import screener_history_collector as sh

    monkeypatch.setattr(sh, "is_non_equity", lambda s: s.startswith("07"))
    assert cn._universe(0) == ["ABC", "LIQUID"]
    assert cn._universe(1) == ["ABC"]


# ---------------------------------------------------------------- main
@pytest.fixture
def env(tmp_path, monkeypatch):
    out_dir = tmp_path / "fh"
    monkeypatch.setattr(cn, "OUT_DIR", str(out_dir))
    monkeypatch.setattr(cn, "OUT", str(out_dir / "IN_nse_results.parquet"))
    monkeypatch.setattr(cn, "BASE", str(tmp_path))
    monkeypatch.setattr(cn.time, "sleep", lambda s: None)
    sessions = []
    monkeypatch.setattr(cn, "_session", lambda: sessions.append(1) or object())
    return sessions


def go(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["x", *argv])
    cn.main()


def rec(t):
    return {"ticker": t, "filed": pd.Timestamp("2025-01-01"), "revenue": 1.0}


def test_main_tickers_fetch_and_resume(env, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        cn, "fetch_ticker", lambda s, t: calls.append(t) or ([rec(t)] if t != "BB" else [])
    )
    go(monkeypatch, "--tickers", "aa", "bb")
    out = capsys.readouterr().out
    assert "2 tickers, 0 already done, 2 to fetch (~0 min at 1.2s/req)" in out
    assert "2/2 | 1 with data | 1 rows" in out and "done -> " in out
    assert calls == ["AA", "BB"]
    df = pd.read_parquet(cn.OUT)
    assert list(df["ticker"]) == ["AA"]
    calls.clear()
    go(monkeypatch, "--tickers", "AA", "CC")
    assert calls == ["CC"] and "1 already done, 1 to fetch" in capsys.readouterr().out
    assert set(pd.read_parquet(cn.OUT)["ticker"]) == {"AA", "CC"}


def test_main_universe_used_when_no_tickers(env, monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(cn, "_universe", lambda n: seen.append(n) or ["Z"])
    monkeypatch.setattr(cn, "fetch_ticker", lambda s, t: [rec(t)])
    go(monkeypatch, "--limit", "5")
    assert seen == [5]
    assert "1 tickers" in capsys.readouterr().out


def test_main_checkpoint_every_50(env, monkeypatch, capsys):
    monkeypatch.setattr(cn, "fetch_ticker", lambda s, t: [rec(t)])
    go(monkeypatch, "--tickers", *[f"T{i:03d}" for i in range(51)])
    out = capsys.readouterr().out
    assert "  50/51 | 50 with data | 50 rows" in out and "  51/51 | 51 with data | 51 rows" in out


def test_main_block_retry_succeeds(env, monkeypatch, capsys):
    n = {"c": 0}

    def ft(s, t):
        n["c"] += 1
        if n["c"] == 1:
            raise cn.Blocked("HTTP 403")
        return [rec(t)]

    monkeypatch.setattr(cn, "fetch_ticker", ft)
    go(monkeypatch, "--tickers", "A")
    out = capsys.readouterr().out
    assert "possible block (HTTP 403) at A" in out and "ABORT" not in out
    assert len(env) == 2  # initial + refreshed session
    assert list(pd.read_parquet(cn.OUT)["ticker"]) == ["A"]


def test_main_double_block_aborts_and_checkpoints(env, monkeypatch, capsys):
    def ft(s, t):
        if t == "B":
            raise cn.Blocked("HTTP 401")
        return [rec(t)]

    monkeypatch.setattr(cn, "fetch_ticker", ft)
    go(monkeypatch, "--tickers", "A", "B", "C")
    out = capsys.readouterr().out
    assert (
        "ABORT: still blocked after session refresh (HTTP 401) at B. Collected 1 before this."
        in out
    )
    assert list(pd.read_parquet(cn.OUT)["ticker"]) == ["A"]  # partial saved, C never tried
    assert "done -> " in out


def test_main_consecutive_empty_breaker(env, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        cn, "fetch_ticker", lambda s, t: calls.append(t) or ([] if t != "T00" else [rec(t)])
    )
    names = [f"T{i:02d}" for i in range(60)]
    go(monkeypatch, "--tickers", *names)
    out = capsys.readouterr().out
    # T00 ok, then 40 empties (T01..T40) trips; T41 never fetched
    assert len(calls) == 41 and calls[-1] == "T40"
    assert "ABORT: 40 consecutive tickers returned nothing" in out and "Collected 1 before" in out
    assert len(pd.read_parquet(cn.OUT)) == 1


def test_main_39_empty_does_not_abort_and_success_resets(env, monkeypatch, capsys):
    names = [f"T{i:02d}" for i in range(80)]
    ok = {"T39", "T00"}  # 39 empties between T00 and T40? ensure reset on success
    monkeypatch.setattr(cn, "fetch_ticker", lambda s, t: [rec(t)] if t in ok else [])
    # T01..T38 empty (38) -> T39 ok resets -> T40..T78 empty (39) -> T79 would be 40th
    go(monkeypatch, "--tickers", *names[:79])
    assert "ABORT" not in capsys.readouterr().out


def test_main_nothing_collected_writes_nothing(env, monkeypatch, capsys):
    monkeypatch.setattr(cn, "fetch_ticker", lambda s, t: [])
    go(monkeypatch, "--tickers", "A")
    import os

    assert not os.path.exists(cn.OUT)
    assert "1/1 | 0 with data | 0 rows" in capsys.readouterr().out
