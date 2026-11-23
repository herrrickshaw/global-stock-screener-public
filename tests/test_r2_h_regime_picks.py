import sys
import types

import numpy as np
import pandas as pd
import pytest

import regime_picks as rp


def _df(n=250, price=100.0, vol=1000.0, slope=0.0, with_vol=True):
    close = price + slope * np.arange(n)
    d = {"Close": close}
    if with_vol:
        d["Volume"] = np.full(n, vol)
    return pd.DataFrame(d)


class _DL:
    def __init__(self, ltm=None, full=None):
        self.ltm, self.full, self.calls = ltm, full, []

    def load_market(self, m, tier=None):
        self.calls.append((m, tier))
        return self.ltm if tier == "ltm" else self.full


@pytest.fixture
def setup(monkeypatch):
    def go(data, pred=lambda f: True, feats=None, market_screen="rsi_oversold"):
        dl = _DL(ltm=data)
        monkeypatch.setattr(rp, "datalink", dl)
        monkeypatch.setattr(rp.backtest, "SCREENS", {market_screen: pred})
        monkeypatch.setattr(
            rp.backtest,
            "_pit",
            feats or (lambda close, t: {"rsi": 25.0, "pct_from_high": -3.0}),
        )
        return dl

    return go


def test_picks_no_datalink(monkeypatch):
    monkeypatch.setattr(rp, "datalink", None)
    assert rp.picks_for("US") == []


def test_picks_unknown_market_or_screen(setup, monkeypatch):
    setup({"A": _df()})
    assert rp.picks_for("XX") == []  # not in PREFERRED
    monkeypatch.setattr(rp.backtest, "SCREENS", {})
    assert rp.picks_for("US") == []


def test_picks_no_data_and_tier_fallback(setup, monkeypatch):
    dl = setup({})
    dl.full = {"A": _df()}
    out = rp.picks_for("US")
    assert [p["symbol"] for p in out] == ["A"]
    assert dl.calls == [("US", "ltm"), ("US", None)]
    dl.full = {}
    assert rp.picks_for("US") == []


def test_picks_filters_short_nan_zero_and_nofeat(setup):
    data = {
        "short": _df(219),
        "ok": _df(220),
        "none": None,
        "zero": _df(250, price=0.0),
        "nan": _df(250, price=np.nan),
        "nofeat": _df(250),
    }

    def feats(close, t):
        return None if close[0] == 100.0 and len(close) == 250 else {"rsi": 1}

    setup(data, feats=feats)
    out = rp.picks_for("US", top=10)
    assert [p["symbol"] for p in out] == ["ok"]


def test_picks_predicate_false_and_exception(setup):
    setup({"A": _df()}, pred=lambda f: False)
    assert rp.picks_for("US") == []

    def boom(f):
        raise KeyError

    setup({"A": _df()}, pred=boom)
    assert rp.picks_for("US") == []


def test_picks_metrics_and_sorting_and_top(setup):
    n = 250
    d1 = _df(n, price=50.0, vol=2_000_000.0)
    d2 = _df(n, price=10.0, vol=1_000_000.0)
    d3 = _df(n, price=10.0, vol=3_000_000.0, with_vol=True)
    setup({"A": d1, "B": d2, "C": d3})
    out = rp.picks_for("US", top=2)
    assert [p["symbol"] for p in out] == ["A", "C"]  # dvol 100M, 30M
    a = out[0]
    assert a["price"] == 50.0
    assert a["dvol_m"] == pytest.approx(100.0)
    assert a["ret1m"] == pytest.approx(0.0)
    assert a["rsi"] == 25.0 and a["from_high"] == -3.0
    assert len(rp.picks_for("US", top=3)) == 3
    assert len(rp.picks_for("US")) == 3  # default top 6


def test_picks_ret1m_and_no_volume(setup):
    n = 250
    close = np.full(n, 100.0)
    close[-1] = 110.0
    close[-1 - 21] = 100.0
    df = pd.DataFrame({"Close": close})
    setup({"A": df})
    out = rp.picks_for("US")[0]
    assert out["ret1m"] == pytest.approx(10.0)
    assert out["dvol_m"] == 0.0


def test_picks_prev_close_zero_gives_nan(setup):
    close = np.full(250, 100.0)
    close[-1 - 21] = 0.0
    setup({"A": pd.DataFrame({"Close": close})})
    assert np.isnan(rp.picks_for("US")[0]["ret1m"])


def test_enrich_trendlyne(monkeypatch):
    tl = types.ModuleType("trendlyne_auth")
    tl.session = lambda: "S"

    def scores(sym, s):
        assert s == "S"
        if sym == "B":
            raise RuntimeError
        return {"piotroski": 8, "dvm_durability": 55}

    tl.scores = scores
    monkeypatch.setitem(sys.modules, "trendlyne_auth", tl)
    picks = [{"symbol": "A"}, {"symbol": "B"}]
    out = rp._enrich_trendlyne(picks)
    assert out[0]["piotroski"] == 8 and out[0]["dvm_dur"] == 55
    assert "piotroski" not in out[1]


def test_enrich_trendlyne_session_failure(monkeypatch):
    tl = types.ModuleType("trendlyne_auth")

    def bad():
        raise RuntimeError

    tl.session = bad
    monkeypatch.setitem(sys.modules, "trendlyne_auth", tl)
    picks = [{"symbol": "A"}]
    assert rp._enrich_trendlyne(picks) == [{"symbol": "A"}]


def _pick(sym="A", price=1234.5, ret=2.0, dvol=3.0, **kw):
    return {"symbol": sym, "price": price, "ret1m": ret, "dvol_m": dvol, **kw}


def test_rows_html_basic():
    h = rp._rows_html("US", [_pick(ret=2.0), _pick("B", ret=-1.25), _pick("C", ret=float("nan"))])
    assert "<b>A</b>" in h and "$1,234.50" in h and "+2.0%" in h
    assert "-1.2%" in h or "-1.3%" in h
    assert "—" in h
    assert "$3.0M" in h
    assert "Piotroski" not in h


def test_rows_html_currency_and_quality():
    h = rp._rows_html("IN", [_pick(piotroski=7, dvm_dur=60), _pick("B", piotroski=6, dvm_dur=None)])
    assert "₹1,234.50" in h
    assert "Piotroski" in h and "Durability" in h
    assert "7 ⭐" in h
    assert "6</td>" in h and "6 ⭐" not in h
    assert "#2e7d32" in h and ">60<" in h and ">—<" in h
    assert rp._rows_html("ZZ", [_pick()]).count("1,234.50") == 1  # no currency symbol
    # piotroski None for all -> no quality columns
    assert "Piotroski" not in rp._rows_html("IN", [_pick(piotroski=None)])
    # only some picks have it -> present, None rendered as dash
    h = rp._rows_html("IN", [_pick(piotroski=None), _pick("B", piotroski=7)])
    assert "Piotroski" in h


def test_rows_html_star_boundary():
    assert "⭐" in rp._rows_html("IN", [_pick(piotroski=7)])
    assert "⭐" not in rp._rows_html("IN", [_pick(piotroski=6)])


def test_section_html(monkeypatch):
    calls = []

    def pf(m, top):
        calls.append((m, top))
        return [_pick(m)] if m != "TW" else []

    monkeypatch.setattr(rp, "picks_for", pf)
    enriched = []
    monkeypatch.setattr(rp, "_enrich_trendlyne", lambda p: enriched.append(1) or p)
    h = rp.section_html(["IN", "US", "TW", "QQ"], top=3)
    assert calls == [("IN", 3), ("US", 3), ("TW", 3), ("QQ", 3)]
    assert enriched == [1]  # only IN with picks
    assert "no current matches" in h
    assert "#c62828" in h  # mean reversion badge for US
    assert "#2e7d32" in h
    assert "Mean-reversion" in h and "Trend" in h
    assert "Screener guide" in h
    assert "Why here: ." in h  # unknown market has empty why
    assert "🇮🇳" in h and "near_high" in h
    calls.clear()
    rp.section_html()
    assert [c[0] for c in calls] == rp.DEFAULT_MARKETS
    assert calls[0][1] == 6


def test_section_html_in_without_picks_skips_enrich(monkeypatch):
    monkeypatch.setattr(rp, "picks_for", lambda m, t: [])
    monkeypatch.setattr(rp, "_enrich_trendlyne", lambda p: pytest.fail("no"))
    assert "no current matches" in rp.section_html(["IN"])


def test_section_text(monkeypatch):
    monkeypatch.setattr(
        rp, "picks_for", lambda m, t: [{"symbol": "X"}, {"symbol": "Y"}] if m == "US" else []
    )
    t = rp.section_text(["US", "IN", "ZZ"])
    assert t.splitlines() == [
        "Regime Picks (each market screened by its best-backtested filter):",
        "  US [rsi_oversold]: X, Y",
        "  IN [near_high]: none",
        "  ZZ [—]: none",
    ]
    assert rp.section_text().count("\n") == len(rp.DEFAULT_MARKETS)


def test_main(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(rp, "BASE", str(tmp_path))
    seen = []
    monkeypatch.setattr(rp, "section_text", lambda m, t: seen.append(("t", m, t)) or "TXT")
    monkeypatch.setattr(rp, "section_html", lambda m, t: seen.append(("h", m, t)) or "<p>x</p>")
    monkeypatch.setattr(sys, "argv", ["rp", "--markets", "us, in", "--top", "2"])
    rp.main()
    assert seen == [("t", ["US", "IN"], 2), ("h", ["US", "IN"], 2)]
    out = capsys.readouterr().out
    assert "TXT" in out and "html bytes: 8" in out
    assert (tmp_path / "regime_picks_section.html").read_text() == "<p>x</p>"
    seen.clear()
    monkeypatch.setattr(sys, "argv", ["rp"])
    rp.main()
    assert seen[0] == ("t", None, 6)


def test_tables_consistent():
    for m, s in rp.PREFERRED.items():
        assert s in rp.WHY
        assert m in rp.FLAG
    assert set(rp.DEFAULT_MARKETS) <= set(rp.PREFERRED)
