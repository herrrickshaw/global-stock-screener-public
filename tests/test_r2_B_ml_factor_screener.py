from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.r2b_support import load_with_fake_sklearn, make_prices

SRC = str(Path(__file__).resolve().parent.parent / "ml_factor_screener.py")


@pytest.fixture
def mf(monkeypatch, tmp_path):
    mod = load_with_fake_sklearn(SRC, "mlf_r2", monkeypatch)
    monkeypatch.setattr(mod, "BASE", str(tmp_path))
    monkeypatch.setattr(mod, "LTM", str(tmp_path / "cache_seed" / "ltm"))
    monkeypatch.setattr(mod, "OUT", str(tmp_path / "data" / "curated" / "picks.parquet"))
    monkeypatch.setattr(mod, "WEIGHTS_OUT", str(tmp_path / "data" / "curated" / "weights.parquet"))
    (tmp_path / "cache_seed" / "ltm").mkdir(parents=True)
    return mod


@pytest.fixture(scope="module")
def big():
    return make_prices(n_sym=90, start="2023-01-02", end="2024-12-31", short=0)


PANDAS3 = int(pd.__version__.split(".")[0]) >= 3


@pytest.fixture
def compat(mf, monkeypatch):
    """`_panel` rebuilt without groupby.apply so the downstream logic is testable on pandas 3
    (where the real `_panel` loses the grouping column -- see test_panel_all_and_sampled)."""
    real = mf._panel

    def panel(market, sample):
        p = Path(mf.LTM) / f"{market}.parquet"
        if not p.exists():
            return None
        d = pd.read_parquet(p, columns=["Date", "Close", "Volume", "Symbol"])
        d["Date"] = pd.to_datetime(d["Date"])
        syms = d["Symbol"].drop_duplicates()
        if sample and len(syms) > sample:
            syms = syms.sample(sample, random_state=7)
            d = d[d["Symbol"].isin(set(syms))]
        d = d.sort_values(["Symbol", "Date"])
        parts = [mf._add_factors(g.copy()) for _, g in d.groupby("Symbol")]
        return pd.concat(parts) if parts else d

    monkeypatch.setattr(mf, "_panel", panel)
    return real


def _write(mf, market, df):
    df.to_parquet(Path(mf.LTM) / f"{market}.parquet", index=False)


# ---- _rsi / _add_factors ---------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="source bug: ml_factor_screener.py:_rsi replaces a zero average loss with NaN, so a "
    "pure up-trend yields RSI=NaN instead of 100 (such rows then vanish from training)",
)
def test_rsi_all_gains_is_100(mf):
    assert mf._rsi(pd.Series(np.arange(1.0, 40.0))).iloc[-1] == 100.0


def test_rsi_extremes(mf):
    flat = pd.Series([5.0] * 40)
    assert np.isnan(mf._rsi(flat).iloc[-1])  # 0/0 -> NaN
    down = pd.Series(np.arange(40.0, 1.0, -1))
    assert mf._rsi(down).iloc[-1] == pytest.approx(0.0)


def test_rsi_value(mf):
    c = pd.Series(
        [10, 11, 10.5, 11.5, 12, 11, 12.5, 13, 12, 13.5, 14, 13, 14.5, 15, 14, (15.5).__float__()]
    )
    d = c.diff()
    up = d.clip(lower=0).rolling(14).mean().iloc[-1]
    dn = (-d.clip(upper=0)).rolling(14).mean().iloc[-1]
    assert mf._rsi(c).iloc[-1] == pytest.approx(100 - 100 / (1 + up / dn))
    assert mf._rsi(c, n=5).iloc[-1] != mf._rsi(c).iloc[-1]


def test_add_factors(mf):
    n = 300
    c = pd.Series(10.0 * 1.002 ** np.arange(n))
    g = pd.DataFrame({"Close": c, "Volume": 100.0})
    out = mf._add_factors(g)
    assert out["mom_63"].iloc[-1] == pytest.approx(c.iloc[-1] / c.iloc[-64] - 1)
    assert out["mom_21"].iloc[-1] == pytest.approx(c.iloc[-1] / c.iloc[-22] - 1)
    assert out["trend"].iloc[-1] == pytest.approx(c.iloc[-1] / c.iloc[-200:].mean() - 1)
    assert out["near_high"].iloc[-1] == pytest.approx(0.0)
    assert out["near_low"].iloc[-1] == pytest.approx(c.iloc[-1] / c.iloc[-126:].min() - 1)
    assert out["vol21"].iloc[-1] == pytest.approx(c.pct_change().iloc[-21:].std() * np.sqrt(252))
    assert out["dvol"].iloc[-1] == pytest.approx((c * 100.0).iloc[-63:].mean())
    assert np.isnan(out["trend"].iloc[198]) and not np.isnan(out["trend"].iloc[199])
    assert np.isnan(out["near_high"].iloc[124]) and not np.isnan(out["near_high"].iloc[125])


# ---- _panel ------------------------------------------------------------------------------
def test_panel_missing(mf):
    assert mf._panel("NOPE", 0) is None


@pytest.mark.xfail(
    PANDAS3,
    strict=True,
    raises=KeyError,
    reason="source bug (pandas>=3): ml_factor_screener.py:_panel uses groupby('Symbol').apply, "
    "which no longer passes the grouping column through, so every later d['Symbol'] raises",
)
def test_panel_all_and_sampled(mf, big):
    _write(mf, "AA", big)
    full = mf._panel("AA", 0)
    assert full["Symbol"].nunique() == 90 and "mom_63" in full.columns
    assert full["Date"].dtype.kind == "M"
    s = mf._panel("AA", 30)
    assert s["Symbol"].nunique() == 30
    # deterministic sample (random_state=7)
    assert set(s["Symbol"]) == set(mf._panel("AA", 30)["Symbol"])
    # sample >= number of symbols -> all kept
    assert mf._panel("AA", 90)["Symbol"].nunique() == 90
    assert mf._panel("AA", 500)["Symbol"].nunique() == 90


# ---- train_weights -----------------------------------------------------------------------
def test_train_weights_ok(compat, mf, big):
    _write(mf, "AA", big)
    fit = mf.train_weights("AA", 21, sample=500)
    assert set(fit) == {"weights", "mean", "scale", "ic", "n"}
    assert list(fit["weights"]) == mf.FACTORS
    assert fit["n"] >= 5000
    assert abs(fit["ic"]) <= 1.0
    assert max(abs(v) for v in fit["weights"].values()) <= 0.5


def test_train_weights_missing_market(compat, mf):
    assert mf.train_weights("NOPE", 21) is None


def test_train_weights_empty_frame(compat, mf, big):
    _write(mf, "AA", big.iloc[0:0])
    assert mf.train_weights("AA", 21) is None


def test_train_weights_too_few_symbols(compat, mf, big):
    keep = sorted(big["Symbol"].unique())[:59]
    _write(mf, "AA", big[big["Symbol"].isin(keep)])
    assert mf.train_weights("AA", 21) is None  # 59 < 60 symbols
    keep = sorted(big["Symbol"].unique())[:60]
    _write(mf, "AB", big[big["Symbol"].isin(keep)])
    assert mf.train_weights("AB", 21) is not None


def test_train_weights_too_few_rows(compat, mf, big):
    _write(mf, "AA", big[big["Date"] < "2023-12-01"])
    # ~230 rows/symbol minus 200 warm-up -> too few training rows
    assert mf.train_weights("AA", 21) is None


def test_train_weights_rejects_large_coef(compat, mf, big, monkeypatch):
    _write(mf, "AA", big)

    class Huge(type(mf.Ridge(alpha=1.0))):
        def fit(self, X, y):
            super().fit(X, y)
            self.coef_ = np.full(X.shape[1], 0.51)
            return self

    monkeypatch.setattr(mf, "Ridge", Huge)
    assert mf.train_weights("AA", 21) is None

    # exactly 0.5 is accepted (strict >)
    class Edge(Huge):
        def fit(self, X, y):
            super().fit(X, y)
            self.coef_ = np.full(X.shape[1], 0.5)
            return self

    monkeypatch.setattr(mf, "Ridge", Edge)
    assert mf.train_weights("AA", 21) is not None


def test_train_weights_horizon_shifts_target(compat, mf, big):
    _write(mf, "AA", big)
    a = mf.train_weights("AA", 5)
    b = mf.train_weights("AA", 63)
    assert a["n"] > b["n"]  # longer horizon drops more trailing rows


# ---- score_current -------------------------------------------------------------------------
def _fit(mf):
    return {
        "weights": {f: (1.0 if f == "mom_21" else 0.0) for f in mf.FACTORS},
        "mean": {f: 0.0 for f in mf.FACTORS},
        "scale": {f: 1.0 for f in mf.FACTORS},
    }


def test_score_current_missing(compat, mf):
    out = mf.score_current("NOPE", _fit(mf), 5)
    assert out.empty


def test_score_current_tiers_and_topn(compat, mf, big):
    _write(mf, "AA", big)
    out = mf.score_current("AA", _fit(mf), 3)
    assert (
        list(out.columns)
        == ["market", "liq_tier", "Symbol", "Date", "Close", "dvol", "score"] + mf.FACTORS
    )
    assert set(out["liq_tier"]) == {"High", "Medium", "Low"}
    assert (out.groupby("liq_tier").size() <= 3).all() and len(out) == 9
    assert (out["market"] == "AA").all()
    # score == mom_21 (weight 1, z identity)
    assert np.allclose(out["score"], out["mom_21"])
    # tiers split on dvol terciles; higher tier has higher dvol
    m = out.groupby("liq_tier")["dvol"].min()
    assert m["High"] > m["Low"]
    # within each tier the picks are the highest-score names
    full = mf._panel("AA", 0)
    assert full["Symbol"].nunique() == 90
    latest = full.dropna(subset=mf.FACTORS + ["dvol"]).groupby("Symbol").tail(1)
    q1, q2 = latest["dvol"].quantile([1 / 3, 2 / 3])
    hi = latest[latest["dvol"] >= q2].sort_values("mom_21", ascending=False).head(3)
    assert set(out[out["liq_tier"] == "High"]["Symbol"]) == set(hi["Symbol"])
    lo = latest[latest["dvol"] < q1]
    assert set(out[out["liq_tier"] == "Low"]["Symbol"]) <= set(lo["Symbol"])
    med = latest[(latest["dvol"] >= q1) & (latest["dvol"] < q2)]
    assert set(out[out["liq_tier"] == "Medium"]["Symbol"]) <= set(med["Symbol"])


def test_score_current_all_nan_returns_empty(compat, mf, big):
    # too short for a 200-day trend -> no complete rows
    _write(mf, "AA", big[big["Date"] < "2023-06-01"])
    assert mf.score_current("AA", _fit(mf), 3).empty


# ---- main --------------------------------------------------------------------------------------
def _fake_fit(mf, ic=0.0123):
    return {
        "weights": {f: 0.001 * (i + 1) for i, f in enumerate(mf.FACTORS)},
        "mean": {},
        "scale": {},
        "ic": ic,
        "n": 7777,
    }


def _picks(mf, market):
    rows = []
    for tier in ("High", "Medium", "Low"):
        for k in range(7):
            r = {
                "market": market,
                "liq_tier": tier,
                "Symbol": f"{tier[0]}{k}",
                "Date": pd.Timestamp("2024-12-31"),
                "Close": 1.0,
                "dvol": 1.0,
                "score": 1.0 - 0.1 * k,
            }
            r.update({f: 0.0 for f in mf.FACTORS})
            rows.append(r)
    return pd.DataFrame(rows)


def test_main_all_markets(mf, monkeypatch, capsys, tmp_path):
    for m in ("US", "IN", "EU"):
        (Path(mf.LTM) / f"{m}.parquet").write_bytes(b"x")
    seen = []
    monkeypatch.setattr(
        mf,
        "train_weights",
        lambda m, h, s: seen.append((m, h, s)) or (None if m == "EU" else _fake_fit(mf)),
    )
    monkeypatch.setattr(
        mf,
        "score_current",
        lambda m, fit, top: _picks(mf, m).head(0) if m == "IN" else _picks(mf, m),
    )
    monkeypatch.setattr("sys.argv", ["x"])
    mf.main()
    out = capsys.readouterr().out
    assert [s[0] for s in seen] == ["EU", "IN", "US"]  # sorted glob
    assert seen[0][1:] == (21, 500)
    assert "EU     (insufficient data)" in out
    assert "IN    0.012" in out and "+1.00" in out and "+7.00" in out
    assert "═══ High liquidity movers (top 5/market) ═══" in out
    assert "US   H0(+1.00)  H1(+0.90)  H2(+0.80)  H3(+0.70)  H4(+0.60)" in out
    assert "H5" not in out
    assert out.index("High liquidity") < out.index("Medium liquidity") < out.index("Low liquidity")
    assert "picks → data/curated/picks.parquet (21 rows · 1 markets · High/Med/Low tiers)" in out
    w = pd.read_parquet(Path(mf.WEIGHTS_OUT))
    assert list(w["market"]) == ["IN", "US"] and w.loc[0, "n"] == 7777
    assert w.loc[0, "w_mom_63"] == pytest.approx(0.001)
    assert len(pd.read_parquet(Path(mf.OUT))) == 21


def test_main_options(mf, monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(
        mf, "train_weights", lambda m, h, s: seen.append((m, h, s)) or _fake_fit(mf)
    )
    monkeypatch.setattr(
        mf, "score_current", lambda m, fit, top: seen.append(("top", top)) or _picks(mf, m)
    )
    monkeypatch.setattr(
        "sys.argv", ["x", "--markets", "us, in", "--horizon", "63", "--top", "3", "--sample", "9"]
    )
    mf.main()
    out = capsys.readouterr().out
    assert seen[:2] == [("US", 63, 9), ("top", 3)]
    assert "63d horizon" in out
    assert "top 3/market" in out and "US   H0(+1.00)  H1(+0.90)  H2(+0.80)" in out
    assert "H3" not in out


def test_main_weights_only(mf, monkeypatch, capsys):
    monkeypatch.setattr(mf, "train_weights", lambda m, h, s: _fake_fit(mf))
    monkeypatch.setattr(mf, "score_current", lambda *a: pytest.fail("no picks in weights-only"))
    monkeypatch.setattr("sys.argv", ["x", "--markets", "US", "--weights-only"])
    mf.main()
    out = capsys.readouterr().out
    assert "weights → data/curated/weights.parquet" in out
    assert "movers" not in out
    assert not Path(mf.OUT).exists()


def test_main_no_markets_no_files(mf, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["x"])
    mf.main()
    out = capsys.readouterr().out
    assert "weights ×1000" in out
    assert not Path(mf.WEIGHTS_OUT).exists() and not Path(mf.OUT).exists()


def test_main_all_insufficient(mf, monkeypatch, capsys):
    monkeypatch.setattr(mf, "train_weights", lambda *a: None)
    monkeypatch.setattr("sys.argv", ["x", "--markets", "A,B"])
    mf.main()
    out = capsys.readouterr().out
    assert out.count("(insufficient data)") == 2 and not Path(mf.WEIGHTS_OUT).exists()
