import pandas as pd
import pytest

import reward_screener_all as A


def _res(robust=True, test=True, chosen=None):
    return {
        "tickers": 100,
        "n_panel": 5000,
        "train": {"med": 0.05, "win": 0.6, "n": 10},
        "test": {"med": 0.03, "win": 0.55, "n": 7} if test else None,
        "chosen": chosen if chosen is not None else [("mom", ">=", 1.0, 0.3)],
        "per_yr": 12.6,
        "robust": robust,
    }


def test_fmt():
    assert A._fmt(None) == "   —   "
    assert A._fmt({"med": 0.0312, "win": 0.555}) == " +3.1% / 56%"
    assert A._fmt({"med": -0.02, "win": 0.4}) == " -2.0% / 40%"


def test_screen_str():
    assert A._screen_str([]) == "(none robust)"
    assert A._screen_str(None) == "(none robust)"
    s = A._screen_str([("mom", ">=", 1, 0.3), ("vol", "<=", 2, 0.25)])
    assert s == "mom↑30% & vol↓25%"


@pytest.fixture
def run(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(A.R, "BASE", str(tmp_path))
    calls = []

    def go(fake, argv=()):
        def disc(mkt, src, h, split, mf):
            calls.append((mkt, src, h, split, mf))
            return fake(mkt)

        monkeypatch.setattr(A.R, "discover", disc)
        monkeypatch.setattr("sys.argv", ["x", *argv])
        A.main()
        return capsys.readouterr().out

    go.calls = calls
    return go


def test_main_all_markets(run, tmp_path):
    out = run(lambda m: _res(robust=m in ("US", "JP")))
    assert len(run.calls) == 20
    assert run.calls[0] == ("US", "local", 252, 2021, 3)
    assert run.calls[1] == ("CN", "warehouse", 252, 2023, 3)
    assert "ROBUST out-of-sample: 2/20 markets — US, JP" in out
    df = pd.read_csv(tmp_path / "reports" / "reward_screeners_252d.csv")
    assert len(df) == 20 and df.loc[0, "per_yr"] == 13
    assert "written → reports/reward_screeners_252d.csv" in out


def test_main_args_and_skips(run, tmp_path):
    def fake(m):
        if m == "CN":
            raise SystemExit("no data")
        return _res(robust=False, test=False, chosen=[])

    out = run(fake, ["--horizon", "63", "--max-filters", "2", "--wh-split", "2020"])
    assert "CN   skipped: no data" in out
    assert run.calls[2] == ("JP", "warehouse", 63, 2020, 2)
    assert "ROBUST out-of-sample: 0/19 markets — none" in out
    assert "(none robust)" in out
    df = pd.read_csv(tmp_path / "reports" / "reward_screeners_63d.csv")
    assert len(df) == 19
    assert (df["oos_n"] == 0).all() and df["oos_med"].isna().all()
