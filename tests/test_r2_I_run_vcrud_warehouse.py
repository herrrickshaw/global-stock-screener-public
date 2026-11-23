import pathlib
import subprocess
import sys
import types

import pandas as pd
import pytest

from tests._r2i import fresh


def _stub(monkeypatch, name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    monkeypatch.setitem(sys.modules, name, m)
    return m


# ---------------- run.py ----------------
@pytest.fixture
def run(monkeypatch, tmp_path):
    m = fresh(monkeypatch, tmp_path, "run", {"BHAV_CACHE": tmp_path / "bc"})
    monkeypatch.setattr(m, "RESULTS", tmp_path / "res")
    (tmp_path / "res").mkdir()
    return m


def _argv(monkeypatch, *a):
    monkeypatch.setattr(sys, "argv", ["run.py", *a])


def test_bhav_cache_default(monkeypatch, tmp_path):
    monkeypatch.setenv("BHAV_CACHE", "x")
    monkeypatch.delenv("BHAV_CACHE")
    m = fresh(monkeypatch, tmp_path, "run")
    import os

    assert os.environ["BHAV_CACHE"] == str(tmp_path / "Downloads" / "data" / "bhavcopy_cache")
    assert m.os.environ["BHAV_CACHE"]


def test_bhav_cache_colab(monkeypatch, tmp_path):
    monkeypatch.setenv("BHAV_CACHE", "x")
    monkeypatch.delenv("BHAV_CACHE")
    real = pathlib.Path.exists
    monkeypatch.setattr(pathlib.Path, "exists", lambda self: str(self) == "/content" or real(self))
    fresh(monkeypatch, tmp_path, "run")
    import os

    assert os.environ["BHAV_CACHE"] == "/content/cache"


def test_bhav_cache_respects_env(run):
    import os

    assert os.environ["BHAV_CACHE"].endswith("bc")


def test_ensure_ready_store_populated(run, monkeypatch):
    kit = _stub(monkeypatch, "screener_kit", bootstrap=lambda verbose: pytest.fail("no bootstrap"))
    _stub(monkeypatch, "bhavcopy_store", info=lambda: {"symbols": 3})
    assert run._ensure_ready() is kit


def test_ensure_ready_bootstraps(run, monkeypatch, capsys):
    calls = []
    kit = _stub(monkeypatch, "screener_kit", bootstrap=lambda verbose: calls.append(verbose))
    _stub(monkeypatch, "bhavcopy_store", info=lambda: {"symbols": 0})
    assert run._ensure_ready() is kit
    assert calls == [True]
    assert "First run" in capsys.readouterr().out
    # info raising also bootstraps; quiet prints nothing
    _stub(monkeypatch, "bhavcopy_store", info=lambda: 1 / 0)
    assert run._ensure_ready(verbose=False) is kit
    assert calls == [True, False]
    assert capsys.readouterr().out == ""


def test_main_list(run, monkeypatch, capsys):
    _stub(monkeypatch, "strategies", STRATEGIES={"darvas": 1, "garp": 2})
    _stub(monkeypatch, "screener_kit", MARKETS=["IN", "US"])
    _argv(monkeypatch, "--list")
    run.main()
    out = capsys.readouterr().out
    assert "Strategies: darvas, garp" in out and "Markets:    IN, US" in out


class _Kit:
    def __init__(self, df=None):
        self.df = df
        self.args = None

    def screen(self, slug, market, top, min_turnover_usd):
        self.args = (slug, market, top, min_turnover_usd)
        return self.df


def test_main_brief(run, monkeypatch, capsys, tmp_path):
    _stub(monkeypatch, "strategies", STRATEGIES={})
    monkeypatch.setattr(run, "_ensure_ready", lambda: _Kit())
    _stub(monkeypatch, "build_mailer", build=lambda: ("Subj", "txt", "<html>x</html>"))
    _argv(monkeypatch, "--brief")
    run.main()
    assert (tmp_path / "res" / "daily_brief.html").read_text() == "<html>x</html>"
    assert capsys.readouterr().out == f"Subj\nsaved → {tmp_path / 'res' / 'daily_brief.html'}\n"


def test_main_global(run, monkeypatch, capsys, tmp_path):
    _stub(monkeypatch, "strategies", STRATEGIES={})
    monkeypatch.setattr(run, "_ensure_ready", lambda: _Kit())
    seen = {}
    g = pd.DataFrame({"Symbol": list("ABC"), "ret_126": [3, 2, 1]})

    def analyse(min_turnover_usd):
        seen["mt"] = min_turnover_usd
        return {"global": g}

    _stub(monkeypatch, "run_global_analysis", analyse=analyse)
    _argv(monkeypatch, "--global", "--top", "2", "--min-turnover", "5e6")
    run.main()
    out = capsys.readouterr().out
    assert seen["mt"] == 5e6
    assert "A" in out and "B" in out and " C" not in out
    assert len(pd.read_csv(tmp_path / "res" / "global_momentum.csv")) == 3


def test_main_default_is_global_and_empty(run, monkeypatch, capsys, tmp_path):
    _stub(monkeypatch, "strategies", STRATEGIES={})
    monkeypatch.setattr(run, "_ensure_ready", lambda: _Kit())
    seen = {}
    _stub(
        monkeypatch,
        "run_global_analysis",
        analyse=lambda min_turnover_usd: seen.setdefault("mt", min_turnover_usd)
        and {"global": pd.DataFrame()},
    )
    _argv(monkeypatch)
    run.main()
    assert seen["mt"] == 1_000_000
    assert not (tmp_path / "res" / "global_momentum.csv").exists()
    assert capsys.readouterr().out == ""


def test_main_strategy_hits(run, monkeypatch, capsys, tmp_path):
    _stub(monkeypatch, "strategies", STRATEGIES={})
    kit = _Kit(pd.DataFrame({"Symbol": ["X"]}))
    monkeypatch.setattr(run, "_ensure_ready", lambda: kit)
    _argv(monkeypatch, "--strategy", "darvas", "--market", "US")
    run.main()
    assert kit.args == ("darvas", "US", 20, 1_000_000)
    assert (tmp_path / "res" / "darvas_US.csv").exists()
    out = capsys.readouterr().out
    assert "X" in out and "NOT investment advice" in out


def test_main_strategy_no_hits(run, monkeypatch, capsys, tmp_path):
    _stub(monkeypatch, "strategies", STRATEGIES={})
    monkeypatch.setattr(run, "_ensure_ready", lambda: _Kit(pd.DataFrame()))
    _argv(monkeypatch, "--strategy", "garp")
    run.main()
    assert capsys.readouterr().out == "No hits for garp on IN.\n"
    assert not (tmp_path / "res" / "garp_IN.csv").exists()


# ---------------- vcrud_index_ltm ----------------
class Rec:
    def __init__(self, path, size, comp, cks, ratio=0.5):
        self.path, self.size_bytes, self.compressed_size = path, size, comp
        self.checksum, self.compression_ratio = cks, ratio


@pytest.fixture
def vil(monkeypatch, tmp_path):
    state = {"recs": [], "fail": set(), "stats": {"n": 1}, "created": []}

    class DB:
        def __init__(self, url):
            state["url"] = url

        def create_file(self, rec):
            if rec.path in state["fail"]:
                raise RuntimeError("dup " * 30)
            state["created"].append(rec.path)

        def get_branch_stats(self, b):
            if state["stats"] is None:
                raise RuntimeError
            return state["stats"]

    class Idx:
        def __init__(self, root):
            state["root"] = root

        def scan_directory(self, branch, patterns):
            state["scan"] = (branch, patterns)
            return state["recs"]

    _stub(monkeypatch, "db_handler", DatabaseHandler=DB)
    _stub(monkeypatch, "vcrud_manager", LocalFileIndexer=Idx)
    m = fresh(monkeypatch, tmp_path, "vcrud_index_ltm", {"DB": "pg://x", "BRANCH": "b1"})
    m.state = state
    return m


def test_vil_defaults(monkeypatch, tmp_path):
    _stub(monkeypatch, "db_handler", DatabaseHandler=object)
    _stub(monkeypatch, "vcrud_manager", LocalFileIndexer=object)
    monkeypatch.delenv("DB", raising=False)
    monkeypatch.delenv("BRANCH", raising=False)
    m = fresh(monkeypatch, tmp_path, "vcrud_index_ltm")
    assert m.DB_URL == "postgresql://umashankar@localhost/vcrud" and m.BRANCH == "karz"


def test_vil_no_files(vil, capsys):
    assert vil.main() == 1
    assert "no LTM parquet files" in capsys.readouterr().out
    assert vil.state["scan"] == ("b1", ["cache_seed/ltm/*.parquet"])
    assert vil.state["url"] == "pg://x"


def test_vil_full(vil, capsys):
    vil.state["recs"] = [
        Rec("cache_seed/ltm/IN.parquet", 2_000_000, 1_000_000, "a" * 20),
        Rec("cache_seed/ltm/US.parquet", 8_000_000, 6_000_000, "b" * 20, 0.75),
        Rec("cache_seed/ltm/UK.parquet", 1_000_000, 500_000, "a" * 20),
    ]
    vil.state["fail"] = {"cache_seed/ltm/UK.parquet"}
    assert vil.main() == 0
    out = capsys.readouterr().out
    assert "indexed 2/3 LTM files on branch 'b1'" in out
    assert "raw: 11.0 MB | gzip-on-parquet: 7.5 MB | extra gzip ratio: 31.8%" in out
    assert "skip cache_seed/ltm/UK.parquet: " + ("dup " * 30)[:60] in out
    assert ("dup " * 30)[:61] not in out
    # biggest first
    assert out.index("US    ") < out.index("IN    ")
    assert "(75.0%)" in out and "b" * 12 in out
    assert "⚠ dup: cache_seed/ltm/UK.parquet == cache_seed/ltm/IN.parquet" in out
    assert "distinct checksums: 2 | duplicate files: 1" in out
    assert "VCRUD branch stats: {'n': 1}" in out


def test_vil_stats_failure_ignored(vil, capsys):
    vil.state["recs"] = [Rec("cache_seed/ltm/IN.parquet", 10, 5, "c" * 20)]
    vil.state["stats"] = None
    assert vil.main() == 0
    assert "branch stats" not in capsys.readouterr().out


# ---------------- vcrud_precheck ----------------
@pytest.fixture
def vp(monkeypatch, tmp_path):
    return fresh(monkeypatch, tmp_path, "vcrud_precheck")


def test_vp_git_runs_and_returns_stdout(vp, monkeypatch):
    seen = {}

    def run(cmd, capture_output, text):
        seen.update(cmd=cmd, co=capture_output, t=text)
        return types.SimpleNamespace(stdout="out")

    monkeypatch.setattr(subprocess, "run", run)
    assert vp._git("status", "-s") == "out"
    assert seen == {"cmd": ["git", "status", "-s"], "co": True, "t": True}


def test_vp_local_branches(vp, monkeypatch):
    monkeypatch.setattr(vp, "_git", lambda *a: " main\n\n dev \n")
    assert vp.local_branches() == ["main", "dev"]


def test_vp_blobs(vp, monkeypatch):
    out = "\n".join(
        [
            "100644 blob abc123     42\tdir/a b.py",
            "160000 commit def456      -\tsub",
            "100644 blob ghi789      -\tweird",
            "bad",
        ]
    )
    monkeypatch.setattr(vp, "_git", lambda *a: out)
    assert list(vp.blobs("main")) == [("dir/a b.py", "abc123", 42), ("weird", "ghi789", 0)]


def _fake_repo(vp, monkeypatch, trees):
    monkeypatch.setattr(vp, "local_branches", lambda: list(trees))
    monkeypatch.setattr(vp, "blobs", lambda b: iter(trees[b]))


def test_vp_scan(vp, monkeypatch):
    _fake_repo(
        vp,
        monkeypatch,
        {
            "a": [("f", "s1", 1_000_000), ("g", "s2", 5)],
            "b": [("f2", "s1", 1_000_000), ("h", "s3", 7)],
            "c": [("f3", "s1", 1_000_000)],
        },
    )
    r = vp.scan()
    assert r["branches"] == ["a", "b", "c"]
    assert list(r["duplicates"]) == ["s1"] and len(r["duplicates"]["s1"]) == 3
    assert r["wasted_bytes"] == 2_000_000  # (n-1) copies


def test_vp_scan_none(vp, monkeypatch):
    _fake_repo(vp, monkeypatch, {"a": [("f", "s1", 3)], "b": [("g", "s2", 3)]})
    r = vp.scan()
    assert r["duplicates"] == {} and r["wasted_bytes"] == 0


def test_vp_main_warn_only(vp, monkeypatch, capsys):
    _fake_repo(vp, monkeypatch, {"a": [("f", "s1", 2_000_000)], "b": [("g", "s1", 2_000_000)]})
    assert vp.WARN_ONLY is True
    assert vp.main() == 0
    out = capsys.readouterr().out
    assert "1 duplicated blobs across 2 local branches; 2.0 MB" in out
    assert "2.0 MB ×2  f  [a, b]" in out


def test_vp_main_strict(vp, monkeypatch, capsys):
    monkeypatch.setattr(vp, "WARN_ONLY", False)
    _fake_repo(vp, monkeypatch, {"a": [("f", "s1", 10)], "b": [("g", "s1", 10)]})
    assert vp.main() == 1
    assert "FAILED" in capsys.readouterr().err
    _fake_repo(vp, monkeypatch, {"a": [("f", "s1", 10)], "b": [("g", "s2", 10)]})
    assert vp.main() == 0


def test_vp_strict_env(monkeypatch, tmp_path):
    monkeypatch.setenv("VCRUD_STRICT", "1")
    assert fresh(monkeypatch, tmp_path, "vcrud_precheck").WARN_ONLY is False


def test_vp_main_top5_sorted(vp, monkeypatch, capsys):
    trees = {"a": [], "b": []}
    for i in range(7):
        trees["a"].append((f"f{i}", f"s{i}", (i + 1) * 1000))
        trees["b"].append((f"g{i}", f"s{i}", (i + 1) * 1000))
    _fake_repo(vp, monkeypatch, trees)
    vp.main()
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("   ")]
    assert len(lines) == 5
    assert "f6" in lines[0] and "f2" in lines[-1]


# ---------------- duckdb-backed warehouse modules ----------------
class FakeCon:
    def __init__(self, log, counts=(7, 9)):
        self.log, self.counts, self._last = log, list(counts), None

    def execute(self, sql):
        self.log.append(sql)
        self._last = sql
        return self

    def fetchone(self):
        return (self.counts.pop(0),)

    def close(self):
        self.log.append("CLOSE")

    def df(self):
        return pd.DataFrame(
            {
                "market": ["US", "IN"],
                "q": [self._last, self._last],
                "Date": ["2026-01-01", "2026-01-02"],
            }
        )

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.log.append("EXIT")


def _fake_duckdb(monkeypatch, log, **kw):
    d = _stub(monkeypatch, "duckdb")

    def connect(path, **k):
        log.append(("connect", path, k))
        if not k:
            pathlib.Path(path).write_bytes(b"0" * 1024 * 1024)
        return FakeCon(log, **kw)

    d.connect = connect
    return d


def test_build_warehouse(monkeypatch, tmp_path, capsys):
    log = []
    _fake_duckdb(monkeypatch, log)
    monkeypatch.setenv("FUND_HIST_DIR", str(tmp_path / "fund"))
    monkeypatch.setenv("LTM_DIR", str(tmp_path / "ltm"))
    bw = fresh(monkeypatch, tmp_path, "build_warehouse")
    (tmp_path / "ltm").mkdir()
    (tmp_path / "ltm" / "US.parquet").write_text("x")
    (tmp_path / "ltm" / "IN.parquet").write_text("x")
    (tmp_path / "ltm" / "notes.txt").write_text("x")
    ssot = tmp_path / "ssot.parquet"
    ssot.write_text("x")
    monkeypatch.setattr(bw, "SSOT", str(ssot))
    out = tmp_path / "wh.duckdb"
    out.write_bytes(b"0" * 2 * 1024 * 1024)  # pre-existing file must be removed first
    bw.build(str(out))
    assert log[0] == ("connect", str(out), {})
    sqls = [x for x in log if isinstance(x, str)]
    inserts = [s for s in sqls if s.startswith("INSERT")]
    assert len(inserts) == 2
    assert "SELECT 'IN'" in inserts[0] and "SELECT 'US'" in inserts[1]  # sorted
    assert any("CREATE TABLE fundamentals" in s and str(tmp_path / "fund") in s for s in sqls)
    assert any("DROP COLUMN filename" in s for s in sqls)
    assert any("CREATE TABLE exchange_reference" in s for s in sqls)
    assert "CLOSE" in sqls
    assert out.stat().st_size == 1024 * 1024  # old 2MB file was removed then recreated
    assert "2 price markets · 7 price rows · 9 fundamental rows" in capsys.readouterr().out


def test_build_warehouse_no_ssot_and_main(monkeypatch, tmp_path, capsys):
    log = []
    _fake_duckdb(monkeypatch, log)
    monkeypatch.setenv("FUND_HIST_DIR", "")
    monkeypatch.setenv("LTM_DIR", "")
    bw = fresh(monkeypatch, tmp_path, "build_warehouse")
    assert bw.FUND_DIR.endswith("cache_seed/fundamentals_history")  # empty env -> default
    assert bw.LTM.endswith("cache_seed/ltm")
    monkeypatch.setattr(bw, "LTM", str(tmp_path / "none"))
    monkeypatch.setattr(bw, "SSOT", str(tmp_path / "nossot"))
    f = tmp_path / "x.duckdb"
    monkeypatch.setattr(sys, "argv", ["bw", "--out", str(f)])
    bw.main()
    assert not any("exchange_reference" in s for s in log if isinstance(s, str))
    assert "0 price markets" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["bw"])
    pathlib.Path(bw.BASE).mkdir(parents=True)
    bw.main()
    assert any(
        isinstance(x, tuple) and x[1] == bw.os.path.join(bw.BASE, "market_data.duckdb") for x in log
    )


def test_warehouse(monkeypatch, tmp_path):
    log = []
    _fake_duckdb(monkeypatch, log)
    monkeypatch.setenv("MARKET_DB", str(tmp_path / "m.duckdb"))
    wh = fresh(monkeypatch, tmp_path, "warehouse")
    assert wh.DB == str(tmp_path / "m.duckdb")
    assert wh.markets() == ["IN", "US"]
    assert log[0] == ("connect", wh.DB, {"read_only": True})
    p = wh.prices("jp")
    assert "WHERE market = 'JP'" in p["q"][0] and "date AS Date" in p["q"][0]
    assert "Date" in p.columns
    assert "WHERE market = 'KR'" in wh.fundamentals("kr")["q"][0]
    assert log.count("EXIT") == 3


def test_warehouse_prices_converts_date(monkeypatch, tmp_path):
    _fake_duckdb(monkeypatch, [])
    wh = fresh(monkeypatch, tmp_path, "warehouse")
    monkeypatch.setattr(wh, "sql", lambda q: pd.DataFrame({"Date": ["2026-01-02"], "Close": [1.0]}))
    assert str(wh.prices("us")["Date"].dtype).startswith("datetime64")
    monkeypatch.delenv("MARKET_DB", raising=False)
    assert fresh(monkeypatch, tmp_path, "warehouse").DB.endswith("python_files/market_data.duckdb")


# ---------------- update_bhavcopy_daily ----------------
@pytest.fixture
def ubd(monkeypatch, tmp_path):
    import bhavcopy_history
    import bhavcopy_store

    monkeypatch.setattr(bhavcopy_history, "fetch_history", lambda n_days, verbose: {"A": 1, "B": 2})
    monkeypatch.setattr(bhavcopy_store, "build", lambda verbose: 99)
    return fresh(monkeypatch, tmp_path, "update_bhavcopy_daily")


def test_ubd_update_incremental(ubd, monkeypatch, capsys):
    seen = {}
    import datalink

    def inc(verbose):
        seen["v"] = verbose
        return {"symbols": 5}

    monkeypatch.setattr(datalink, "build_store_incremental", inc)
    assert ubd.update(n_days=30, verbose=True) == 5
    out = capsys.readouterr().out
    assert "fetching up to 30 days back" in out and "cache refreshed: 2 India symbols" in out
    assert "(5 symbols touched)" in out and seen["v"] is True


def test_ubd_update_missing_key_and_fallback(ubd, monkeypatch, capsys):
    import datalink

    monkeypatch.setattr(datalink, "build_store_incremental", lambda verbose: {})
    assert ubd.update(verbose=False) == 0
    monkeypatch.setattr(datalink, "build_store_incremental", lambda verbose: 1 / 0)
    assert ubd.update(verbose=False) == 99
    assert capsys.readouterr().out == ""


def test_ubd_main(ubd, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(ubd, "update", lambda n_days, verbose: calls.append((n_days, verbose)))
    monkeypatch.setattr(sys, "argv", ["u"])
    assert ubd.main() == 0
    monkeypatch.setattr(sys, "argv", ["u", "--days", "7", "--quiet"])
    assert ubd.main() == 0
    assert calls == [(400, True), (7, False)]

    def boom(**k):
        raise RuntimeError("bad")

    monkeypatch.setattr(ubd, "update", boom)
    assert ubd.main() == 1
    assert "bhavcopy daily update FAILED: bad" in capsys.readouterr().err


# ---------------- deepen_ltm ----------------
def test_deepen(monkeypatch):
    import datalink
    import deepen_ltm as dl
    import market_memory as mm

    seen = {}
    monkeypatch.setattr(
        datalink,
        "load_market",
        lambda m, tier=None: ({} if tier == "ltm" else {"X": 1, "Y": 2, "Z": 3}),
    )

    def fetch(syms, period, min_bars, verbose):
        seen.update(s=syms, p=period, mb=min_bars, v=verbose)
        return {"X": pd.DataFrame()}

    _stub(monkeypatch, "data_sources", fetch=fetch)
    monkeypatch.setattr(mm, "commit", lambda m, long, verbose: {"market": m, "v": verbose})
    r = dl.deepen("US", years=3, limit=2, verbose=False)
    assert seen == {"s": ["X", "Y"], "p": "3y", "mb": 1, "v": False}
    assert r == {"market": "US", "v": False}
    dl.deepen("US", verbose=True)
    assert seen["s"] == ["X", "Y", "Z"] and seen["p"] == "5y"


def test_deepen_no_symbols_and_main(monkeypatch, capsys):
    import datalink
    import deepen_ltm as dl

    monkeypatch.setattr(datalink, "load_market", lambda m, tier=None: {})
    assert dl.deepen("US") == {"market": "US", "error": "no LTM symbols"}
    monkeypatch.setattr(sys, "argv", ["d", "--market", "IN", "--years", "2", "--limit", "9"])
    monkeypatch.setattr(dl, "deepen", lambda m, y, l: {"got": (m, y, l)})
    assert dl.main() == 0
    assert "{'got': ('IN', 2, 9)}" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["d"])
    dl.main()
    assert "('US', 5, 0)" in capsys.readouterr().out
