import json
import os
import pathlib
import sys
import types

import pandas as pd
import pytest

import flows_regime as fr
import portfolio_analytics as pa
import preflight as pf


def _stub(monkeypatch, name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    monkeypatch.setitem(sys.modules, name, m)
    return m


# ---------------- portfolio_analytics ----------------
def test_weights_list_and_dict():
    assert pa._weights(["a", "b"]) == {"A": 0.5, "B": 0.5}
    assert pa._weights([]) == {}
    w = pa._weights({"x": 2, "y": 6})
    assert w == {"X": 0.25, "Y": 0.75}
    assert pa._weights({"x": 0, "y": 0}) == {"X": 0, "Y": 0}  # zero total falls back to 1.0


def test_hhi():
    assert pa._hhi({"a": 1.0}) == 1.0
    assert pa._hhi({"a": 0.5, "b": 0.5}) == 0.5
    assert pa._hhi({}) == 0
    assert pa._hhi({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}) == 0.333


def _row(sym, w, sig="BUY", flags="", sector="IT"):
    return {"symbol": sym, "weight": w, "signal": sig, "flags": flags, "sector": sector}


def test_rebalance_actions_each_rule():
    rows = [
        _row("A", 0.10, "BUY"),  # exactly at name cap: no action
        _row("B", 0.11, "BUY", sector="Fin"),  # over cap
        _row("C", 0.05, "SELL", flags="LOW"),
        _row("D", 0.05, "HOLD", flags="IMPAIRED x"),
        _row("E", 0.05, "BUY", sector="?"),
    ]
    acts = pa.rebalance_actions(rows)
    assert acts == [
        "REDUCE B — 11% > 10% single-name cap",
        "TRIM/EXIT C — SELL LOW",
        "TRIM/EXIT D — HOLD IMPAIRED x",
    ]
    assert len(acts) == 3  # sector IT = 0.10+... below the cap; '?' ignored


def test_rebalance_sector_cap_and_trim_strip():
    rows = [_row("A", 0.2, "SELL", flags="", sector="IT"), _row("B", 0.2, "BUY", sector="IT")]
    acts = pa.rebalance_actions(rows, max_name=0.5, max_sector=0.30)
    assert acts == ["TRIM/EXIT A — SELL", "REDUCE sector IT — 40% > 30% sector cap"]
    # exactly at the sector cap: no action
    assert pa.rebalance_actions([_row("A", 0.3), _row("B", 0.0)], max_name=1, max_sector=0.3) == []
    # a missing sector key groups under '?' and is never flagged
    assert pa.rebalance_actions([{"symbol": "A", "weight": 0.9, "signal": "BUY"}], max_name=1) == []
    # empty-string sector ignored
    assert pa.rebalance_actions([_row("A", 0.9, sector="")], max_name=1) == []


def _ev(rows):
    return pd.DataFrame(rows, columns=["Symbol", "Signal", "PE", "Flags"])


def _patch_analyze(monkeypatch, ev, smap):
    seen = {}

    def evaluate(syms, market, fetch, verbose):
        seen.update(syms=syms, market=market, fetch=fetch, verbose=verbose)
        return ev

    _stub(monkeypatch, "portfolio_evaluator", evaluate=evaluate)
    if smap is None:
        _stub(monkeypatch, "sector_map", sector_map=lambda m: 1 / 0)
    else:
        _stub(monkeypatch, "sector_map", sector_map=lambda m: smap)
    return seen


def test_analyze_full(monkeypatch, capsys):
    ev = _ev(
        [
            ["A", "BUY", 20.0, ""],
            ["B", "SELL", 40.0, "IMPAIRED"],
            ["C", "HOLD", float("nan"), ""],
            ["D", "BUY", 1000.0, ""],  # pe == 1000 excluded (< 1000)
            ["E", "HOLD", 0.0, ""],  # pe 0 excluded
        ]
    )
    seen = _patch_analyze(
        monkeypatch, ev, {"A": "IT", "B": "IT", "C": "Fin", "D": "Fin", "E": "Fin"}
    )
    out = pa.analyze(["a", "b", "c", "d", "e", "ghost"], "US")
    assert seen == {
        "syms": ["A", "B", "C", "D", "E", "GHOST"],
        "market": "US",
        "fetch": False,
        "verbose": False,
    }
    assert out["n"] == 6 and out["market"] == "US"
    assert out["weighted_pe"] == 30.0  # (20+40)/2 equal weight
    assert out["signal_mix"] == {
        "BUY": round(2 / 6, 3),
        "HOLD": round(2 / 6, 3),
        "SELL": round(1 / 6, 3),
        "NO DATA": round(1 / 6, 3),
    }
    assert out["flagged"] == ["B"]
    assert out["name_hhi"] == round(6 * (1 / 6) ** 2, 3)
    assert list(out["top_sectors"]) == ["Fin", "IT", "?"]
    assert "TRIM/EXIT B — SELL IMPAIRED" in out["actions"]
    txt = capsys.readouterr().out
    assert "=== Portfolio analytics (US) — 6 holdings ===" in txt
    assert "weighted P/E: 30.0" in txt and "• TRIM/EXIT B — SELL IMPAIRED" in txt


def test_analyze_quiet_no_sector_map_no_pe(monkeypatch, capsys):
    ev = _ev([["A", "BUY", float("nan"), ""]])
    _patch_analyze(monkeypatch, ev, None)
    out = pa.analyze({"A": 1.0}, verbose=False)
    assert out["weighted_pe"] is None and out["top_sectors"] == {"?": 1.0}
    assert out["sector_hhi"] == 1.0
    assert capsys.readouterr().out == ""
    assert out["actions"] == ["REDUCE A — 100% > 10% single-name cap"]


def test_analyze_balanced_message(monkeypatch, capsys):
    syms = [f"S{i}" for i in range(20)]
    ev = _ev([[s, "BUY", 10.0, ""] for s in syms])
    _patch_analyze(monkeypatch, ev, {s: f"Sec{i}" for i, s in enumerate(syms)})
    out = pa.analyze(syms)
    assert out["actions"] == []
    assert "(none — book looks balanced)" in capsys.readouterr().out


def test_analyze_top_five_sectors(monkeypatch):
    syms = [f"S{i}" for i in range(7)]
    ev = _ev([[s, "BUY", 10.0, ""] for s in syms])
    _patch_analyze(monkeypatch, ev, {s: f"Sec{i}" for i, s in enumerate(syms)})
    assert len(pa.analyze(syms, verbose=False)["top_sectors"]) == 5


def test_pa_main(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(pa, "analyze", lambda t, m: calls.append((t, m)))
    monkeypatch.setattr(sys, "argv", ["pa", "--tickers", "A, B,,C", "--market", "US"])
    assert pa.main() == 0
    assert calls == [(["A", "B", "C"], "US")]
    monkeypatch.setattr(sys, "argv", ["pa", "--tickers", "X"])
    pa.main()
    assert calls[-1] == (["X"], "IN")


# ---------------- flows_regime ----------------
@pytest.fixture
def flows(monkeypatch, tmp_path):
    monkeypatch.setattr(fr, "STORE", tmp_path / "IN.json")
    return fr


def _nse(monkeypatch, vix=None, fiidii=None, vix_exc=False, fd_exc=False):
    def quote(name):
        assert name == "INDIA VIX"
        if vix_exc:
            raise RuntimeError
        return {"lastPrice": vix}

    def fd():
        if fd_exc:
            raise RuntimeError
        return fiidii

    _stub(monkeypatch, "nsepython", nse_get_index_quote=quote, nse_fiidii=fd)


def test_fetch_success(flows, monkeypatch, capsys):
    _nse(
        monkeypatch,
        vix="14.5",
        fiidii=[
            {
                "category": "FII/FPI *",
                "netValue": "-1,0",
            },  # unparsable -> whole fiidii block skipped
        ],
    )
    out = flows.fetch()
    assert out["vix"] == 14.5 and "fii_net" not in out
    _nse(
        monkeypatch,
        vix="20",
        fiidii=[
            {"category": "DII **", "netValue": "300"},
            {"category": "fpi", "net": 500.5},
            {"category": "other", "netValue": 9},
            {"category": "FII", "netValue": None},
        ],
    )
    out = flows.fetch(verbose=False)
    assert out["vix"] == 20.0 and out["dii_net"] == 300.0 and out["fii_net"] == 0.0
    assert out["asof"] == __import__("datetime").date.today().isoformat()
    assert json.loads(flows.STORE.read_text()) == out
    assert capsys.readouterr().out.count("flows:") == 1


def test_fetch_net_fallback_key(flows, monkeypatch):
    _nse(monkeypatch, vix_exc=True, fiidii=[{"category": "FPI", "net": 7}])
    out = flows.fetch(verbose=False)
    assert out["fii_net"] == 7.0 and "vix" not in out


def test_fetch_non_list_and_failures(flows, monkeypatch, capsys):
    _nse(monkeypatch, vix=15, fiidii={"not": "a list"})
    assert flows.fetch(verbose=False)["vix"] == 15.0
    _nse(monkeypatch, vix_exc=True, fd_exc=True)
    flows.STORE.unlink()
    assert flows.fetch() == {}
    assert not flows.STORE.exists()  # nothing written when empty
    assert "fetch failed (nsepython/network)" in capsys.readouterr().out
    monkeypatch.setitem(sys.modules, "nsepython", None)
    assert flows.fetch(verbose=False) == {}


def test_load(flows):
    assert flows._load() == {}
    flows.STORE.write_text('{"vix": 1}')
    assert flows._load() == {"vix": 1}


def test_regime_scoring_boundaries(flows):
    r = fr.regime({"vix": 17})
    assert r["score"] == 0 and r["regime"] == "Neutral" and r["drivers"] == ["VIX 17.0"]
    assert fr.regime({"vix": 10})["score"] == 28.0 and fr.regime({"vix": 10})["regime"] == "Risk-On"
    assert fr.regime({"vix": 0})["score"] == 40  # clamp +40
    assert fr.regime({"vix": 100})["score"] == -40  # clamp -40
    assert fr.regime({"vix": 17, "fii_net": 100000})["score"] == 40
    assert fr.regime({"fii_net": -100000})["score"] == -40
    assert fr.regime({"fii_net": 1000})["score"] == 20.0
    assert fr.regime({"dii_net": 100000})["score"] == 20
    assert fr.regime({"dii_net": -100000})["score"] == -20
    assert fr.regime({"dii_net": 1000})["score"] == 10.0
    # thresholds are strict: exactly 25 is Neutral, exactly -25 is Neutral
    assert fr.regime({"fii_net": 1250})["regime"] == "Neutral"
    assert fr.regime({"fii_net": 1300})["regime"] == "Risk-On"
    assert fr.regime({"fii_net": -1250})["regime"] == "Neutral"
    assert fr.regime({"fii_net": -1300})["regime"] == "Risk-Off"


def test_regime_drivers_and_asof(flows):
    r = fr.regime({"vix": 21.0, "fii_net": -1500.4, "dii_net": 2000, "asof": "2026-01-01"})
    assert r["drivers"] == ["VIX 21.0", "FII net ₹-1,500cr", "DII net ₹2,000cr"]
    assert r["asof"] == "2026-01-01" and r["score"] == round(-16 - 30.008 + 20, 1)
    assert fr.regime({})["asof"] == "" and fr.regime({})["drivers"] == []


def test_regime_loads_cache_and_bias(flows):
    assert fr.regime()["regime"] == "Neutral"
    flows.STORE.write_text(json.dumps({"vix": 10}))
    assert fr.regime()["score"] == 28.0
    assert fr.regime_bias() == 0.28
    assert fr.regime({})["score"] == 0  # explicit empty dict is used, not the cache


def test_flows_main(flows, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["f"])
    assert fr.main() == 0
    assert "Neutral  (score 0.0) — no data — run --fetch" in capsys.readouterr().out
    called = []
    monkeypatch.setattr(fr, "fetch", lambda: called.append(1) or {})
    flows.STORE.write_text(json.dumps({"vix": 10}))
    monkeypatch.setattr(sys, "argv", ["f", "--fetch"])
    fr.main()
    assert called == [1]
    assert "Risk-On  (score 28.0) — VIX 10.0" in capsys.readouterr().out


# ---------------- preflight ----------------
@pytest.fixture
def pre(monkeypatch, tmp_path):
    monkeypatch.setenv("BHAV_CACHE", str(tmp_path / "cache"))
    return pf


def _mm(monkeypatch, exc=False):
    calls = []

    def ens(verbose):
        calls.append(verbose)
        if exc:
            raise RuntimeError("derive " * 20)

    _stub(monkeypatch, "market_memory", ensure_stm_seeds=ens)
    return calls


def _sl(monkeypatch, built=("IN",), exc=False):
    calls = []

    def build(m, verbose):
        calls.append(("build", m, verbose))
        if exc:
            raise RuntimeError("serve " * 20)
        return m in built

    _stub(
        monkeypatch,
        "serving_layer",
        build_serving=build,
        capture_deltas=lambda m, verbose: calls.append(("delta", m, verbose)),
        serving=lambda m: pd.DataFrame({"a": [1]}) if m in built else pd.DataFrame(),
    )
    return calls


def test_ensure_happy(pre, monkeypatch, tmp_path, capsys):
    mm = _mm(monkeypatch)
    sl = _sl(monkeypatch, built=("IN",))
    out = pre.ensure(["IN", "US"])
    assert out == {"cache": str(tmp_path / "cache"), "serving_built": 1}
    assert (tmp_path / "cache").is_dir() and mm == [False]
    assert sl == [
        ("build", "IN", False),
        ("delta", "IN", False),
        ("build", "US", False),
        ("delta", "US", False),
    ]
    assert (
        f"cache={tmp_path / 'cache'}, serving views built for 1 market(s)"
        in capsys.readouterr().out
    )


def test_ensure_default_market_and_skip_serving(pre, monkeypatch):
    _mm(monkeypatch)
    sl = _sl(monkeypatch)
    assert pre.ensure(build_serving=False, verbose=False)["serving_built"] == 0
    assert sl == []
    pre.ensure(verbose=False)
    assert [c[1] for c in sl if c[0] == "build"] == ["IN"]


def test_ensure_failures_reported_truncated(pre, monkeypatch, capsys):
    _mm(monkeypatch, exc=True)
    _sl(monkeypatch, exc=True)
    out = pre.ensure()
    txt = capsys.readouterr().out
    assert "STM derive skipped (" + ("derive " * 20)[:50] + ")" in txt
    assert "serving build skipped (" + ("serve " * 20)[:50] + ")" in txt
    assert out["serving_built"] == 0
    pre.ensure(verbose=False)
    assert capsys.readouterr().out == ""


def test_ensure_sets_cache_default(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("BHAV_CACHE", "x")
    monkeypatch.delenv("BHAV_CACHE")
    _mm(monkeypatch)
    out = pf.ensure(build_serving=False, verbose=False)
    assert out["cache"] == str(tmp_path / "Downloads" / "data" / "bhavcopy_cache")
    assert os.environ["BHAV_CACHE"] == out["cache"]
    os.environ.pop("BHAV_CACHE")  # monkeypatch cannot undo a var it saw as absent


def test_ensure_colab_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("BHAV_CACHE", "x")
    monkeypatch.delenv("BHAV_CACHE")
    _mm(monkeypatch)
    real_exists, real_mkdir = pathlib.Path.exists, pathlib.Path.mkdir
    monkeypatch.setattr(pathlib.Path, "exists", lambda s: str(s) == "/content" or real_exists(s))
    monkeypatch.setattr(
        pathlib.Path,
        "mkdir",
        lambda s, *a, **k: None if str(s) == "/content/cache" else real_mkdir(s, *a, **k),
    )
    out = pf.ensure(build_serving=False, verbose=False)
    assert out["cache"] == "/content/cache"
    os.environ.pop("BHAV_CACHE")


def test_is_lfs_pointer(tmp_path):
    p = tmp_path / "a"
    p.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:x")
    assert pf._is_lfs_pointer(p) is True
    p.write_bytes(b"PAR1" + b"0" * 60)
    assert pf._is_lfs_pointer(p) is False
    assert pf._is_lfs_pointer(tmp_path / "missing") is False
    p.write_bytes(b"version https://git-lf")  # truncated prefix
    assert pf._is_lfs_pointer(p) is False


def test_importable():
    assert pf._importable("json") is True
    assert pf._importable("definitely_not_a_module_xyz") is False


def _check_env(monkeypatch, tmp_path, ltm_files=("IN.parquet",), pointer=False, core_ok=True):
    ltm = tmp_path / "ltm"
    ltm.mkdir(exist_ok=True)
    for f in ltm_files:
        (ltm / f).write_bytes(
            b"version https://git-lfs.github.com/spec/v1\n" if pointer else b"PAR1" + b"0" * 60
        )
    monkeypatch.setattr(pf, "LTM_DIR", ltm)
    monkeypatch.setattr(pf, "REQUIRED_PKGS", ["json"])
    monkeypatch.setattr(pf, "OPTIONAL_PKGS", ["json"])
    monkeypatch.setattr(pf, "CORE_MODULES", ["json"] if core_ok else ["json", "no_such_mod_zz"])
    monkeypatch.setenv("BHAV_CACHE", str(tmp_path / "cache"))


def test_check_all_ok(monkeypatch, tmp_path, capsys):
    _check_env(monkeypatch, tmp_path)
    _sl(monkeypatch, built=("IN",))
    ok, r = pf.check(["IN"])
    assert ok is True
    assert r["required packages"] == "OK" and r["optional packages"] == "all present"
    assert r["LFS data"] == "OK (1 LTM seeds pulled)"
    assert (
        r["cache writable"] == f"OK ({tmp_path / 'cache'})"
        and not (tmp_path / "cache" / ".w").exists()
    )
    assert r["core imports"] == "OK" and r["serving views"] == "1/1 ready"
    out = capsys.readouterr().out
    assert "READY ✓" in out and "✓ required packages" in out


def test_check_failures(monkeypatch, tmp_path, capsys):
    _check_env(monkeypatch, tmp_path, ltm_files=(), core_ok=False)
    monkeypatch.setattr(pf, "REQUIRED_PKGS", ["json", "nope_pkg_zz"])
    monkeypatch.setattr(pf, "OPTIONAL_PKGS", ["nope_opt_zz"])
    _sl(monkeypatch, built=())
    ok, r = pf.check()
    assert ok is False
    assert r["required packages"] == "MISSING ['nope_pkg_zz']"
    assert r["optional packages"] == "missing (some features off): ['nope_opt_zz']"
    assert r["LFS data"].startswith("CRITICAL: no LTM seeds")
    assert r["core imports"].startswith('FAIL ["no_such_mod_zz:')
    assert r["serving views"] == "none built — call preflight.ensure()"
    out = capsys.readouterr().out
    assert "✗ required packages" in out and "• optional packages" in out and "NOT READY ✗" in out
    assert pf.check(verbose=False)[0] is False and capsys.readouterr().out == ""


def test_check_lfs_pointer(monkeypatch, tmp_path):
    _check_env(monkeypatch, tmp_path, ltm_files=("IN.parquet", "US.parquet"), pointer=True)
    _sl(monkeypatch)
    ok, r = pf.check(verbose=False)
    assert not ok and r["LFS data"].startswith("CRITICAL: LTM are LFS pointers")


def test_check_cache_unwritable_and_serving_unknown(monkeypatch, tmp_path):
    _check_env(monkeypatch, tmp_path)
    blocker = tmp_path / "blocker"
    blocker.write_text("file")
    monkeypatch.setenv("BHAV_CACHE", str(blocker / "sub"))
    monkeypatch.setitem(sys.modules, "serving_layer", None)
    ok, r = pf.check(verbose=False)
    assert not ok and r["cache writable"].startswith("FAIL (")
    assert r["serving views"] == "unknown"


def test_check_default_cache_env(monkeypatch, tmp_path):
    _check_env(monkeypatch, tmp_path)
    monkeypatch.setenv("BHAV_CACHE", "x")
    monkeypatch.delenv("BHAV_CACHE")
    monkeypatch.setenv("HOME", str(tmp_path))
    _sl(monkeypatch)
    ok, r = pf.check(verbose=False)
    assert r["cache writable"] == f"OK ({tmp_path / 'Downloads' / 'data' / 'bhavcopy_cache'})"


def test_check_partial_serving(monkeypatch, tmp_path):
    _check_env(monkeypatch, tmp_path)
    _sl(monkeypatch, built=("IN",))
    assert pf.check(["IN", "US"], verbose=False)[1]["serving views"] == "1/2 ready"


def test_preflight_main(monkeypatch):
    calls = []
    monkeypatch.setattr(pf, "ensure", lambda m: calls.append(m))
    monkeypatch.setattr(pf, "check", lambda m: (True, {}))
    monkeypatch.setattr(sys, "argv", ["p"])
    assert pf.main() == 0 and calls == []
    monkeypatch.setattr(sys, "argv", ["p", "--ensure", "--markets", "IN", "US"])
    assert pf.main() == 0 and calls == [["IN", "US"]]
    monkeypatch.setattr(pf, "check", lambda m: (False, {}))
    assert pf.main() == 1
