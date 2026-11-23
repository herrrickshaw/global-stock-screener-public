"""Offline tests for dep_map (static import / data dependency map)."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import dep_map as dm


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.setattr(dm, "HERE", tmp_path)
    monkeypatch.setattr(dm, "OUT", tmp_path / "DEPENDENCIES.md")
    return tmp_path


def _w(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return rel


def test_modules_filters_tests_and_uses_git(monkeypatch, tmp_path):
    monkeypatch.setattr(dm, "HERE", tmp_path)
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        seen["kw"] = kw
        return SimpleNamespace(stdout="a.py\ntests/test_a.py\nsub/b.py\ntestsx/c.py\n")

    monkeypatch.setattr(dm.subprocess, "run", fake_run)
    assert dm._modules() == ["a.py", "sub/b.py", "testsx/c.py"]
    assert seen["cmd"] == ["git", "-C", str(tmp_path), "ls-files", "*.py"]
    assert seen["kw"]["capture_output"] and seen["kw"]["text"]
    assert subprocess  # module imported


def test_local_names():
    assert dm._local_names(["a.py", "pkg/b.py", "pkg/c.py", "x/y/z.py"]) == {"a", "pkg", "y"}
    assert dm._local_names([]) == set()


def test_code_graph_imports(proj):
    mods = [
        _w(proj, "a.py", "import b\nimport os\nimport c.sub\nfrom b import x\n"),
        _w(proj, "b.py", "import b\nfrom a import q\nimport numpy as np\n"),
        _w(proj, "c.py", "x = 1\n"),
        _w(proj, "d.py", "from . import a\nfrom pkg.mod import f\n"),
        _w(proj, "pkg/m.py", "import a\n"),
        _w(proj, "bad.py", "def (:\n"),
    ]
    fwd, rev = dm.code_graph(mods)
    assert fwd["a"] == {"b", "c"}
    assert fwd["b"] == {"a"}  # self import and numpy ignored
    assert fwd["d"] == {"pkg"}  # relative import has no module; pkg is local dir name
    assert fwd["m"] == {"a"}
    assert "bad" not in fwd and "c" not in fwd
    assert rev["b"] == {"a"} and rev["a"] == {"b", "m"}
    assert rev["pkg"] == {"d"}


def test_code_graph_missing_file(proj):
    fwd, rev = dm.code_graph(["ghost.py"])
    assert dict(fwd) == {} and dict(rev) == {}


def test_data_graph_literals(proj):
    mods = [
        _w(
            proj,
            "w.py",
            'df.to_parquet("cache_seed/cleaned_long.parquet")\n'
            "x = json.dump(1, f)  # liquidity_index\n"
            'open("data_manifest.json")\n',
        ),
        _w(
            proj,
            "r.py",
            'df = pd.read_parquet("cache_seed/cleaned_long.parquet")\n'
            'if Path("serving/x").exists(): pass\n'
            'print("liquidity_index")\n',
        ),
    ]
    produces, consumes = dm.data_graph(mods)
    assert produces["STM seeds (1y)"] == {"w"}
    assert consumes["STM seeds (1y)"] == {"r"}
    assert produces["liquidity index"] == {"w"}
    assert produces["data manifest"] == {"w"}  # open( is a write hint
    assert consumes["serving views"] == {"r"}
    # pure print of an artifact name is neither read nor write
    assert "r" not in consumes["liquidity index"] and "r" not in produces["liquidity index"]


def test_data_graph_write_wins_over_read(proj):
    mods = [_w(proj, "m.py", 'df.to_csv(pd.read_csv("company_list"))\n')]
    produces, consumes = dm.data_graph(mods)
    assert produces["company list"] == {"m"}
    assert "company list" not in consumes


def test_data_graph_path_variables(proj):
    mods = [
        _w(
            proj,
            "v.py",
            'CLEANED, OTHER = Path("x") / "cleaned_long", 1\n'
            "def f():\n"
            "    pd.read_parquet(CLEANED)\n"
            "    write_seed(OTHER)\n"
            "    CLEANEDX = 1\n"
            "    df.to_parquet(CLEANED)\n",
        ),
    ]
    produces, consumes = dm.data_graph(mods)
    assert consumes["STM seeds (1y)"] == {"v"}
    assert produces["STM seeds (1y)"] == {"v"}
    assert set(produces) | set(consumes) == {"STM seeds (1y)"}


def test_data_graph_word_boundary(proj):
    mods = [_w(proj, "v.py", 'CLEANED = "cleaned_long"\nread_parquet(CLEANED_2)\nx = "ltm/"\n')]
    produces, consumes = dm.data_graph(mods)
    assert "v" not in consumes.get("STM seeds (1y)", set())  # CLEANED_2 != CLEANED


def test_data_graph_unreadable(proj):
    produces, consumes = dm.data_graph(["ghost.py"])
    assert dict(produces) == {} and dict(consumes) == {}


def test_table():
    assert dm._table(["a", "b"]) == "a\nb"


def _mini_repo(proj, monkeypatch):
    mods = [
        _w(proj, "a.py", "import b\n"),
        _w(proj, "b.py", 'df.to_parquet("cache_seed/cleaned_long.parquet")\n'),
        _w(proj, "c.py", 'x = pd.read_parquet("cleaned_long")\nimport b\n'),
        _w(proj, "d.py", "import b\nimport a\n"),
    ]
    monkeypatch.setattr(dm, "_modules", lambda: mods)
    return mods


def test_build_writes_markdown(proj, monkeypatch):
    _mini_repo(proj, monkeypatch)
    dm.build()
    md = (proj / "DEPENDENCIES.md").read_text()
    assert md.startswith("# Interdependency map (code + data)\n")
    assert md.endswith("\n") and not md.endswith("\n\n\n")
    assert "| `a` | `d` |" in md
    assert "| `b` | `a`, `c`, `d` |" in md
    assert "**Leaf modules** (nothing imports them — safe to change in isolation): `c`, `d`" in md
    assert "| `d` | `a`, `b` |" in md  # section 2 forward deps sorted
    assert "| STM seeds (1y) | `b` | `c` |" in md
    assert "## How to use" in md


def test_build_dash_placeholders(proj, monkeypatch):
    mods = [_w(proj, "w.py", 'df.to_csv("company_list")\n')]
    monkeypatch.setattr(dm, "_modules", lambda: mods)
    dm.build()
    assert "| company list | `w` | — |" in (proj / "DEPENDENCIES.md").read_text()
    mods2 = [_w(proj, "r.py", 'pd.read_csv("nse_holidays")\n')]
    monkeypatch.setattr(dm, "_modules", lambda: mods2)
    dm.build()
    assert "| NSE holidays | — | `r` |" in (proj / "DEPENDENCIES.md").read_text()


def test_build_verbose(proj, monkeypatch, capsys):
    _mini_repo(proj, monkeypatch)
    dm.build(verbose=True)
    out = capsys.readouterr().out
    assert "wrote DEPENDENCIES.md: 4 modules" in out
    assert "most-depended-on modules:" in out
    assert "b" in out and "← 3 importers" in out
    dm.build(verbose=False)
    assert capsys.readouterr().out == ""


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="dep_map.py:~215: verbose line uses `len(produces)|len(consumes)` (bitwise OR) "
    "instead of the size of the key union, so the artifact count is wrong",
)
def test_build_verbose_artifact_count(proj, monkeypatch, capsys):
    mods = [
        _w(proj, "w.py", 'df.to_csv("company_list")\n'),
        _w(proj, "r.py", 'pd.read_csv("nse_holidays")\n'),
    ]
    monkeypatch.setattr(dm, "_modules", lambda: mods)
    dm.build(verbose=True)
    assert "2 data artifacts" in capsys.readouterr().out


def test_main(proj, monkeypatch, capsys):
    _mini_repo(proj, monkeypatch)
    monkeypatch.setattr("sys.argv", ["dep_map.py"])
    assert dm.main() == 0
    assert capsys.readouterr().out == ""
    assert (proj / "DEPENDENCIES.md").exists()
    monkeypatch.setattr("sys.argv", ["dep_map.py", "--print"])
    assert dm.main() == 0
    assert "wrote DEPENDENCIES.md" in capsys.readouterr().out


def test_constants():
    assert Path(dm.__file__).name == "dep_map.py"
    assert "to_parquet" in dm.WRITE_HINTS and "read_parquet" in dm.READ_HINTS
