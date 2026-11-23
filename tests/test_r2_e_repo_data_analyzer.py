"""Offline tests for repo_data_analyzer (static script analysis)."""

import ast
import json
from datetime import datetime

import pandas as pd
import pytest

import repo_data_analyzer as rda


@pytest.fixture
def an(tmp_path):
    return rda.RepositoryDataAnalyzer(str(tmp_path))


def analyse(an, tmp_path, src, name="s.py"):
    p = tmp_path / name
    p.write_text(src)
    return an._analyze_script(p)


def test_discover_scripts_filters_and_sorts(an, tmp_path):
    for d in ["pkg", "venv", "site-packages", "__pycache__", ".git", "sub/env"]:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
        (tmp_path / d / "m.py").write_text("")
    (tmp_path / "b.py").write_text("")
    (tmp_path / "a.py").write_text("")
    (tmp_path / "pkg" / "m.py").write_text("")
    found = an._discover_scripts()
    names = [str(p.relative_to(tmp_path)) for p in found]
    assert names == ["a.py", "b.py", "pkg/m.py"]


def test_discover_scripts_caps_at_100(an, tmp_path):
    for i in range(105):
        (tmp_path / f"f{i:03d}.py").write_text("")
    found = an._discover_scripts()
    assert len(found) == 100
    assert found[0].name == "f000.py" and found[-1].name == "f099.py"


def test_get_func_name_and_extract_string():
    g = rda.RepositoryDataAnalyzer._get_func_name
    e = rda.RepositoryDataAnalyzer._extract_string_arg
    call = ast.parse("pd.read_csv('a.csv')").body[0].value
    assert g(call.func) == "read_csv"
    assert g(ast.parse("f(1)").body[0].value.func) == "f"
    assert g(ast.parse("f(1)(2)").body[0].value.func) is None
    assert e(call.args[0]) == "a.csv"
    assert e(ast.parse("x").body[0].value) is None
    assert e(ast.parse("5").body[0].value) is None


def test_ast_loads_and_creates(an, tmp_path):
    a = analyse(
        an,
        tmp_path,
        "import pandas as pd\n"
        "a = pd.read_parquet('in.parquet')\n"
        "b = pd.read_csv('b.csv')\n"
        "c = pd.read_excel('c.xlsx')\n"
        "d = pd.read_csv(path_var)\n"
        "e = pd.read_csv()\n"
        "a.to_parquet('out.parquet')\n"
        "a.to_csv('out.csv')\n"
        "a.to_csv(var)\n"
        "a.to_csv()\n",
    )
    assert a.loaded_data_sources == ["in.parquet", "b.csv", "c.xlsx"]
    assert a.created_data_sources == ["out.parquet", "out.csv"]
    assert a.script_name == "s.py" and a.file_path.endswith("s.py")


def test_ast_split_and_sample(an, tmp_path):
    a = analyse(an, tmp_path, "train_test_split(X, y, test_size=0.2)\n")
    assert a.has_train_test_split and a.split_method == "random"
    b = analyse(an, tmp_path, "train_test_split(X, y)\n")
    assert b.has_train_test_split and b.split_method == "random"  # from pattern pass
    c = analyse(an, tmp_path, "x = df.sample(3)\n")
    assert c.uses_samples
    d = analyse(an, tmp_path, "shuffle(x)\n")
    assert d.uses_samples


def test_patterns_backtest_dates(an, tmp_path):
    a = analyse(an, tmp_path, "s='2020-01-01'\ne='2021-06-30'\nm='2020-09-09'\n")
    assert a.backtest_dates == ("2020-01-01", "2020-09-09")
    b = analyse(an, tmp_path, "s='2020-01-01'\n")
    assert b.backtest_dates is None
    c = analyse(an, tmp_path, "x = datetime(2020, 1, 1)\ny = datetime(2021, 1, 1)\n")
    assert c.backtest_dates is not None


def test_patterns_split_methods(an, tmp_path):
    a = analyse(an, tmp_path, "tr = df.loc[:split_point]\n")
    assert a.split_method == "chronological"
    b = analyse(an, tmp_path, "tr = df.iloc[:split_idx]\n")
    assert b.split_method == "chronological"
    c = analyse(an, tmp_path, "tr = df\n")
    assert c.split_method is None


def test_patterns_sample_size_and_exhaustive(an, tmp_path):
    a = analyse(an, tmp_path, "df.head(25)\n")
    assert a.uses_samples and a.sample_size == 25 and not a.is_exhaustive
    b = analyse(an, tmp_path, "pd.read_csv('x', nrows=500)\n")
    assert b.sample_size == 500
    c = analyse(an, tmp_path, "df.sample(n=7)\n")
    assert c.uses_samples and c.sample_size is None  # plain .sample() regex has no n group match
    d = analyse(an, tmp_path, "x = 1\n")
    assert d.is_exhaustive and not d.uses_samples
    e = analyse(an, tmp_path, "limit = 5\n")
    assert e.uses_samples


def test_patterns_all_data_detection(an, tmp_path):
    a = analyse(an, tmp_path, "X = df[cols]\ny = df['t']\n")
    assert a.uses_all_data and a.trains_on_test_data
    b = analyse(an, tmp_path, "X = df[cols]\ntrain_test_split(X)\n")
    assert not b.uses_all_data and not b.trains_on_test_data
    c = analyse(an, tmp_path, "z = df.copy()\n")
    assert c.uses_all_data
    d = analyse(an, tmp_path, "z = 1\n")
    assert not d.uses_all_data


def test_patterns_transformations(an, tmp_path):
    a = analyse(an, tmp_path, "StandardScaler()\nnormalize(x)\ndf.fillna(0)\ndf.drop(nan)\n")
    assert a.data_transformations == [
        "scaling",
        "normalization",
        "null_imputation",
        "null_removal",
    ]
    b = analyse(an, tmp_path, "df.drop(null)\n")
    assert b.data_transformations == ["null_removal"]
    c = analyse(an, tmp_path, "df.drop(x)\n")
    assert c.data_transformations == []
    d = analyse(an, tmp_path, "nan = 1\n")
    assert d.data_transformations == []


def test_issue_all_data_no_split(an, tmp_path):
    a = analyse(an, tmp_path, "X = df\n")
    assert any(i.startswith("Uses all data without train/test split") for i in a.issues)
    b = analyse(
        an, tmp_path, "X = df\ntrain_test_split(X, test_size=1, random_state=1)\nassert 1\n"
    )
    assert b.issues == []


def test_warning_samples(an, tmp_path):
    a = analyse(an, tmp_path, "df.head(10)\nassert 1\n")
    assert any("Testing on samples only (10 rows)" in w for w in a.warnings)
    # is_exhaustive false + uses_samples true is the only trigger
    b = analyse(an, tmp_path, "x = 1\nassert 1\n")
    assert not any("samples only" in w for w in b.warnings)


def test_warning_random_state(an, tmp_path):
    a = analyse(an, tmp_path, "train_test_split(X, test_size=0.2)\nassert 1\n")
    assert any("without fixed random_state" in w for w in a.warnings)
    b = analyse(an, tmp_path, "train_test_split(X, test_size=0.2, random_state=1)\nassert 1\n")
    assert not any("random_state" in w for w in b.warnings)


def test_warning_backtest_dates_without_split(an, tmp_path):
    a = analyse(an, tmp_path, "a='2020-01-01'\nb='2021-01-01'\nassert 1\n")
    assert any("Backtest on date range ('2020-01-01', '2021-01-01')" in w for w in a.warnings)
    b = analyse(an, tmp_path, "a='2020-01-01'\nb='2021-01-01'\ndf.loc[:split_point]\nassert 1\n")
    assert not any("Backtest on date range" in w for w in b.warnings)


def test_warning_no_validation(an, tmp_path):
    a = analyse(an, tmp_path, "x = 1\n")
    assert any("No data validation" in w for w in a.warnings)
    assert not any("No data validation" in w for w in analyse(an, tmp_path, "assert x\n").warnings)
    assert not any(
        "No data validation" in w for w in analyse(an, tmp_path, "def Validate(): pass\n").warnings
    )


def test_warning_multi_source(an, tmp_path):
    src = "pd.read_csv('a')\npd.read_csv('b')\n"
    a = analyse(an, tmp_path, src)
    assert any("Uses 2 data sources" in w for w in a.warnings)
    assert not any("data sources" in w for w in analyse(an, tmp_path, src + "# Compare\n").warnings)
    assert not any("data sources" in w for w in analyse(an, tmp_path, src + "assert 1\n").warnings)
    one = analyse(an, tmp_path, "pd.read_csv('a')\n")
    assert not any("data sources" in w for w in one.warnings)


def test_analyze_script_syntax_error(an, tmp_path, capsys):
    assert analyse(an, tmp_path, "def (:\n", "bad.py") is None
    assert "Error analyzing bad.py" in capsys.readouterr().out
    assert an._analyze_script(tmp_path / "missing.py") is None


def _sa(name, **kw):
    return rda.ScriptAnalysis(file_path=name, script_name=name, **kw)


def test_compile_report_counts_and_graph(an):
    a = _sa("a.py", trains_on_test_data=True, uses_all_data=True, is_exhaustive=True)
    b = _sa("b.py", issues=["x"], uses_samples=True, loaded_data_sources=["l1", "l2"])
    b.created_data_sources = ["o"]
    c = _sa("c.py", has_train_test_split=True, uses_all_data=True)
    c.created_data_sources = ["o"]
    r = an._compile_report([a, b, c])
    assert r.total_scripts == 3
    assert r.scripts_with_issues == 2
    assert r.scripts_missing_train_val_split == 1  # a only (c has split)
    assert r.scripts_training_on_test == 1
    assert r.exhaustive_vs_sample_breakdown == {"exhaustive_testing": 1, "sample_testing": 1}
    assert r.data_dependency_graph == {"b.py": ["l1", "l2"]}
    assert r.source_to_users == {"o": ["b.py", "c.py"]}
    datetime.fromisoformat(r.generated_at)


def test_recommendations_all(an):
    a = _sa("a.py", uses_samples=True, loaded_data_sources=["x", "y"])
    recs = an._generate_recommendations([a], 2, 3)
    assert recs[0].startswith("🔴 CRITICAL: 3 scripts")
    assert recs[1].startswith("⚠️  2 scripts lack")
    assert "1 scripts test on samples. Examples: a.py." in recs[2]
    assert recs[3].startswith("🔧")
    assert recs[4].startswith("🔗")
    assert recs[5].startswith("📌")
    assert len(recs) == 6


def test_recommendations_minimal(an):
    a = _sa("a.py", data_transformations=["scaling"], loaded_data_sources=["x"])
    recs = an._generate_recommendations([a], 0, 0)
    assert len(recs) == 1 and recs[0].startswith("📌")
    # no-transform script without loads does not trigger the transformation hint
    b = _sa("b.py")
    assert len(an._generate_recommendations([b], 0, 0)) == 1
    many = [_sa(f"s{i}.py", uses_samples=True) for i in range(5)]
    rec = an._generate_recommendations(many, 0, 0)[0]
    assert "Examples: s0.py, s1.py, s2.py." in rec and "s3.py" not in rec


def test_analyze_all_end_to_end(an, tmp_path, capsys):
    (tmp_path / "good.py").write_text("assert 1\n")
    (tmp_path / "bad.py").write_text("def (:\n")
    (tmp_path / "leaky.py").write_text("X = df\n")
    r = an.analyze_all()
    assert an.report is r
    assert r.total_scripts == 2
    assert {s.script_name for s in r.scripts_analyzed} == {"good.py", "leaky.py"}
    out = capsys.readouterr().out
    assert "✓ good.py" in out and "⚠️ leaky.py" in out
    assert "Found 3 scripts" in out


def test_report_save_and_print(an, tmp_path, capsys):
    s = _sa("x.py", issues=["boom"], warnings=["hmm"])
    t = _sa("t.py", trains_on_test_data=True)
    u = _sa("clean.py", warnings=["ignored"])
    r = rda.RepositoryAnalysisReport(
        generated_at="g",
        total_scripts=3,
        scripts_analyzed=[s, t, u],
        scripts_with_issues=2,
        exhaustive_vs_sample_breakdown={"exhaustive_testing": 4},
        recommendations=[f"rec{i}" for i in range(7)],
    )
    r.save_report(str(tmp_path / "o.json"))
    assert json.loads((tmp_path / "o.json").read_text())["total_scripts"] == 3
    r.print_report()
    out = capsys.readouterr().out
    assert "Scripts with issues: 2" in out
    assert "exhaustive_testing: 4 scripts" in out
    assert "❌ boom" in out and "⚠️  hmm" in out
    assert "t.py" in out
    assert "clean.py" not in out and "ignored" not in out
    assert "5. rec4" in out and "rec5" not in out


def _df():
    return pd.DataFrame(
        {"d": pd.date_range("2020-01-01", periods=10, freq="D")[::-1], "v": range(10)}
    )


def test_validate_chronological_split():
    s = rda.DataSplitValidator.validate_chronological_split(_df(), "d")
    assert s["train"] == (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-06"))
    assert s["validation"] == (pd.Timestamp("2020-01-07"), pd.Timestamp("2020-01-08"))
    assert s["test"] == (pd.Timestamp("2020-01-09"), pd.Timestamp("2020-01-10"))
    s2 = rda.DataSplitValidator.validate_chronological_split(
        _df(), "d", train_ratio=0.5, val_ratio=0.3, test_ratio=0.2
    )
    assert s2["train"][1] == pd.Timestamp("2020-01-05")
    assert s2["validation"][0] == pd.Timestamp("2020-01-06")
    assert s2["test"][0] == pd.Timestamp("2020-01-09")


def test_check_data_leakage_boundaries():
    f = rda.DataSplitValidator.check_data_leakage
    assert (
        f(("2020-01-01", "2020-01-05"), ("2020-01-06", "2020-01-09"), ("2020-01-10", "x")) is False
    )
    # equal boundary counts as leakage
    assert (
        f(("2020-01-01", "2020-01-06"), ("2020-01-06", "2020-01-09"), ("2020-01-10", "x")) is True
    )
    assert (
        f(("2020-01-01", "2020-01-05"), ("2020-01-06", "2020-01-10"), ("2020-01-10", "x")) is True
    )
    assert (
        f(("2020-01-01", "2020-01-09"), ("2020-01-06", "2020-01-08"), ("2020-01-10", "x")) is True
    )
