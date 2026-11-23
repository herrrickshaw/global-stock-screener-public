"""Offline tests for inventory_all_data (file scan / dedup report)."""

import json

import pytest

import inventory_all_data as ia

MB = 1024**2


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "cache_seed").mkdir(parents=True)
    (r / "other").mkdir()
    (r / ".git").mkdir()
    (r / "__pycache__").mkdir()
    (r / ".git" / "obj").write_bytes(b"gitjunk")
    (r / "__pycache__" / "x.pyc").write_bytes(b"pyc")
    return r


def _w(p, data):
    p.write_bytes(data)
    return p


def test_compute_hash(tmp_path):
    a = ia.DataInventoryAnalyzer(str(tmp_path))
    f = _w(tmp_path / "f", b"hello")
    assert a.compute_hash(f) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    big = _w(tmp_path / "big", b"z" * 200_000)
    assert len(a.compute_hash(big)) == 64
    assert a.compute_hash(tmp_path / "missing") is None
    assert a.compute_hash(_w(tmp_path / "e", b"")) == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_scan_skips_git_and_pycache(repo):
    _w(repo / "a.py", b"x")
    a = ia.DataInventoryAnalyzer(str(repo))
    rep = a.scan_repository()
    assert rep["summary"]["total_files"] == 1
    assert a.total_size == 1
    assert list(a.files_by_extension) == [".py"]


def test_scan_extension_and_no_ext(repo):
    _w(repo / "Makefile", b"abc")
    _w(repo / "d.csv", b"1,2")
    a = ia.DataInventoryAnalyzer(str(repo))
    a.scan_repository()
    assert set(a.files_by_extension) == {"no_extension", ".csv"}
    rec = a.files_by_extension[".csv"][0]
    assert rec["path"] == "d.csv" and rec["size"] == 3 and len(rec["hash"]) == 64


def test_scan_progress_every_100(repo, capsys):
    for i in range(100):
        _w(repo / "other" / f"f{i}.txt", str(i).encode())
    ia.DataInventoryAnalyzer(str(repo)).scan_repository()
    out = capsys.readouterr().out
    assert "Scanned 100 files..." in out
    assert "✓ Scanned 100 files total" in out


def test_scan_unreadable_file_recorded_without_hash(repo, monkeypatch):
    _w(repo / "a.txt", b"x")
    a = ia.DataInventoryAnalyzer(str(repo))
    monkeypatch.setattr(a, "compute_hash", lambda p: None)
    a.scan_repository()
    assert a.files_by_hash == {}
    assert a.files_by_extension[".txt"][0]["hash"] is None


def test_scan_skipped_on_exception(repo, monkeypatch, capsys):
    _w(repo / "a.txt", b"x")
    a = ia.DataInventoryAnalyzer(str(repo))

    def boom(p):
        raise OSError("nope")

    monkeypatch.setattr(a, "compute_hash", boom)
    a.scan_repository()
    assert "Skipped" in capsys.readouterr().out
    assert a.total_size == 0 and not a.files_by_extension


def test_duplicates_report(repo):
    _w(repo / "cache_seed" / "a.parquet", b"D" * 2 * MB)
    _w(repo / "other" / "b.parquet", b"D" * 2 * MB)
    _w(repo / "other" / "c.parquet", b"D" * 2 * MB)
    _w(repo / "other" / "u1.txt", b"same")
    _w(repo / "other" / "u2.txt", b"same")
    _w(repo / "other" / "solo.txt", b"unique")
    a = ia.DataInventoryAnalyzer(str(repo))
    rep = a.scan_repository()
    s = rep["summary"]
    assert s["total_files"] == 6
    assert s["duplicate_files"] == 2
    assert s["duplicate_size_mb"] == pytest.approx(4.0 + 4 / MB)
    assert s["potential_savings_mb"] == s["duplicate_size_mb"]
    assert s["total_size_mb"] == pytest.approx(6.0 + 14 / MB)
    dups = rep["duplicates"]
    assert dups[0]["copies"] == 3  # sorted by wasted space desc
    assert dups[0]["total_wasted_mb"] == pytest.approx(4.0)
    assert dups[0]["size_per_copy_mb"] == pytest.approx(2.0)
    assert dups[0]["hash"].endswith("...") and len(dups[0]["hash"]) == 11
    assert len(dups[0]["paths"]) == 3
    assert dups[1]["copies"] == 2
    assert dups[1]["recommendation"] == "Keep original, delete/symlink others"


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="inventory_all_data.py:~84: generate_report() does `self.duplicate_size +=` "
    "without resetting, so calling it twice (scan_repository then save_report) doubles the wasted space",
)
def test_generate_report_is_idempotent(repo):
    _w(repo / "a.txt", b"same")
    _w(repo / "b.txt", b"same")
    a = ia.DataInventoryAnalyzer(str(repo))
    r1 = a.generate_report()
    a.scan_repository()
    r2 = a.generate_report()
    assert r1["summary"]["duplicate_size_mb"] == r2["summary"]["duplicate_size_mb"]


def test_dup_recommendation(tmp_path):
    a = ia.DataInventoryAnalyzer(str(tmp_path))
    assert "Keep in cache_seed/" in a._get_dup_recommendation([{"path": "cache_seed/x.parquet"}])
    assert "Git LFS" in a._get_dup_recommendation([{"path": "d/x.parquet"}])
    assert "delete/symlink" in a._get_dup_recommendation([{"path": "d/x.csv"}])


def test_categorize(tmp_path):
    a = ia.DataInventoryAnalyzer(str(tmp_path))
    exts = {
        ".parquet": "parquet_files",
        ".csv": "csv_data",
        ".tsv": "csv_data",
        ".json": "json_config",
        ".yml": "json_config",
        ".yaml": "json_config",
        ".py": "code_files",
        ".sh": "code_files",
        ".md": "documentation",
        ".txt": "documentation",
        ".pkl": "binary_data",
        ".db": "binary_data",
        ".xyz": "other",
        "no_extension": "other",
    }
    for i, ext in enumerate(exts):
        a.files_by_extension[ext].append({"path": f"p{i}{ext}", "size": MB, "hash": "h"})
    cats = a._categorize_files()
    for ext, cat in exts.items():
        assert any(f["path"].endswith(ext) for f in cats[cat]["files"]), ext
    assert cats["other"]["count"] == 2
    assert cats["csv_data"]["count"] == 2 and cats["csv_data"]["size_mb"] == pytest.approx(2.0)
    assert cats["cache_data"]["count"] == 0


def test_categorize_shows_top5_only(tmp_path):
    a = ia.DataInventoryAnalyzer(str(tmp_path))
    for i in range(8):
        a.files_by_extension[".csv"].append({"path": f"{i}.csv", "size": 1, "hash": None})
    c = a._categorize_files()["csv_data"]
    assert c["count"] == 8 and len(c["files"]) == 5


def test_file_type_summary(tmp_path):
    a = ia.DataInventoryAnalyzer(str(tmp_path))
    a.files_by_extension[".a"] = [{"path": "x", "size": 1024}, {"path": "y", "size": 3072}]
    a.files_by_extension[".b"] = []
    s = a._file_type_summary()
    assert s[".a"] == {"count": 2, "size_mb": 4096 / MB, "avg_size_kb": 2.0}
    assert s[".b"]["avg_size_kb"] == 0 and s[".b"]["count"] == 0


def test_redundant_data(tmp_path):
    a = ia.DataInventoryAnalyzer(str(tmp_path))
    assert a._identify_redundant_data() == []
    a.files_by_extension[".csv"] = [
        {"path": "cache/a.csv", "size": MB},
        {"path": "screening_x.csv", "size": 2 * MB},
    ]
    a.files_by_extension[".parquet"] = [{"path": "p.parquet", "size": 3 * MB}]
    r = {x["type"]: x for x in a._identify_redundant_data()}
    assert r["Cache Data"]["count"] == 1 and r["Cache Data"]["size_mb"] == pytest.approx(1.0)
    assert r["Cache Data"]["priority"] == "HIGH"
    assert r["Parquet Data"]["count"] == 1 and r["Parquet Data"]["size_mb"] == pytest.approx(3.0)
    assert r["Screening Results"]["priority"] == "MEDIUM"
    assert r["Screening Results"]["size_mb"] == pytest.approx(2.0)
    # order: cache, parquet, screening
    assert [x["type"] for x in a._identify_redundant_data()] == [
        "Cache Data",
        "Parquet Data",
        "Screening Results",
    ]


def test_strategy_savings(tmp_path):
    a = ia.DataInventoryAnalyzer(str(tmp_path))
    s = a._create_strategy()
    assert s["expected_savings"]["git_repo_reduction_percent"] == 0
    assert [p["phase"] for p in s["phases"]] == [1, 2, 3, 4, 5]
    assert s["phases"][0]["status"] == "COMPLETE"
    a.total_size, a.duplicate_size = 200, 50
    e = a._create_strategy()["expected_savings"]
    assert e["git_repo_reduction_percent"] == 25.0
    assert e["redundant_data_mb"] == pytest.approx(50 / MB)


def test_empty_repo_report(tmp_path):
    rep = ia.DataInventoryAnalyzer(str(tmp_path)).scan_repository()
    assert rep["summary"]["total_files"] == 0
    assert rep["duplicates"] == [] and rep["redundant_data"] == []
    assert rep["repository"] == str(tmp_path)


def test_save_report_writes_json_and_prints(repo, tmp_path, capsys):
    _w(repo / "a.txt", b"same")
    _w(repo / "b.txt", b"same")
    a = ia.DataInventoryAnalyzer(str(repo))
    a.scan_repository()
    out = tmp_path / "rep.json"
    rep = a.save_report(str(out))
    data = json.loads(out.read_text())
    assert data["summary"]["total_files"] == 2 == rep["summary"]["total_files"]
    txt = capsys.readouterr().out
    assert "Total Files: 2" in txt and "Duplicate Files: 1" in txt
    assert f"Report saved: {out}" in txt


def test_main(repo, tmp_path, monkeypatch, capsys):
    for i in range(5):
        _w(repo / "other" / f"d{i}.txt", b"dup")
    monkeypatch.chdir(tmp_path)
    orig = ia.DataInventoryAnalyzer
    monkeypatch.setattr(ia, "DataInventoryAnalyzer", lambda _p: orig(str(repo)))
    ia.main()
    assert (tmp_path / "data_inventory_report.json").exists()
    out = capsys.readouterr().out
    assert "TOP DUPLICATES" in out
    assert "1. Wasted:" in out and "(5x copies)" in out
    assert "... and 2 more" in out
    assert "Action: Keep original, delete/symlink others" in out
    assert "NEXT STEPS" in out


def test_main_no_duplicates(repo, tmp_path, monkeypatch, capsys):
    _w(repo / "a.txt", b"1")
    monkeypatch.chdir(tmp_path)
    orig = ia.DataInventoryAnalyzer
    monkeypatch.setattr(ia, "DataInventoryAnalyzer", lambda _p: orig(str(repo)))
    ia.main()
    out = capsys.readouterr().out
    assert "1. Wasted" not in out and "NEXT STEPS" in out
