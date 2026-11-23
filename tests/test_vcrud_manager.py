"""Offline tests for .vcrud_bin/vcrud_manager.py. The database handler is replaced by a recording
double; the real FileRecord / CompressionManager from db_handler are used with real tmp_path files
(``psycopg2`` is faked), and git is never invoked."""

import hashlib
import types
from datetime import datetime

import pytest

from tests.vcrud_support import load_vcrud

MB = 1024**2


class FakeDB:
    def __init__(self):
        self.calls = []
        self.create_ids = iter(range(100, 1000))
        self.create_error = None
        self.update_result = True
        self.delete_result = True
        self.read_result = None
        self.stats = {
            "total_files": 5,
            "total_size_mb": 1.5,
            "avg_compression_ratio": 0.5,
        }
        self.top = []
        self.dups = []
        self.files = []
        self.compare = {"only_in_branch1": ["a"], "only_in_branch2": [], "common_files": 3}
        self.history = [{"operation": "CREATE"}]

    def create_file(self, record):
        self.calls.append(("create", record))
        if self.create_error and self.create_error(record):
            raise RuntimeError("db down")
        return next(self.create_ids)

    def update_file(self, record):
        self.calls.append(("update", record))
        return self.update_result

    def delete_file(self, path, branch, commit):
        self.calls.append(("delete", path, branch, commit))
        return self.delete_result

    def read_file(self, path, branch):
        self.calls.append(("read", path, branch))
        return self.read_result

    def get_branch_stats(self, branch):
        self.calls.append(("stats", branch))
        return self.stats

    def get_top_retrieval_files(self, branch, limit=10):
        self.calls.append(("top", branch, limit))
        return self.top

    def find_duplicates(self):
        return self.dups

    def list_files_by_branch(self, branch):
        return self.files

    def compare_branches(self, b1, b2):
        self.calls.append(("compare", b1, b2))
        return self.compare

    def get_file_history(self, path, branch):
        self.calls.append(("history", path, branch))
        return self.history


@pytest.fixture
def v(monkeypatch):
    return load_vcrud(monkeypatch)


@pytest.fixture
def mgr(v, tmp_path):
    db = FakeDB()
    return v.mgr.VersionedCRUDManager(db, str(tmp_path)), db


def made(v, **kw):
    base = dict(
        path="p", size_bytes=1, checksum="c", branch="b", last_modified=datetime(2026, 1, 1)
    )
    base.update(kw)
    return v.db.FileRecord(**base)


# ---- LocalFileIndexer ----------------------------------------------------------------------


def test_scan_directory_indexes_files_with_relative_paths_and_skips_git(v, tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / ".git").mkdir()
    (tmp_path / "a.txt").write_bytes(b"hello" * 50)
    (tmp_path / "d" / "b.txt").write_bytes(b"x")
    (tmp_path / ".git" / "HEAD").write_bytes(b"ref")
    recs = v.mgr.LocalFileIndexer(str(tmp_path)).scan_directory("dev")
    by_path = {r.path: r for r in recs}
    assert sorted(by_path) == ["a.txt", "d/b.txt"]  # dirs and .git contents are excluded
    a = by_path["a.txt"]
    assert a.branch == "dev" and a.size_bytes == 250
    assert a.checksum == hashlib.sha256(b"hello" * 50).hexdigest()
    assert a.compressed_size < 250 and a.compression_ratio == a.compressed_size / 250
    assert a.git_commit is None and isinstance(a.last_modified, datetime)


def test_scan_directory_honours_patterns(v, tmp_path):
    (tmp_path / "a.py").write_bytes(b"1")
    (tmp_path / "b.txt").write_bytes(b"2")
    recs = v.mgr.LocalFileIndexer(str(tmp_path)).scan_directory("dev", patterns=["*.py"])
    assert [r.path for r in recs] == ["a.py"]
    assert v.mgr.LocalFileIndexer(str(tmp_path)).scan_directory("dev", patterns=[]) == []


def test_scan_directory_skips_unreadable_files_but_keeps_the_rest(v, tmp_path, monkeypatch, caplog):
    (tmp_path / "bad.txt").write_bytes(b"1")
    (tmp_path / "ok.txt").write_bytes(b"2")
    real = v.mgr.CompressionManager.compute_checksum

    def flaky(path):
        if path.endswith("bad.txt"):
            raise OSError("denied")
        return real(path)

    monkeypatch.setattr(v.mgr.CompressionManager, "compute_checksum", staticmethod(flaky))
    recs = v.mgr.LocalFileIndexer(str(tmp_path)).scan_directory("dev")
    assert [r.path for r in recs] == ["ok.txt"]
    assert "Skipped" in caplog.text and "denied" in caplog.text


def test_scan_directory_on_an_empty_tree(v, tmp_path):
    assert v.mgr.LocalFileIndexer(str(tmp_path)).scan_directory("dev") == []


def test_github_remote_sync_is_a_stub_that_stores_its_settings(v):
    s = v.mgr.GitHubRemoteSync("https://example.test/r.git", token="t")
    assert (s.repo_url, s.token) == ("https://example.test/r.git", "t")
    assert s.get_remote_branches() == []
    assert s.get_remote_file_size("main", "a") is None
    assert v.mgr.GitHubRemoteSync("u").token is None


# ---- create / update / delete --------------------------------------------------------------


def test_create_file_missing_returns_minus_one_and_never_touches_the_db(mgr):
    m, db = mgr
    assert m.create_file("nope.txt", "main") == -1
    assert db.calls == []


def test_create_file_records_size_checksum_compression_and_commit(mgr, tmp_path):
    m, db = mgr
    (tmp_path / "f.txt").write_bytes(b"z" * 500)
    assert m.create_file("f.txt", "main", git_commit="abc") == 100
    ((op, r),) = db.calls
    assert op == "create" and r.path == "f.txt" and r.branch == "main"
    assert r.size_bytes == 500 and r.git_commit == "abc"
    assert r.checksum == hashlib.sha256(b"z" * 500).hexdigest()
    assert 0 < r.compression_ratio < 1 and r.compressed_size == round(r.compression_ratio * 500)


def test_create_file_defaults_git_commit_to_none(mgr, tmp_path):
    m, db = mgr
    (tmp_path / "f.txt").write_bytes(b"1")
    m.create_file("f.txt", "main")
    assert db.calls[0][1].git_commit is None


def test_create_many_collects_per_file_results_and_isolates_failures(mgr, tmp_path):
    m, db = mgr
    (tmp_path / "a").write_bytes(b"1")
    (tmp_path / "b").write_bytes(b"2")
    db.create_error = lambda rec: rec.path == "b"
    out = m.create_many("main", ["a", "b"], git_commit="c1")
    assert out == [("a", 100, "✓"), ("b", -1, "db down")]
    assert db.calls[0][1].git_commit == "c1"
    assert m.create_many("main", []) == []


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="create_many marks a missing file '✓' (create_file returns -1 without raising), so the "
    "batch log counts it as a successful create",
)
def test_create_many_does_not_mark_a_missing_file_successful(mgr):
    m, _ = mgr
    ((path, file_id, status),) = m.create_many("main", ["missing.txt"])
    assert file_id == -1 and status != "✓"


def test_update_file_missing_returns_false_without_db_call(mgr):
    m, db = mgr
    assert m.update_file("nope", "main") is False
    assert db.calls == []


@pytest.mark.parametrize("result", [True, False])
def test_update_file_returns_the_db_result_and_sends_a_fresh_record(mgr, tmp_path, result):
    m, db = mgr
    db.update_result = result
    (tmp_path / "f.txt").write_bytes(b"abcabc")
    assert m.update_file("f.txt", "dev", git_commit="c2") is result
    ((op, r),) = db.calls
    assert op == "update" and (r.path, r.branch, r.git_commit, r.size_bytes) == (
        "f.txt",
        "dev",
        "c2",
        6,
    )
    assert r.checksum == hashlib.sha256(b"abcabc").hexdigest()


@pytest.mark.parametrize("result", [True, False])
def test_delete_file_passes_through_to_the_db(mgr, result):
    m, db = mgr
    db.delete_result = result
    assert m.delete_file("a", "main", "c3") is result
    assert db.calls == [("delete", "a", "main", "c3")]
    m.delete_file("a", "main")
    assert db.calls[-1] == ("delete", "a", "main", None)


# ---- read ----------------------------------------------------------------------------------


def test_read_file_returns_record_or_none(mgr, v):
    m, db = mgr
    assert m.read_file("a", "main") is None
    rec = made(v, retrieval_count=4)
    db.read_result = rec
    assert m.read_file("a", "main") is rec
    assert db.calls[-1] == ("read", "a", "main")


def test_retrieval_hints_without_history_say_so(mgr):
    m, db = mgr
    out = m.read_with_retrieval_hints("main")
    assert out["branch"] == "main" and out["stats"] is db.stats
    assert out["top_retrieval_candidates"] == []
    assert out["recommendation"] == "No retrieval history yet. Start tracking file accesses."
    assert ("top", "main", 10) in db.calls


def test_retrieval_hints_recommend_the_first_file_with_traffic_share(mgr):
    m, db = mgr
    db.top = [
        {"file_path": "big.parquet", "retrieval_count": 3, "size_bytes": 2 * MB},
        {"file_path": "small", "retrieval_count": 1, "size_bytes": 10},
    ]
    rec = m.read_with_retrieval_hints("main")["recommendation"]
    assert rec == (
        "Pre-cache 'big.parquet' (2.0 MB, 75.0% of traffic). Total 5 files, 1.5 MB, 50.0% avg compression."
    )


# ---- scan_branch_and_index -----------------------------------------------------------------


class SizedPath:
    """A path_obj whose reported size is fake but whose bytes live in a real file."""

    def __init__(self, real, size):
        self.real, self.size = real, size

    def stat(self):
        return types.SimpleNamespace(st_size=self.size)

    def __str__(self):
        return str(self.real)


def scan_files(tmp_path, spec):
    out = {}
    for name, size in spec.items():
        real = tmp_path / name
        real.write_bytes(b"data" * 10)
        out[name] = {
            "size": size,
            "checksum": f"sum-{name}",
            "git_commit": f"commit-{name}",
            "path_obj": SizedPath(real, size),
        }
    return out


def test_scan_branch_compresses_only_files_under_100mb(mgr, tmp_path):
    m, db = mgr
    limit = 100 * 1024 * 1024
    m.scanner = types.SimpleNamespace(
        get_branch_files=lambda b: scan_files(
            tmp_path, {"under": limit - 1, "at": limit, "over": limit + 1}
        )
    )
    out = m.scan_branch_and_index("dev")
    recs = {r.path: r for _, r in (c for c in db.calls if c[0] == "create")}
    assert recs["under"].compressed_size is not None and recs["under"].compression_ratio > 0
    assert recs["at"].compressed_size is None and recs["over"].compression_ratio is None
    assert recs["under"].git_commit == "commit-under" and recs["under"].checksum == "sum-under"
    assert recs["under"].size_bytes == limit - 1 and recs["under"].branch == "dev"
    assert out == {"branch": "dev", "files_indexed": 3, "stats": db.stats}


def test_scan_branch_skips_files_that_fail_and_counts_the_rest(mgr, tmp_path, caplog):
    m, db = mgr
    m.scanner = types.SimpleNamespace(
        get_branch_files=lambda b: scan_files(tmp_path, {"a": 1, "b": 2, "c": 3})
    )
    db.create_error = lambda rec: rec.path == "b"
    out = m.scan_branch_and_index("dev")
    assert out["files_indexed"] == 2 and "Skipped b" in caplog.text


def test_scan_branch_with_no_files(mgr):
    m, db = mgr
    m.scanner = types.SimpleNamespace(get_branch_files=lambda b: {})
    assert m.scan_branch_and_index("dev") == {
        "branch": "dev",
        "files_indexed": 0,
        "stats": db.stats,
    }


def test_manager_constructor_wires_scanner_to_the_repo_path(mgr, tmp_path):
    m, _ = mgr
    assert m.repo_path == tmp_path and m.scanner.repo_path == str(tmp_path)


# ---- analysis ------------------------------------------------------------------------------


def test_find_duplicates_returns_db_rows_and_logs_waste(mgr, caplog):
    m, db = mgr
    db.dups = [
        {
            "checksum": "abcdef0123456789",
            "duplicate_count": 3,
            "total_wasted": 5 * MB,
            "branches": ["a", "b"],
        }
    ]
    with caplog.at_level("INFO"):
        assert m.find_duplicates_across_branches() is db.dups
    assert "Checksum abcdef01...: 3x, 5.0 MB wasted, branches: a, b" in caplog.text
    db.dups = []
    assert m.find_duplicates_across_branches() == []


def test_compare_branches_returns_the_db_comparison(mgr):
    m, db = mgr
    assert m.compare_branches("x", "y") is db.compare
    assert db.calls[-1] == ("compare", "x", "y")


def test_get_file_history_passes_through(mgr):
    m, db = mgr
    assert m.get_file_history("a", "main") is db.history
    assert db.calls[-1] == ("history", "a", "main")


def test_export_summary_shapes_every_file(mgr, v):
    m, db = mgr
    db.files = [
        made(
            v, path="a", size_bytes=2 * MB, checksum="s1", compression_ratio=0.5, retrieval_count=7
        ),
        made(v, path="b", size_bytes=MB // 2, checksum="s2"),
    ]
    db.top = [{"file_path": "a"}]
    out = m.export_summary("main")
    assert out["branch"] == "main" and out["total_files"] == 2
    assert out["statistics"] is db.stats and out["top_accessed_files"] == [{"file_path": "a"}]
    assert out["all_files"][0] == {
        "path": "a",
        "size_mb": 2.0,
        "compression_ratio": 0.5,
        "checksum": "s1",
        "retrieval_count": 7,
    }
    assert (
        out["all_files"][1]["size_mb"] == 0.5 and out["all_files"][1]["compression_ratio"] is None
    )
    assert ("top", "main", 20) in db.calls
    datetime.fromisoformat(out["timestamp"])


def test_optimize_storage_flags_only_ratios_strictly_above_the_threshold(mgr, v):
    m, db = mgr
    db.files = [
        made(v, path="exact", compression_ratio=0.85),
        made(v, path="over", compression_ratio=0.86, size_bytes=MB),
        made(v, path="none", compression_ratio=None),
        made(v, path="zero", compression_ratio=0),
    ]
    out = m.optimize_storage("main")
    assert [p["path"] for p in out["poor_compression_candidates"]] == ["over"]
    assert out["poor_compression_candidates"][0]["ratio"] == 0.86
    assert out["poor_compression_candidates"][0]["size_mb"] == 1.0
    assert out["optimization_potential"]["poor_compression_files"] == 1


def test_optimize_storage_custom_threshold_and_savings_math(mgr, v):
    m, db = mgr
    db.files = [made(v, path="a", compression_ratio=0.5, size_bytes=2 * MB)]
    db.dups = [{"total_wasted": 3 * MB}, {"total_wasted": None}]
    out = m.optimize_storage("main", min_compression_ratio=0.4)
    assert out["total_duplicate_waste_mb"] == 3.0
    assert out["optimization_potential"] == {
        "poor_compression_files": 1,
        "total_duplicate_files": 2,
        "estimated_savings_mb": 5.0,
    }
    assert m.optimize_storage("main")["optimization_potential"]["poor_compression_files"] == 0


def test_optimize_storage_lists_at_most_ten_candidates_but_counts_all(mgr, v):
    m, db = mgr
    db.files = [made(v, path=f"f{i}", compression_ratio=0.9) for i in range(12)]
    out = m.optimize_storage("main")
    assert len(out["poor_compression_candidates"]) == 10
    assert out["optimization_potential"]["poor_compression_files"] == 12
    assert out["poor_compression_candidates"][0]["note"].startswith("Consider alternative")


def test_update_file_logs_success_only_when_the_db_reports_it(mgr, tmp_path, caplog):
    m, db = mgr
    (tmp_path / "f.txt").write_bytes(b"1")
    with caplog.at_level("INFO"):
        db.update_result = False
        m.update_file("f.txt", "main")
        assert "Updated" not in caplog.text
        db.update_result = True
        m.update_file("f.txt", "main")
    assert "Updated: f.txt" in caplog.text
