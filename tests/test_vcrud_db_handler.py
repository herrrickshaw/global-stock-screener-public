"""Offline tests for .vcrud_bin/db_handler.py. ``psycopg2`` is faked (see vcrud_support), so the SQL
is only recorded - never executed - and subprocess/git is faked where used."""

import gzip
import hashlib
import subprocess
import types
from datetime import datetime

import pytest

from tests.vcrud_support import FakeConn, load_vcrud

NOW = datetime(2026, 1, 2, 3, 4, 5)


@pytest.fixture
def v(monkeypatch):
    return load_vcrud(monkeypatch)


def handler(v, results=()):
    h = v.db.DatabaseHandler("postgresql://example.test/x")
    v.conn.log.clear()
    v.conn.results.extend(results)
    v.conn.commits = 0
    return h


def rec(v, **kw):
    base = dict(path="a/b.txt", size_bytes=10, checksum="c" * 64, branch="main", last_modified=NOW)
    base.update(kw)
    return v.db.FileRecord(**base)


# ---- FileRecord / construction -------------------------------------------------------------


def test_filerecord_defaults(v):
    r = rec(v)
    assert r.git_commit is None and r.compressed_size is None
    assert r.compression_ratio is None and r.retrieval_count == 0


def test_init_connects_and_creates_every_table_and_index(v):
    h = v.db.DatabaseHandler("postgresql://example.test/x")
    assert v.connect_calls == ["postgresql://example.test/x"] and h.conn is v.conn
    created = [e[0] for e in v.conn.statements("CREATE TABLE IF NOT EXISTS")]
    names = [s.split("IF NOT EXISTS ")[1].split(" ")[0] for s in created]
    assert names == [
        "branches",
        "files",
        "file_versions",
        "file_dedup",
        "retrieval_cache",
        "compression_stats",
    ]
    assert len(v.conn.statements("CREATE INDEX IF NOT EXISTS")) == 4
    assert v.conn.commits == 1


def test_connect_failure_is_logged_and_reraised(monkeypatch, caplog):
    boom = RuntimeError("no db")
    with pytest.raises(RuntimeError, match="no db"):
        # load_vcrud only fails when a handler is constructed, so construct inside the raise
        v = load_vcrud(monkeypatch, connect_error=boom)
        v.db.DatabaseHandler("postgresql://example.test/x")
    assert "Connection failed: no db" in caplog.text


def test_close_closes_connection_and_tolerates_none(v):
    h = handler(v)
    h.close()
    assert v.conn.closed
    h.conn = None
    h.close()  # no AttributeError


# ---- create_branch / create_file -----------------------------------------------------------


def test_create_branch_returns_id_and_commits(v):
    h = handler(v, [(7,)])
    assert h.create_branch("dev") == 7
    sql, params, _ = v.conn.log[0]
    assert sql.startswith("INSERT INTO branches") and params == ("dev",)
    assert v.conn.commits == 1


def test_create_file_without_commit_inserts_only_the_file_row(v):
    h = handler(v, [(3,), (42,)])
    r = rec(v, compressed_size=4, compression_ratio=0.4)
    assert h.create_file(r) == 42
    (ins,) = v.conn.statements("INSERT INTO files")
    assert ins[1] == ("a/b.txt", 3, 10, "c" * 64, None, 4, 0.4, NOW)
    assert not v.conn.statements("file_versions")


def test_create_file_with_commit_also_logs_a_create_version(v):
    h = handler(v, [(3,), (42,)])
    assert h.create_file(rec(v, git_commit="abc", compressed_size=4)) == 42
    (ver,) = v.conn.statements("INSERT INTO file_versions")
    assert "'CREATE'" in ver[0]
    assert ver[1][:4] == (42, "abc", 10, 4) and isinstance(ver[1][4], datetime)


# ---- read_file / list_files_by_branch ------------------------------------------------------


def row_for_record(**kw):
    row = dict(
        path="a/b.txt",
        size_bytes=10,
        checksum="c" * 64,
        branch="main",
        last_modified=NOW,
        git_commit="abc",
        compressed_size=5,
        compression_ratio=0.5,
        retrieval_count=2,
    )
    row.update(kw)
    return row


def test_read_file_miss_returns_none_without_bumping_the_counter(v):
    h = handler(v, [(3,), None])
    assert h.read_file("a/b.txt", "main") is None
    assert not v.conn.statements("UPDATE files")
    sel = v.conn.statements("SELECT * FROM files")[0]
    assert sel[1] == ("a/b.txt", 3) and sel[2] is v.extras.RealDictCursor


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="read_file does FileRecord(**row) on a SELECT * row, whose columns (id, file_path, "
    "branch_id, created_at, ...) are not FileRecord fields -> TypeError on every real hit",
)
def test_read_file_builds_a_record_from_a_real_schema_row(v):
    real_row = {
        "id": 9,
        "file_path": "a/b.txt",
        "branch_id": 3,
        "size_bytes": 10,
        "checksum": "c" * 64,
        "git_commit": "abc",
        "compressed_size": 5,
        "compression_ratio": 0.5,
        "last_modified": NOW,
        "retrieval_count": 2,
        "created_at": NOW,
        "updated_at": NOW,
    }
    h = handler(v, [(3,), real_row])
    got = h.read_file("a/b.txt", "main")
    assert v.conn.statements("retrieval_count=retrieval_count+1")[0][1] == (9,)
    assert got.path == "a/b.txt" and got.size_bytes == 10


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="list_files_by_branch has the same FileRecord(**row) mismatch with SELECT * rows",
)
def test_list_files_by_branch_builds_records_from_real_schema_rows(v):
    real_row = {"id": 1, "file_path": "x", "branch_id": 3, "size_bytes": 1}
    h = handler(v, [(3,), [real_row]])
    assert [r.path for r in h.list_files_by_branch("main")] == ["x"]


def test_read_file_hit_updates_counter_when_row_has_only_record_fields(v):
    # a row restricted to FileRecord fields (e.g. an aliased query) works; the counter UPDATE then
    # needs row["id"], which such a row lacks -> KeyError. Documents the id coupling.
    h = handler(v, [(3,), row_for_record()])
    with pytest.raises(KeyError):
        h.read_file("a/b.txt", "main")


def test_list_files_by_branch_orders_by_path_and_builds_records(v):
    rows = [row_for_record(path="a"), row_for_record(path="b", retrieval_count=0)]
    h = handler(v, [(3,), rows])
    out = h.list_files_by_branch("main")
    assert [r.path for r in out] == ["a", "b"] and out[0].retrieval_count == 2
    sql, params, factory = v.conn.statements("SELECT * FROM files WHERE branch_id")[0]
    assert (
        sql.endswith("ORDER BY file_path") and params == (3,) and factory is v.extras.RealDictCursor
    )


def test_list_files_by_branch_empty(v):
    assert handler(v, [(3,), []]).list_files_by_branch("main") == []


# ---- update_file / delete_file -------------------------------------------------------------


def test_update_file_missing_row_returns_false_and_does_not_commit_or_version(v):
    h = handler(v, [(3,), None])
    assert h.update_file(rec(v, git_commit="abc")) is False
    assert v.conn.commits == 1  # only create_branch committed
    assert not v.conn.statements("file_versions")


def test_update_file_with_commit_logs_update_version_with_the_found_id(v):
    h = handler(v, [(3,), (11,)])
    r = rec(v, git_commit="abc", compressed_size=4, compression_ratio=0.4)
    assert h.update_file(r) is True
    (upd,) = v.conn.statements("UPDATE files SET")
    assert upd[1] == (10, "c" * 64, 4, 0.4, NOW, "a/b.txt", 3)
    (ver,) = v.conn.statements("INSERT INTO file_versions")
    assert "'UPDATE'" in ver[0] and ver[1][:4] == (11, "abc", 10, 4)
    assert v.conn.commits == 2


def test_update_file_without_commit_skips_the_version_row_but_commits(v):
    h = handler(v, [(3,), (11,)])
    assert h.update_file(rec(v)) is True
    assert not v.conn.statements("file_versions") and v.conn.commits == 2


def test_delete_file_missing_returns_false(v):
    h = handler(v, [(3,), None])
    assert h.delete_file("a", "main", "abc") is False
    assert not v.conn.statements("DELETE FROM files") and v.conn.commits == 1


def test_delete_file_with_commit_logs_delete_version_then_removes_row(v):
    h = handler(v, [(3,), (11,)])
    assert h.delete_file("a/b.txt", "main", "abc") is True
    kinds = [e[0] for e in v.conn.log if "file_versions" in e[0] or "DELETE FROM files" in e[0]]
    assert "'DELETE'" in kinds[0] and kinds[1] == "DELETE FROM files WHERE id=%s"
    ver = v.conn.statements("INSERT INTO file_versions")[0]
    assert ver[1][:2] == (11, "abc")
    assert v.conn.statements("DELETE FROM files")[0][1] == (11,)
    assert v.conn.commits == 2  # create_branch + the delete itself


def test_delete_file_without_commit_has_no_version_row(v):
    h = handler(v, [(3,), (11,)])
    assert h.delete_file("a/b.txt", "main") is True
    assert not v.conn.statements("file_versions")
    assert v.conn.statements("DELETE FROM files")[0][1] == (11,)


# ---- stats / queries -----------------------------------------------------------------------


def test_get_branch_stats_converts_to_megabytes(v):
    mb = 1024**2
    stats = {
        "total_files": 4,
        "total_size": 8 * mb,
        "total_compressed": 2 * mb,
        "avg_ratio": 0.25,
        "best_ratio": 0.1,
        "worst_ratio": 0.9,
        "bytes_saved": 6 * mb,
    }
    out = handler(v, [(3,), stats]).get_branch_stats("main")
    assert out == {
        "branch": "main",
        "total_files": 4,
        "total_size_mb": 8.0,
        "compressed_size_mb": 2.0,
        "avg_compression_ratio": 0.25,
        "best_ratio": 0.1,
        "worst_ratio": 0.9,
        "bytes_saved_mb": 6.0,
    }


def test_get_branch_stats_treats_nulls_as_zero(v):
    nulls = dict.fromkeys(
        ("total_files", "total_size", "total_compressed", "avg_ratio", "best_ratio", "worst_ratio"),
        None,
    )
    nulls["bytes_saved"] = None
    out = handler(v, [(3,), nulls]).get_branch_stats("empty")
    assert out["total_files"] == 0 and out["total_size_mb"] == 0
    assert out["avg_compression_ratio"] == 0 and out["bytes_saved_mb"] == 0
    assert out["branch"] == "empty"


def test_find_duplicates_returns_rows_and_requires_two_or_more(v):
    rows = [{"checksum": "x", "duplicate_count": 2}]
    h = handler(v, [rows])
    assert h.find_duplicates() is rows
    sql = v.conn.log[0][0]
    assert "HAVING COUNT(*) > 1" in sql and "ORDER BY total_wasted DESC" in sql


def test_get_file_history_scopes_to_path_and_branch(v):
    rows = [{"operation": "UPDATE"}]
    h = handler(v, [(3,), rows])
    assert h.get_file_history("a/b.txt", "main") is rows
    sel = v.conn.statements("FROM file_versions fv")[0]
    assert sel[1] == ("a/b.txt", 3) and sel[0].endswith("ORDER BY fv.changed_at DESC")


def test_get_top_retrieval_files_default_and_explicit_limit(v):
    h = handler(v, [(3,), [{"file_path": "a"}], (3,), []])
    assert h.get_top_retrieval_files("main") == [{"file_path": "a"}]
    assert v.conn.statements("LIMIT")[0][1] == (3, 10)
    h.get_top_retrieval_files("main", limit=2)
    assert v.conn.statements("LIMIT")[1][1] == (3, 2)


def test_compare_branches_queries_both_directions(v):
    h = handler(
        v,
        [(1,), (2,), [{"file_path": "a"}], [{"file_path": "b"}, {"file_path": "c"}], {"count": 5}],
    )
    out = h.compare_branches("one", "two")
    assert out == {
        "branch1": "one",
        "branch2": "two",
        "only_in_branch1": ["a"],
        "only_in_branch2": ["b", "c"],
        "common_files": 5,
    }
    excepts = v.conn.statements("EXCEPT")
    assert excepts[0][1] == (1, 2) and excepts[1][1] == (2, 1)
    assert v.conn.statements("SELECT COUNT(*) as count")[0][1] == (1, 2)


# ---- CompressionManager --------------------------------------------------------------------


def test_compress_file_roundtrips_and_reports_ratio(v, tmp_path):
    p = tmp_path / "f.txt"
    data = b"abc" * 1000
    p.write_bytes(data)
    comp, ratio = v.db.CompressionManager.compress_file(str(p))
    assert gzip.decompress(comp) == data
    assert ratio == len(comp) / len(data) and 0 < ratio < 0.1
    assert v.db.CompressionManager.decompress_file(comp) == data


def test_compress_file_empty_file_has_zero_ratio_not_a_division_error(v, tmp_path):
    p = tmp_path / "empty"
    p.write_bytes(b"")
    comp, ratio = v.db.CompressionManager.compress_file(str(p))
    assert ratio == 0 and gzip.decompress(comp) == b""


def test_compute_checksum_matches_sha256_across_chunk_boundaries(v, tmp_path):
    for size in (0, 3, 4096, 4097, 10000):
        p = tmp_path / f"f{size}"
        data = bytes(range(256)) * (size // 256 + 1)
        data = data[:size]
        p.write_bytes(data)
        assert v.db.CompressionManager.compute_checksum(str(p)) == hashlib.sha256(data).hexdigest()


# ---- GitBranchScanner ----------------------------------------------------------------------


def fake_subprocess(v, monkeypatch, handler_fn):
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw))
        return handler_fn(cmd, kw)

    monkeypatch.setattr(
        v.db,
        "subprocess",
        types.SimpleNamespace(run=run, CalledProcessError=subprocess.CalledProcessError),
    )
    return calls


def test_get_all_branches_strips_star_markers_and_blank_lines(v, monkeypatch):
    out = "* main\n  dev\n\n  remotes/origin/x\n"
    calls = fake_subprocess(v, monkeypatch, lambda c, k: types.SimpleNamespace(stdout=out))
    assert v.db.GitBranchScanner("/repo").get_all_branches() == ["main", "dev", "remotes/origin/x"]
    assert calls[0][0] == ["git", "-C", "/repo", "branch", "-a"]
    assert calls[0][1] == {"capture_output": True, "text": True}


def test_get_branch_files_collects_metadata_and_skips_git_dir(v, monkeypatch, tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / ".git").mkdir()
    (tmp_path / "a.txt").write_bytes(b"hello")
    (tmp_path / "sub" / "b.txt").write_bytes(b"xy")
    (tmp_path / ".git" / "config").write_bytes(b"secret")

    def fn(cmd, kw):
        if "checkout" in cmd:
            return types.SimpleNamespace(stdout="")
        path = cmd[-1]
        return types.SimpleNamespace(stdout="deadbeef\n" if path == "a.txt" else "")

    calls = fake_subprocess(v, monkeypatch, fn)
    files = v.db.GitBranchScanner(str(tmp_path)).get_branch_files("dev")
    assert sorted(files) == ["a.txt", "sub/b.txt"]
    a = files["a.txt"]
    assert a["size"] == 5 and a["git_commit"] == "deadbeef"
    assert a["checksum"] == hashlib.sha256(b"hello").hexdigest()
    assert a["path_obj"] == tmp_path / "a.txt"
    assert files["sub/b.txt"]["git_commit"] is None  # empty git log output -> None
    assert calls[0][0] == ["git", "-C", str(tmp_path), "checkout", "dev"]
    assert calls[0][1] == {"capture_output": True, "check": True}


def test_get_branch_files_returns_empty_dict_when_checkout_fails(v, monkeypatch, caplog):
    def fn(cmd, kw):
        raise subprocess.CalledProcessError(1, cmd)

    fake_subprocess(v, monkeypatch, fn)
    assert v.db.GitBranchScanner("/repo").get_branch_files("nope") == {}
    assert "Git error on nope" in caplog.text


def test_get_commit_history_splits_lines_and_handles_empty(v, monkeypatch):
    out = {"v": "abc one\ndef two\n"}
    calls = fake_subprocess(v, monkeypatch, lambda c, k: types.SimpleNamespace(stdout=out["v"]))
    s = v.db.GitBranchScanner("/repo")
    assert s.get_commit_history("main", "a.txt") == ["abc one", "def two"]
    assert calls[0][0] == ["git", "-C", "/repo", "log", "--oneline", "main", "--", "a.txt"]
    out["v"] = "  \n"
    assert s.get_commit_history("main", "a.txt") == []


def test_fakeconn_helper_is_importable():
    assert FakeConn([1]).results == [1]


def test_compress_file_uses_gzip_level_six(v, tmp_path, monkeypatch):
    p = tmp_path / "f"
    p.write_bytes(b"abc")
    seen = []
    real = gzip.compress

    def spy(data, compresslevel=9):
        seen.append(compresslevel)
        return real(data, compresslevel=compresslevel)

    monkeypatch.setattr(v.db, "gzip", types.SimpleNamespace(compress=spy))
    v.db.CompressionManager.compress_file(str(p))
    assert seen == [6]


def test_compute_checksum_reads_in_4096_byte_chunks(v, monkeypatch):
    reads = []

    class FakeFile:
        def __init__(self):
            self.left = [b"x" * 4096, b"y"]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n):
            reads.append(n)
            return self.left.pop(0) if self.left else b""

    monkeypatch.setattr(v.db, "open", lambda path, mode: FakeFile(), raising=False)
    got = v.db.CompressionManager.compute_checksum("ignored")
    assert reads == [4096, 4096, 4096]
    assert got == hashlib.sha256(b"x" * 4096 + b"y").hexdigest()
