# Tests for the initial-conditions preflight — pure logic, no network/heavy build.
import os
from pathlib import Path

import preflight as pf


def test_importable_helper():
    assert pf._importable("pandas") is True
    assert pf._importable("no_such_module_xyz") is False


def test_lfs_pointer_detection(tmp_path):
    ptr = tmp_path / "p.parquet"
    ptr.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\n")
    assert pf._is_lfs_pointer(ptr) is True
    real = tmp_path / "r.parquet"
    real.write_bytes(b"PAR1\x00\x00realparquetbytes")
    assert pf._is_lfs_pointer(real) is False


def test_ensure_sets_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("BHAV_CACHE", raising=False)
    monkeypatch.setattr(pf, "SEED_DIR", tmp_path)  # avoid touching real data
    monkeypatch.setattr(pf, "LTM_DIR", tmp_path / "ltm")
    # build_serving False → no heavy work; just verify cache condition is established
    out = pf.ensure(["IN"], build_serving=False, verbose=False)
    assert os.environ.get("BHAV_CACHE")
    assert Path(out["cache"]).exists()


def test_check_returns_report_and_flags_missing_lfs(monkeypatch, tmp_path):
    # point LTM_DIR at an empty dir → LFS condition must be CRITICAL / not ready
    monkeypatch.setattr(pf, "LTM_DIR", tmp_path / "empty_ltm")
    (tmp_path / "empty_ltm").mkdir()
    ok, report = pf.check(["IN"], verbose=False)
    assert "LFS data" in report and "CRITICAL" in report["LFS data"]
    assert ok is False
