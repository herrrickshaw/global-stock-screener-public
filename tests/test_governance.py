# Tests for the TOGAF governance compliance check — pure logic.
import governance as g


def test_run_covers_all_controls():
    r = g.run(verbose=False)
    assert len(r) == len(g.CHECKS)
    assert all(isinstance(v, tuple) and isinstance(v[0], bool) for v in r.values())


def test_docs_check_present():
    ok, msg = g._check_docs()
    assert ok is True and "docs" in msg.lower() or "content" in msg.lower()


def test_dep_map_and_tests_controls():
    assert g._check_dep_map()[0] is True  # DEPENDENCIES.md exists on the branch
    ok, msg = g._check_tests()
    assert ok is True and "test" in msg


def test_no_secrets_control():
    ok, msg = g._check_no_secrets()
    assert ok is True  # .env untracked, no hard-coded secrets
