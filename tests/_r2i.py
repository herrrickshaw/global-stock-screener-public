"""Helper for group-I tests: import a repo module fresh under a sandboxed HOME."""

import importlib
import sys


def fresh(monkeypatch, tmp_path, name, env=None):
    """Import ``name`` anew with HOME redirected into tmp_path; restore sys.modules after."""
    monkeypatch.setenv("HOME", str(tmp_path))
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, str(v))
    monkeypatch.delitem(sys.modules, name, raising=False)
    mod = importlib.import_module(name)
    # monkeypatch.delitem on teardown restores the previous entry (or removes ours)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod
