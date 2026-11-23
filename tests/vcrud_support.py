"""Shared offline fakes for the .vcrud_bin tests: a fake ``psycopg2`` plus a loader that imports the
real ``db_handler`` / ``vcrud_manager`` source files under it. Nothing connects to a database."""

import importlib.util
import sys
import types
from pathlib import Path

VCRUD_BIN = Path(__file__).resolve().parent.parent / ".vcrud_bin"


class FakeCursor:
    """Records every execute(); fetchone()/fetchall() pop from the connection's scripted results."""

    def __init__(self, conn, factory=None):
        self.conn, self.factory = conn, factory

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.conn.log.append((" ".join(sql.split()), params, self.factory))

    def fetchone(self):
        return self.conn.results.pop(0)

    def fetchall(self):
        return self.conn.results.pop(0)


class FakeConn:
    def __init__(self, results=()):
        self.results = list(results)
        self.log = []
        self.commits = 0
        self.closed = False

    def cursor(self, cursor_factory=None):
        return FakeCursor(self, cursor_factory)

    def commit(self):
        self.commits += 1

    def close(self):
        self.closed = True

    def statements(self, needle):
        return [e for e in self.log if needle in e[0]]


def load_vcrud(monkeypatch, conn=None, connect_error=None):
    """Import the real db_handler + vcrud_manager with a fake psycopg2; sys.modules is restored."""
    conn = conn if conn is not None else FakeConn()
    calls = []

    def connect(url):
        calls.append(url)
        if connect_error:
            raise connect_error
        return conn

    extras = types.ModuleType("psycopg2.extras")
    extras.RealDictCursor = object()
    pg = types.ModuleType("psycopg2")
    pg.connect, pg.extras = connect, extras
    monkeypatch.setitem(sys.modules, "psycopg2", pg)
    monkeypatch.setitem(sys.modules, "psycopg2.extras", extras)
    mods = {}
    for name in ("db_handler", "vcrud_manager"):
        spec = importlib.util.spec_from_file_location(name, VCRUD_BIN / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, mod)
        spec.loader.exec_module(mod)
        mods[name] = mod
    return types.SimpleNamespace(
        db=mods["db_handler"],
        mgr=mods["vcrud_manager"],
        conn=conn,
        connect_calls=calls,
        extras=extras,
    )
