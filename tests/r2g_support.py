"""Shared helper for group-G tests: import a script module fresh with fakes, restore afterwards."""

import importlib
import sys
import types

import pytest


@pytest.fixture
def loader():
    """loader(name, fakes=None) -> freshly imported module; sys.modules restored at teardown."""
    saved = {}
    touched = []

    def _load(name, fakes=None):
        for k, v in (fakes or {}).items():
            if k not in saved:
                saved[k] = sys.modules.get(k, None)
                touched.append(k)
            sys.modules[k] = v
        if name not in saved:
            saved[name] = sys.modules.get(name, None)
            touched.append(name)
        sys.modules.pop(name, None)
        return importlib.import_module(name)

    yield _load
    for k in touched:
        if saved[k] is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = saved[k]


def fake_module(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m
