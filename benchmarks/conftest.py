"""Isolate each benchmark's flat ``src``/``run`` modules during whole-tree runs.

Each benchmark under this directory is a self-contained project that imports its
own top-level ``src`` and ``run`` modules. When the whole tree is collected in a
single pytest session those names would otherwise collide in ``sys.modules`` and
a later benchmark would import an earlier one's modules. Evicting the shared
names before importing each test module forces every benchmark's tests to import
their own package. Per-benchmark runs (a single project on the path) are
unaffected.
"""

import sys

import pytest

_SHARED_ROOTS = ("src", "run")
_SHARED_PREFIXES = tuple(f"{root}." for root in _SHARED_ROOTS)


def _evict_shared_modules() -> None:
    for name in list(sys.modules):
        if name in _SHARED_ROOTS or name.startswith(_SHARED_PREFIXES):
            del sys.modules[name]


@pytest.hookimpl(wrapper=True)
def pytest_pycollect_makemodule(module_path, parent):
    _evict_shared_modules()
    return (yield)
