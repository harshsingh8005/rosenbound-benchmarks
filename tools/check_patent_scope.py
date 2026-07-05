"""Fail if a proprietary-module name leaks outside the boundary documents.

The reproducer substitutes public methods for the platform's proprietary
modules and must never ship an implementation, import, or stray reference to
those modules. This checker scans the tracked tree for the proprietary module
names and exits non-zero if any appears outside the small set of documents
whose explicit job is to describe the public/proprietary boundary.

Run from the repository root::

    python tools/check_patent_scope.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# Proprietary module short-names and their internal sub-components.
_FORBIDDEN = [
    "dcie", "vbsm", "psim", "hnsi", "smce", "ncaap",
    "tarnet", "cfrnet", "dragonnet", "sinkhorn",
]

# Documents whose purpose is to state the boundary may name the modules, and
# this checker necessarily lists them.
_ALLOWED = {
    "README.md",
    "docs/patent_scope.md",
    "tools/check_patent_scope.py",
}

# The MIMIC reproducer is maintained separately; this checker guards the tree
# this reproducer owns and does not police a sibling benchmark's files.
_SKIP_PREFIXES = ("benchmarks/mimic_iv_w1_mortality/",)

_SCAN_SUFFIXES = {".py", ".ipynb", ".md", ".json", ".yml", ".yaml", ".cff", ".txt"}
_PATTERN = re.compile(r"\b(" + "|".join(_FORBIDDEN) + r")\b", re.IGNORECASE)


def _tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout
    return [line for line in out.splitlines() if line]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    offences: list[str] = []
    for rel in _tracked_files():
        if rel in _ALLOWED or Path(rel).suffix.lower() not in _SCAN_SUFFIXES:
            continue
        if rel.startswith(_SKIP_PREFIXES):
            continue
        path = root / rel
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if _PATTERN.search(line):
                offences.append(f"{rel}:{i}: {line.strip()[:120]}")

    if offences:
        print("proprietary-module names leaked outside boundary docs:", file=sys.stderr)
        for o in offences:
            print(f"  {o}", file=sys.stderr)
        return 1
    print("patent-scope check: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
