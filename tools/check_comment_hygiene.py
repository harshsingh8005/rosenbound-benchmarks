"""Fail if internal process labels leak into the public tree.

The public repository must read like a mature, independently authored
reproducer: no internal sprint labels, no work-item identifiers, no private
strategy-document references, no session dates or authoring-tool mentions.
This checker scans the tracked tree for those markers and exits non-zero on
any hit.

Run from the repository root::

    python tools/check_comment_hygiene.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# Each pattern targets an internal-process marker that must not ship publicly.
_PATTERNS = [
    (r"\bsub-?sprint\b", "sprint label"),
    (r"\bsprint\s*[0-9A-Z]", "sprint label"),
    (r"\bPR-\d+\b", "work-item id"),
    (r"\bW1[0-9]\b", "internal work-lane label"),
    (r"\bfounder\b", "founder reference"),
    (r"\bcowork\b", "authoring-tool reference"),
    (r"docs/strategy", "private strategy-doc reference"),
    (r"\bpath\s+d\b", "internal decision label"),
    (r"\boption\s+b\b", "internal decision label"),
    (r"founder-locked", "internal decision label"),
    (r"CLAUDE\.md", "authoring-tool reference"),
    (r"AGENTS\.md", "authoring-tool reference"),
    (r"\btier\s*[0-9]", "tier label"),
    (r"\bphase[- ][0-9]", "phase label"),
    (r"\bsub-?agent\b", "subagent reference"),
    (r"docs/(audits|handoffs)", "private doc reference"),
    (r"session_memory", "session-scratchpad reference"),
]
_COMPILED = [(re.compile(p, re.IGNORECASE), why) for p, why in _PATTERNS]

_SCAN_SUFFIXES = {".py", ".ipynb", ".md", ".json", ".yml", ".yaml", ".cff", ".txt"}
_SELF = "tools/check_comment_hygiene.py"


def _tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout
    return [line for line in out.splitlines() if line]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    offences: list[str] = []
    for rel in _tracked_files():
        # The checker itself necessarily contains the very patterns it hunts.
        if rel == _SELF or Path(rel).suffix.lower() not in _SCAN_SUFFIXES:
            continue
        path = root / rel
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            for rx, why in _COMPILED:
                if rx.search(line):
                    offences.append(f"{rel}:{i}: [{why}] {line.strip()[:120]}")

    if offences:
        print("internal process labels leaked into the public tree:", file=sys.stderr)
        for o in offences:
            print(f"  {o}", file=sys.stderr)
        return 1
    print("comment-hygiene check: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
