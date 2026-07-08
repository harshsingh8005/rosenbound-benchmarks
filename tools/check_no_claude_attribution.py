"""Block AI-assistant attribution from commits, code, and documentation.

Every artifact in this repository is reviewed and owned by its named human
maintainers. Tool-generated authorship trailers, "generated with" footers,
and vendor-name references must not ship in the public tree or in commit
messages.

Two invocation modes
--------------------
1. Content scan (default)::

       python tools/check_no_claude_attribution.py <path> [<path> ...]

   Each path may be a file or a directory (scanned recursively). Every
   matching line is reported and the process exits 1.

2. Commit-message scan (pre-commit ``commit-msg`` stage)::

       python tools/check_no_claude_attribution.py --commit-msg <FILE>

   ``FILE`` is the path Git passes to the ``commit-msg`` hook (i.e.
   ``.git/COMMIT_EDITMSG``). The message body is scanned for the same
   patterns.

Exit codes: 0 = clean, 1 = forbidden pattern found, 2 = usage error.

A line carrying the inline escape token (matched as a substring, so either
``# hygiene-allow`` or ``// hygiene-allow`` works) is exempt — reserved for
legitimate references such as a genuine runtime dependency, never to
silence a real attribution leak.

This file, ``check_comment_hygiene.py``, and ``.gitignore`` necessarily
contain the forbidden literals (as detection patterns or as blocked
filenames), so they exclude themselves from the scan.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Case-insensitive. Each entry is (human-readable label, compiled pattern).
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("AI co-authorship trailer", re.compile(r"Co-?Authored-?By:\s*Claude", re.I)),
    ("AI co-authorship trailer", re.compile(r"Co-?Authored-?By:\s*claude\.ai", re.I)),
    ("vendor noreply address", re.compile(r"noreply@anthropic\.com", re.I)),
    ("'generated with' footer", re.compile(r"Generated with\s*\[?Claude", re.I)),
    ("'generated with' robot footer", re.compile(r"\U0001f916\s*Generated with", re.I)),
    ("'powered by' vendor footer", re.compile(r"Powered by Anthropic", re.I)),
    ("'built with' vendor footer", re.compile(r"Built with Claude", re.I)),
    ("'drafted/written with' footer", re.compile(r"(?:Drafted|Written)\s+(?:with|by)\s+Claude", re.I)),
    ("model-id string in prose", re.compile(r"claude-(?:opus|sonnet|haiku|fable|mythos)-[0-9]", re.I)),
    # Work attributed to the assistant as a subject ("Claude Code caught this",
    # "Claude wrote the fix"). Catches narrative attribution the trailer/footer
    # patterns above miss.
    ("AI work-attribution",
     re.compile(r"\bClaude(?:\s+Code)?\s+(?:caught|wrote|added|fixed|implemented|"
                r"generated|built|drafted|created|authored|made|did|produced|patched)\b", re.I)),
    # Bare vendor/tool names. This repository has no legitimate runtime use of
    # either word; any future exception must carry the inline escape token.
    ("vendor/tool name", re.compile(r"\b(?:anthropic|claude)\b", re.I)),
]

# Files that legitimately contain the literals (as detection patterns or as
# blocked filenames in the ignore list).
SELF_EXCLUDE = {
    "check_no_claude_attribution.py",
    "check_comment_hygiene.py",
    ".gitignore",
}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", "htmlcov", ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints"}
TEXT_SUFFIXES = {
    ".py", ".ipynb", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".yml", ".yaml", ".toml", ".cff",
    ".md", ".txt", ".json", ".cfg", ".ini", ".sh", ".ps1", ".html", ".css",
}

# Inline escape hatch (substring match, so it works in either comment
# style). Use ONLY for legitimate references, never to silence a real
# attribution leak.
ALLOW_DIRECTIVE = "hygiene-allow"


def iter_files(paths: list[Path]):
    """Yield the files to scan (expanding directories recursively)."""
    for p in paths:
        if p.is_dir():
            for child in sorted(p.rglob("*")):
                if any(part in SKIP_DIRS for part in child.parts):
                    continue
                if child.is_file() and child.suffix.lower() in TEXT_SUFFIXES and child.name not in SELF_EXCLUDE:
                    yield child
        elif p.is_file() and p.name not in SELF_EXCLUDE:
            yield p


def scan_text(text: str, label_prefix: str, findings: list[str]) -> None:
    """Append any attribution matches found in ``text`` to ``findings``."""
    for lineno, line in enumerate(text.splitlines(), 1):
        if ALLOW_DIRECTIVE in line:
            continue
        for label, pat in PATTERNS:
            if pat.search(line):
                findings.append(f"{label_prefix}:{lineno}: [{label}] {line.strip()[:160]}")
                break


def main() -> int:
    """CLI entry: fail if any AI-attribution pattern is present."""
    ap = argparse.ArgumentParser(description="Block AI-assistant attribution.")
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--commit-msg", type=Path, default=None,
                    help="Path to the commit-message file (commit-msg hook stage).")
    args = ap.parse_args()

    findings: list[str] = []

    if args.commit_msg is not None:
        try:
            scan_text(args.commit_msg.read_text(encoding="utf-8", errors="replace"),
                      "commit-message", findings)
        except OSError as exc:
            print(f"error: cannot read commit message {args.commit_msg}: {exc}", file=sys.stderr)
            return 2

    for f in iter_files(args.paths):
        try:
            scan_text(f.read_text(encoding="utf-8", errors="replace"), str(f), findings)
        except OSError:
            continue

    if findings:
        print("Forbidden AI-attribution found (remove it; do NOT bypass with --no-verify):\n", file=sys.stderr)
        for line in findings:
            print(f"  {line}", file=sys.stderr)
        print(f"\n{len(findings)} match(es).", file=sys.stderr)
        return 1
    print("attribution check: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
