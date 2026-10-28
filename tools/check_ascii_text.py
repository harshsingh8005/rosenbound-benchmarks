"""Fail if a tracked text file contains any character outside 7-bit ASCII.

Typographic characters (em and en dashes, arrows, box-drawing trees, curly
quotes, plus-minus and multiplication signs, non-breaking and zero-width
spaces) enter prose, docstrings and metadata through copy-paste and editor
autocorrect. They render differently across terminals, fonts and diff tools,
and an invisible or look-alike character can hide inside an identifier or a
pinned value. This checker keeps every tracked text file plain ASCII, so the
public tree reads, greps and diffs the same everywhere.

Invocation
----------
Run from the repository root::

    python tools/check_ascii_text.py [<path> ...]

Each path must be a regular file (the pre-commit hook passes the staged
files). With no path, every file listed by ``git ls-files`` that exists in the
working tree is scanned.

A file whose first 8 KB contain a NUL byte is treated as binary and skipped.
Every other file must decode as UTF-8; a file that does not is a violation.
Each character above U+007F is reported on its own line as
``path:line:col: U+XXXX NAME``, with 1-based line and column numbers and the
column counted in characters, not bytes. There is no allowlist and no inline
escape token.

Exit codes: 0 = clean, 1 = violation found, 2 = usage error (a path that is
not a readable regular file, an unknown option, or ``git ls-files`` failing
when no path is given).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import unicodedata
from pathlib import Path

# Prefix inspected for a NUL byte when deciding a file is binary. Git's own
# binary detection uses the same signal (a NUL in the first 8000 bytes).
BINARY_SNIFF_BYTES = 8192


class UsageError(Exception):
    """A path or environment problem that is not a finding (exit code 2)."""


def tracked_files() -> list[Path]:
    """Return the tracked files that exist in the working tree.

    Runs ``git ls-files`` in the current directory. Entries absent from the
    working tree (deleted but not yet staged) and non-files such as submodule
    gitlinks are skipped, so the default scan never fails on a path the user
    cannot see.

    Raises:
        UsageError: git is unavailable or the directory is not a repository.
    """
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"], capture_output=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise UsageError(f"cannot list tracked files with git ls-files: {exc}") from exc
    names = [name for name in out.decode("utf-8", "surrogateescape").split("\0") if name]
    return [Path(name) for name in names if Path(name).is_file()]


def scan_bytes(data: bytes, label: str) -> list[str]:
    """Return one finding per non-ASCII character in ``data``.

    Args:
        data: the raw file contents.
        label: the path string used as the prefix of each finding.

    Returns:
        An empty list when ``data`` is binary (a NUL in its first
        ``BINARY_SNIFF_BYTES``) or pure ASCII; a single finding when it is not
        valid UTF-8; otherwise one ``label:line:col: U+XXXX NAME`` string per
        character above U+007F, in file order. Lines split on ``\\n`` only, so
        line numbers match git and editors for both LF and CRLF files.
    """
    if b"\0" in data[:BINARY_SNIFF_BYTES]:
        return []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        line = data.count(b"\n", 0, exc.start) + 1
        return [f"{label}:{line}: not valid UTF-8 (byte 0x{data[exc.start]:02x} at offset {exc.start})"]
    if text.isascii():
        return []
    findings: list[str] = []
    for lineno, line in enumerate(text.split("\n"), start=1):
        if line.isascii():
            continue
        for col, ch in enumerate(line, start=1):
            if ord(ch) > 0x7F:
                name = unicodedata.name(ch, "<unnamed>")
                findings.append(f"{label}:{lineno}:{col}: U+{ord(ch):04X} {name}")
    return findings


def main(argv: list[str] | None = None) -> int:
    """CLI entry: scan the given paths (or every tracked file) and report.

    Findings and usage errors go to stderr; the clean message goes to stdout.
    Files are only read, never modified.

    Args:
        argv: arguments without the program name; ``None`` reads ``sys.argv``.

    Returns:
        The process exit code: 0 clean, 1 violation found, 2 usage error.
        ``argparse`` itself exits with 2 on an unknown option.
    """
    ap = argparse.ArgumentParser(
        description="Fail on non-ASCII characters in text files."
    )
    ap.add_argument("paths", nargs="*", type=Path,
                    help="Files to scan (default: every file from git ls-files).")
    args = ap.parse_args(argv)

    findings: list[str] = []
    files_with_findings = 0
    try:
        for path in args.paths or tracked_files():
            if not path.is_file():
                raise UsageError(f"not a regular file: {path}")
            try:
                data = path.read_bytes()
            except OSError as exc:
                raise UsageError(f"cannot read {path}: {exc}") from exc
            found = scan_bytes(data, str(path))
            findings.extend(found)
            files_with_findings += bool(found)
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if findings:
        for line in findings:
            print(line, file=sys.stderr)
        print(f"\n{len(findings)} non-ASCII finding(s) in {files_with_findings} file(s).",
              file=sys.stderr)
        return 1
    print("ascii check: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
