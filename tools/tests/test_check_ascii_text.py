"""Contract tests for ``tools/check_ascii_text.py``.

The checker is exercised as a subprocess, exactly as the pre-commit hook and
the hygiene workflow invoke it, so these tests pin the command-line contract:
the exit code, the ``path:line:col: U+XXXX NAME`` report format, binary
skipping and UTF-8 strictness. Non-ASCII inputs are built with ``chr`` and
byte escapes so this file stays ASCII under the checker it tests.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CHECKER = Path(__file__).resolve().parents[1] / "check_ascii_text.py"


def _run(*paths: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), *map(str, paths)],
        capture_output=True,
        text=True,
    )


def test_clean_ascii_file_passes(tmp_path):
    """A file of plain ASCII text exits 0 and reports nothing."""
    path = tmp_path / "clean.md"
    path.write_bytes(b"# Title\n\nPlain ASCII: dashes - quotes ' \" and +/-0.02.\n")

    result = _run(path)

    assert result.returncode == 0
    assert result.stdout.strip() == "ascii check: clean"
    assert result.stderr == ""


def test_em_dash_reported_with_line_and_column(tmp_path):
    """Each non-ASCII character is reported at its 1-based line and character column.

    The em dash sits behind a two-byte plus-minus sign, so a byte-based column
    would read 4; the contract is the character column, 3.
    """
    path = tmp_path / "dash.md"
    line_two = chr(0x00B1) + "x" + chr(0x2014) + " tail"
    path.write_text("first line\n" + line_two + "\n", encoding="utf-8")

    result = _run(path)

    assert result.returncode == 1
    reported = result.stderr.splitlines()
    assert f"{path}:2:1: U+00B1 PLUS-MINUS SIGN" in reported
    assert f"{path}:2:3: U+2014 EM DASH" in reported


def test_binary_file_is_skipped(tmp_path):
    """A file with a NUL byte in its first 8 KB is skipped, even with high bytes."""
    path = tmp_path / "slice.gz"
    path.write_bytes(b"\x1f\x8b\x08\x00\x00\x00" + b"\xe2\x80\x94\xff\xfe")

    result = _run(path)

    assert result.returncode == 0
    assert result.stderr == ""


def test_invalid_utf8_is_a_violation(tmp_path):
    """A text file that does not decode as UTF-8 fails and names the offending line."""
    path = tmp_path / "latin1.txt"
    path.write_bytes(b"ok\nna\xefve\n")

    result = _run(path)

    assert result.returncode == 1
    assert f"{path}:2: not valid UTF-8" in result.stderr


def test_missing_path_is_a_usage_error(tmp_path):
    """A path that is not a readable regular file exits 2, not 1."""
    result = _run(tmp_path / "does_not_exist.md")

    assert result.returncode == 2
    assert "not a regular file" in result.stderr
