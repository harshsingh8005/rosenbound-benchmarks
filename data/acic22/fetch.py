"""Idempotent acquisition helper for the ACIC22 Track-2 data.

Running this script leaves the machine in a state where the reproducer can
find the extracted Track-2 cohorts under ``data/acic22/``. It is safe to run
repeatedly:

1. If the cohorts are already extracted, it verifies the file counts and
   exits without touching anything.
2. If the challenge archive is present but not yet extracted, it verifies the
   archive against the pinned SHA-256 and extracts it.
3. If neither is present, it prints the manual download instructions and exits
   non-zero. The Track-2 archive is distributed from the challenge site behind
   a click-through; this script never guesses a mirror URL.

The pinned checksum is the SHA-256 of the corrected Track-2 archive
(``track2_20220404.zip``) as published by the challenge organisers.
"""

from __future__ import annotations

import hashlib
import os
import sys
import zipfile
from pathlib import Path

EXPECTED_ZIP_SHA256 = (
    "188890aa7ff3f7a90d20569bf7320d547edd03eea31c772c4148ab0a3785f0a7"
)
_ARCHIVE_GLOB = "track2*.zip"
_EXPECTED_COHORTS = 3400
_CHALLENGE_URL = "https://acic2022.mathematica.org/data"


def _target_dir() -> Path:
    env = os.environ.get("ACIC22_DATA_ROOT")
    return Path(env) if env else Path(__file__).resolve().parent


def _extracted_root(base: Path) -> Path | None:
    """Return the directory containing practice/ + practice_year/, or None."""
    for cand in (base / "track2", base):
        if (cand / "practice").is_dir() and (cand / "practice_year").is_dir():
            return cand
    return None


def _count_cohorts(root: Path) -> int:
    return len(list((root / "practice").glob("acic_practice_*.csv")))


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    base = _target_dir()
    base.mkdir(parents=True, exist_ok=True)

    root = _extracted_root(base)
    if root is not None:
        n = _count_cohorts(root)
        if n >= _EXPECTED_COHORTS:
            print(f"ACIC22 Track-2 already extracted at {root} ({n} cohorts); skipping.")
            return 0
        print(
            f"warning: found {n} cohorts at {root}, expected {_EXPECTED_COHORTS}. "
            "The extraction looks incomplete; re-extract the archive.",
            file=sys.stderr,
        )

    archives = sorted(base.glob(_ARCHIVE_GLOB))
    if not archives:
        print(
            "ACIC22 Track-2 data not found.\n"
            f"  1. Download the Track-2 archive from {_CHALLENGE_URL}\n"
            f"  2. Place it in {base}\n"
            "  3. Re-run this script to verify the checksum and extract.\n"
            "See data/acic22/README.md for details.",
            file=sys.stderr,
        )
        return 1

    archive = archives[0]
    print(f"verifying {archive.name} against pinned SHA-256 ...")
    digest = _sha256(archive)
    if digest != EXPECTED_ZIP_SHA256:
        print(
            f"CHECKSUM MISMATCH for {archive.name}:\n"
            f"  expected {EXPECTED_ZIP_SHA256}\n"
            f"  got      {digest}\n"
            "Refusing to extract a mismatched archive.",
            file=sys.stderr,
        )
        return 1

    print(f"checksum ok; extracting {archive.name} -> {base} ...")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(base)

    root = _extracted_root(base)
    if root is None:
        print("error: extraction did not produce the expected layout.", file=sys.stderr)
        return 1
    print(f"extracted {_count_cohorts(root)} cohorts to {root}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
