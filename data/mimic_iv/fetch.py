"""Idempotent fetch for the open-access MIMIC-IV Clinical Database Demo v2.2.

The W1 reproducer's default (non-credentialed) data path is the **MIMIC-IV
Clinical Database Demo v2.2**: a 100-patient open-access subset released under
the Open Data Commons Open Database Licence, carrying the same ``hosp/`` and
``icu/`` table layout as the full corpus. This script downloads and extracts it
under ``data/mimic_iv/demo/`` and verifies every extracted file against the
SHA-256 manifest that ships inside the dataset.

It is safe to run repeatedly:

1. If the tables are already extracted and their checksums verify, it exits 0
   without downloading anything.
2. Otherwise it downloads the dataset zip, extracts it, and verifies the
   checksums, failing non-zero on any mismatch.

The full credentialed MIMIC-IV corpus is not fetched here; see
``data/mimic_iv/README.md`` for the credentialed path.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

# Open-access zip endpoint for the demo (PhysioNet). Anonymous download is
# permitted for this dataset under its ODbL licence.
DEMO_ZIP_URL = "https://physionet.org/content/mimic-iv-demo/get-zip/2.2/"
# Files whose presence marks a complete extraction.
_REQUIRED = (
    "hosp/admissions.csv.gz",
    "hosp/patients.csv.gz",
    "hosp/labevents.csv.gz",
    "hosp/diagnoses_icd.csv.gz",
    "icu/icustays.csv.gz",
    "icu/chartevents.csv.gz",
)
_MANIFEST = "SHA256SUMS.txt"


def _target_dir() -> Path:
    env = os.environ.get("MIMIC_IV_DEMO_PATH")
    return Path(env) if env else Path(__file__).resolve().parent / "demo"


def _dataset_root(base: Path) -> Path | None:
    """Return the directory holding hosp/admissions.csv.gz under ``base``."""
    for cand in (base, *sorted(p for p in base.glob("*") if p.is_dir())):
        if (cand / "hosp" / "admissions.csv.gz").is_file():
            return cand
    return None


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _verify_manifest(root: Path) -> list[str]:
    """Check extracted files against the dataset's SHA256SUMS manifest.

    Returns a list of human-readable problems (empty means every listed file
    that exists on disk matched). Manifest entries for files not present on
    disk are skipped; missing required tables are reported separately.
    """
    manifest = root / _MANIFEST
    if not manifest.is_file():
        return [f"manifest {_MANIFEST} missing under {root}"]
    problems: list[str] = []
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, rel = line.partition(" ")
        rel = rel.strip().lstrip("*").strip()
        if not rel:
            continue
        path = root / rel
        if not path.is_file():
            continue  # not every manifest entry is required by this reproducer
        actual = _sha256(path)
        if actual != digest:
            problems.append(f"checksum mismatch: {rel}\n    expected {digest}\n    got      {actual}")
    return problems


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "rosenbound-benchmarks/fetch"})
    with urllib.request.urlopen(req) as resp, dest.open("wb") as out:  # noqa: S310 (pinned https host)
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            out.write(block)


def main() -> int:
    base = _target_dir()
    base.mkdir(parents=True, exist_ok=True)

    root = _dataset_root(base)
    if root is not None:
        problems = _verify_manifest(root)
        if not problems:
            print(f"MIMIC-IV demo already present and verified at {root}; skipping.")
            return 0
        print("existing extraction failed verification; re-fetching:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)

    print(f"downloading MIMIC-IV demo from {DEMO_ZIP_URL} ...")
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "mimic-iv-demo.zip"
        try:
            _download(DEMO_ZIP_URL, zip_path)
        except OSError as exc:
            print(
                f"download failed: {exc}\n"
                f"Download the demo manually from {DEMO_ZIP_URL} and extract it "
                f"under {base}. See data/mimic_iv/README.md.",
                file=sys.stderr,
            )
            return 1
        print(f"extracting into {base} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(base)

    root = _dataset_root(base)
    if root is None:
        print(
            f"error: extraction did not produce the expected hosp/ + icu/ layout under {base}.",
            file=sys.stderr,
        )
        return 1

    problems = _verify_manifest(root)
    missing = [t for t in _REQUIRED if not (root / t).is_file()]
    if missing:
        print(f"error: required tables missing after extraction: {missing}", file=sys.stderr)
        return 1
    if problems:
        print("error: checksum verification failed:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    print(f"MIMIC-IV demo ready at {root} (checksums verified).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
