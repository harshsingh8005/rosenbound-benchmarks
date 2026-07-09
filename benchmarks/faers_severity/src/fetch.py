"""Fetch FAERS adverse-event reports for the severity benchmark.

Two acquisition modes, mirroring how a cloner would obtain the data:

- **demo** (default): page the public **openFDA** drug-event API
  (``api.fda.gov/drug/event.json``) over a fixed historical date range and
  write a slimmed, gzip-compressed JSONL slice. openFDA needs no credentials
  and imposes no data-use agreement; a modest slice fetches in a couple of
  minutes. The committed demo slice was produced by exactly this path.
- **full**: the openFDA API caps deep pagination (``skip`` <= 25000), so the
  full-corpus reproduction instead consumes the FDA quarterly ASCII dumps
  published at ``fis.fda.gov/extensions/FPD-QDE-FAERS/``. This module only
  prints the acquisition instructions for that mode; ingesting the multi-GB
  quarterly files is out of scope for the public demo.

Only the fields the modeling pipeline needs are retained, so the slice stays
small enough to commit. FAERS is a work of the U.S. federal government and is
in the public domain.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_OPENFDA_ENDPOINT = "https://api.fda.gov/drug/event.json"
_PAGE_LIMIT = 100          # openFDA hard cap per request without special access
_SKIP_CAP = 25000          # openFDA hard cap on skip-based pagination
_MAX_RETRIES = 5

_HERE = Path(__file__).resolve().parent
DEMO_SLICE = _HERE.parent / "data" / "faers_demo_slice.jsonl.gz"

# Fixed historical windows for the demo slice. Older receive-date ranges are
# stable (no ongoing backfill), so the same query returns the same reports and
# the demo stays reproducible. The split boundary sits between them.
DEMO_TRAIN_RANGE = ("20120101", "20121231")
DEMO_TEST_RANGE = ("20130101", "20131231")
DEMO_TRAIN_N = 2600
DEMO_TEST_N = 1100


def _slim_report(rec: dict) -> dict | None:
    """Reduce a raw openFDA report to the fields the pipeline consumes.

    Returns ``None`` for a report with no reactions or no receive date, which
    cannot contribute a usable row.
    """
    received = rec.get("receivedate")
    patient = rec.get("patient") or {}
    reactions = [
        r.get("reactionmeddrapt", "").strip().upper()
        for r in patient.get("reaction") or []
        if r.get("reactionmeddrapt")
    ]
    if not received or not reactions:
        return None

    drugs = []
    for d in patient.get("drug") or []:
        name = (d.get("medicinalproduct") or "").strip()
        if not name:
            continue
        drugs.append({
            "name": name.upper(),
            "char": str(d.get("drugcharacterization") or ""),
            "indication": (d.get("drugindication") or "").strip().upper() or None,
        })

    primary = rec.get("primarysource") or {}
    return {
        "safetyreportid": rec.get("safetyreportid"),
        "receivedate": received,
        "serious": str(rec.get("serious") or ""),
        "sex": str(patient.get("patientsex") or ""),
        "age": patient.get("patientonsetage"),
        "age_unit": str(patient.get("patientonsetageunit") or ""),
        "reporter_qualification": str(primary.get("qualification") or ""),
        "reporter_country": (primary.get("reportercountry")
                             or rec.get("occurcountry") or "").strip().upper(),
        "reporttype": str(rec.get("reporttype") or ""),
        "drugs": drugs,
        "reactions": reactions,
    }


def _fetch_page(date_range: tuple[str, str], skip: int, api_key: str | None) -> list[dict]:
    """Fetch one page of reports for a receive-date range, with backoff."""
    lo, hi = date_range
    search = f"receivedate:[{lo}+TO+{hi}]"
    params = f"search={search}&sort=receivedate:asc&limit={_PAGE_LIMIT}&skip={skip}"
    if api_key:
        params += f"&api_key={api_key}"
    url = f"{_OPENFDA_ENDPOINT}?{params}"

    for attempt in range(_MAX_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.load(resp).get("results", [])
        except urllib.error.HTTPError as exc:
            if exc.code == 404:            # openFDA returns 404 past the last page
                return []
            if exc.code == 429 and attempt < _MAX_RETRIES - 1:
                time.sleep(2.0 * (2 ** attempt))    # exponential backoff on rate limit
                continue
            raise
        except urllib.error.URLError:
            if attempt < _MAX_RETRIES - 1:
                time.sleep(2.0 * (2 ** attempt))
                continue
            raise
    return []


def _fetch_window(date_range: tuple[str, str], target: int, api_key: str | None) -> list[dict]:
    """Page a receive-date window until ``target`` slim reports are collected."""
    out: list[dict] = []
    skip = 0
    while len(out) < target and skip <= _SKIP_CAP:
        page = _fetch_page(date_range, skip, api_key)
        if not page:
            break
        for rec in page:
            slim = _slim_report(rec)
            if slim is not None:
                out.append(slim)
                if len(out) >= target:
                    break
        skip += _PAGE_LIMIT
        time.sleep(0.2)                     # stay well under the rate limit
    return out


def fetch_demo(out_path: Path = DEMO_SLICE, api_key: str | None = None) -> Path:
    """Build the demo slice from openFDA and write it as gzip JSONL."""
    print(f"fetching demo train window {DEMO_TRAIN_RANGE} (target {DEMO_TRAIN_N}) ...")
    train = _fetch_window(DEMO_TRAIN_RANGE, DEMO_TRAIN_N, api_key)
    print(f"  got {len(train)} reports")
    print(f"fetching demo test window {DEMO_TEST_RANGE} (target {DEMO_TEST_N}) ...")
    test = _fetch_window(DEMO_TEST_RANGE, DEMO_TEST_N, api_key)
    print(f"  got {len(test)} reports")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as fh:
        for rec in train + test:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    print(f"wrote {out_path} ({len(train) + len(test)} reports)")
    return out_path


_FULL_INSTRUCTIONS = """\
Full-corpus FAERS acquisition (public, no data-use agreement)
-------------------------------------------------------------
The openFDA API caps skip-based pagination at 25,000 records, so full-corpus
reproduction uses the FDA quarterly ASCII dumps instead:

  1. Download the quarterly files from
       https://fis.fda.gov/extensions/FPD-QDE-FAERS/
     Each quarter ships DEMO, DRUG, REAC, OUTC, THER, INDI, RPSR tables.
  2. Unpack them into a directory, one subdirectory per quarter.
  3. Point the loader at that directory (see benchmarks/faers_severity/README.md).

The quarterly dumps total multiple gigabytes and are not committed here.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("demo", "full"), default="demo")
    parser.add_argument("--output", type=Path, default=DEMO_SLICE)
    parser.add_argument("--api-key", default=None,
                        help="optional openFDA API key (raises the rate limit)")
    args = parser.parse_args(argv)

    if args.mode == "full":
        print(_FULL_INSTRUCTIONS)
        return 0
    try:
        fetch_demo(args.output, api_key=args.api_key)
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        print(f"error: openFDA fetch failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
