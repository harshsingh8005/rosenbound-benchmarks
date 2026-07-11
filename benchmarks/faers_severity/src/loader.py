"""Parse the FAERS demo slice into a modeling table.

Reads the gzip-JSONL slice written by :mod:`.fetch` (one slimmed openFDA report
per line) and returns one row per report: the severity label, demographics and
reporter context, the receive year that drives the temporal split, and the raw
drug and reaction lists that :mod:`.features` turns into the model matrix.
"""

from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
DEMO_SLICE = _HERE.parent / "data" / "faers_demo_slice.jsonl.gz"

LABEL_COL = "serious"
YEAR_COL = "year"

# openFDA patientonsetageunit codes -> multiplier to convert the age value to
# years. Codes outside this map (or a missing value) yield an unknown age.
_AGE_UNIT_TO_YEARS: dict[str, float] = {
    "800": 10.0,       # decade
    "801": 1.0,        # year
    "802": 1.0 / 12,   # month
    "803": 1.0 / 52,   # week
    "804": 1.0 / 365,  # day
    "805": 1.0 / (365 * 24),  # hour
}


@dataclass(frozen=True)
class FaersCohort:
    """The parsed report-level severity cohort.

    Attributes
    ----------
    frame
        One row per report. Carries the ``serious`` label, ``year`` (for the
        temporal split), demographic and reporter columns, and the ``drugs`` and
        ``reactions`` object columns consumed by feature construction.
    label_col, year_col
        Names of the label and split columns in ``frame``.
    """

    frame: pd.DataFrame
    label_col: str = LABEL_COL
    year_col: str = YEAR_COL

    @property
    def n(self) -> int:
        return int(len(self.frame))

    @property
    def prevalence(self) -> float:
        return float(self.frame[self.label_col].mean())


def resolve_demo_slice(path: str | os.PathLike[str] | None = None) -> Path:
    """Locate the demo slice: the argument, then ``FAERS_DEMO_SLICE``, then default."""
    for cand in (path, os.environ.get("FAERS_DEMO_SLICE"), DEMO_SLICE):
        if cand:
            p = Path(cand)
            if p.is_file():
                return p
    raise FileNotFoundError(
        f"FAERS demo slice not found (looked at {DEMO_SLICE}). Fetch it first: "
        "python -m benchmarks.faers_severity.src.fetch — see "
        "benchmarks/faers_severity/README.md."
    )


def _age_to_years(age, unit: str) -> float:
    """Convert an openFDA (age, unit) pair to years, or NaN when unusable."""
    if age is None:
        return np.nan
    try:
        value = float(age)
    except (TypeError, ValueError):
        return np.nan
    return value * _AGE_UNIT_TO_YEARS.get(str(unit), 1.0)


def load_faers_cohort(path: str | os.PathLike[str] | None = None) -> FaersCohort:
    """Load the demo slice into a report-level :class:`FaersCohort`."""
    slice_path = resolve_demo_slice(path)
    with gzip.open(slice_path, "rt", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    return build_cohort(records)


def build_cohort(records: list[dict]) -> FaersCohort:
    """Assemble a :class:`FaersCohort` from slimmed report dicts.

    Kept separate from :func:`load_faers_cohort` so tests can drive the pipeline
    on synthetic reports without touching disk.
    """
    rows = []
    for rec in records:
        received = str(rec.get("receivedate") or "")
        if len(received) < 4 or not received[:4].isdigit():
            continue
        rows.append({
            "safetyreportid": rec.get("safetyreportid"),
            YEAR_COL: int(received[:4]),
            LABEL_COL: 1 if str(rec.get("serious")) == "1" else 0,
            "sex": str(rec.get("sex") or ""),
            "age_yr": _age_to_years(rec.get("age"), rec.get("age_unit") or ""),
            "reporter_qualification": str(rec.get("reporter_qualification") or ""),
            "reporter_country": str(rec.get("reporter_country") or ""),
            "reporttype": str(rec.get("reporttype") or ""),
            "drugs": rec.get("drugs") or [],
            "reactions": rec.get("reactions") or [],
        })
    frame = pd.DataFrame(rows)
    return FaersCohort(frame=frame)


__all__ = [
    "FaersCohort",
    "LABEL_COL",
    "YEAR_COL",
    "DEMO_SLICE",
    "resolve_demo_slice",
    "load_faers_cohort",
    "build_cohort",
]
