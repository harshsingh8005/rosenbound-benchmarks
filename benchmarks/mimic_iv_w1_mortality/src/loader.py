"""Assemble the MIMIC-IV W1 in-hospital-mortality cohort.

The W1 cohort is one row per **first ICU stay of a hospital admission**. The
prediction target is in-hospital death (``hospital_expire_flag`` on the
admission). Every predictor is measured inside the first 24 hours of the ICU
stay, so no information from after the prediction point (discharge time, death
time, later labs) can leak into the features.

Data source
-----------
MIMIC-IV v3.1 (Johnson et al., 2023, PhysioNet). The reproducer runs on the
open-access **MIMIC-IV Clinical Database Demo v2.2** (100 patients, ODbL
licence), fetched by ``data/mimic_iv/fetch.py``; the identical code path runs
on the full credentialed corpus when pointed at it (see the module README).

Only these tables are read, all gzip-compressed CSVs:

- ``hosp/admissions``  — admit/discharge times, admission type/location,
                          insurance, ``hospital_expire_flag`` (the label).
- ``hosp/patients``    — ``gender`` and ``anchor_age``.
- ``hosp/labevents``   — laboratory measurements (subset of itemids below).
- ``hosp/diagnoses_icd`` — coded diagnoses, reduced to a comorbidity count.
- ``icu/icustays``     — ICU stay windows (``intime``/``outtime``).
- ``icu/chartevents``  — bedside vitals (subset of itemids below).

The loader returns numeric first-24h aggregates plus the categorical
admission descriptors; :mod:`.features` encodes them into the model matrix.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Length of the post-admission observation window. Predictors are aggregated
# over ``[intime, intime + OBS_WINDOW]``; anything charted later is discarded so
# the feature set is available at the 24h prediction point.
OBS_WINDOW = pd.Timedelta(hours=24)

# ICU chartevents itemids (MIMIC-IV ``icu/d_items``). Several physiological
# signals are charted under more than one itemid (invasive vs non-invasive
# pressures, Fahrenheit vs Celsius temperature); each logical vital maps to the
# set of itemids that record it, aggregated together.
VITAL_ITEMIDS: dict[str, tuple[int, ...]] = {
    "heart_rate": (220045,),
    "sbp": (220050, 220179),           # arterial / non-invasive systolic
    "dbp": (220051, 220180),           # arterial / non-invasive diastolic
    "mbp": (220052, 220181),           # mean arterial / non-invasive
    "resp_rate": (220210, 224690),
    "spo2": (220277,),
    "temp_c": (223762,),               # Celsius; Fahrenheit handled below
    "temp_f": (223761,),               # Fahrenheit, converted to Celsius
}

# Hospital labevents itemids (MIMIC-IV ``hosp/d_labitems``). First-24h mean of
# each analyte; itemid variants (e.g. two glucose assays) are pooled.
LAB_ITEMIDS: dict[str, tuple[int, ...]] = {
    "creatinine": (50912,),
    "wbc": (51301, 51300),
    "hemoglobin": (51222,),
    "platelet": (51265,),
    "glucose": (50931, 50809),
    "sodium": (50983,),
    "potassium": (50971,),
    "bicarbonate": (50882,),
    "bun": (51006,),
    "lactate": (50813,),
}

# Physiologically plausible ranges; values outside are charting errors and are
# dropped before aggregation. Keys match VITAL_ITEMIDS / LAB_ITEMIDS logical
# names (temperature validated in Celsius after conversion).
_PLAUSIBLE: dict[str, tuple[float, float]] = {
    "heart_rate": (10, 300), "sbp": (20, 300), "dbp": (5, 200),
    "mbp": (10, 250), "resp_rate": (1, 80), "spo2": (10, 100),
    "temp": (25, 45),
    "creatinine": (0, 30), "wbc": (0, 200), "hemoglobin": (0, 30),
    "platelet": (0, 2000), "glucose": (0, 2000), "sodium": (80, 200),
    "potassium": (0, 15), "bicarbonate": (0, 60), "bun": (0, 300),
    "lactate": (0, 40),
}

CATEGORICAL_COLS = ("gender", "admission_type", "admission_location", "insurance")
NUMERIC_BASE_COLS = ("anchor_age", "n_diagnoses")
LABEL_COL = "hospital_expire_flag"


@dataclass(frozen=True)
class W1Cohort:
    """The assembled first-24h ICU-admission cohort.

    Attributes
    ----------
    frame
        One row per first ICU stay of an admission. Holds the categorical
        admission descriptors, ``anchor_age``, ``n_diagnoses``, the first-24h
        vital and lab means, and the ``hospital_expire_flag`` label.
    label_col
        Name of the binary in-hospital-mortality column in ``frame``.
    """

    frame: pd.DataFrame
    label_col: str = LABEL_COL

    @property
    def n(self) -> int:
        return int(len(self.frame))

    @property
    def prevalence(self) -> float:
        return float(self.frame[self.label_col].mean())


def resolve_demo_root(data_root: str | os.PathLike[str] | None = None) -> Path:
    """Locate the directory that holds ``hosp/`` and ``icu/`` MIMIC-IV tables.

    Resolution order: the ``data_root`` argument, then the
    ``MIMIC_IV_DEMO_PATH`` environment variable, then ``data/mimic_iv/demo``
    under the repository root. Each candidate and its immediate subdirectories
    are searched for one containing ``hosp/admissions.csv.gz``.

    Raises
    ------
    FileNotFoundError
        If no candidate contains the expected MIMIC-IV layout.
    """
    candidates: list[Path] = []
    if data_root is not None:
        candidates.append(Path(data_root))
    env = os.environ.get("MIMIC_IV_DEMO_PATH")
    if env:
        candidates.append(Path(env))
    candidates.append(Path(__file__).resolve().parents[3] / "data" / "mimic_iv" / "demo")

    for base in candidates:
        for cand in (base, *sorted(p for p in base.glob("*") if p.is_dir())):
            if (cand / "hosp" / "admissions.csv.gz").is_file():
                return cand
    raise FileNotFoundError(
        "MIMIC-IV demo tables not found (looked for hosp/admissions.csv.gz "
        f"under {[str(c) for c in candidates]}). Fetch them first: "
        "python data/mimic_iv/fetch.py — see data/mimic_iv/README.md."
    )


def _read(root: Path, module: str, table: str, **kw) -> pd.DataFrame:
    """Read one gzip CSV table from a MIMIC-IV module directory."""
    return pd.read_csv(root / module / f"{table}.csv.gz", **kw)


def _window_means(
    events: pd.DataFrame,
    itemid_map: dict[str, tuple[int, ...]],
    key: str,
) -> pd.DataFrame:
    """Mean ``valuenum`` per logical signal within each stay's 24h window.

    ``events`` must carry columns ``[key, "itemid", "valuenum", "_keep"]`` where
    ``_keep`` flags rows already restricted to the observation window. Returns a
    frame indexed by ``key`` with one mean column per entry in ``itemid_map``.
    Signals absent for a stay are left as NaN for downstream imputation.
    """
    keep = events[events["_keep"]]
    id_to_name = {iid: name for name, ids in itemid_map.items() for iid in ids}
    keep = keep.assign(_signal=keep["itemid"].map(id_to_name))
    keep = keep.dropna(subset=["_signal", "valuenum"])

    out = pd.DataFrame(index=pd.Index([], name=key))
    for name in itemid_map:
        rows = keep[keep["_signal"] == name]
        lo, hi = _PLAUSIBLE.get(name, _PLAUSIBLE.get(name.rstrip("_cf"), (-np.inf, np.inf)))
        rows = rows[(rows["valuenum"] >= lo) & (rows["valuenum"] <= hi)]
        means = rows.groupby(key)["valuenum"].mean()
        out = out.join(means.rename(name), how="outer")
    return out


def load_w1_cohort(data_root: str | os.PathLike[str] | None = None) -> W1Cohort:
    """Build the first-24h ICU-admission W1 cohort from MIMIC-IV tables.

    Parameters
    ----------
    data_root
        Optional path to the MIMIC-IV tables (see :func:`resolve_demo_root`).

    Returns
    -------
    W1Cohort
        One row per first ICU stay of an admission, with the in-hospital
        mortality label and all first-24h predictors.
    """
    root = resolve_demo_root(data_root)

    admissions = _read(
        root, "hosp", "admissions",
        usecols=["subject_id", "hadm_id", "admittime", "admission_type",
                 "admission_location", "insurance", "hospital_expire_flag"],
        parse_dates=["admittime"],
    )
    patients = _read(
        root, "hosp", "patients", usecols=["subject_id", "gender", "anchor_age"]
    )
    icustays = _read(
        root, "icu", "icustays",
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
        parse_dates=["intime", "outtime"],
    )

    # First ICU stay per admission (earliest intime).
    icustays = icustays.sort_values("intime").drop_duplicates("hadm_id", keep="first")

    cohort = (
        icustays.merge(admissions, on=["subject_id", "hadm_id"], how="inner")
        .merge(patients, on="subject_id", how="left")
    )
    cohort["_win_end"] = cohort["intime"] + OBS_WINDOW

    # Comorbidity load: number of distinct diagnosis codes recorded for the
    # admission (a coarse, leakage-free Charlson/Elixhauser proxy).
    dx = _read(root, "hosp", "diagnoses_icd", usecols=["hadm_id", "icd_code"])
    n_dx = dx.groupby("hadm_id")["icd_code"].nunique().rename("n_diagnoses")
    cohort = cohort.merge(n_dx, on="hadm_id", how="left")
    cohort["n_diagnoses"] = cohort["n_diagnoses"].fillna(0).astype(int)

    stay_window = cohort[["stay_id", "intime", "_win_end"]]
    cohort = cohort.set_index("stay_id")

    vitals = _read(
        root, "icu", "chartevents",
        usecols=["stay_id", "charttime", "itemid", "valuenum"],
        parse_dates=["charttime"],
    )
    vitals = vitals.merge(stay_window, on="stay_id", how="inner")
    vitals["_keep"] = (vitals["charttime"] >= vitals["intime"]) & (
        vitals["charttime"] <= vitals["_win_end"]
    )
    vit = _window_means(vitals, VITAL_ITEMIDS, key="stay_id")
    # Fold Fahrenheit temperature into a single Celsius column.
    temp_c = vit.get("temp_c")
    temp_f = vit.get("temp_f")
    temp = temp_c.copy() if temp_c is not None else pd.Series(index=vit.index, dtype=float)
    if temp_f is not None:
        conv = (temp_f - 32.0) * 5.0 / 9.0
        temp = temp.fillna(conv)
    temp = temp.clip(*_PLAUSIBLE["temp"])
    vit = vit.drop(columns=[c for c in ("temp_c", "temp_f") if c in vit.columns])
    vit["temperature"] = temp
    cohort = cohort.join(vit, how="left")

    # Labs are keyed on hadm_id and windowed against the ICU intime.
    labs = _read(
        root, "hosp", "labevents",
        usecols=["hadm_id", "charttime", "itemid", "valuenum"],
        parse_dates=["charttime"],
    )
    lab_window = cohort.reset_index()[["hadm_id", "intime", "_win_end"]]
    labs = labs.merge(lab_window, on="hadm_id", how="inner")
    labs["_keep"] = (labs["charttime"] >= labs["intime"]) & (
        labs["charttime"] <= labs["_win_end"]
    )
    lab = _window_means(labs, LAB_ITEMIDS, key="hadm_id")
    cohort = cohort.merge(lab, on="hadm_id", how="left")

    # Deterministic column order for reproducible feature matrices.
    ordered = [c for c in _canonical_columns() if c in cohort.columns]
    frame = cohort.reset_index(drop=True)[ordered]
    frame[LABEL_COL] = frame[LABEL_COL].astype(int)
    return W1Cohort(frame=frame)


def _canonical_columns() -> list[str]:
    """Deterministic feature/label column order for the cohort frame."""
    vitals = [v for v in VITAL_ITEMIDS if v not in ("temp_c", "temp_f")] + ["temperature"]
    return (
        list(CATEGORICAL_COLS)
        + list(NUMERIC_BASE_COLS)
        + vitals
        + list(LAB_ITEMIDS.keys())
        + [LABEL_COL]
    )
