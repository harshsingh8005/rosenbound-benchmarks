# MIMIC-IV v3.1 data

The MIMIC-IV W1 in-hospital mortality reproducer reads **MIMIC-IV v3.1**,
obtained through PhysioNet under credentialed access. This dataset is **not**
public and cannot be redistributed here; each user must provision it under
their own PhysioNet credentials.

## Prerequisites

1. A [PhysioNet](https://physionet.org/) account.
2. A completed **CITI "Data or Specimens Only Research"** training certificate,
   submitted to PhysioNet.
3. A **signed data use agreement (DUA)** for MIMIC-IV.

Once approved, download MIMIC-IV v3.1 from:

- <https://physionet.org/content/mimiciv/3.1/>

## Tables read by the W1 reproducer

The in-hospital mortality reproducer builds an admission-level feature matrix
from the hospital and ICU modules:

- `hosp/admissions` — admission/discharge times, admission type, mortality flag
- `hosp/patients` — demographics (anchor age, sex)
- `hosp/labevents` — laboratory measurements
- `icu/chartevents` — bedside vitals and charted observations
- `icu/icustays` — ICU stay windows

## Expected cohort size

MIMIC-IV v3.1 contains **546K admissions across 364K patients**. The W1 cohort
is derived from this population after the admission-level inclusion filters
documented in `docs/data_provenance.md`.

## On-disk structure

```
data/mimic_iv/
├── hosp/                       gzip-compressed hospital-module CSVs
│   ├── admissions.csv.gz
│   ├── patients.csv.gz
│   └── labevents.csv.gz
└── icu/                        gzip-compressed ICU-module CSVs
    ├── chartevents.csv.gz
    └── icustays.csv.gz
```

Everything under this directory except this README is git-ignored.

## Disk footprint

MIMIC-IV v3.1 is large — the compressed `chartevents` and `labevents` tables
dominate. Budget roughly **80-100 GB** of free disk for the compressed
download, plus additional working space for any decompressed intermediates.
