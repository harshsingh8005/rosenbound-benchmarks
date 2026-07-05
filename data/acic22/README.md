# ACIC22 Track-2 data

The ACIC22 Track-2 causal-estimation challenge data is public. The reproducer
in `benchmarks/acic22_track2/` reads the **3,400-cohort canonical** used for
the V3 lock.

## Obtain the data

The Track-2 datasets are distributed through the American Causal Inference
Conference 2022 data challenge:

- Challenge page: <https://acic2022.mathematica.org/>

Download the Track-2 archive (`track2_20220404.zip`) and place it in this
directory, then run the fetch helper to verify and extract it:

```bash
python data/acic22/fetch.py
```

`fetch.py` is idempotent: it verifies the archive against the pinned checksum,
extracts it if the cohorts are not already present, and does nothing if they
are. Set `ACIC22_DATA_ROOT` to keep the data outside this directory.

## Expected on-disk structure

After extraction, this directory contains the per-cohort files that the
reproducer iterates over:

```
data/acic22/
├── track2_20220404.zip         the downloaded archive (git-ignored)
├── track2/
│   ├── practice/               practice-level covariates, per cohort
│   └── practice_year/          practice-by-year outcome + treatment, per cohort
└── ACIC_estimand_truths.csv    ground-truth SATT per cohort
```

There are 3,400 cohorts (`acic_practice_0001.csv` … `acic_practice_3400.csv`).
The reproducer resolves the extracted `track2/` root relative to this
directory. Keep the archive alongside the extracted tree so the checksum can be
re-verified.

## Integrity

Verify the downloaded archive against the pinned checksum before running:

```bash
sha256sum track2_20220404.zip
# expected: 188890aa7ff3f7a90d20569bf7320d547edd03eea31c772c4148ab0a3785f0a7
```

`fetch.py` performs this check automatically and refuses to extract a
mismatched archive.

## Disk footprint

The extracted 3,400-cohort canonical is on the order of a few hundred MB. Budget
roughly **1 GB** to hold both the archive and the extracted tree during a run.
