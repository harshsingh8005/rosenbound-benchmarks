# ACIC22 Track-2 data

The ACIC22 Track-2 causal-estimation challenge data is public. The reproducer
in `benchmarks/acic22_track2/` reads the **3,400-cohort canonical** used for
the V3 lock.

## Obtain the data

The Track-2 datasets are distributed through the American Causal Inference
Conference 2022 data challenge:

- Challenge page: <https://acic2022.mathematica.org/>

Download the Track-2 archive and place it in this directory.

## Expected on-disk structure

After extraction, this directory should contain the per-cohort files that the
reproducer iterates over:

```
data/acic22/
├── track2.zip                  the downloaded archive (git-ignored)
└── track2/                     extracted cohorts
    ├── practice/               per-cohort covariate + treatment + outcome CSVs
    └── ...                     3,400 cohorts total
```

The reproducer resolves the extracted `track2/` root relative to this
directory. Keep the archive alongside the extracted tree so the checksum can
be re-verified.

## Integrity

Verify the downloaded archive against the recorded checksum before running:

```bash
sha256sum track2.zip
# expected: {{ EXPECTED_ZIP_SHA256 }}
```

The `{{ EXPECTED_ZIP_SHA256 }}` placeholder is filled once the archive has been
downloaded and hashed a single time; the value is then pinned here so future
downloads can be validated against it.

## Disk footprint

The extracted 3,400-cohort canonical is on the order of a few hundred MB. Budget
roughly **1 GB** to hold both the archive and the extracted tree during a run.
