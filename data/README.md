# Data

This repository does **not** redistribute any dataset. The ACIC22 Track-2
challenge data is publicly downloadable, and MIMIC-IV is available under a
PhysioNet credentialed-access agreement. Both carry their own licenses and
terms of use, and neither may be committed here.

Each subdirectory documents how to obtain one source, where to place it on
disk, and how the reproducers expect it to be laid out:

- [`acic22/`](acic22/README.md) — ACIC22 Track-2 3,400-cohort canonical.
- [`mimic_iv/`](mimic_iv/README.md) — MIMIC-IV v3.1 via PhysioNet.

The `.gitignore` at the repository root excludes the data files themselves, so
a populated `data/` tree stays local to your machine. Only these README files
are tracked.
