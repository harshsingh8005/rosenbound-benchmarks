# FAERS severity — data provenance

## Source

FDA Adverse Event Reporting System (FAERS), a database of adverse-event and
medication-error reports submitted to the FDA. FAERS is a work of the U.S.
federal government and is in the public domain; it carries no data-use
agreement and needs no credentials.

Two acquisition paths, both public:

| Path        | Source                                        | Used for      |
|-------------|-----------------------------------------------|---------------|
| openFDA API | `https://api.fda.gov/drug/event.json`         | demo slice    |
| FDA dumps   | `https://fis.fda.gov/extensions/FPD-QDE-FAERS/` | full corpus |

## Demo slice

The committed demo slice (`data/faers_demo_slice.jsonl.gz`, ~3,700 reports) was
built by `src/fetch.py` from the openFDA API over two fixed historical
receive-date windows — 2012 (training) and 2013 (test). Older windows are stable
(no ongoing backfill), so the same query returns the same reports and the demo
stays reproducible. Only the fields the pipeline consumes are retained, so the
slice is small enough to commit. Because FAERS is public domain, the slice is
committed directly rather than fetched at test time, which keeps continuous
integration offline and deterministic.

## Cohort and label

One row per report. The label is the openFDA `serious` field (1 = serious:
death, hospitalization, disability, life-threatening, or congenital anomaly;
0 otherwise). Reports with no reactions or no receive date are dropped, as they
cannot contribute a usable row.

## Split and determinism

The cohort is split temporally on receive year (train `< split_year`, test
`>= split_year`; the canonical demo uses 2013). Feature-vocabulary selection and
imputation are fit on the training window only. The pipeline pins its random
seed (LightGBM bagging and feature sampling) and runs single-threaded, so a
given `(seed, split_year)` reproduces the metrics in `expected_results.json`
bit-for-bit. A metric that drifts past tolerance fails the run.
