# FAERS adverse-event severity reproducer

## What this reproduces

This benchmark predicts whether an **FDA Adverse Event Reporting System
(FAERS)** report is **serious** — an event flagged for death, hospitalization,
disability, a life-threatening condition, or a congenital anomaly — from the
drug, reaction, reporter, and demographic fields of the report. It is the public
reproducer for the portfolio's FAERS severity headline number.

The pipeline uses only published, non-proprietary methods (symbolic features and
a LightGBM classifier), so the demo number can be checked without any access to
the closed platform. The data is public: FAERS is a work of the U.S. federal
government, distributed through the openFDA API and FDA quarterly dumps with no
data-use agreement.

## Leak-safe temporal split

Discrimination is measured on a **temporal split**: reports received before the
split year train the model, reports received on or after it are the held-out
test set. The top-K drug / reaction / indication vocabulary is ranked on the
**training window only**, so no later-era statistic (for example, a drug that
becomes frequent only after the split) can influence which features exist. This
is the same leak-safety principle behind the W1 Charlson fix, enforced on the
time axis; `tests/test_leak_free.py` asserts it directly.

## Methods

- **Fetch** (`src/fetch.py`) — pages the openFDA `/drug/event.json` API over a
  fixed historical range and writes a slimmed, gzip-compressed JSONL slice, with
  exponential backoff on rate limits. `--mode full` prints instructions for the
  FDA quarterly-dump path used at full-corpus scale.
- **Loader** (`src/loader.py`) — parses the slice into one row per report with
  the severity label, receive year, demographics, and drug/reaction lists.
- **Features** (`src/features.py`) — symbolic feature blocks (demographics,
  reporter context, drug-role counts, a known-high-risk-drug flag, and top-K
  drug/reaction/indication bags), with a fit/apply split so the vocabulary is
  learned on the training window only.
- **Model** (`src/model.py`) — a LightGBM classifier; raw (uncalibrated)
  probabilities.
- **Evaluation** (`src/evaluate.py`) — AUROC and AUPRC for discrimination,
  Brier score and 10-bin ECE for calibration.

## How to run (demo)

```bash
pip install -r requirements.txt
cd benchmarks/faers_severity
python run.py                 # canonical demo run (seed 42, split year 2013)
```

The committed demo slice (`data/faers_demo_slice.jsonl.gz`) makes the run work
offline and deterministically. To refetch it from openFDA:

```bash
python -m benchmarks.faers_severity.src.fetch          # writes the demo slice
```

`python run.py --ci` runs the same fixed-seed canonical configuration used by
continuous integration; it writes `results.json` and asserts the metrics against
`expected_results.json`, exiting non-zero on drift.

## How to run (full corpus)

The openFDA API caps deep pagination, so full-corpus reproduction consumes the
FDA quarterly ASCII dumps instead:

```bash
python -m benchmarks.faers_severity.src.fetch --mode full   # prints instructions
```

Download the quarterly files from
<https://fis.fda.gov/extensions/FPD-QDE-FAERS/> (DEMO, DRUG, REAC, OUTC, THER,
INDI tables per quarter). The dumps total multiple gigabytes and are not
committed here.

## Expected numbers

The canonical demo run reproduces the contract in `expected_results.json`
(3,700-report openFDA slice, temporal split, seed 42):

| Metric        | Demo target | Tolerance |
|---------------|------------:|----------:|
| AUROC         | 0.8885      | ±0.02     |
| AUPRC         | 0.9546      | ±0.02     |
| Brier score   | 0.1462      | ±0.02     |
| ECE (10-bin)  | 0.1142      | ±0.03     |

The demo slice is small and its serious-label prevalence shifts across the split
(0.86 train vs 0.71 test), so calibration is loose — the public-method pipeline
applies no post-hoc calibrator. Discrimination (AUROC/AUPRC) is the headline
measure.

## Internal vs public

> The full internal FAERS severity pipeline augments this LightGBM baseline with
> proprietary Rosenbound components (a patented causal-inference layer and an
> ensemble head) on a multi-quarter corpus. The number in
> `expected_results.json` → `internal_reference` (AUROC 0.8872) is from that
> system and is not reproducible here. This reproducer is the leak-safe
> public-method baseline only; the patented components are out of scope (see
> [`../../docs/patent_scope.md`](../../docs/patent_scope.md)).

## Citations

- U.S. Food and Drug Administration. FDA Adverse Event Reporting System (FAERS)
  Quarterly Data Files; openFDA drug-event API. Public domain.
- Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision
  tree. *Advances in Neural Information Processing Systems*, 30.
