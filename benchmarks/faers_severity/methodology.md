# FAERS severity — methodology

This document describes the public methods the FAERS reproducer uses. None are
proprietary; the patented components of the Rosenbound platform (a causal-
inference layer and the ensemble head) are out of scope for this repository and
are enumerated in [`../../docs/patent_scope.md`](../../docs/patent_scope.md).

## Task

Predict whether an FAERS report is **serious** from the report's structured
fields. The label is the openFDA `serious` flag, which is set when the report
records any of: death, hospitalization, disability, a life-threatening
condition, or a congenital anomaly.

## Data

Public openFDA `/drug/event.json` reports (demo) or the FDA quarterly ASCII
dumps (full corpus). One modeling row per report. FAERS is a U.S. federal
government work in the public domain, with no data-use agreement.

## Temporal split and leakage control

Reports are split by **receive year**: earlier years train, the split year and
later test. The design goal is that nothing observable only after the prediction
point can inform the model:

- The top-K drug / reaction / indication vocabularies are ranked over the
  **training window only**. Ranking them over the full corpus would let
  test-era frequencies (for example, a newly frequent drug) select feature
  columns — a temporal leak. `tests/test_leak_free.py` asserts a token seen only
  in the test window never becomes a feature.
- The age-imputation median is fit on the training window and replayed on the
  test window.
- No feature is derived from the label.

## Features

All features are classical and symbolic — no learned text embeddings:

- **Demographics** — age in years (normalized from the openFDA age unit,
  median-imputed), an age-missing flag, one-hot sex.
- **Reporter context** — healthcare-professional and consumer flags (from the
  reporter qualification code) and a US-country flag.
- **Report structure** — distinct drug-role counts (suspect / concomitant /
  interacting, from `drugcharacterization`), a known-high-risk-drug flag
  (case-insensitive substring match against a fixed list of narrow-therapeutic-
  index drugs), and distinct reaction and indication counts.
- **Top-K bags** — presence indicators for the most frequent drugs, reactions,
  and indications in the training window.

## Model and metrics

A **LightGBM** gradient-boosted-trees classifier produces raw probabilities; no
post-hoc calibrator is applied in the public baseline. Metrics on the held-out
later window:

- **AUROC** and **AUPRC** (average precision) — discrimination. AUPRC is
  reported alongside AUROC because the severity classes are imbalanced.
- **Brier score** and **10-bin expected calibration error (ECE)** — calibration.

The expected values and tolerances are pinned in `expected_results.json`.

## Known limitations of FAERS

FAERS is a spontaneous-reporting system with well-documented biases: serious
events are over-reported relative to mild ones; the same event may be reported
multiple times with slight variation; confounding by indication and missing
demographic fields are common; and safety signals accumulate before label
changes, creating temporal bias. Predicted probabilities reflect FAERS report
frequencies, not population incidence, and should be read as a triage signal for
expert review rather than a diagnosis.
