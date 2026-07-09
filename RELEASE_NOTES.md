# Release Notes

## 2026-07-09 — Leak fix + FAERS reproducer

### Data-leakage fix in the W1 mortality reproducer

During an external audit, `n_diagnoses` was identified as a data-leakage
feature. It was sourced from `hosp/diagnoses_icd`, whose columns are
`subject_id, hadm_id, seq_num, icd_code, icd_version` — no timestamp. Those rows
are discharge-coded billing diagnoses, unknowable at the 24-hour prediction
window the methodology targets. An ablation on the demo cohort confirmed the
feature contributed spuriously to discrimination.

**Change.** `n_diagnoses` was replaced with `charlson_history`, a Charlson
comorbidity index computed only from a subject's admissions whose discharge
precedes the current admission's admit time — history that is fully coded and
available before the current stay begins. First-in-cohort admissions score 0.

We chose this replacement over simply dropping the feature because it is both
leak-safe and signal-bearing: 42% of the demo cohort (54 of 128 admissions) has
at least one qualifying prior admission, well above the threshold below which a
prior-history feature would be mostly zeros. The index uses the Quan (2005)
ICD-9-CM / ICD-10 code sets with the original Charlson (1987) weights and the
standard severity hierarchy.

**Re-locked demo targets** (calibrated metrics, 40-seed × 5-fold protocol):

| Metric            | Before | After  |
|-------------------|-------:|-------:|
| AUROC             | 0.6642 | 0.6197 |
| Average precision | 0.2284 | 0.2069 |
| Brier score       | 0.1047 | 0.1056 |
| ECE (10-bin)      | 0.0491 | 0.0472 |

`expected_results.json` and the test suite were updated, including new
data-free regression tests covering the index weights, the severity hierarchy,
and the prior-only time windowing. The new AUROC is lower than the leaked
baseline, which is the expected direction. A public-method 24-hour-window
MIMIC-IV mortality model in the ~0.62–0.68 AUROC range is consistent with the
published literature.

Credit: this leak was surfaced by an external reviewer. Finding and correcting
methodological bugs before they propagate is central to reproducible research.

### FAERS severity benchmark added

Added `benchmarks/faers_severity/` — a public reproducer for the FAERS severity
classifier using the openFDA API plus the FDA quarterly dumps. It uses symbolic
drug / reaction / demographic features and a LightGBM classifier on a leak-safe
temporal split: reports received before the split year train the model, later
reports test it, and the top-K feature vocabulary is ranked over the training
window only so no later-era statistic can select feature columns. A dedicated
`test_leak_free.py` asserts that a token seen only in the test window never
becomes a feature — the check that would have caught the W1 leak.

FAERS is public domain (no data-use agreement), so a 3,700-report demo slice is
committed and the smoke test runs offline and deterministically.

Demo metrics: AUROC 0.8885, AUPRC 0.9546, Brier 0.1462, ECE 0.1142. The internal
full-corpus reference (AUROC 0.8872, augmented with proprietary components) is
recorded under `internal_reference` and is not reproducible here.

### Hygiene checker renamed

`tools/check_no_claude_attribution.py` → `tools/check_no_ai_attribution.py`.
Vendor-neutral naming; the pre-commit hook ids and the hygiene workflow were
updated to match. No functional change.

### Reproducibility tiers documented

`README.md` and `docs/methodology.md` now describe four verification levels the
repository offers, so a reader knows what is checkable versus what stays
proprietary:

1. **Demo** — runs on open demo data or a sampled slice; fully verifiable in CI.
2. **Credentialed public-method** — the same public pipeline on a full
   credentialed corpus; expected baseline recorded as future work and not yet
   gated.
3. **Externally verified (planned Q3–Q4 2026)** — third-party-certified metrics
   for the full-corpus proprietary pipeline without exposing source, via a
   MedPerf (MLCommons) container submission and an ACIC Data Challenge
   predictions-only entry. Both routes produce a certified number without
   releasing the proprietary components; both are in setup and will be linked
   here when live.
4. **Internal** — the closed pipeline, not reproducible here; full-corpus figures
   are labelled `internal_reference`, with external verification routed through
   the externally-verified level above.

### External validation roadmap

Internal-reference numbers are claims until externally certified. The routes in
setup, tracked separately by the maintainers:

- **ACIC Data Challenge** (target ~Q3 2026) — submit generated predictions to the
  organizers; scored against held-out truth. Predictions-only, zero source
  exposure.
- **MedPerf (MLCommons)** (target ~Q4 2026) — a versioned Docker container runs
  on data-owner infrastructure against fixed data; MLCommons signs the certified
  metric. The container is a black box to reviewers.
- **NDA replication** (contingent on partner outreach) — proprietary code shared
  under signed NDA; a partner runs it on their credentialed instance and the
  result is published as a co-authored certified number, with the code kept
  confidential.

These routes convert an `internal_reference` figure from a bare claim into a
third-party-certified metric without releasing the proprietary components.
Results will be linked into this file when live.
