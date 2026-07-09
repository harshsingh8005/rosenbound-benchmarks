# Rosenbound Benchmarks

## What this repository is

This is a public reproducer for specific benchmark claims made by the
Rosenbound Clinical AI platform. It exists so that anyone — a reviewer, a
prospective design partner, an independent researcher — can clone the code,
provision the underlying public or credentialed-public datasets, run a single
notebook, and independently confirm a headline number rather than take it on
trust. The scope of this repository is benchmark **verification only**; the
full Rosenbound platform, including its causal-inference and self-improving-
memory components, is proprietary and is not published here.

Three benchmarks are covered:

- **ACIC22 Track-2** — the 2022 Atlantic Causal Inference Conference causal
  estimation challenge (public data).
- **MIMIC-IV W1 in-hospital mortality** — a calibrated in-hospital mortality
  classifier on MIMIC-IV v3.1 (credentialed-public data).
- **FAERS severity** — a serious-adverse-event classifier on FDA Adverse Event
  Reporting System data (public data, openFDA API + FDA quarterly dumps).

## Recent updates

**2026-07-09 — Leak fix + FAERS addition.** During external review,
`n_diagnoses` (from `hosp/diagnoses_icd`) was identified as a data-leakage
feature — that table has no timestamp and its records are discharge-coded
billing diagnoses, unknowable at the 24-hour prediction point. We replaced it
with `charlson_history`, a Charlson comorbidity index computed only from a
subject's prior admissions, and re-locked the demo targets (the AUROC decreased,
the expected direction after removing a leak). We also added a public reproducer
for the FAERS severity classifier under `benchmarks/faers_severity/`, and renamed
the attribution hygiene checker to a vendor-neutral name. See
[`RELEASE_NOTES.md`](RELEASE_NOTES.md) for details.

## Reproducing the results

Each benchmark is self-contained under `benchmarks/` and validates its output
against a checked-in `expected_results.json` contract.

- **ACIC22 Track-2**

  ```bash
  cd benchmarks/acic22_track2
  python run.py
  ```

  Expected outputs match `benchmarks/acic22_track2/expected_results.json`.
  Runs on public data; no credentials required. See
  `data/acic22/README.md` for the download instructions.

- **MIMIC-IV W1 mortality**

  ```bash
  cd benchmarks/mimic_iv_w1_mortality
  python run.py
  ```

  Expected outputs match
  `benchmarks/mimic_iv_w1_mortality/expected_results.json`. Requires a
  completed PhysioNet CITI training certificate and a signed MIMIC-IV data
  use agreement. See `data/mimic_iv/README.md`.

- **FAERS severity**

  ```bash
  cd benchmarks/faers_severity
  python run.py
  ```

  Expected outputs match `benchmarks/faers_severity/expected_results.json`.
  Runs on public data with no credentials; a small demo slice is committed so
  the run works offline. See `benchmarks/faers_severity/README.md`.

Install dependencies first with `pip install -r requirements.txt` (or create
the conda environment from `environment.yml`).

> **ACIC22 CI note.** ACIC22 Track-2 requires a manual data registration and
> click-through download; `data/acic22/fetch.py` gives the steps, but CI cannot
> verify that lane automatically and skips numeric reproduction when the archive
> is absent. See `data/acic22/README.md` for the manual steps.

## Repository structure

```
rosenbound-benchmarks/
├── README.md                     This file
├── LICENSE                       MIT license
├── CITATION.cff                  Machine-readable citation metadata
├── requirements.txt              Pinned pip dependencies
├── environment.yml               Conda environment (Python 3.11)
├── .github/workflows/            CI: reproduces each benchmark on a schedule
├── data/                         Per-source data acquisition instructions
│   ├── acic22/                   ACIC22 Track-2 canonical download
│   └── mimic_iv/                 MIMIC-IV v3.1 / PhysioNet access
├── benchmarks/                   One directory per reproducible claim
│   ├── acic22_track2/            Causal challenge reproducer
│   ├── mimic_iv_w1_mortality/    Mortality classifier reproducer
│   └── faers_severity/           Adverse-event severity reproducer
└── docs/                         Methodology, provenance, patent scope
```

Data files are never committed to this repository — only the instructions to
obtain them. Each benchmark directory carries a `run.py` entry point, an
`expected_results.json` contract, a `notebook.ipynb`, and a local `README.md`.

## Reproducibility tiers

This repository is explicit about what an outside party can verify versus what
remains proprietary. Four levels apply:

- **Demo.** Runs on open MIMIC-IV demo data, a sampled FAERS slice, or the
  public ACIC22 release. Fully verifiable in CI — `git clone` plus
  `pip install` reproduces the pinned numbers. This is the only level the
  committed targets and tolerances gate.
- **Credentialed public-method.** The same public-method pipeline run on a full
  credentialed corpus (e.g. the 546K-admission MIMIC-IV corpus). Expected values
  are recorded as future work — unpinned until validated across independent runs
  — so results here are informational rather than gated. See the
  `credentialed_public_method` key in each `expected_results.json`.
- **Externally verified (planned Q3–Q4 2026).** Third-party-certified metrics
  for the full-corpus proprietary pipeline, obtained without exposing source, via
  two routes now in setup:
  - **MedPerf (MLCommons)** — a versioned Docker container runs on data-owner
    infrastructure against fixed data; MLCommons certifies the output while the
    model internals stay inside the container.
  - **ACIC Data Challenge** — a predictions-only submission (counterfactuals and
    intervals, no code) scored by the organizers against held-out ground truth.
  Both routes are in setup; results will be linked here when live.
- **Internal (proprietary).** The closed Rosenbound pipeline, not reproducible
  from this repository. Full-corpus figures are labelled `internal_reference`
  with method-category rationale in `docs/patent_scope.md`; external
  verification for them goes through the externally-verified level above.

## What is NOT in this repo

> *"This repository contains reproducers for specific benchmark claims. The
> underlying Rosenbound platform includes proprietary components covered by
> USPTO provisional patent AGI-PROV-01 (filed 2026-03-22, docket AGI-PROV-01,
> sole inventor Harsh Singh). The following modules are OUT OF SCOPE for this
> repository: the Differentiable Causal Inference Engine (DCIE — patent
> Claim 2, comprising Neural Feature Extractor + DCSL + IS + NCAAP), the
> Verifiable Bounded Self-Modification framework (VBSM — Claim 3), the
> Persistent Self-Improving Memory (PSIM — Claim 1), the Modular Cognitive
> Substrate (MCS — Claim 4, including IMCP and SMCE), and the Hybrid Neuro-
> Symbolic Integration mechanisms (HNSI — Claim 5, comprising EPT + BGB +
> CGETA + MCLR). Where a Rosenbound benchmark uses one of those modules in the
> closed platform, this reproducer substitutes a public equivalent (e.g.,
> AIPW / DR-ATT / IV-2SLS / Rosenbaum sensitivity for the causal Pentagon's
> non-DCIE arms; LightGBM classical for W-lane baselines; medspaCy + scispaCy
> for clinical NER). Numbers reproduced here match the corresponding claim in
> the platform WITHOUT invoking the patented mechanism."*

A claim-by-claim breakdown of the module boundary is in
`docs/patent_scope.md`.

## License

Released under the MIT License. See [LICENSE](LICENSE).

## Citation

Machine-readable citation metadata is in [CITATION.cff](CITATION.cff). To cite
this repository:

```bibtex
@software{singh_rosenbound_benchmarks_2026,
  author  = {Singh, Harsh},
  title   = {Rosenbound Benchmarks — Reproducers for Clinical AI Platform Claims},
  version = {0.1.0},
  year    = {2026},
  url     = {https://github.com/harshsingh8005/rosenbound-benchmarks}
}
```

## Contact

Harsh Singh — <harsh22@bu.edu>

Questions, reproduction issues, and discrepancy reports are welcome on the
[issue tracker](https://github.com/harshsingh8005/rosenbound-benchmarks/issues).
