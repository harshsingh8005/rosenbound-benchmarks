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

Two flagship benchmarks are covered:

- **ACIC22 Track-2** — the 2022 Atlantic Causal Inference Conference causal
  estimation challenge (public data).
- **MIMIC-IV W1 in-hospital mortality** — a calibrated in-hospital mortality
  classifier on MIMIC-IV v3.1 (credentialed-public data).

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

Install dependencies first with `pip install -r requirements.txt` (or create
the conda environment from `environment.yml`).

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
│   └── mimic_iv_w1_mortality/    Mortality classifier reproducer
└── docs/                         Methodology, provenance, patent scope
```

Data files are never committed to this repository — only the instructions to
obtain them. Each benchmark directory carries a `run.py` entry point, an
`expected_results.json` contract, a `notebook.ipynb`, and a local `README.md`.

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
