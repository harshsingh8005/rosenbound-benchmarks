# ACIC22 Track-2 reproducer

## What this reproduces

The **2022 Atlantic Causal Inference Conference (ACIC22) data challenge**,
Track 2, scores treatment-effect estimation on semi-synthetic panels where the
ground-truth effect is known. Each of the 3,400 cohorts is a panel of roughly
500 healthcare practices observed over four years: practice-level covariates, a
practice-level treatment indicator, and a continuous per-practice-year
expenditure outcome, with treatment acting only in the post-intervention years.
The scored estimand is the sample average treatment effect on the treated
(SATT).

This reproducer estimates the SATT on a fixed, seeded 100-cohort sample using
three published, non-proprietary methods and reports how accurate and how
well-calibrated the estimates are. It exists so the headline numbers can be
checked against public data without any access to the closed platform. Pass
`--full` to run all 3,400 cohorts.

## Methods

Each cohort is first reduced to a cross-sectional `(covariates, treatment,
outcome)` problem with a practice-level difference-in-differences change score:
the outcome is the post-period mean expenditure minus the pre-period mean, so
each practice's time-invariant level differences out and, under parallel trends
conditional on the covariates, the effect of treatment on this change score
identifies the SATT.

- **AIPW** (`src/aipw.py`) — cross-fitted augmented inverse-probability
  weighting for the effect on the treated; consistent if either the propensity
  model or the outcome model is correct. Robins, Rotnitzky & Zhao (1994);
  Bang & Robins (2005); Chernozhukov et al. (2018).
- **DR-ATT** (`src/dr_att.py`) — doubly-robust ATT restricted to the region of
  covariate overlap via a Crump trim, which bounds the inverse-propensity
  weights. Hahn (1998); Crump et al. (2009); Farrell (2015).
- **Rosenbaum sensitivity** (`src/rosenbaum.py`) — the Gamma bound: how strong
  an unmeasured confounder would have to be to explain the estimate away,
  computed with a documented widening-bound simplification. Rosenbaum (2002),
  Ch. 4.

## How to run

```bash
pip install -r requirements.txt
python data/acic22/fetch.py                       # verify/extract the data
cd benchmarks/acic22_track2
python run.py                                     # 100-cohort canonical sample
```

`python run.py --ci` runs the same fixed-seed canonical sample used by
continuous integration; `python run.py --full` runs all 3,400 cohorts. The run
writes `results.json` and asserts it against `expected_results.json`, exiting
non-zero on any drift.

## Or use the notebook

```bash
jupyter notebook notebook.ipynb
```

The notebook walks through a single cohort, each estimator, the sensitivity
bound, and the aggregate comparison, cell by cell.

## Expected numbers

The canonical 100-cohort sample (seed 42) reproduces the contract in
`expected_results.json`:

| Method | Bias | RMSE | Coverage @ 95% | Mean CI width |
|--------|-----:|-----:|---------------:|--------------:|
| AIPW   | 26.01 | 38.79 | 0.72 | 91.59 |
| DR-ATT | 20.68 | 30.63 | 0.74 | 71.26 |

Rosenbaum median Gamma-to-zero: 1.00 (most cohort effects are not individually
significant at Gamma = 1 on this sample). Every metric is checked against a
tolerance band; the run fails on drift beyond it.

## Interpretation

The estimand is small relative to the year-to-year variation in the change
score, so per-cohort estimates are noisy and the intervals under-cover — an
honest reflection of what public doubly-robust methods achieve on these panels.
DR-ATT, which trims to the overlap region, is the more accurate and better
calibrated of the two arms. The reproduced numbers approach, without matching,
the internal figures recorded in `expected_results.json`.

## Internal vs public

> The full internal system (which additionally uses an instrumental-variables
> estimator, a deep counterfactual estimator, and a paired Rosenbaum inversion)
> achieves tighter numbers than the three methods reproduced here — see
> `expected_results.json` → `internal_reference`. Those extensions are
> commercial; contact the maintainers for a platform demonstration.

## Data provenance

Acquisition and integrity details are in
[`data/acic22/README.md`](../../data/acic22/README.md); the cohort filters and
determinism controls are in [`docs/data_provenance.md`](../../docs/data_provenance.md).

## Citations

- Mathematica (2022). *The 2022 American Causal Inference Conference Data
  Challenge.* <https://acic2022.mathematica.org/>
- Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of regression
  coefficients when some regressors are not always observed. *JASA*, 89(427).
- Bang, H., & Robins, J. M. (2005). Doubly robust estimation in missing data
  and causal inference models. *Biometrics*, 61(4).
- Chernozhukov, V., et al. (2018). Double/debiased machine learning for
  treatment and structural parameters. *The Econometrics Journal*, 21(1).
- Hahn, J. (1998). On the role of the propensity score in efficient
  semiparametric estimation of average treatment effects. *Econometrica*, 66(2).
- Crump, R. K., Hotz, V. J., Imbens, G. W., & Mitnik, O. A. (2009). Dealing
  with limited overlap in estimation of average treatment effects.
  *Biometrika*, 96(1).
- Farrell, M. H. (2015). Robust inference on average treatment effects with
  possibly more covariates than observations. *Journal of Econometrics*, 189(1).
- Rosenbaum, P. R. (2002). *Observational Studies* (2nd ed.). Springer.
