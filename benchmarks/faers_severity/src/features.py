"""Turn the FAERS report cohort into a numeric model matrix.

Feature construction is split into a fit step and an apply step. Everything
data-dependent — the top-K drug / reaction / indication vocabularies and the
age-imputation median — is learned in the fit step, which the caller runs on the
**training window only**. The apply step replays that fitted vocabulary on any
fold. Because the vocabulary never sees the test window, no test-era statistic
(e.g. a drug that only becomes frequent later) can influence which columns exist
— the temporal analogue of the leak this repository's W1 fix removed.

Feature blocks (all classical and symbolic, no learned embeddings):

- demographics: age in years (median-imputed), an age-missing flag, and
  one-hot sex;
- reporter context: healthcare-professional / consumer flags and a US-country
  flag;
- report structure: distinct drug-role counts, a known-high-risk-drug flag, and
  distinct reaction / indication counts;
- top-K bags: presence indicators for the most frequent drugs, reactions, and
  indications in the training window.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Default vocabulary sizes for the demo. Small enough to stay well-conditioned
# on a few-thousand-row slice; the full-corpus pipeline uses larger bags.
DEFAULT_TOP_DRUGS = 60
DEFAULT_TOP_REACS = 40
DEFAULT_TOP_INDIS = 25

# openFDA drugcharacterization codes: 1 suspect, 2 concomitant, 3 interacting.
_ROLE_SUSPECT, _ROLE_CONCOMITANT, _ROLE_INTERACTING = "1", "2", "3"

# openFDA primarysource.qualification: 1 physician, 2 pharmacist, 3 other HP,
# 4 lawyer, 5 consumer.
_HEALTHCARE_QUALS = {"1", "2", "3"}
_CONSUMER_QUALS = {"5"}

_US_COUNTRY = {"US", "USA", "UNITED STATES"}

# Well-known narrow-therapeutic-index / serious-outcome drugs, matched as
# case-insensitive substrings of the reported product name. Interpretable flag,
# not a learned feature.
HIGH_RISK_SUBSTRINGS = (
    "WARFARIN", "METHOTREXATE", "DIGOXIN", "INSULIN", "LITHIUM",
    "AMIODARONE", "TACROLIMUS", "CYCLOSPORINE", "HEPARIN",
    "CLOZAPINE", "VALPROATE", "CARBAMAZEPINE", "PHENYTOIN",
)


@dataclass(frozen=True)
class FeatureState:
    """Fitted vocabulary and imputation values learned on the training window."""

    top_drugs: list[str]
    top_reacs: list[str]
    top_indis: list[str]
    age_median: float
    columns: list[str] = field(default_factory=list)


def _drug_names(drugs: list[dict]) -> list[str]:
    return [str(d.get("name", "")).strip().upper() for d in drugs if d.get("name")]


def _indications(drugs: list[dict]) -> list[str]:
    return [str(d.get("indication", "")).strip().upper()
            for d in drugs if d.get("indication")]


def _top_k(counter: Counter, k: int) -> list[str]:
    """Most common ``k`` keys, ordered by count then name for determinism."""
    return [name for name, _ in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def build_features_fit(
    frame: pd.DataFrame,
    top_drugs: int = DEFAULT_TOP_DRUGS,
    top_reacs: int = DEFAULT_TOP_REACS,
    top_indis: int = DEFAULT_TOP_INDIS,
) -> tuple[pd.DataFrame, FeatureState]:
    """Learn the vocabulary + age median on ``frame`` and transform it.

    ``frame`` must be the training window; the caller is responsible for passing
    only training-era rows so the vocabulary stays leak-free.
    """
    drug_counts: Counter = Counter()
    reac_counts: Counter = Counter()
    indi_counts: Counter = Counter()
    for drugs in frame["drugs"]:
        drug_counts.update(set(_drug_names(drugs)))
        indi_counts.update(set(_indications(drugs)))
    for reacs in frame["reactions"]:
        reac_counts.update({str(r).strip().upper() for r in reacs})

    ages = pd.to_numeric(frame["age_yr"], errors="coerce")
    age_median = float(ages.median()) if ages.notna().any() else 0.0

    state = FeatureState(
        top_drugs=_top_k(drug_counts, top_drugs),
        top_reacs=_top_k(reac_counts, top_reacs),
        top_indis=_top_k(indi_counts, top_indis),
        age_median=age_median,
    )
    X = _transform(frame, state)
    object.__setattr__(state, "columns", list(X.columns))
    return X, state


def build_features_apply(frame: pd.DataFrame, state: FeatureState) -> pd.DataFrame:
    """Transform any fold with a vocabulary fitted on the training window."""
    X = _transform(frame, state)
    return X.reindex(columns=state.columns, fill_value=0.0)


def _transform(frame: pd.DataFrame, state: FeatureState) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)

    ages = pd.to_numeric(frame["age_yr"], errors="coerce")
    out["age_yr"] = ages.fillna(state.age_median).astype(np.float64)
    out["age_missing"] = ages.isna().astype(np.float64)

    sex = frame["sex"].astype(str)
    out["sex_M"] = (sex == "1").astype(np.float64)
    out["sex_F"] = (sex == "2").astype(np.float64)
    out["sex_UNK"] = (~sex.isin(["1", "2"])).astype(np.float64)

    qual = frame["reporter_qualification"].astype(str)
    out["is_healthcare_prof"] = qual.isin(_HEALTHCARE_QUALS).astype(np.float64)
    out["is_consumer"] = qual.isin(_CONSUMER_QUALS).astype(np.float64)
    out["reporter_country_is_US"] = (
        frame["reporter_country"].astype(str).str.upper().isin(_US_COUNTRY)
    ).astype(np.float64)

    drugs_col = frame["drugs"]
    out["n_drugs"] = drugs_col.apply(len).astype(np.float64)
    out["n_suspect"] = drugs_col.apply(
        lambda ds: sum(str(d.get("char")) == _ROLE_SUSPECT for d in ds)
    ).astype(np.float64)
    out["n_concomitant"] = drugs_col.apply(
        lambda ds: sum(str(d.get("char")) == _ROLE_CONCOMITANT for d in ds)
    ).astype(np.float64)
    out["n_interacting"] = drugs_col.apply(
        lambda ds: sum(str(d.get("char")) == _ROLE_INTERACTING for d in ds)
    ).astype(np.float64)
    out["has_high_risk_drug"] = drugs_col.apply(_has_high_risk).astype(np.float64)
    out["n_distinct_indications"] = drugs_col.apply(
        lambda ds: len(set(_indications(ds)))
    ).astype(np.float64)
    out["n_distinct_reactions"] = frame["reactions"].apply(
        lambda rs: len({str(r).strip().upper() for r in rs})
    ).astype(np.float64)

    drug_sets = drugs_col.apply(lambda ds: set(_drug_names(ds)))
    reac_sets = frame["reactions"].apply(lambda rs: {str(r).strip().upper() for r in rs})
    indi_sets = drugs_col.apply(lambda ds: set(_indications(ds)))

    # Build the sparse top-K bags in one block and concatenate once, rather than
    # inserting hundreds of columns individually (which fragments the frame).
    bags: dict[str, pd.Series] = {}
    for name in state.top_drugs:
        bags[f"drug={name}"] = drug_sets.apply(lambda s, n=name: float(n in s))
    for name in state.top_reacs:
        bags[f"reac={name}"] = reac_sets.apply(lambda s, n=name: float(n in s))
    for name in state.top_indis:
        bags[f"indi={name}"] = indi_sets.apply(lambda s, n=name: float(n in s))

    bag_frame = pd.DataFrame(bags, index=frame.index)
    return pd.concat([out, bag_frame], axis=1).astype(np.float64)


def _has_high_risk(drugs: list[dict]) -> float:
    for name in _drug_names(drugs):
        if any(sub in name for sub in HIGH_RISK_SUBSTRINGS):
            return 1.0
    return 0.0


__all__ = [
    "FeatureState",
    "HIGH_RISK_SUBSTRINGS",
    "DEFAULT_TOP_DRUGS",
    "DEFAULT_TOP_REACS",
    "DEFAULT_TOP_INDIS",
    "build_features_fit",
    "build_features_apply",
]
