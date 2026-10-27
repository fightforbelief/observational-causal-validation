"""Primary WSC specification from protocol candidate v2.

These functions estimate only when the frame is marked as a synthetic fixture.
They do not open the WSC data file. Real confirmatory fitting stays behind
the approval gate and is still disabled in the confirmatory entry.
"""

from __future__ import annotations

import pandas as pd

from p4_causal_validation.data.estimands import ESTIMANDS, subset_for_estimand
from p4_causal_validation.data.wsc import POST_TREATMENT, RAW_DUPLICATES_OF_LOGS, resolve_covariate_columns
from p4_causal_validation.estimators.aipw import NUMERICAL_FLOOR, crossfit_aipw
from p4_causal_validation.estimators.common import DataContractError, EstimateResult
from p4_causal_validation.estimators.unadjusted import difference_in_means
from p4_causal_validation.inference.bootstrap import PROTOCOL_BOOTSTRAP_REPLICATES, PROTOCOL_BOOTSTRAP_SEED

SYNTHETIC_SOURCE = "synthetic_fixture"
PRIMARY_OUTCOME = "mathPost"
PRIMARY_RANDOMIZED = "RCT_effect_prefer_math"
PRIMARY_OBSERVATIONAL = "QED_concordant_ATT"
PRIMARY_SEED = PROTOCOL_BOOTSTRAP_SEED
PRIMARY_FOLDS = 5
PRIMARY_BOOTSTRAP_REPLICATES = PROTOCOL_BOOTSTRAP_REPLICATES

SECONDARY_ESTIMANDS = (
    "RCT_ATE_full",
    "QED_concordant_ATE",
    "RCT_effect_prefer_vocab",
)
NOT_IMPLEMENTED_THIS_ROUND = ("QED_concordant_ATU",)


class ConfirmatoryNotAuthorized(RuntimeError):
    """Real WSC outcome comparisons are outside this preparation build."""


def assert_synthetic_fixture(frame: pd.DataFrame) -> None:
    source = frame.attrs.get("source")
    if source != SYNTHETIC_SOURCE:
        raise ConfirmatoryNotAuthorized(
            "Refusing to estimate. This build accepts source='synthetic_fixture' only. "
            "Real WSC confirmatory analysis is not authorized."
        )


def assert_primary_outcome(outcome: str) -> None:
    if outcome != PRIMARY_OUTCOME:
        raise DataContractError(
            f"The primary comparison uses {PRIMARY_OUTCOME}. {outcome} is not in the primary comparison. "
            "A vocabulary-training effect would need the reversed assignment 1-mathGrp and a separate estimand."
        )


def reject_blocked_covariates(names: list[str]) -> None:
    blocked = set(POST_TREATMENT).union(RAW_DUPLICATES_OF_LOGS).union({"mathGrp", "mathSel", "row_id"})
    found = [name for name in names if name in blocked]
    if found:
        raise DataContractError(f"Covariate list contains blocked columns: {found}.")


def mark_synthetic(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy marked as a fixture. Does not read external data."""
    copied = frame.copy()
    copied.attrs["source"] = SYNTHETIC_SOURCE
    return copied


def randomized_math_preference_benchmark(frame: pd.DataFrame) -> EstimateResult:
    """Neyman difference in means of mathPost inside mathSel == 1.

    Treated rows are G1 (mathGrp == 1). Control rows are G2 (mathGrp == 0).
    The interval is sampling uncertainty, not a noiseless benchmark.
    """
    assert_synthetic_fixture(frame)
    assert_primary_outcome(PRIMARY_OUTCOME)
    subset = subset_for_estimand(frame, PRIMARY_RANDOMIZED)
    if not subset["mathSel"].eq(1).all():
        raise DataContractError("RCT_effect_prefer_math must be restricted to mathSel == 1.")
    result = difference_in_means(
        subset[PRIMARY_OUTCOME].to_numpy(dtype=float),
        subset["mathGrp"].to_numpy(dtype=int),
        estimand=PRIMARY_RANDOMIZED,
        outcome=PRIMARY_OUTCOME,
        target_population="Participants with mathSel=1. Treated G1 versus control G2.",
        role="Randomized Neyman benchmark. The point estimate is not a noiseless truth.",
    )
    result.diagnostics["rows_deleted_for_overlap"] = 0
    result.diagnostics["scientific_trimming"] = "none"
    return result


def qed_concordant_att(frame: pd.DataFrame) -> EstimateResult:
    """Five-fold linear ATT AIPW on the concordant sample, synthetic frames only.

    The estimation sample is G1 union G4. mathGrp is the treatment. Inside this
    sample mathSel equals mathGrp, so mathSel is not a covariate. Overlap
    outside [0.1, 0.9] is reported and does not delete rows.
    """
    assert_synthetic_fixture(frame)
    assert_primary_outcome(PRIMARY_OUTCOME)
    subset = subset_for_estimand(frame, PRIMARY_OBSERVATIONAL).reset_index(drop=True)
    if not subset["mathGrp"].eq(subset["mathSel"]).all():
        raise DataContractError("In the concordant sample mathGrp must equal mathSel.")
    covariates = resolve_covariate_columns(subset)
    reject_blocked_covariates(covariates)
    if "mathSel" in covariates:
        raise DataContractError("mathSel cannot enter the concordant propensity model.")
    features = subset[covariates].to_numpy(dtype=float)
    participant_ids = subset["row_id"].tolist() if "row_id" in subset.columns else None
    result = crossfit_aipw(
        subset[PRIMARY_OUTCOME].to_numpy(dtype=float),
        subset["mathGrp"].to_numpy(dtype=int),
        features,
        estimand=PRIMARY_OBSERVATIONAL,
        outcome=PRIMARY_OUTCOME,
        target_population="Concordant G1 union G4. ATT target is G1, the treated rows.",
        n_folds=PRIMARY_FOLDS,
        seed=PRIMARY_SEED,
        weight_type="att",
        participant_ids=participant_ids,
    )
    if result.diagnostics["numerical_floor"] != NUMERICAL_FLOOR:
        raise DataContractError("ATT AIPW numerical floor drifted from 1e-6.")
    if result.diagnostics["rows_deleted_for_overlap"] != 0:
        raise DataContractError("Overlap reporting must not delete rows.")
    result.diagnostics["covariates"] = covariates
    result.diagnostics["mathSel_in_propensity"] = False
    return result


def secondary_subset(frame: pd.DataFrame, estimand: str) -> pd.DataFrame:
    """Prepare a secondary sample without fitting it.

    QED ATU is not implemented in this round. The returned frame is not an estimate.
    """
    if estimand in NOT_IMPLEMENTED_THIS_ROUND:
        raise NotImplementedError("QED ATU is not part of this preparation round.")
    if estimand not in SECONDARY_ESTIMANDS:
        raise KeyError(f"Not a prepared secondary estimand: {estimand}. Known labels: {sorted(ESTIMANDS)}.")
    assert_synthetic_fixture(frame)
    subset = subset_for_estimand(frame, estimand)
    subset.attrs["source"] = SYNTHETIC_SOURCE
    subset.attrs["fitted"] = False
    return subset
