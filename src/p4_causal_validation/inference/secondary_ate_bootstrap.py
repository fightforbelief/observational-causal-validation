"""Secondary full-sample ATE bootstrap.

This is not the primary confirmatory bootstrap. It resamples all four design
cells, keeps the same G1 and G4 draws in both estimators, and refits the ATE
nuisance models. It does not change which estimands the primary bootstrap accepts.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from p4_causal_validation.data.estimands import subset_for_estimand
from p4_causal_validation.data.wsc import resolve_covariate_columns
from p4_causal_validation.estimators.aipw import NUMERICAL_FLOOR, FitFailure, crossfit_aipw
from p4_causal_validation.estimators.common import DataContractError, EmptyStratumError
from p4_causal_validation.estimators.unadjusted import difference_in_means
from p4_causal_validation.inference.bootstrap import ESTIMATOR_FAILURES

SECONDARY_REPLICATES = 2000
SECONDARY_SEED = 20261026
SECONDARY_FOLDS = 5
RCT_LABEL = "RCT_ATE_full"
QED_LABEL = "QED_concordant_ATE"


def secondary_zero_failure_rule_met(n_replicates: int, n_success: int, n_failed: int) -> bool:
    """Return whether this secondary interval meets its own zero-failure rule.

    The rule is true only for a request of exactly 2,000 replicates with every
    replicate succeeding. A shorter run is false. This does not make the
    secondary analysis confirmatory and it does not prove coverage.
    """
    if n_replicates != SECONDARY_REPLICATES:
        return False
    if n_success < 0 or n_failed < 0:
        return False
    if n_success + n_failed != n_replicates:
        return False
    return n_failed == 0 and n_success == SECONDARY_REPLICATES


def _cell_positions(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    preference = frame["mathSel"].to_numpy(dtype=int)
    assignment = frame["mathGrp"].to_numpy(dtype=int)
    return {
        "G1": np.flatnonzero((preference == 1) & (assignment == 1)),
        "G2": np.flatnonzero((preference == 1) & (assignment == 0)),
        "G3": np.flatnonzero((preference == 0) & (assignment == 1)),
        "G4": np.flatnonzero((preference == 0) & (assignment == 0)),
    }


def _drawn_frame(frame: pd.DataFrame, positions: np.ndarray, local_draw: np.ndarray) -> pd.DataFrame:
    return frame.iloc[positions[local_draw]].copy()


def _rct_estimate(frame: pd.DataFrame):
    sample = subset_for_estimand(frame, RCT_LABEL)
    return difference_in_means(
        sample["mathPost"].to_numpy(dtype=float),
        sample["mathGrp"].to_numpy(dtype=int),
        estimand=RCT_LABEL,
        outcome="mathPost",
        target_population="All 2200 participants when the frame is the full WSC sample. Treated mathGrp=1 versus control mathGrp=0.",
        role="Unadjusted Neyman difference in means. Observed arm sizes are not reweighted to 1/2.",
    )


def _qed_estimate(frame: pd.DataFrame):
    sample = subset_for_estimand(frame, QED_LABEL).reset_index(drop=True)
    if not sample["mathGrp"].eq(sample["mathSel"]).all():
        raise DataContractError("QED_concordant_ATE must be restricted to mathSel == mathGrp.")
    covariates = resolve_covariate_columns(sample)
    if "mathSel" in covariates:
        raise DataContractError("mathSel cannot enter the concordant ATE propensity model.")
    result = crossfit_aipw(
        sample["mathPost"].to_numpy(dtype=float),
        sample["mathGrp"].to_numpy(dtype=int),
        sample[covariates].to_numpy(dtype=float),
        estimand=QED_LABEL,
        outcome="mathPost",
        target_population="Concordant G1 union G4. The ATE score uses both outcome regressions.",
        n_folds=SECONDARY_FOLDS,
        seed=SECONDARY_SEED,
        weight_type="ate",
        participant_ids=sample["row_id"].tolist(),
    )
    result.diagnostics["covariates"] = covariates
    result.diagnostics["mathSel_in_propensity"] = False
    return result


def _public_estimate(point) -> dict:
    diagnostics = point.diagnostics if isinstance(point.diagnostics, dict) else {}
    return {
        "estimand": point.estimand,
        "estimator": point.estimator,
        "estimate": float(point.estimate),
        "se": float(point.se),
        "normal_interval_low": float(point.ci_low),
        "normal_interval_high": float(point.ci_high),
        "n": int(point.n),
        "n_treated": int(point.n_treated),
        "n_control": int(point.n_control),
        "warnings": [str(item) for item in point.warnings],
        "numerical_floor": diagnostics.get("numerical_floor", NUMERICAL_FLOOR),
        "rows_deleted_for_overlap": diagnostics.get("rows_deleted_for_overlap"),
        "n_probabilities_moved_by_floor": diagnostics.get("n_probabilities_moved_by_floor"),
        "scientific_trimming": diagnostics.get("scientific_trimming"),
        "overlap_report": diagnostics.get("overlap_report"),
        "ate_arm_weights": diagnostics.get("ate_arm_weights"),
        "out_of_fold_propensity": diagnostics.get("out_of_fold_propensity"),
        "fold_summaries": diagnostics.get("fold_summaries"),
        "covariates": diagnostics.get("covariates"),
        "normal_interval_is_not_the_bootstrap_interval": True,
    }


def stratified_secondary_ate_bootstrap(
    frame: pd.DataFrame,
    *,
    n_replicates: int = SECONDARY_REPLICATES,
    seed: int = SECONDARY_SEED,
    observer: Callable[[dict], None] | None = None,
) -> dict:
    """Cell-size-preserving bootstrap for QED_concordant_ATE minus RCT_ATE_full."""
    if n_replicates < 2:
        raise ValueError("Bootstrap needs at least 2 replicates.")
    if "row_id" not in frame.columns or frame["row_id"].isna().any() or frame["row_id"].duplicated().any():
        raise DataContractError("Secondary ATE bootstrap requires unique non-missing row_id values on the source frame.")
    cells = _cell_positions(frame)
    counts = {name: int(index.size) for name, index in cells.items()}
    short = [name for name in ("G1", "G2", "G3", "G4") if counts[name] < 2]
    if short:
        raise EmptyStratumError(f"Secondary ATE cells must each have at least 2 rows; short cells: {short}. Counts are {counts}.")
    point_rct = _rct_estimate(frame)
    point_qed = _qed_estimate(frame)
    if point_rct.estimand != RCT_LABEL or point_qed.estimand != QED_LABEL:
        raise DataContractError("Secondary point estimates returned unexpected labels.")
    if point_qed.diagnostics.get("weight_type") != "ate" or point_qed.diagnostics.get("treated_outcome_model_used") is not True:
        raise DataContractError("Secondary observational estimate did not use the ATE score.")
    if point_qed.diagnostics.get("numerical_floor") != NUMERICAL_FLOOR or point_qed.diagnostics.get("rows_deleted_for_overlap") != 0:
        raise DataContractError("Secondary ATE changed the numerical floor or deleted rows for overlap.")
    difference_point = float(point_qed.estimate - point_rct.estimate)
    rng = np.random.default_rng(seed)
    successes = []
    failures = []
    g3_source_ids = set(frame.iloc[cells["G3"]]["row_id"].tolist())
    for replicate in range(n_replicates):
        draws = {name: rng.integers(0, counts[name], size=counts[name]) for name in ("G1", "G2", "G3", "G4")}
        parts = {name: _drawn_frame(frame, cells[name], draws[name]) for name in draws}
        rct_frame = pd.concat([parts["G1"], parts["G2"], parts["G3"], parts["G4"]], ignore_index=True)
        qed_frame = pd.concat([parts["G1"], parts["G4"]], ignore_index=True)
        g1_ids = parts["G1"]["row_id"].tolist()
        g4_ids = parts["G4"]["row_id"].tolist()
        if rct_frame["row_id"].iloc[: counts["G1"]].tolist() != g1_ids:
            raise RuntimeError("The RCT replicate did not keep the shared G1 draw.")
        if qed_frame["row_id"].iloc[: counts["G1"]].tolist() != g1_ids:
            raise RuntimeError("The QED replicate did not keep the shared G1 draw.")
        if qed_frame["row_id"].iloc[counts["G1"] :].tolist() != g4_ids:
            raise RuntimeError("The QED replicate did not keep the shared G4 draw.")
        if set(qed_frame["row_id"].tolist()).intersection(g3_source_ids):
            raise RuntimeError("G3 entered the observational ATE replicate.")
        if not set(parts["G3"]["row_id"].tolist()).issubset(set(rct_frame["row_id"].tolist())):
            raise RuntimeError("G3 was missing from the randomized ATE replicate.")
        if len(rct_frame) != sum(counts.values()) or len(qed_frame) != counts["G1"] + counts["G4"]:
            raise RuntimeError("A secondary bootstrap replicate changed a cell count.")
        if observer is not None:
            observer({"replicate": replicate, "g3_in_rct": True, "g3_in_qed": False})
        try:
            rct = _rct_estimate(rct_frame)
            qed = _qed_estimate(qed_frame)
            if not np.isfinite(rct.estimate) or not np.isfinite(qed.estimate):
                raise FitFailure(f"Replicate {replicate} produced a non-finite estimate.")
            if qed.diagnostics["n_folds"] != SECONDARY_FOLDS or qed.diagnostics["seed"] != SECONDARY_SEED:
                raise RuntimeError("A secondary replicate changed the five-fold seed.")
            if qed.diagnostics["fold_assignment"] != "distinct_participant_id":
                raise RuntimeError("A secondary replicate did not group folds by participant id.")
            successes.append(
                {
                    "replicate": replicate,
                    "rct_estimate": float(rct.estimate),
                    "qed_estimate": float(qed.estimate),
                    "difference_qed_minus_rct": float(qed.estimate - rct.estimate),
                    "rct_recomputed": True,
                    "propensity_refit": True,
                    "both_outcome_models_refit": True,
                    "crossfit_refit": True,
                }
            )
        except ESTIMATOR_FAILURES as exc:
            failures.append({"replicate": replicate, "error_type": type(exc).__name__, "message": str(exc)})
    if len(successes) + len(failures) != n_replicates:
        raise RuntimeError("Secondary bootstrap replicate accounting does not match the requested count.")
    rule_met = secondary_zero_failure_rule_met(n_replicates, len(successes), len(failures))
    result = {
        "secondary_analysis": True,
        "replaces_primary_analysis": False,
        "not_fully_preregistered": True,
        "method_choices_are_a_new_owner_decision": True,
        "contrast": "QED_concordant_ATE - RCT_ATE_full",
        "contrast_is_not_pure_confounding": True,
        "resampling": "stratified with replacement inside G1, G2, G3, and G4; cell sizes held fixed; one shared G1 draw and one shared G4 draw",
        "conditioning": "cell counts fixed at the observed sample sizes; this is not randomization inference",
        "g3_in_randomized_ate": True,
        "g3_in_observational_ate": False,
        "refits_ate_nuisance_each_replicate": True,
        "aipw_seed": SECONDARY_SEED,
        "n_folds": SECONDARY_FOLDS,
        "seed": seed,
        "n_replicates": n_replicates,
        "n_success": len(successes),
        "n_failed": len(failures),
        "failures": failures,
        "failures_not_redrawn": True,
        "failure_rule": "all_2000_requested_replicates_succeed",
        "secondary_zero_failure_rule_met": rule_met,
        "interval_is_diagnostic_only": not rule_met,
        "this_is_not_the_primary_confirmatory_analysis": True,
        "zero_failure_rule_does_not_prove_coverage": True,
        "cell_counts": counts,
        "successful_replicates": successes,
        "rct_estimate": float(point_rct.estimate),
        "rct_neyman_se": float(point_rct.se),
        "qed_estimate": float(point_qed.estimate),
        "qed_influence_se": float(point_qed.se),
        "difference_qed_minus_rct": difference_point,
        "difference_is_not_automatically_confounding": True,
        "similar_points_do_not_establish_equivalence": True,
        "reweighted_to_one_half": False,
        "p_z_one_half_independently_verified": False,
        "same_parameter_claim": False,
        "rct_point": _public_estimate(point_rct),
        "qed_point": _public_estimate(point_qed),
    }
    if len(successes) >= 2:
        differences = np.array([item["difference_qed_minus_rct"] for item in successes], dtype=float)
        rct_draws = np.array([item["rct_estimate"] for item in successes], dtype=float)
        qed_draws = np.array([item["qed_estimate"] for item in successes], dtype=float)
        low = float(np.quantile(differences, 0.025))
        high = float(np.quantile(differences, 0.975))
        if low > high or not np.isfinite(low) or not np.isfinite(high):
            raise RuntimeError("Secondary bootstrap percentile interval is not finite and ordered.")
        result["bootstrap_se"] = float(differences.std(ddof=1))
        result["percentile_ci_low"] = low
        result["percentile_ci_high"] = high
        result["bootstrap_covariance"] = float(np.cov(rct_draws, qed_draws, ddof=1)[0, 1])
        result["interval_status"] = (
            "secondary_zero_failure_rule_met" if rule_met else "diagnostic_only_secondary_zero_failure_rule_not_met"
        )
    else:
        result["bootstrap_se"] = None
        result["percentile_ci_low"] = None
        result["percentile_ci_high"] = None
        result["bootstrap_covariance"] = None
        result["interval_status"] = "fewer than 2 successful replicates; no interval was invented"
    return result
