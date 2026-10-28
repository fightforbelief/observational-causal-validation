"""Bootstrap utilities for the primary WSC contrast.

``participant_bootstrap`` resamples indexes and evaluates a caller-supplied
statistic. It does not refit AIPW. The primary contrast uses
``stratified_primary_contrast_bootstrap``, which redraws design cells, shares
the G1 draw, and refits the nuisance models. Neither function reads the WSC
file. Percentile intervals are not equivalence tests.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from p4_causal_validation.estimators.aipw import EmptyFoldError, FitFailure
from p4_causal_validation.estimators.common import DataContractError, EmptyStratumError, EstimandMismatchError

ESTIMATOR_FAILURES = (FitFailure, EmptyFoldError, EmptyStratumError)

PROTOCOL_BOOTSTRAP_REPLICATES = 2000
PROTOCOL_BOOTSTRAP_SEED = 20261026


def confirmatory_interval_rule_met(n_replicates: int, n_success: int, n_failed: int) -> bool:
    """Return whether the zero-failure confirmatory rule is met.

    The percentile interval meets that rule only when the caller requested the
    protocol count of 2,000 replicates and every one of those replicates
    succeeded. Any failure makes the result false. A shorter run is false even
    when it has no failures. This is not a numeric failure-share threshold and
    it does not authorize a real WSC analysis.
    """
    if n_replicates != PROTOCOL_BOOTSTRAP_REPLICATES:
        return False
    if n_success < 0 or n_failed < 0:
        return False
    if n_success + n_failed != n_replicates:
        return False
    return n_failed == 0 and n_success == PROTOCOL_BOOTSTRAP_REPLICATES


def participant_bootstrap(
    n_participants: int,
    statistic: Callable[[np.ndarray], float],
    *,
    n_replicates: int = PROTOCOL_BOOTSTRAP_REPLICATES,
    seed: int = PROTOCOL_BOOTSTRAP_SEED,
) -> dict:
    """Resample participant indexes with replacement.

    ``statistic`` receives one replicate's indexes, shape ``(n_participants,)``.
    This does not refit nuisance models. It is not the primary-contrast bootstrap.
    """
    if n_participants < 2:
        raise ValueError("Bootstrap needs at least 2 participants.")
    if n_replicates < 2:
        raise ValueError("Bootstrap needs at least 2 replicates.")
    rng = np.random.default_rng(seed)
    values = np.empty(n_replicates, dtype=float)
    for replicate in range(n_replicates):
        draw = rng.integers(0, n_participants, size=n_participants)
        value = float(statistic(draw))
        if not np.isfinite(value):
            raise ValueError(f"Bootstrap replicate {replicate} was not finite.")
        values[replicate] = value
    low = float(np.quantile(values, 0.025))
    high = float(np.quantile(values, 0.975))
    if low > high:
        raise ValueError("Bootstrap percentile endpoints are unordered.")
    return {
        "n_participants": n_participants,
        "n_replicates": n_replicates,
        "seed": seed,
        "resampling": "one with-replacement participant index draw per replicate, shared by every functional",
        "refits_nuisance_models": False,
        "replicates_mean": float(values.mean()),
        "se": float(values.std(ddof=1)),
        "percentile_ci_low": low,
        "percentile_ci_high": high,
        "interval_overlap_is_not_equivalence": True,
        "not_a_wsc_result": True,
        "randomized_point_is_not_noiseless_truth": True,
    }


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
    part = frame.iloc[positions[local_draw]].copy()
    part.attrs["source"] = frame.attrs.get("source")
    return part


def _public_point(point) -> dict:
    diagnostics = getattr(point, "diagnostics", {}) or {}
    return {
        "estimand": point.estimand,
        "estimate": float(point.estimate),
        "se": float(point.se),
        "normal_interval_low": float(point.ci_low),
        "normal_interval_high": float(point.ci_high),
        "n": int(point.n),
        "n_treated": int(point.n_treated),
        "n_control": int(point.n_control),
        "warnings": [str(item) for item in getattr(point, "warnings", ())],
        "rows_deleted_for_overlap": diagnostics.get("rows_deleted_for_overlap"),
        "numerical_floor": diagnostics.get("numerical_floor"),
        "n_probabilities_moved_by_floor": diagnostics.get("n_probabilities_moved_by_floor"),
        "scientific_trimming": diagnostics.get("scientific_trimming"),
        "overlap_report": diagnostics.get("overlap_report"),
        "att_control_weights": diagnostics.get("att_control_weights"),
        "covariates": diagnostics.get("covariates"),
        "fold_summaries": diagnostics.get("fold_summaries"),
        "normal_interval_is_not_the_bootstrap_interval": True,
        "score_se_holds_treated_share_fixed": diagnostics.get("treated_share_held_fixed_in_score_se"),
    }


def stratified_primary_contrast_bootstrap(
    frame: pd.DataFrame,
    *,
    n_replicates: int = PROTOCOL_BOOTSTRAP_REPLICATES,
    seed: int = PROTOCOL_BOOTSTRAP_SEED,
    randomized_estimand: str = "RCT_effect_prefer_math",
    observational_estimand: str = "QED_concordant_ATT",
    observer: Callable[[dict], None] | None = None,
    authorized_real_run: bool = False,
    failure_rule_approved: bool = False,
) -> dict:
    """Conditional cell bootstrap for the primary RCT-minus-QED contrast.

    G1, G2, and G4 are redrawn with replacement at their original sizes.
    The same G1 draw is concatenated into both the RCT sample (with G2) and
    the QED sample (with G4). G3 is not passed to either estimator. Each
    replicate refits the five-fold ATT AIPW. Failed replicates are counted
    and are not replaced.
    """
    from p4_causal_validation.experiments import primary_specification as spec

    if randomized_estimand != spec.PRIMARY_RANDOMIZED or observational_estimand != spec.PRIMARY_OBSERVATIONAL:
        raise EstimandMismatchError(
            "The primary bootstrap only compares RCT_effect_prefer_math with QED_concordant_ATT."
        )
    if n_replicates < 2:
        raise ValueError("Bootstrap needs at least 2 replicates.")
    if authorized_real_run:
        spec.assert_authorized_real_frame(frame)
    else:
        spec.assert_synthetic_fixture(frame)
    if "row_id" not in frame.columns or frame["row_id"].isna().any() or frame["row_id"].duplicated().any():
        raise DataContractError(
            "Stratified bootstrap requires unique non-missing row_id values on the source frame. "
            "Copies are created inside each replicate and must stay together in cross-fitting."
        )
    cells = _cell_positions(frame)
    counts = {name: int(index.size) for name, index in cells.items()}
    too_small = [name for name in ("G1", "G2", "G4") if counts[name] < 2]
    if too_small:
        raise EmptyStratumError(
            f"Primary contrast cells must each have at least 2 rows; empty or short cells: {too_small}. "
            f"Counts are {counts}. G3 is not part of the primary contrast."
        )
    point_rct = spec.randomized_math_preference_benchmark(frame)
    point_qed = spec.qed_concordant_att(frame)
    if point_rct.estimand != randomized_estimand or point_qed.estimand != observational_estimand:
        raise EstimandMismatchError("Point estimates returned unexpected estimand labels.")
    difference_point = float(point_qed.estimate - point_rct.estimate)
    rng = np.random.default_rng(seed)
    successes = []
    failures = []
    for replicate in range(n_replicates):
        g1_draw = rng.integers(0, counts["G1"], size=counts["G1"])
        g2_draw = rng.integers(0, counts["G2"], size=counts["G2"])
        g4_draw = rng.integers(0, counts["G4"], size=counts["G4"])
        g1 = _drawn_frame(frame, cells["G1"], g1_draw)
        g2 = _drawn_frame(frame, cells["G2"], g2_draw)
        g4 = _drawn_frame(frame, cells["G4"], g4_draw)
        rct_frame = pd.concat([g1, g2], ignore_index=True)
        qed_frame = pd.concat([g1, g4], ignore_index=True)
        if authorized_real_run:
            rct_frame.attrs["source"] = spec.AUTHORIZED_REAL_SOURCE
            qed_frame.attrs["source"] = spec.AUTHORIZED_REAL_SOURCE
            rct_frame.attrs["real_execution_authorized"] = True
            qed_frame.attrs["real_execution_authorized"] = True
        else:
            rct_frame.attrs["source"] = spec.SYNTHETIC_SOURCE
            qed_frame.attrs["source"] = spec.SYNTHETIC_SOURCE
        if len(rct_frame) != counts["G1"] + counts["G2"] or len(qed_frame) != counts["G1"] + counts["G4"]:
            raise RuntimeError("A bootstrap replicate changed a primary cell count.")
        g1_ids = g1["row_id"].tolist()
        if rct_frame["row_id"].iloc[: counts["G1"]].tolist() != g1_ids:
            raise RuntimeError("The RCT frame did not keep the shared G1 draw.")
        if qed_frame["row_id"].iloc[: counts["G1"]].tolist() != g1_ids:
            raise RuntimeError("The QED frame did not keep the shared G1 draw.")
        g3_ids = set(frame.iloc[cells["G3"]]["row_id"].tolist())
        used_ids = set(rct_frame["row_id"].tolist()).union(qed_frame["row_id"].tolist())
        if g3_ids.intersection(used_ids):
            raise RuntimeError("G3 entered a primary bootstrap sample.")
        if observer is not None:
            observer(
                {
                    "replicate": replicate,
                    "rct_row_ids": rct_frame["row_id"].tolist() if "row_id" in rct_frame.columns else None,
                    "qed_row_ids": qed_frame["row_id"].tolist() if "row_id" in qed_frame.columns else None,
                    "g3_excluded": True,
                }
            )
        try:
            rct = spec.randomized_math_preference_benchmark(rct_frame)
            qed = spec.qed_concordant_att(qed_frame)
            if not np.isfinite(rct.estimate) or not np.isfinite(qed.estimate):
                raise FitFailure(f"Replicate {replicate} produced a non-finite estimate.")
            if qed.diagnostics["n_folds"] != spec.PRIMARY_FOLDS or qed.diagnostics["seed"] != spec.PRIMARY_SEED:
                raise RuntimeError("A bootstrap replicate changed the five-fold seed or fold count.")
            if qed.diagnostics["fold_assignment"] != "distinct_participant_id":
                raise RuntimeError("A bootstrap replicate did not group cross-fitting by participant id.")
            successes.append(
                {
                    "replicate": replicate,
                    "rct_estimate": float(rct.estimate),
                    "qed_estimate": float(qed.estimate),
                    "difference_qed_minus_rct": float(qed.estimate - rct.estimate),
                    "rct_recomputed": True,
                    "propensity_refit": True,
                    "control_outcome_refit": True,
                    "crossfit_refit": True,
                    "qed_n_folds": int(qed.diagnostics["n_folds"]),
                    "qed_seed": int(qed.diagnostics["seed"]),
                }
            )
        except ESTIMATOR_FAILURES as exc:
            failures.append(
                {
                    "replicate": replicate,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
    if len(successes) + len(failures) != n_replicates:
        raise RuntimeError("Bootstrap replicate accounting does not match the requested count.")
    rule_met = confirmatory_interval_rule_met(n_replicates, len(successes), len(failures))
    result = {
        "not_a_wsc_result": not authorized_real_run,
        "resampling": "stratified with replacement inside G1, G2, and G4; cell sizes held fixed; one shared G1 draw",
        "conditioning": "cell counts fixed at the observed sample sizes; this is not an iid redraw of all 2200 rows",
        "g3_in_primary_estimators": False,
        "refits_aipw_each_replicate": True,
        "aipw_seed": spec.PRIMARY_SEED,
        "n_folds": spec.PRIMARY_FOLDS,
        "seed": seed,
        "protocol_replicate_count": PROTOCOL_BOOTSTRAP_REPLICATES,
        "n_replicates": n_replicates,
        "n_success": len(successes),
        "n_failed": len(failures),
        "failures": failures,
        "failures_not_redrawn": True,
        "estimator_failures_recorded": ["FitFailure", "EmptyFoldError", "EmptyStratumError"],
        "unexpected_exceptions_are_not_recorded_as_failures": True,
        "failure_share_threshold": None,
        "failure_share_threshold_status": "not_applicable_zero_failure_rule_is_not_a_numeric_share",
        "confirmatory_failure_rule": "all_2000_requested_replicates_succeed",
        "confirmatory_interval_authorized": bool(rule_met and authorized_real_run),
        "interval_is_not_a_confirmatory_interval": not (rule_met and authorized_real_run),
        "zero_failure_rule_does_not_authorize_real_wsc": True,
        "real_wsc_execution_authorized": bool(authorized_real_run),
        "machine_approval_of_failure_rule": bool(authorized_real_run and failure_rule_approved),
        "fixed_cell_bootstrap_is_not_randomization_inference": True,
        "bootstrap_does_not_test_unmeasured_confounding": True,
        "cell_counts": counts,
        "successful_replicates": successes,
        "randomized_estimand": point_rct.estimand,
        "observational_estimand": point_qed.estimand,
        "rct_estimate": float(point_rct.estimate),
        "rct_neyman_se": float(point_rct.se),
        "qed_estimate": float(point_qed.estimate),
        "qed_score_se_treats_share_as_fixed": float(point_qed.se),
        "difference_qed_minus_rct": difference_point,
        "randomized_point_is_not_noiseless_truth": True,
        "difference_is_not_automatically_confounding": True,
        "interval_overlap_is_not_equivalence": True,
        "interval_uses_only_successful_replicates": True,
    }
    result["rct_point"] = _public_point(point_rct)
    result["qed_point"] = _public_point(point_qed)
    if len(successes) >= 2:
        differences = np.array([item["difference_qed_minus_rct"] for item in successes], dtype=float)
        rct_draws = np.array([item["rct_estimate"] for item in successes], dtype=float)
        qed_draws = np.array([item["qed_estimate"] for item in successes], dtype=float)
        low = float(np.quantile(differences, 0.025))
        high = float(np.quantile(differences, 0.975))
        if low > high or not np.isfinite(low) or not np.isfinite(high):
            raise RuntimeError("Bootstrap percentile interval is not finite and ordered.")
        result["bootstrap_se"] = float(differences.std(ddof=1))
        result["percentile_ci_low"] = low
        result["percentile_ci_high"] = high
        result["bootstrap_covariance"] = float(np.cov(rct_draws, qed_draws, ddof=1)[0, 1])
        if rule_met and authorized_real_run:
            result["interval_status"] = "zero_failure_rule_met"
        elif rule_met:
            result["interval_status"] = "zero_failure_rule_met_this_is_not_a_real_wsc_result"
        else:
            result["interval_status"] = "diagnostic_only_zero_failure_rule_not_met"
    else:
        result["bootstrap_se"] = None
        result["percentile_ci_low"] = None
        result["percentile_ci_high"] = None
        result["bootstrap_covariance"] = None
        result["interval_status"] = "fewer than 2 successful replicates; no interval was invented"
    return result
