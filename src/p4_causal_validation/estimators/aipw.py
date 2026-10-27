"""Cross-fit AIPW with linear and logistic nuisances.

Double robustness is not unconfoundedness. These scores do not remove
unmeasured confounding, and the normal interval is not an equivalence test.
"""

from __future__ import annotations

import warnings

import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold

from p4_causal_validation.diagnostics.overlap import overlap_summary
from p4_causal_validation.estimators.common import (
    EstimateResult,
    as_float_array,
    as_treatment,
    require_arms,
    validate_result,
)

NUMERICAL_FLOOR = 1e-6
OVERLAP_REPORT_LOW = 0.1
OVERLAP_REPORT_HIGH = 0.9


class EmptyFoldError(ValueError):
    """A cross-fitting fold does not contain both treatment arms."""


class FitFailure(RuntimeError):
    """A nuisance model failed. Rows are not dropped to hide the failure."""


def apply_numerical_floor(propensity, floor: float = NUMERICAL_FLOOR) -> tuple[np.ndarray, int]:
    """Clip probabilities only so that a weight is defined. This is not scientific trimming."""
    probability = np.asarray(propensity, dtype=float)
    if np.any((probability < 0) | (probability > 1) | ~np.isfinite(probability)):
        raise FitFailure("Propensity values must be finite and inside [0, 1] before the numerical floor.")
    floored = np.clip(probability, floor, 1.0 - floor)
    moved = int(np.sum(probability != floored))
    return floored, moved


def att_influence_scores(y, treatment, mu0, propensity, treated_share: float) -> np.ndarray:
    """ATT scores with the treated share held fixed.

    psi = (D / p) (Y - mu0) - ((1 - D) e / (p (1 - e))) (Y - mu0)

    p is the treated share in the estimation sample. e and mu0 are nuisance
    predictions for the same rows. The mean estimates E[Y(1) - Y(0) | D = 1]
    when the nuisances are correct, assignment is unconfounded given X, and
    probabilities stay inside (0, 1). The score does not include an outcome
    model for the treated arm because those outcomes are observed.
    """
    outcome = as_float_array(y)
    assignment = as_treatment(treatment)
    control_mean = as_float_array(mu0)
    probability, _moved = apply_numerical_floor(propensity)
    if not np.isfinite(treated_share) or treated_share <= 0 or treated_share >= 1:
        raise FitFailure("ATT scores need a treated share strictly between 0 and 1.")
    if not (outcome.shape == assignment.shape == control_mean.shape == probability.shape):
        raise ValueError("ATT score inputs must have the same length.")
    residual = outcome - control_mean
    odds = probability / (1.0 - probability)
    return (assignment / treated_share) * residual - ((1.0 - assignment) * odds / treated_share) * residual


def _ols_predict(x_train, y_train, x_test) -> np.ndarray:
    if np.unique(y_train).size < 1 or x_train.shape[0] < 2:
        raise FitFailure("Outcome regression does not have enough training rows.")
    design = sm.add_constant(x_train, has_constant="add")
    try:
        model = sm.OLS(y_train, design).fit()
    except (np.linalg.LinAlgError, ValueError) as exc:
        raise FitFailure(f"Outcome regression failed: {type(exc).__name__}: {exc}") from exc
    test = sm.add_constant(x_test, has_constant="add")
    predicted = np.asarray(model.predict(test), dtype=float)
    if not np.isfinite(predicted).all():
        raise FitFailure("Outcome regression returned a non-finite prediction.")
    return predicted


def propensity_warning_failure_reason(warning_category: str) -> str | None:
    """Classify one warning from the propensity fit.

    Perfect separation, quasi-complete separation, and non-convergence are fit
    failures. Ordinary numeric warnings are not failures by themselves.
    """
    if warning_category == "PerfectSeparationWarning":
        return "perfect or quasi-complete separation"
    if warning_category == "ConvergenceWarning":
        return "optimizer did not converge"
    return None


def _statistical_fit_error(exc: BaseException) -> bool:
    if isinstance(exc, (np.linalg.LinAlgError, ValueError)):
        return True
    return type(exc).__name__ in {"PerfectSeparationError", "ConvergenceError"}


def _failure_reason_from_warnings(caught) -> str | None:
    for item in caught:
        reason = propensity_warning_failure_reason(item.category.__name__)
        if reason is not None:
            return reason
    return None


def _logit_predict(x_train, d_train, x_test) -> np.ndarray:
    if np.unique(d_train).size < 2:
        raise FitFailure("Logistic propensity training fold lost an arm.")
    design = sm.add_constant(x_train, has_constant="add")
    caught: list[warnings.WarningMessage] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = sm.Logit(d_train, design).fit(disp=False, maxiter=200)
    except FitFailure:
        raise
    except Exception as exc:
        if not _statistical_fit_error(exc):
            raise
        reason = _failure_reason_from_warnings(caught)
        if reason is not None:
            raise FitFailure(f"Logistic propensity fit failed: {reason}. {type(exc).__name__}: {exc}") from exc
        raise FitFailure(f"Logistic propensity fit failed: {type(exc).__name__}: {exc}") from exc
    reason = _failure_reason_from_warnings(caught)
    if reason is not None:
        raise FitFailure(f"Logistic propensity fit failed: {reason}.")
    converged = bool(getattr(model, "mle_retvals", {}).get("converged", False))
    if not converged:
        raise FitFailure("Logistic propensity did not converge.")
    test = sm.add_constant(x_test, has_constant="add")
    predicted = np.asarray(model.predict(test), dtype=float)
    if not np.isfinite(predicted).all():
        raise FitFailure("Logistic propensity returned a non-finite probability.")
    return predicted


def _assign_crossfit_folds(participant_ids, n_folds: int, seed: int) -> np.ndarray:
    """Assign every row to one test fold by distinct participant id.

    ``KFold(n_splits=n_folds, shuffle=True, random_state=seed)`` is applied to
    the first-appearance list of ids. Every copy of an id receives that id's
    fold. The fold count and seed are not reduced when a sample cannot be split.
    """
    ids = list(participant_ids)
    if any(value is None or (isinstance(value, float) and not np.isfinite(value)) for value in ids):
        raise ValueError("participant ids must be non-missing.")
    unique_ids = list(dict.fromkeys(ids))
    if len(unique_ids) < n_folds:
        raise EmptyFoldError(
            f"Cross-fitting keeps {n_folds} folds and seed {seed}, but only {len(unique_ids)} distinct "
            "participants are available. The fold count was not reduced. No row was deleted and the "
            "bootstrap sample was not redrawn."
        )
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    id_to_fold: dict = {}
    for fold, (_, test_positions) in enumerate(splitter.split(np.arange(len(unique_ids)))):
        for position in test_positions:
            id_to_fold[unique_ids[int(position)]] = fold
    assigned = np.fromiter((id_to_fold[value] for value in ids), dtype=int, count=len(ids))
    if set(np.unique(assigned).tolist()) != set(range(n_folds)):
        raise EmptyFoldError(
            f"Cross-fitting did not produce {n_folds} non-empty test folds. No row was deleted and the sample was not redrawn."
        )
    return assigned


def crossfit_aipw(
    y,
    treatment,
    covariates,
    *,
    estimand: str,
    outcome: str,
    target_population: str,
    n_folds: int = 5,
    seed: int = 20261026,
    weight_type: str = "ate",
    participant_ids=None,
) -> EstimateResult:
    """Cross-fitted augmented IPW.

    Nuisances are main-effects linear outcome regressions and a main-effects
    logistic propensity model. ``weight_type='att'`` targets treated units in
    the supplied estimation sample. Folds are assigned on distinct participant
    ids so bootstrap copies of one person stay in one fold. The [0.1, 0.9]
    window is stored as a report. No row is deleted because of it. The
    numerical floor is 1e-6.
    """
    if weight_type not in {"ate", "att"}:
        raise ValueError("weight_type must be 'ate' or 'att'.")
    outcome_values = as_float_array(y)
    assignment = as_treatment(treatment)
    features = np.asarray(covariates, dtype=float)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if features.shape[0] != outcome_values.shape[0] or assignment.shape[0] != outcome_values.shape[0]:
        raise ValueError("Outcome, treatment, and covariates must have the same number of rows.")
    n_treated, n_control = require_arms(assignment)
    if n_folds < 2:
        raise ValueError("Cross-fitting needs at least two folds.")
    if min(n_treated, n_control) < n_folds:
        raise EmptyFoldError(f"Not enough observations per arm for {n_folds} folds.")
    if participant_ids is None:
        ids = list(range(outcome_values.size))
    else:
        ids = list(participant_ids)
        if len(ids) != outcome_values.size:
            raise ValueError("participant_ids must have one entry per estimation row.")
    fold_id = _assign_crossfit_folds(ids, n_folds, seed)
    n_unique_participants = len(dict.fromkeys(ids))
    scores = np.empty(outcome_values.size)
    out_of_fold_propensity = np.empty(outcome_values.size)
    n_floored = 0
    treated_share = float(assignment.mean())
    fold_summaries = []
    for fold in range(n_folds):
        test_index = np.flatnonzero(fold_id == fold)
        train_index = np.flatnonzero(fold_id != fold)
        train_ids = {ids[int(index)] for index in train_index}
        test_ids = {ids[int(index)] for index in test_index}
        if train_ids.intersection(test_ids):
            raise RuntimeError(
                f"Fold {fold} placed the same participant in training and testing. No row was deleted."
            )
        train_assignment = assignment[train_index]
        train_treated = train_assignment == 1
        train_control = ~train_treated
        test_assignment = assignment[test_index]
        summary = {
            "fold": fold,
            "n_train": int(train_index.size),
            "n_test": int(test_index.size),
            "n_unique_participants_train": len(train_ids),
            "n_unique_participants_test": len(test_ids),
            "n_treated_train": int(train_treated.sum()),
            "n_control_train": int(train_control.sum()),
            "n_treated_test": int(test_assignment.sum()),
            "n_control_test": int(test_assignment.size - test_assignment.sum()),
            "seed": seed,
            "status": "not_fit",
        }
        if train_treated.sum() < 2 or train_control.sum() < 2:
            summary["status"] = "failed_empty_training_arm"
            fold_summaries.append(summary)
            raise EmptyFoldError(f"Fold {fold} training data lost an arm. {summary}")
        try:
            # ATT uses the observed treated outcomes, so a treated-arm regression
            # is not part of the score. Fitting it would let an unused nuisance
            # abort the ATT estimate.
            mu0 = _ols_predict(
                features[train_index][train_control],
                outcome_values[train_index][train_control],
                features[test_index],
            )
            propensity = _logit_predict(features[train_index], train_assignment, features[test_index])
            mu1 = None
            if weight_type == "ate":
                mu1 = _ols_predict(
                    features[train_index][train_treated],
                    outcome_values[train_index][train_treated],
                    features[test_index],
                )
        except FitFailure as exc:
            summary["status"] = "failed_nuisance"
            summary["error"] = str(exc)
            fold_summaries.append(summary)
            raise FitFailure(f"Fold {fold} nuisance fit failed: {exc}") from exc
        floored, moved = apply_numerical_floor(propensity)
        n_floored += moved
        d_test = test_assignment
        y_test = outcome_values[test_index]
        if weight_type == "ate":
            scores[test_index] = (mu1 - mu0) + d_test * (y_test - mu1) / floored - (1.0 - d_test) * (y_test - mu0) / (
                1.0 - floored
            )
        else:
            scores[test_index] = att_influence_scores(y_test, d_test, mu0, floored, treated_share)
        out_of_fold_propensity[test_index] = floored
        fold_id[test_index] = fold
        summary["status"] = "fit"
        summary["n_probabilities_moved_by_floor"] = moved
        fold_summaries.append(summary)
    if not np.isfinite(scores).all():
        raise FitFailure("AIPW scores are not finite. Rows were not deleted.")
    estimate = float(scores.mean())
    se = float(np.std(scores, ddof=1) / np.sqrt(scores.size))
    overlap = overlap_summary(out_of_fold_propensity, assignment, OVERLAP_REPORT_LOW, OVERLAP_REPORT_HIGH)
    result = EstimateResult(
        estimand=estimand,
        estimator=f"crossfit_aipw_{weight_type}_linear",
        estimate=estimate,
        se=se,
        n=int(scores.size),
        n_treated=n_treated,
        n_control=n_control,
        inference=(
            "Cross-fit AIPW score SD/sqrt(n); normal 95% interval. "
            "The treated share is held fixed in the ATT score. "
            "This interval is not an equivalence test and does not remove unmeasured confounding."
        ),
        target_population=target_population,
        outcome=outcome,
        warnings=(
            "Double robustness does not protect against unmeasured confounding or a mismatched target population.",
            "Nuisance models are linear/logistic, not the product-release AutoML ensemble.",
            "The overlap window is a report. No row was removed because of it.",
        ),
        diagnostics={
            "n_folds": n_folds,
            "seed": seed,
            "fold_assignment": "distinct_participant_id",
            "n_unique_participants": n_unique_participants,
            "participant_ids_disjoint_across_train_and_test": True,
            "weight_type": weight_type,
            "treated_outcome_model_used": weight_type == "ate",
            "treated_share_held_fixed_in_score_se": weight_type == "att",
            "n_probabilities_moved_by_floor": n_floored,
            "numerical_floor": NUMERICAL_FLOOR,
            "scientific_trimming": "none",
            "rows_deleted_for_overlap": 0,
            "fold_id": [int(value) for value in fold_id.tolist()],
            "fold_summaries": fold_summaries,
            "overlap_report": overlap,
        },
    )
    validate_result(result)
    return result
