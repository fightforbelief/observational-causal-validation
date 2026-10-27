"""Separate linear outcome regressions and standardization to a declared target sample."""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm

from p4_causal_validation.estimators.common import (
    EstimateResult,
    as_float_array,
    as_treatment,
    require_arms,
    validate_result,
)


def _fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_target: np.ndarray) -> tuple[np.ndarray, float]:
    if y_train.size <= x_train.shape[1] + 1:
        raise ValueError("An outcome arm has too few rows for its linear model.")
    design = sm.add_constant(x_train, has_constant="add")
    model = sm.OLS(y_train, design).fit()
    target = sm.add_constant(x_target, has_constant="add")
    if target.shape[1] != design.shape[1]:
        raise ValueError("Target design columns do not match the fitted outcome model.")
    residual_mse = float(np.sum(model.resid**2) / model.df_resid)
    return np.asarray(model.predict(target), dtype=float), residual_mse


def g_computation(
    y,
    treatment,
    covariates,
    *,
    estimand: str,
    outcome: str,
    target_population: str,
    target_covariates=None,
) -> EstimateResult:
    """Standardize arm-specific linear predictions to ``target_covariates``.

    The residual standard error is the notebook-style approximation
    sqrt(MSE1/n1 + MSE0/n0). It does not include all uncertainty from estimating
    the regression coefficients, and it is not doubly robust.
    """
    outcome_values = as_float_array(y)
    assignment = as_treatment(treatment)
    features = np.asarray(covariates, dtype=float)
    if target_covariates is None:
        target = features
        population_note = "empirical covariate distribution of the estimation sample"
    else:
        target = np.asarray(target_covariates, dtype=float)
        population_note = "externally supplied target covariate distribution"
    if features.ndim != 2 or target.ndim != 2:
        raise ValueError("Covariate matrices must be two-dimensional.")
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError("Covariates contain non-finite values.")
    n_treated, n_control = require_arms(assignment)
    mu1, mse1 = _fit_predict(features[assignment == 1], outcome_values[assignment == 1], target)
    mu0, mse0 = _fit_predict(features[assignment == 0], outcome_values[assignment == 0], target)
    estimate = float(np.mean(mu1 - mu0))
    se = float(np.sqrt(mse1 / n_treated + mse0 / n_control))
    result = EstimateResult(
        estimand=estimand,
        estimator="linear_g_computation",
        estimate=estimate,
        se=se,
        n=int(target.shape[0]),
        n_treated=n_treated,
        n_control=n_control,
        inference="Residual-MSE approximation; coefficient uncertainty is not fully included",
        target_population=f"{target_population}; standardized to {population_note}",
        outcome=outcome,
        warnings=("Separate linear outcome models. Misspecification is not corrected by double robustness.",),
        diagnostics={"mse_treated": mse1, "mse_control": mse0, "n_target": int(target.shape[0])},
    )
    validate_result(result)
    return result
