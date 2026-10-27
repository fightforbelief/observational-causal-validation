"""Linear regression adjustment. The coefficient is a constant-effect summary, not automatically a standardized ATE."""

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


def linear_adjustment(
    y,
    treatment,
    covariates,
    *,
    estimand: str,
    outcome: str,
    target_population: str,
) -> EstimateResult:
    """OLS of Y on treatment and main-effect covariates, HC1 standard error.

    Without treatment-by-covariate interactions, the treatment coefficient equals
    the g-computation contrast from this same additive model. It is not a
    saturated or machine-learning standardized effect.
    """
    outcome_values = as_float_array(y)
    assignment = as_treatment(treatment)
    features = np.asarray(covariates, dtype=float)
    if features.ndim != 2:
        raise ValueError("Covariates must be a two-dimensional array.")
    if features.shape[0] != outcome_values.size or assignment.size != outcome_values.size:
        raise ValueError("Row counts differ.")
    if not np.isfinite(features).all():
        raise ValueError("Covariates contain non-finite values.")
    n_treated, n_control = require_arms(assignment)
    if outcome_values.size <= features.shape[1] + 2:
        raise ValueError("Too few rows for the additive regression design.")
    design = sm.add_constant(np.column_stack([assignment, features]), has_constant="add")
    model = sm.OLS(outcome_values, design).fit(cov_type="HC1")
    estimate = float(model.params[1])
    se = float(model.bse[1])
    result = EstimateResult(
        estimand=estimand,
        estimator="linear_regression_hc1",
        estimate=estimate,
        se=se,
        n=int(outcome_values.size),
        n_treated=n_treated,
        n_control=n_control,
        inference="HC1 robust SE for the treatment coefficient in an additive OLS model",
        target_population=target_population,
        outcome=outcome,
        warnings=(
            "This is the additive treatment coefficient. It matches g-computation only under this linear specification.",
        ),
        diagnostics={"r_squared": float(model.rsquared), "n_covariates": int(features.shape[1])},
    )
    validate_result(result)
    return result
