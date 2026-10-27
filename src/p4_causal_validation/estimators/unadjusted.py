"""Difference in means. Descriptive on observational samples; a randomized benchmark when assignment is random."""

from __future__ import annotations

import numpy as np

from p4_causal_validation.estimators.common import (
    EstimateResult,
    as_float_array,
    as_treatment,
    require_arms,
    validate_result,
)


def difference_in_means(
    y,
    treatment,
    *,
    estimand: str,
    outcome: str,
    target_population: str,
    role: str,
) -> EstimateResult:
    """Neyman difference in means with unequal-variance standard error.

    The variance uses sample variances with divisor n-1, matching the usual
    Neyman / Welch formula. The randomized point estimate is not a noiseless truth.
    """
    outcome_values = as_float_array(y)
    assignment = as_treatment(treatment)
    if outcome_values.shape != assignment.shape:
        raise ValueError("Outcome and treatment lengths differ.")
    n_treated, n_control = require_arms(assignment)
    treated = outcome_values[assignment == 1]
    control = outcome_values[assignment == 0]
    estimate = float(treated.mean() - control.mean())
    variance = float(treated.var(ddof=1) / n_treated + control.var(ddof=1) / n_control)
    result = EstimateResult(
        estimand=estimand,
        estimator="unadjusted_difference_in_means",
        estimate=estimate,
        se=float(np.sqrt(variance)),
        n=int(outcome_values.size),
        n_treated=n_treated,
        n_control=n_control,
        inference="Neyman unequal-variance SE; normal 95% interval",
        target_population=target_population,
        outcome=outcome,
        warnings=(role,),
        diagnostics={
            "mean_treated": float(treated.mean()),
            "mean_control": float(control.mean()),
            "role": role,
        },
    )
    validate_result(result)
    return result
