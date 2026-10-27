"""Propensity weighting with explicit estimands. No data-driven weight trimming."""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm

from p4_causal_validation.diagnostics.overlap import effective_sample_size
from p4_causal_validation.estimators.common import (
    EstimateResult,
    as_float_array,
    as_treatment,
    require_arms,
    validate_result,
)

NUMERICAL_FLOOR = 1e-6


def fit_logit_propensity(treatment: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    design = sm.add_constant(covariates, has_constant="add")
    model = sm.Logit(treatment, design).fit(disp=False, maxiter=200)
    probability = np.asarray(model.predict(design), dtype=float)
    if not np.isfinite(probability).all():
        raise ValueError("Propensity model returned non-finite probabilities.")
    return probability


def _hajek(y: np.ndarray, weights: np.ndarray) -> float:
    if np.sum(weights) <= 0:
        raise ValueError("Weights sum to zero.")
    return float(np.sum(weights * y) / np.sum(weights))


def propensity_weighting(
    y,
    treatment,
    covariates,
    *,
    estimand: str,
    outcome: str,
    target_population: str,
    weight_type: str = "ate",
) -> EstimateResult:
    """Hajek propensity weighting.

    ``weight_type='ate'`` targets the combined sample. ``weight_type='att'``
    targets the treated units in this sample. The numerical floor is only a
    division guard; observations moved by the floor are counted and do not
    define a new scientific trimming rule.
    """
    if weight_type not in {"ate", "att"}:
        raise ValueError("weight_type must be 'ate' or 'att'.")
    outcome_values = as_float_array(y)
    assignment = as_treatment(treatment)
    features = np.asarray(covariates, dtype=float)
    if features.ndim != 2 or features.shape[0] != outcome_values.size:
        raise ValueError("Covariate shape does not match the outcome.")
    n_treated, n_control = require_arms(assignment)
    raw = fit_logit_propensity(assignment, features)
    floored = np.clip(raw, NUMERICAL_FLOOR, 1 - NUMERICAL_FLOOR)
    n_floored = int(np.sum(raw != floored))
    if weight_type == "ate":
        weights = np.where(assignment == 1, 1.0 / floored, 1.0 / (1.0 - floored))
        treated_mean = _hajek(outcome_values, weights * assignment)
        control_mean = _hajek(outcome_values, weights * (1 - assignment))
        estimate = treated_mean - control_mean
        score = assignment * (outcome_values - treated_mean) / floored - (1 - assignment) * (outcome_values - control_mean) / (
            1 - floored
        )
        inference = "Hajek ATE weights; influence-function SE treats the propensity fit as fixed"
    else:
        odds = floored / (1.0 - floored)
        weights = np.where(assignment == 1, 1.0, odds)
        treated_mean = _hajek(outcome_values, weights * assignment)
        control_mean = _hajek(outcome_values, weights * (1 - assignment))
        estimate = treated_mean - control_mean
        treated_share = float(assignment.mean())
        control_mass = float(np.mean((1 - assignment) * odds))
        score = assignment * (outcome_values - treated_mean) / treated_share - (1 - assignment) * odds * (
            outcome_values - control_mean
        ) / control_mass
        inference = "Hajek ATT weights; influence-function SE treats the propensity fit as fixed"
    se = float(np.std(score, ddof=1) / np.sqrt(outcome_values.size))
    result = EstimateResult(
        estimand=estimand,
        estimator=f"hajek_ipw_{weight_type}",
        estimate=float(estimate),
        se=se,
        n=int(outcome_values.size),
        n_treated=n_treated,
        n_control=n_control,
        inference=inference,
        target_population=target_population,
        outcome=outcome,
        warnings=(
            "Propensity weighting does not remove unmeasured confounding.",
            "The reported SE does not include all uncertainty from estimating the propensity model.",
        ),
        diagnostics={
            "weight_type": weight_type,
            "numerical_floor": NUMERICAL_FLOOR,
            "n_probabilities_moved_by_floor": n_floored,
            "propensity_min": float(raw.min()),
            "propensity_max": float(raw.max()),
            "weight_min": float(weights.min()),
            "weight_max": float(weights.max()),
            "ess_treated": effective_sample_size(weights[assignment == 1]),
            "ess_control": effective_sample_size(weights[assignment == 0]),
            "propensity": raw,
            "weights": weights,
        },
    )
    validate_result(result)
    return result
