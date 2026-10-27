"""Contrasts that keep the randomized benchmark's sampling uncertainty."""

from __future__ import annotations

import math

from p4_causal_validation.estimators.common import EstimateResult, require_same_estimand, validate_result
from p4_causal_validation.inference.intervals import normal_interval


def contrast_estimates(
    benchmark: EstimateResult,
    comparison: EstimateResult,
    *,
    dependence: str,
    scale_sd: float | None = None,
    override_reason: str | None = None,
) -> dict:
    """Compare two estimates of the same estimand and outcome.

    ``dependence='independent'`` uses the sum of variances. Overlapping WSC
    samples must use ``dependence='overlapping_diagnostic'``; that SE is labeled
    invalid for inference because it ignores covariance. Interval overlap is not
    an equivalence test.
    """
    if dependence not in {"independent", "overlapping_diagnostic"}:
        raise ValueError("dependence must be independent or overlapping_diagnostic.")
    require_same_estimand(benchmark, comparison, override_reason)
    validate_result(benchmark)
    validate_result(comparison)
    difference = float(comparison.estimate - benchmark.estimate)
    se = math.sqrt(benchmark.se**2 + comparison.se**2)
    low, high = normal_interval(difference, se)
    result = {
        "estimand": benchmark.estimand if override_reason is None else f"{benchmark.estimand}__vs__{comparison.estimand}",
        "outcome": benchmark.outcome,
        "benchmark_estimator": benchmark.estimator,
        "comparison_estimator": comparison.estimator,
        "benchmark_estimate": benchmark.estimate,
        "benchmark_se": benchmark.se,
        "comparison_estimate": comparison.estimate,
        "comparison_se": comparison.se,
        "difference_comparison_minus_benchmark": difference,
        "difference_se": se,
        "difference_ci_low": low,
        "difference_ci_high": high,
        "dependence": dependence,
        "independence_se_valid": dependence == "independent",
        "interval_overlap_is_not_equivalence": True,
        "override_reason": override_reason,
    }
    if scale_sd is not None:
        if scale_sd <= 0 or not math.isfinite(scale_sd):
            raise ValueError("Scaling SD must be a positive finite number.")
        result["standardized_difference"] = difference / scale_sd
        result["scale_sd"] = scale_sd
    if dependence == "overlapping_diagnostic":
        result["warning"] = (
            "The difference SE assumes independence. Overlapping RCT and QED subsets require a "
            "participant-level bootstrap or stacked scores before this interval is used inferentially."
        )
    return result
