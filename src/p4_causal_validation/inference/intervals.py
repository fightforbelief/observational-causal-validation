"""Interval helpers."""

from __future__ import annotations

from p4_causal_validation.estimators.common import Z95, EstimateResult, validate_result


def normal_interval(estimate: float, se: float) -> tuple[float, float]:
    low = estimate - Z95 * se
    high = estimate + Z95 * se
    if not (low <= high):
        raise ValueError("Interval endpoints are unordered.")
    return float(low), float(high)


def interval_contains(result: EstimateResult, value: float) -> bool:
    validate_result(result)
    return bool(result.ci_low <= value <= result.ci_high)
