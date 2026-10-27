"""Overlap and weight diagnostics."""

from __future__ import annotations

import numpy as np


def effective_sample_size(weights) -> float:
    values = np.asarray(weights, dtype=float)
    if values.size == 0:
        raise ValueError("Cannot compute effective sample size for an empty weight vector.")
    if np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("Weights must be finite and non-negative.")
    denominator = float(np.sum(values**2))
    if denominator == 0:
        return 0.0
    return float(np.sum(values) ** 2 / denominator)


def overlap_summary(propensity, treatment, low: float = 0.1, high: float = 0.9) -> dict:
    """Report common support. The 0.1/0.9 bounds are a reporting screen, not a trim."""
    probability = np.asarray(propensity, dtype=float)
    assignment = np.asarray(treatment, dtype=int)
    if probability.shape != assignment.shape:
        raise ValueError("Propensity and treatment lengths differ.")
    outside = (probability < low) | (probability > high)
    return {
        "reporting_low": low,
        "reporting_high": high,
        "min": float(probability.min()),
        "max": float(probability.max()),
        "share_outside_reporting_window": float(outside.mean()),
        "n_outside_reporting_window": int(outside.sum()),
        "min_treated": float(probability[assignment == 1].min()) if np.any(assignment == 1) else None,
        "max_treated": float(probability[assignment == 1].max()) if np.any(assignment == 1) else None,
        "min_control": float(probability[assignment == 0].min()) if np.any(assignment == 0) else None,
        "max_control": float(probability[assignment == 0].max()) if np.any(assignment == 0) else None,
    }
