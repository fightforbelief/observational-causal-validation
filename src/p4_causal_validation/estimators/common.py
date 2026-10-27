"""Shared estimate container and estimand checks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Z95 = 1.959963984540054


class EstimandMismatchError(ValueError):
    """Raised when two results target different estimands."""


class EmptyStratumError(ValueError):
    """Raised when a treatment arm or target population cannot support estimation."""


class DataContractError(ValueError):
    """Raised when a data adapter would leak post-treatment information or invent outcomes."""


@dataclass
class EstimateResult:
    estimand: str
    estimator: str
    estimate: float
    se: float
    n: int
    n_treated: int
    n_control: int
    inference: str
    target_population: str
    outcome: str
    warnings: tuple[str, ...] = ()
    diagnostics: dict = field(default_factory=dict)

    @property
    def ci_low(self) -> float:
        return float(self.estimate - Z95 * self.se)

    @property
    def ci_high(self) -> float:
        return float(self.estimate + Z95 * self.se)

    def as_dict(self) -> dict:
        return {
            "estimand": self.estimand,
            "estimator": self.estimator,
            "outcome": self.outcome,
            "target_population": self.target_population,
            "estimate": self.estimate,
            "se": self.se,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n": self.n,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "inference": self.inference,
            "warnings": list(self.warnings),
            "diagnostics": self.diagnostics,
        }


def validate_result(result: EstimateResult) -> None:
    if not np.isfinite(result.estimate) or not np.isfinite(result.se):
        raise ValueError("Estimate and standard error must be finite.")
    if result.se < 0:
        raise ValueError("Standard error must be non-negative.")
    if result.ci_low > result.ci_high:
        raise ValueError("Confidence interval endpoints are unordered.")
    if result.n_treated < 0 or result.n_control < 0 or result.n < result.n_treated:
        raise ValueError("Sample counts are inconsistent.")


def require_same_estimand(left: EstimateResult, right: EstimateResult, override_reason: str | None = None) -> None:
    if left.estimand == right.estimand and left.outcome == right.outcome:
        return
    if override_reason:
        return
    raise EstimandMismatchError(
        f"Refusing to compare {left.estimator}:{left.estimand}/{left.outcome} with "
        f"{right.estimator}:{right.estimand}/{right.outcome}. "
        "Pass an explicit written override only when the estimand difference is the object of study."
    )


def as_float_array(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("Expected a one-dimensional array.")
    if not np.isfinite(array).all():
        raise ValueError("Non-finite values are not silently imputed.")
    return array


def as_treatment(values) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("Treatment must be one-dimensional.")
    if array.dtype == bool:
        coded = array.astype(int)
    else:
        coded = np.asarray(array, dtype=int)
    if not np.isin(coded, [0, 1]).all():
        raise ValueError("Treatment must be coded 0/1.")
    return coded


def require_arms(treatment: np.ndarray, minimum: int = 2) -> tuple[int, int]:
    n_treated = int(treatment.sum())
    n_control = int(treatment.size - n_treated)
    if n_treated < minimum or n_control < minimum:
        raise EmptyStratumError(
            f"Need at least {minimum} observations in each arm; got treated={n_treated}, control={n_control}."
        )
    return n_treated, n_control
