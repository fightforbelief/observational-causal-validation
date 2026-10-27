"""Covariate balance."""

from __future__ import annotations

import numpy as np


def standardized_mean_difference(values, treatment) -> float:
    outcome = np.asarray(values, dtype=float)
    assignment = np.asarray(treatment, dtype=int)
    treated = outcome[assignment == 1]
    control = outcome[assignment == 0]
    if treated.size < 2 or control.size < 2:
        raise ValueError("SMD needs at least two observations in each arm.")
    pooled = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2.0)
    difference = float(treated.mean() - control.mean())
    if pooled == 0:
        return 0.0 if difference == 0 else np.inf
    return difference / float(pooled)


def balance_table(covariates, treatment, names: list[str]) -> list[dict]:
    features = np.asarray(covariates, dtype=float)
    if features.ndim != 2 or features.shape[1] != len(names):
        raise ValueError("Covariate names do not match the matrix.")
    rows = []
    for index, name in enumerate(names):
        rows.append({"covariate": name, "smd": standardized_mean_difference(features[:, index], treatment)})
    return rows
