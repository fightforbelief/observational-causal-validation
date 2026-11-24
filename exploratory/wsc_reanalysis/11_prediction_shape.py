"""Sharp linear sensitivity of discordant-cell residuals at fixed fitted effect."""
from pathlib import Path

import numpy as np
import pandas as pd

import wsc_lib as lib


def sensitivity(x, direction):
    centered = (x - x.mean(axis=0)) / np.sqrt(len(x))
    _, singular, vectors = np.linalg.svd(centered, full_matrices=False)
    keep = singular > max(centered.shape)*np.finfo(float).eps*singular[0]
    basis = vectors[keep]
    projection = basis @ direction
    if not np.allclose(basis.T @ projection, direction, rtol=1e-10, atol=1e-12):
        return np.inf
    return np.linalg.norm(projection / singular[keep])


def shape_table(data, intervals):
    rows = []
    for (outcome, estimand), tests in intervals.groupby(["outcome", "estimand"], sort=False):
        math_pref = int((outcome == "mathPost") == (estimand == "ATT"))
        target = data[data.mathSel == math_pref]
        train, val = target[target.mathGrp == math_pref], target[target.mathGrp != math_pref]
        x = train[lib.COVS_FULL].to_numpy(dtype=float)
        v = val[lib.COVS_FULL].to_numpy(dtype=float)
        pre = outcome.replace("Post", "Pre")
        high = val[pre].to_numpy() > train[pre].median()
        directions = dict(mean=v.mean(axis=0)-x.mean(axis=0),
                          high_minus_low=v[high].mean(axis=0)-v[~high].mean(axis=0))
        for row in tests.itertuples():
            scale = sensitivity(x, directions[row.target])
            radius = abs(row.estimate)/scale if scale else (np.inf if row.estimate else 0.)
            rows.append(dict(outcome=outcome, estimand=estimand, target=row.target,
                             unit_rms_change=scale, rms_to_zero=radius))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    path = Path("results")
    output = shape_table(lib.load(), pd.read_csv(path / "discordant_intervals.csv"))
    output.to_csv(path / "discordant_shape.csv", index=False)
