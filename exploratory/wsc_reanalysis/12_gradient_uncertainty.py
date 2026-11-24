"""Training-inclusive uncertainty for opposite-session subgroup residual differences."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import wsc_lib as lib


def source_variance(x, y, gradient):
    design = np.column_stack([np.ones(len(x)), x])
    contrast = np.r_[0., gradient]
    inverse = np.linalg.pinv(design)
    if not np.allclose(contrast @ inverse @ design, contrast, rtol=1e-8, atol=1e-10):
        raise ValueError("Validation contrast lies outside the estimable source space")
    score_weights = contrast @ inverse
    leverage = np.einsum("ij,ji->i", design, inverse)
    if np.any(leverage >= 1-1e-12):
        raise ValueError("HC3 is undefined for a source observation with unit leverage")
    residual = y - design @ (inverse @ y)
    return np.sum((score_weights * residual / (1-leverage))**2)


def gradient_intervals(data, estimates):
    rows = []
    for row in estimates[estimates.target == "high_minus_low"].itertuples():
        pref = int((row.outcome == "mathPost") == (row.estimand == "ATT"))
        target = data[data.mathSel == pref]
        pretest = row.outcome.replace("Post", "Pre")
        cutoff = target.loc[target.mathGrp == pref, pretest].median()
        val = target[target.mathGrp != pref]
        source = data[(data.mathSel != pref) & (data.mathGrp != pref)]
        high = val[pretest] > cutoff
        x = val[lib.COVS_FULL]
        gradient = (x[high].mean() - x[~high].mean()).to_numpy()
        variance = source_variance(source[lib.COVS_FULL].to_numpy(), source[row.outcome].to_numpy(), gradient)
        se = np.sqrt(row.se**2 + variance)
        radius = norm.ppf(1-.05/16)*se
        rows.append(dict(outcome=row.outcome, estimand=row.estimand, gradient=row.estimate,
                         validation_se=row.se, source_se=np.sqrt(variance), total_se=se,
                         family_lo=row.estimate-radius, family_hi=row.estimate+radius))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    results = Path("results")
    table = gradient_intervals(lib.load(), pd.read_csv(results / "discordant_intervals.csv"))
    table.to_csv(results / "discordant_retrained_gradients.csv", index=False)
