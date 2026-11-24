"""Audit fitted counterfactual predictions using the two discordant randomized cells."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.linear_model import LinearRegression

import wsc_lib as lib


def intervals(estimate, se, df):
    radii = t.ppf([1 - .05 / 2, 1 - .05 / 16], df) * se
    return dict(estimate=estimate, se=se, df=df, lo=estimate-radii[0], hi=estimate+radii[0],
                family_lo=estimate-radii[1], family_hi=estimate+radii[1])


def fit_rules(concordant, outcome):
    # Coding is relative to the session being evaluated.
    x = concordant[lib.COVS_FULL].to_numpy(dtype=float)
    y = concordant[outcome].to_numpy(dtype=float)
    assigned = concordant.mathGrp.to_numpy()
    if outcome == "vocabPost":
        assigned = 1 - assigned
    estimates = lib.aipw_all(x, assigned, y)
    rules = []
    pretest = outcome.replace("Post", "Pre")
    for s, estimand, sign in [(1, "att", 1), (0, "atu", -1)]:
        target = assigned == s
        model = LinearRegression().fit(x[~target], y[~target])
        implied = y[target].mean() - sign * estimates[estimand]
        shift = implied - model.predict(x[target]).mean()
        rules.append((s, estimand.upper(), sign, model, shift, implied,
                      concordant.loc[target, pretest].median(), y[target].mean(), estimates[estimand]))
    return rules


def evaluate(frame):
    concordant = frame.mathSel == frame.mathGrp
    rows, comparisons = [], []
    for outcome in ["mathPost", "vocabPost"]:
        rules = fit_rules(frame.loc[concordant], outcome)
        preference = frame.mathSel if outcome == "mathPost" else 1 - frame.mathSel
        pretest = outcome.replace("Post", "Pre")
        for s, estimand, sign, model, shift, implied, cutoff, factual, qed in rules:
            validation = frame.loc[~concordant & (preference == s)]
            y = validation[outcome].to_numpy()
            prediction = model.predict(validation[lib.COVS_FULL].to_numpy(dtype=float)) + shift
            errors = sign * (y - prediction)
            high = validation[pretest].to_numpy() > cutoff
            low_errors, high_errors = errors[~high], errors[high]
            n0, n1 = len(low_errors), len(high_errors)
            if min(n0, n1) < 2:
                raise ValueError("Both pretest groups need at least two validation participants")
            v0, v1 = low_errors.var(ddof=1)/n0, high_errors.var(ddof=1)/n1
            denominator = v0**2/(n0-1) + v1**2/(n1-1)
            df = (v0 + v1)**2 / denominator if denominator else np.inf
            mean = errors.mean()
            mix = sign * (prediction.mean() - implied)
            gap = qed - sign * (factual - y.mean())
            np.testing.assert_allclose(mean + mix, gap, atol=1e-12)
            base = dict(outcome=outcome, estimand=estimand)
            rows.append(dict(**base, cutoff=cutoff, n_low=n0, n_high=n1,
                             residual_low=low_errors.mean(), residual_high=high_errors.mean(),
                             unshifted_residual=mean + sign*shift, shift=shift,
                             signed_residual=mean, prediction_mix=mix, original_gap=gap))
            for target, result in [
                ("mean", intervals(mean, errors.std(ddof=1)/np.sqrt(len(y)), len(y)-1)),
                ("high_minus_low", intervals(high_errors.mean()-low_errors.mean(), np.sqrt(v0+v1), df)),
            ]:
                comparisons.append(dict(**base, target=target, **result))
    return pd.DataFrame(rows), pd.DataFrame(comparisons)


if __name__ == "__main__":
    rows, comparisons = evaluate(lib.load())
    destination = Path("results")
    destination.mkdir(exist_ok=True)
    rows.to_csv(destination / "discordant_decomposition.csv", index=False)
    comparisons.to_csv(destination / "discordant_intervals.csv", index=False)
