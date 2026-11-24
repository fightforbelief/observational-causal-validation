"""Known-truth conditional calibration experiment for the discordant validation protocol."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


def design(rng, repeats, size, location):
    z = rng.normal(location, 1, (repeats, size))
    group = rng.choice([-1., 1.], size=z.shape)
    return np.stack([np.ones_like(z), z, group], axis=2)


def noise(rng, shape, skewed):
    return rng.exponential(1, shape) - 1 if skewed else rng.normal(size=shape)


def simulate(rng, repeats, n_validation, delta, skewed):
    coefficients = np.array([2., .7, .4])
    train = design(rng, repeats, 160, -.5)
    y_train = train @ coefficients + noise(rng, train.shape[:2], skewed)
    xt = train.transpose(0, 2, 1)
    fit = np.linalg.solve(xt @ train, xt @ y_train[..., None])[..., 0]
    # Linear projection on the source population; the subsequent shift is an intercept.
    inverse = np.linalg.solve(xt @ train, xt)
    leverage = (train * inverse.transpose(0, 2, 1)).sum(axis=2)
    source_residual = y_train - (train * fit[:, None, :]).sum(axis=2)
    calibration = design(rng, repeats, 120, -.5)
    residual = calibration @ coefficients + noise(rng, calibration.shape[:2], skewed)
    residual -= (calibration * fit[:, None, :]).sum(axis=2)
    # Density ratio between N(.5,1) and N(-.5,1) is exp(z).
    fit[:, 0] += (np.exp(calibration[:, :, 1]) * residual).mean(axis=1)
    val = design(rng, repeats, n_validation, .5)
    group = val[:, :, 2]
    errors = (val * (coefficients-fit)[:, None, :]).sum(axis=2)
    errors += delta * group + noise(rng, group.shape, skewed)
    means, var_means, counts = [], [], []
    for mask in [group < 0, group > 0]:
        n = mask.sum(axis=1)
        mean = (errors*mask).sum(axis=1)/n
        variance = (((errors-mean[:, None])**2)*mask).sum(axis=1)/(n-1)
        counts.append(n)
        means.append(mean)
        var_means.append(variance/n)
    v0, v1 = var_means
    welch = (v0+v1)**2 / (v0**2/(counts[0]-1) + v1**2/(counts[1]-1))
    high = group > 0
    low = ~high
    dx = (val * high[..., None]).sum(axis=1)/counts[1][:, None]
    dx -= (val * low[..., None]).sum(axis=1)/counts[0][:, None]
    weights = (dx[:, None, :] @ inverse)[:, 0, :]
    source_variance = ((weights * source_residual / (1-leverage))**2).sum(axis=1)
    true_mean = 2-fit[:, 0] + .5*(.7-fit[:, 1])
    true_diff = 2*(.4-fit[:, 2]+delta)
    return [
        ("mean", errors.mean(axis=1), errors.std(axis=1, ddof=1)/np.sqrt(n_validation), n_validation-1, true_mean),
        ("high_minus_low", means[1]-means[0], np.sqrt(v0+v1), welch, true_diff),
        ("population_gradient", means[1]-means[0], np.sqrt(v0+v1+source_variance), np.inf, 2*delta),
    ]


def run():
    rng = np.random.default_rng(20261124)
    rows = []
    for scenario, n, delta, skew in [("common", 319, 0., False), ("cancellation", 319, 1., False),
                                     ("skewed_small", 90, 1., True)]:
        results = simulate(rng, 5000, n, delta, skew)
        for target, estimate, se, df, truth in results:
            for alpha in [.05, .05/8]:
                radius = t.ppf(1-alpha/2, df)*se
                cover = (np.abs(estimate-truth) <= radius).mean()
                rows.append(dict(scenario=scenario, n=n, trials=len(estimate), target=target, alpha=alpha,
                                 avg_target=np.mean(truth), avg_estimate=estimate.mean(), coverage=cover,
                                 mcse=np.sqrt(cover*(1-cover)/len(estimate)),
                                 excludes_zero=(np.abs(estimate)>radius).mean()))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run().to_csv(args.output, index=False)
