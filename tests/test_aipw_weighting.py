import numpy as np

from p4_causal_validation.estimators.aipw import crossfit_aipw
from p4_causal_validation.estimators.matching import att_nearest_neighbor
from p4_causal_validation.estimators.weighting import propensity_weighting


def _draw(seed: int):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(120, 1))
    propensity = 1 / (1 + np.exp(-0.4 * x[:, 0]))
    treatment = rng.binomial(1, propensity)
    outcome = 2.0 * treatment + 0.5 * x[:, 0] + rng.normal(scale=0.2, size=x.shape[0])
    return outcome, treatment, x


def test_weighting_and_aipw_recover_a_constant_effect_approximately():
    y, d, x = _draw(7)
    ipw = propensity_weighting(y, d, x, estimand="ATE_full", outcome="y", target_population="fixture", weight_type="ate")
    dr = crossfit_aipw(
        y, d, x, estimand="ATE_full", outcome="y", target_population="fixture", n_folds=4, seed=7, weight_type="ate"
    )
    assert abs(ipw.estimate - 2.0) < 0.75
    assert abs(dr.estimate - 2.0) < 0.75
    assert ipw.ci_low < ipw.ci_high
    assert dr.diagnostics["seed"] == 7


def test_matching_seed_is_reproducible():
    y, d, x = _draw(4)
    left = att_nearest_neighbor(y, d, x, estimand="ATT_fixture", outcome="y", target_population="fixture", random_state=4)
    right = att_nearest_neighbor(y, d, x, estimand="ATT_fixture", outcome="y", target_population="fixture", random_state=4)
    assert left.estimate == right.estimate
    assert left.diagnostics["n_pairs"] >= 2
