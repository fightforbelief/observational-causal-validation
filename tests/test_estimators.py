import numpy as np
import pytest

from p4_causal_validation.estimators.common import EstimandMismatchError
from p4_causal_validation.estimators.matching import att_nearest_neighbor
from p4_causal_validation.estimators.outcome_model import g_computation
from p4_causal_validation.estimators.regression import linear_adjustment
from p4_causal_validation.estimators.unadjusted import difference_in_means
from p4_causal_validation.inference.comparisons import contrast_estimates


def test_difference_in_means_matches_hand_calculation():
    result = difference_in_means(
        [1, 3, 0, 2],
        [1, 1, 0, 0],
        estimand="ATE_full",
        outcome="y",
        target_population="fixture",
        role="randomized benchmark",
    )
    treated = np.array([1.0, 3.0])
    control = np.array([0.0, 2.0])
    estimate = treated.mean() - control.mean()
    se = np.sqrt(treated.var(ddof=1) / 2 + control.var(ddof=1) / 2)
    assert result.estimate == pytest.approx(estimate)
    assert result.se == pytest.approx(se)
    assert result.ci_low < result.ci_high


def test_empty_arm_is_explicit():
    with pytest.raises(Exception, match="each arm"):
        difference_in_means([1, 2, 3], [1, 1, 1], estimand="ATE_full", outcome="y", target_population="x", role="x")


def test_regression_coefficient_on_balanced_binary_covariate():
    y = np.array([1, 2, 3, 4], dtype=float)
    d = np.array([0, 0, 1, 1])
    x = np.array([[0], [1], [0], [1]], dtype=float)
    result = linear_adjustment(y, d, x, estimand="ATE_full", outcome="y", target_population="fixture")
    assert result.estimate == pytest.approx(2.0)
    assert result.ci_low < result.estimate < result.ci_high


def test_g_computation_constant_effect_matches_regression_coefficient():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 1))
    d = np.array([0, 1] * 20)
    y = 1.5 + 2.0 * d + 0.3 * x[:, 0]
    reg = linear_adjustment(y, d, x, estimand="ATE_full", outcome="y", target_population="fixture")
    gcomp = g_computation(y, d, x, estimand="ATE_full", outcome="y", target_population="fixture")
    assert gcomp.estimate == pytest.approx(reg.estimate, abs=1e-8)


def test_matching_requires_att_label_and_returns_finite_interval():
    y = np.array([5, 1, 6, 2, 7, 3], dtype=float)
    d = np.array([1, 0, 1, 0, 1, 0])
    x = np.array([[0], [0], [1], [1], [2], [2]], dtype=float)
    with pytest.raises(ValueError, match="ATT"):
        att_nearest_neighbor(y, d, x, estimand="ATE_full", outcome="y", target_population="fixture")
    result = att_nearest_neighbor(y, d, x, estimand="ATT_fixture", outcome="y", target_population="fixture")
    assert result.n_treated == result.n_control
    assert np.isfinite(result.estimate) and result.ci_low <= result.ci_high


def test_contrast_keeps_both_standard_errors_and_blocks_mismatches():
    left = difference_in_means([1, 2, 0, 0], [1, 1, 0, 0], estimand="ATE_full", outcome="mathPost", target_population="a", role="rct")
    right = difference_in_means([3, 4, 0, 1], [1, 1, 0, 0], estimand="ATE_full", outcome="mathPost", target_population="b", role="qed")
    contrast = contrast_estimates(left, right, dependence="independent", scale_sd=2.0)
    assert contrast["difference_se"] == pytest.approx(np.sqrt(left.se**2 + right.se**2))
    assert contrast["standardized_difference"] == pytest.approx(contrast["difference_comparison_minus_benchmark"] / 2)
    other = difference_in_means([3, 4, 0, 1], [1, 1, 0, 0], estimand="ATT_other", outcome="mathPost", target_population="b", role="qed")
    with pytest.raises(EstimandMismatchError):
        contrast_estimates(left, other, dependence="independent")
