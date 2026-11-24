"""Regression checks for the two orientations and separation of training and validation."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "exploratory/wsc_reanalysis"
sys.path.insert(0, str(SOURCE))
spec = importlib.util.spec_from_file_location("discordant_validation", SOURCE / "10_discordant_validation.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture
def trial():
    rng = np.random.default_rng(970)
    x = pd.DataFrame(rng.normal(size=(480, 25)), columns=audit.lib.COVS_FULL)
    x["mathSel"] = np.repeat([0, 1], 240)
    x["mathGrp"] = np.tile([0, 1], 240)
    for outcome, pre in [("mathPost", "mathPre"), ("vocabPost", "vocabPre")]:
        x[outcome] = 2*x[pre] + x.mathGrp + rng.normal(size=len(x))
    return x


def test_discordant_changes_only_validation(trial):
    changed = trial.copy()
    discordant = changed.mathGrp != changed.mathSel
    changed.loc[discordant, ["mathPost", "vocabPost"]] += 7
    before, intervals = audit.evaluate(trial)
    after, revised = audit.evaluate(changed)
    np.testing.assert_allclose(after.signed_residual-before.signed_residual, [7, -7, 7, -7])
    np.testing.assert_allclose(before["shift"], after["shift"])
    np.testing.assert_allclose(before.prediction_mix, after.prediction_mix)
    np.testing.assert_allclose(intervals.se, revised.se)
    assert np.all(intervals.family_lo <= intervals.lo)


def test_constant_response_surface_moves_entire_gap_to_residual(trial):
    trial["mathPost"] = np.where(trial.mathGrp == trial.mathSel, 0., 3.)
    rows, _ = audit.evaluate(trial)
    math = rows[rows.outcome == "mathPost"]
    np.testing.assert_allclose(math.prediction_mix, 0., atol=1e-12)
    np.testing.assert_allclose(math.signed_residual, [3., -3.])
    np.testing.assert_allclose(math.original_gap, math.signed_residual)


def test_sharp_shape_bound_and_off_support_direction():
    spec = importlib.util.spec_from_file_location("prediction_shape", SOURCE / "11_prediction_shape.py")
    shape = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shape)
    x = np.array([[1., 0.], [3., 0.], [5., 0.]])
    # A unit-RMS linear function has slope 1/std(x) along its varying coordinate.
    assert shape.sensitivity(x, np.array([2., 0.])) == pytest.approx(2/x[:, 0].std())
    assert np.isinf(shape.sensitivity(x, np.array([2., 1.])))
    assert shape.sensitivity(x, np.array([0., 0.])) == 0


def test_source_contribution_matches_hc3():
    from statsmodels.api import OLS, add_constant
    spec = importlib.util.spec_from_file_location("gradient_uncertainty", SOURCE / "12_gradient_uncertainty.py")
    gradient = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gradient)
    rng = np.random.default_rng(227)
    x = rng.normal(size=(140, 3))
    y = x[:, 1]**2 + rng.normal(size=len(x))
    d = np.array([.2, -.4, 1.])
    hc3 = OLS(y, add_constant(x)).fit(cov_type="HC3").cov_params()
    assert gradient.source_variance(x, y, d) == pytest.approx(np.r_[0, d] @ hc3 @ np.r_[0, d])
