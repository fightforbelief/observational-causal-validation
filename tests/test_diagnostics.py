import numpy as np
import pandas as pd
import pytest

from p4_causal_validation.diagnostics.balance import standardized_mean_difference
from p4_causal_validation.diagnostics.missingness import complete_case_mask
from p4_causal_validation.diagnostics.overlap import effective_sample_size, overlap_summary
from p4_causal_validation.estimators.common import DataContractError
from p4_causal_validation.data.wsc import resolve_covariate_columns


def test_smd_exact_answer():
    values = np.array([1, 3, 0, 0], dtype=float)
    treatment = np.array([1, 1, 0, 0])
    treated = values[:2]
    control = values[2:]
    expected = (treated.mean() - control.mean()) / np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2)
    assert standardized_mean_difference(values, treatment) == pytest.approx(expected)


def test_effective_sample_size_uniform_weights():
    assert effective_sample_size([2, 2, 2, 2]) == pytest.approx(4)


def test_overlap_summary_counts_reporting_window():
    summary = overlap_summary([0.05, 0.5, 0.95, 0.2], [1, 1, 0, 0])
    assert summary["n_outside_reporting_window"] == 2
    assert summary["reporting_low"] == 0.1


def test_complete_case_mask_does_not_impute():
    frame = pd.DataFrame({"y": [1, None], "x": [1, 2]})
    mask = complete_case_mask(frame, ["y", "x"])
    assert mask.tolist() == [True, False]


def test_covariate_resolver_rejects_post_treatment_alias():
    columns = ["mathPost"]
    frame = pd.DataFrame(columns=columns)
    with pytest.raises(DataContractError):
        resolve_covariate_columns(frame)
