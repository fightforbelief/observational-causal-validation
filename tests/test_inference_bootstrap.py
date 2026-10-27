"""Synthetic checks for the primary-contrast bootstrap.

TEST_FIXTURE_NOT_RESEARCH_EVIDENCE. These fits are not WSC results.
"""

import numpy as np
import pandas as pd
import pytest

from p4_causal_validation.data.estimands import subset_for_estimand
from p4_causal_validation.data.wsc import COVARIATE_CANDIDATES
from p4_causal_validation.estimators.aipw import EmptyFoldError, FitFailure, att_influence_scores, crossfit_aipw
from p4_causal_validation.estimators.common import DataContractError, EmptyStratumError, EstimandMismatchError
from p4_causal_validation.estimators.unadjusted import difference_in_means
from p4_causal_validation.experiments.primary_specification import mark_synthetic, qed_concordant_att
from p4_causal_validation.inference.bootstrap import (
    PROTOCOL_BOOTSTRAP_REPLICATES,
    participant_bootstrap,
    stratified_primary_contrast_bootstrap,
)


def _synthetic_wsc(seed: int, effect: float, n1: int = 120, n2: int = 80, n3: int = 12, n4: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cells = [(1, 1, n1), (1, 0, n2), (0, 1, n3), (0, 0, n4)]
    preference = []
    assignment = []
    origins = []
    for s_value, z_value, count in cells:
        preference.extend([s_value] * count)
        assignment.extend([z_value] * count)
        origins.extend([f"{s_value}{z_value}"] * count)
    n = len(preference)
    preference = np.array(preference)
    assignment = np.array(assignment)
    data = {
        "row_id": np.arange(n),
        "cell_origin": origins,
        "mathSel": preference,
        "mathGrp": assignment,
        "vocabPost": np.zeros(n),
    }
    for candidate in COVARIATE_CANDIDATES:
        name = candidate[0] if isinstance(candidate, tuple) else candidate
        data[name] = rng.normal(size=n)
    data["mathPost"] = effect * assignment + 0.05 * data["mathPre"] + rng.normal(scale=0.05, size=n)
    data["mathPost"] = np.where(np.array(origins) == "01", 1000.0, data["mathPost"])
    return mark_synthetic(pd.DataFrame(data))


def test_neyman_hand_calculation_and_interval_order():
    result = difference_in_means(
        [5.0, 1.0, 2.0, 0.0],
        [1, 1, 0, 0],
        estimand="RCT_effect_prefer_math",
        outcome="mathPost",
        target_population="G1 versus G2",
        role="randomized benchmark",
    )
    treated = np.array([5.0, 1.0])
    control = np.array([2.0, 0.0])
    assert result.estimate == pytest.approx(treated.mean() - control.mean())
    assert result.se == pytest.approx(np.sqrt(treated.var(ddof=1) / 2 + control.var(ddof=1) / 2))
    assert np.isfinite(result.ci_low) and result.ci_low <= result.ci_high


def test_att_point_score_matches_hand_calculation():
    scores = att_influence_scores(
        [4.0, 2.0, 0.0, 0.0],
        [1, 1, 0, 0],
        [1.0, 1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5, 0.5],
        0.5,
    )
    assert scores.tolist() == pytest.approx([6.0, 2.0, 2.0, 2.0])
    assert float(scores.mean()) == pytest.approx(3.0)


def test_shared_g1_draw_refit_and_difference_inside_replicate(monkeypatch):
    frame = _synthetic_wsc(11, effect=0.5)
    seen = []
    calls = {"n": 0}
    import p4_causal_validation.experiments.primary_specification as spec

    real = spec.qed_concordant_att

    def wrapped(data):
        calls["n"] += 1
        return real(data)

    monkeypatch.setattr(spec, "qed_concordant_att", wrapped)
    result = stratified_primary_contrast_bootstrap(
        frame,
        n_replicates=2,
        seed=20261026,
        observer=seen.append,
    )
    assert calls["n"] == 1 + 2
    assert len(seen) == 2
    g3_ids = set(frame.loc[frame["cell_origin"].eq("01"), "row_id"].tolist())
    for item in seen:
        rct_ids = item["rct_row_ids"]
        qed_ids = item["qed_row_ids"]
        assert [row_id for row_id in rct_ids if row_id in g3_ids] == []
        assert [row_id for row_id in qed_ids if row_id in g3_ids] == []
        rct_treated = [row_id for row_id in rct_ids if frame.loc[frame["row_id"].eq(row_id), "mathGrp"].iloc[0] == 1]
        qed_treated = [row_id for row_id in qed_ids if frame.loc[frame["row_id"].eq(row_id), "cell_origin"].iloc[0] == "11"]
        assert rct_treated == qed_treated
        assert len(rct_ids) == 120 + 80
        assert len(qed_ids) == 120 + 240
    assert result["n_success"] == 2
    assert result["n_failed"] == 0
    assert result["n_success"] + result["n_failed"] == 2
    assert result["refits_aipw_each_replicate"] is True
    assert result["g3_in_primary_estimators"] is False
    assert result["difference_is_not_automatically_confounding"] is True
    assert result["confirmatory_interval_authorized"] is False
    assert result["failure_share_threshold"] is None
    assert result["fixed_cell_bootstrap_is_not_randomization_inference"] is True
    assert result["protocol_replicate_count"] == PROTOCOL_BOOTSTRAP_REPLICATES == 2000
    item = result["successful_replicates"][0]
    assert item["difference_qed_minus_rct"] == pytest.approx(item["qed_estimate"] - item["rct_estimate"])
    assert item["rct_recomputed"] and item["propensity_refit"] and item["control_outcome_refit"] and item["crossfit_refit"]
    assert item["qed_n_folds"] == 5 and item["qed_seed"] == 20261026
    assert np.isfinite(result["bootstrap_se"]) and np.isfinite(result["bootstrap_covariance"])
    assert result["percentile_ci_low"] <= result["percentile_ci_high"]
    assert abs(result["rct_estimate"] - 0.5) < 1.5


def test_bootstrap_seed_is_reproducible_on_a_short_run():
    frame = _synthetic_wsc(5, effect=0.4)
    left = stratified_primary_contrast_bootstrap(frame, n_replicates=2, seed=20261026)
    right = stratified_primary_contrast_bootstrap(frame, n_replicates=2, seed=20261026)
    assert left["rct_estimate"] == right["rct_estimate"]
    assert left["qed_estimate"] == right["qed_estimate"]
    assert left["n_failed"] == right["n_failed"]
    assert left["difference_qed_minus_rct"] == right["difference_qed_minus_rct"]
    assert left["not_a_wsc_result"] is True


def test_failed_replicates_are_kept_in_the_accounting(monkeypatch):
    frame = _synthetic_wsc(3, effect=0.2)
    import p4_causal_validation.experiments.primary_specification as spec

    real = spec.qed_concordant_att
    calls = {"n": 0}

    def wrapped(data):
        calls["n"] += 1
        if calls["n"] == 2:
            raise FitFailure("injected: logistic propensity did not converge")
        return real(data)

    monkeypatch.setattr(spec, "qed_concordant_att", wrapped)
    result = stratified_primary_contrast_bootstrap(frame, n_replicates=3, seed=7)
    assert result["n_failed"] == 1
    assert result["n_success"] == 2
    assert result["n_success"] + result["n_failed"] == 3
    assert result["failures"][0]["error_type"] == "FitFailure"
    assert "did not converge" in result["failures"][0]["message"]
    assert result["failures_not_redrawn"] is True
    assert result["percentile_ci_low"] <= result["percentile_ci_high"]


def test_empty_primary_cell_is_reported_and_wrong_estimand_is_refused():
    frame = mark_synthetic(
        pd.DataFrame(
            {
                "row_id": [1, 2, 3, 4],
                "mathSel": [1, 1, 0, 0],
                "mathGrp": [1, 1, 0, 0],
                "mathPost": [1.0, 2.0, 0.0, 1.0],
            }
        )
    )
    with pytest.raises(EmptyStratumError, match="G2"):
        stratified_primary_contrast_bootstrap(frame, n_replicates=2, seed=1)
    full = _synthetic_wsc(9, effect=0.0, n1=24, n2=24, n3=6, n4=40)
    with pytest.raises(EstimandMismatchError):
        stratified_primary_contrast_bootstrap(
            full,
            n_replicates=2,
            seed=1,
            randomized_estimand="RCT_ATE_full",
        )


def test_simple_participant_bootstrap_does_not_claim_to_refit():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    result = participant_bootstrap(4, lambda indexes: float(values[indexes].mean()), n_replicates=10, seed=20261026)
    assert result["refits_nuisance_models"] is False
    assert result["percentile_ci_low"] <= result["percentile_ci_high"]


def test_duplicate_bootstrap_copies_stay_inside_one_fold():
    frame = _synthetic_wsc(17, effect=0.3, n1=40, n2=32, n3=8, n4=80)
    repeated_g1 = frame.loc[frame["cell_origin"].eq("11")].head(6)
    repeated_g4 = frame.loc[frame["cell_origin"].eq("00")].head(6)
    sample = mark_synthetic(pd.concat([frame, repeated_g1, repeated_g1, repeated_g4], ignore_index=True))
    first = qed_concordant_att(sample)
    second = qed_concordant_att(sample)
    subset = subset_for_estimand(sample, "QED_concordant_ATT").reset_index(drop=True)
    folds = np.asarray(first.diagnostics["fold_id"])
    assert subset["row_id"].duplicated().any()
    assert len(folds) == len(subset) == first.n
    assert first.diagnostics["rows_deleted_for_overlap"] == 0
    assert first.diagnostics["n_folds"] == 5
    assert first.diagnostics["seed"] == 20261026
    assert first.diagnostics["fold_assignment"] == "distinct_participant_id"
    assert first.diagnostics["fold_id"] == second.diagnostics["fold_id"]
    paired = subset.assign(fold=folds)
    assert int(paired.groupby("row_id")["fold"].nunique().max()) == 1
    assert set(folds.tolist()) == {0, 1, 2, 3, 4}
    for fold in range(5):
        test_ids = set(paired.loc[paired["fold"].eq(fold), "row_id"].tolist())
        train_ids = set(paired.loc[paired["fold"].ne(fold), "row_id"].tolist())
        assert test_ids
        assert train_ids.isdisjoint(test_ids)
    for summary in first.diagnostics["fold_summaries"]:
        assert summary["status"] == "fit"
        assert summary["n_treated_train"] >= 2 and summary["n_control_train"] >= 2
        assert summary["n_train"] + summary["n_test"] == len(subset)


def test_fewer_than_five_participants_does_not_reduce_the_fold_count():
    outcome = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    treatment = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    covariates = np.ones((12, 1))
    with pytest.raises(EmptyFoldError, match="was not reduced"):
        crossfit_aipw(
            outcome,
            treatment,
            covariates,
            estimand="QED_concordant_ATT",
            outcome="mathPost",
            target_population="fixture",
            n_folds=5,
            seed=20261026,
            weight_type="att",
            participant_ids=[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        )


def test_duplicate_source_ids_are_refused_before_resampling():
    frame = _synthetic_wsc(1, effect=0.0)
    duplicated = mark_synthetic(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))
    with pytest.raises(DataContractError, match="row_id"):
        stratified_primary_contrast_bootstrap(duplicated, n_replicates=2, seed=1)
