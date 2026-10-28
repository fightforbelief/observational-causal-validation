"""Synthetic checks for the prepared primary specification.

Nothing in this file is a WSC empirical result. TEST_FIXTURE_NOT_RESEARCH_EVIDENCE.
"""

import numpy as np
import pandas as pd
import pytest

from p4_causal_validation.data.wsc import COVARIATE_CANDIDATES
from p4_causal_validation.estimators.aipw import (
    FitFailure,
    apply_numerical_floor,
    att_influence_scores,
    crossfit_aipw,
)
from p4_causal_validation.estimators.common import DataContractError, EstimandMismatchError
from p4_causal_validation.estimators.unadjusted import difference_in_means
from p4_causal_validation.experiments.primary_specification import (
    PRIMARY_BOOTSTRAP_REPLICATES,
    ConfirmatoryNotAuthorized,
    assert_primary_outcome,
    mark_synthetic,
    qed_concordant_att,
    randomized_math_preference_benchmark,
    reject_blocked_covariates,
    secondary_subset,
)
from p4_causal_validation.experiments.wsc_confirmatory import main as confirmatory_main
from p4_causal_validation.inference.bootstrap import participant_bootstrap
from p4_causal_validation.inference.comparisons import contrast_estimates


def test_neyman_math_preference_matches_hand_calculation():
    frame = mark_synthetic(
        pd.DataFrame(
            {
                "mathSel": [1, 1, 1, 1, 0],
                "mathGrp": [1, 1, 0, 0, 1],
                "mathPost": [5.0, 1.0, 2.0, 0.0, 9.0],
            }
        )
    )
    result = randomized_math_preference_benchmark(frame)
    treated = np.array([5.0, 1.0])
    control = np.array([2.0, 0.0])
    assert result.estimand == "RCT_effect_prefer_math"
    assert result.outcome == "mathPost"
    assert result.n_treated == 2 and result.n_control == 2
    assert result.estimate == pytest.approx(treated.mean() - control.mean())
    assert result.se == pytest.approx(np.sqrt(treated.var(ddof=1) / 2 + control.var(ddof=1) / 2))
    assert np.isfinite(result.ci_low) and result.ci_low <= result.estimate <= result.ci_high


def test_att_scores_match_hand_calculation():
    scores = att_influence_scores(
        [4.0, 2.0, 0.0, 0.0],
        [1, 1, 0, 0],
        [1.0, 1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5, 0.5],
        0.5,
    )
    # Treated: (Y-1)/0.5. Control: -(Y-1)/(0.5). Mean is the difference in means, 3.
    assert scores.tolist() == pytest.approx([6.0, 2.0, 2.0, 2.0])
    assert float(scores.mean()) == pytest.approx(3.0)


def test_crossfit_folds_are_reproducible_and_keep_every_row():
    rng = np.random.default_rng(20261026)
    n = 40
    x = rng.normal(size=(n, 1))
    propensity = 1 / (1 + np.exp(-0.3 * x[:, 0]))
    treatment = rng.binomial(1, propensity)
    outcome = 1.5 * treatment + 0.2 * x[:, 0] + rng.normal(scale=0.1, size=n)
    kwargs = dict(estimand="QED_concordant_ATT", outcome="mathPost", target_population="fixture", n_folds=5, seed=20261026, weight_type="att")
    left = crossfit_aipw(outcome, treatment, x, **kwargs)
    right = crossfit_aipw(outcome, treatment, x, **kwargs)
    assert left.estimate == right.estimate
    assert left.diagnostics["fold_id"] == right.diagnostics["fold_id"]
    assert left.n == n
    assert left.diagnostics["rows_deleted_for_overlap"] == 0
    assert len(left.diagnostics["fold_summaries"]) == 5
    assert {item["status"] for item in left.diagnostics["fold_summaries"]} == {"fit"}
    assert sum(item["n_test"] for item in left.diagnostics["fold_summaries"]) == n
    assert abs(left.estimate - 1.5) < 1.0
    assert "unmeasured confounding" in " ".join(left.warnings)


def test_empty_arm_and_failed_nuisance_are_reported():
    with pytest.raises(Exception, match="each arm"):
        difference_in_means([1, 2, 3], [1, 1, 1], estimand="RCT_effect_prefer_math", outcome="mathPost", target_population="x", role="x")
    y = np.linspace(0.0, 1.0, 16)
    d = np.array([0, 1] * 8)
    x = np.linspace(0.0, 1.0, 16).reshape(-1, 1)

    def broken(_x_train, _d_train, _x_test):
        raise FitFailure("Logistic propensity did not converge.")

    with pytest.raises(FitFailure, match="did not converge"):
        original = __import__("p4_causal_validation.estimators.aipw", fromlist=["_logit_predict"])
        saved = original._logit_predict
        original._logit_predict = broken
        try:
            crossfit_aipw(y, d, x, estimand="ATT_fixture", outcome="y", target_population="fixture", n_folds=2, seed=1, weight_type="att")
        finally:
            original._logit_predict = saved


def test_extreme_propensity_is_counted_and_not_used_to_drop_rows():
    floored, moved = apply_numerical_floor([0.0, 1.0, 0.2])
    assert moved == 2
    assert floored.tolist() == pytest.approx([1e-6, 1 - 1e-6, 0.2])
    rng = np.random.default_rng(3)
    n = 40
    y = rng.normal(size=n)
    d = np.array([0, 1] * (n // 2))
    x = rng.normal(size=(n, 1))
    result = crossfit_aipw(y, d, x, estimand="ATT_fixture", outcome="y", target_population="fixture", n_folds=2, seed=3, weight_type="att")
    assert result.n == n
    assert result.diagnostics["rows_deleted_for_overlap"] == 0
    assert result.diagnostics["scientific_trimming"] == "none"
    assert result.diagnostics["overlap_report"]["reporting_low"] == 0.1


def test_blocked_covariates_and_vocab_outcome_are_rejected():
    with pytest.raises(DataContractError, match="blocked"):
        reject_blocked_covariates(["logAge", "mathSel"])
    with pytest.raises(DataContractError, match="blocked"):
        reject_blocked_covariates(["mathPost", "logBooks"])
    with pytest.raises(DataContractError, match="blocked"):
        reject_blocked_covariates(["age", "books", "BDI"])
    with pytest.raises(DataContractError, match="vocabPost"):
        assert_primary_outcome("vocabPost")


def test_primary_pair_refuses_unmarked_and_mismatched_frames():
    plain = pd.DataFrame({"mathSel": [1, 1, 0, 0], "mathGrp": [1, 0, 1, 0], "mathPost": [1.0, 0.0, 1.0, 0.0]})
    with pytest.raises(ConfirmatoryNotAuthorized):
        randomized_math_preference_benchmark(plain)
    left = difference_in_means([1, 2, 0, 0], [1, 1, 0, 0], estimand="RCT_effect_prefer_math", outcome="mathPost", target_population="G1G2", role="rct")
    right = difference_in_means([3, 4, 0, 1], [1, 1, 0, 0], estimand="QED_concordant_ATE", outcome="mathPost", target_population="G1G4", role="qed")
    with pytest.raises(EstimandMismatchError):
        contrast_estimates(left, right, dependence="overlapping_diagnostic")


def test_bootstrap_reuses_one_participant_draw_and_keeps_the_protocol_count():
    assert PRIMARY_BOOTSTRAP_REPLICATES == 2000
    values = np.array([10.0, 0.0, 0.0, 0.0])

    def mean_of_draw(indexes):
        assert indexes.shape == (4,)
        return float(values[indexes].mean())

    first = participant_bootstrap(4, mean_of_draw, n_replicates=30, seed=20261026)
    second = participant_bootstrap(4, mean_of_draw, n_replicates=30, seed=20261026)
    assert first["replicates_mean"] == second["replicates_mean"]
    assert first["percentile_ci_low"] <= first["percentile_ci_high"]
    assert first["not_a_wsc_result"] is True
    protocol_run = participant_bootstrap(4, mean_of_draw, n_replicates=2000, seed=20261026)
    assert protocol_run["n_replicates"] == 2000
    assert np.isfinite(protocol_run["se"])

    def paired(indexes):
        return float(values[indexes].mean() - np.array([1.0, 1.0, 1.0, 1.0])[indexes].mean())

    contrast = participant_bootstrap(4, paired, n_replicates=20, seed=1)
    assert contrast["interval_overlap_is_not_equivalence"] is True


def test_missing_outcome_is_not_imputed():
    with pytest.raises(ValueError, match="Non-finite"):
        difference_in_means([1.0, np.nan, 0.0, 0.0], [1, 1, 0, 0], estimand="RCT_effect_prefer_math", outcome="mathPost", target_population="x", role="x")


def test_secondary_interfaces_do_not_fit_and_skip_atu():
    columns = {"mathSel": [1, 1, 0, 0], "mathGrp": [1, 0, 1, 0], "mathPost": [1.0, 0.0, 2.0, 0.0], "vocabPost": [0.0, 1.0, 1.0, 2.0]}
    for candidate in COVARIATE_CANDIDATES:
        name = candidate[0] if isinstance(candidate, tuple) else candidate
        columns[name] = [0.0, 1.0, 0.0, 1.0]
    frame = mark_synthetic(pd.DataFrame(columns))
    ate = secondary_subset(frame, "RCT_ATE_full")
    assert len(ate) == 4 and ate.attrs["fitted"] is False
    qed = secondary_subset(frame, "QED_concordant_ATE")
    assert qed["mathGrp"].tolist() == [1, 0]
    prefer_vocab = secondary_subset(frame, "RCT_effect_prefer_vocab")
    assert prefer_vocab["mathSel"].eq(0).all()
    with pytest.raises(NotImplementedError, match="ATU"):
        secondary_subset(frame, "QED_concordant_ATU")


def test_confirmatory_entry_still_refuses(monkeypatch, tmp_path):
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text("status: frozen\n", encoding="utf-8")
    missing = tmp_path / "missing_approval.json"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"protocol_path: '{protocol}'\napproval_path: '{missing}'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv", ["wsc_confirmatory", "--config", str(config)])
    with pytest.raises(SystemExit, match="approval file is absent"):
        confirmatory_main()
    monkeypatch.setattr(
        "p4_causal_validation.experiments.wsc_confirmatory.assert_owner_approved",
        lambda *_args, **_kwargs: {"approver": "owner", "decision": "approved"},
    )
    with pytest.raises(SystemExit, match="does not fit mathPost"):
        confirmatory_main()


def test_synthetic_att_records_covariates_without_preference():
    rng = np.random.default_rng(1)
    n1, n2, n4 = 40, 12, 80
    rows = []
    for _ in range(n1):
        rows.append((1, 1))
    for _ in range(n2):
        rows.append((1, 0))
    for _ in range(n4):
        rows.append((0, 0))
    preference = np.array([row[0] for row in rows])
    assignment = np.array([row[1] for row in rows])
    data = {"mathSel": preference, "mathGrp": assignment, "vocabPost": np.zeros(len(rows))}
    for candidate in COVARIATE_CANDIDATES:
        name = candidate[0] if isinstance(candidate, tuple) else candidate
        data[name] = rng.normal(size=len(rows))
    signal = data["mathPre"]
    data["mathPost"] = 0.4 * assignment + 0.1 * signal + rng.normal(scale=0.05, size=len(rows))
    frame = mark_synthetic(pd.DataFrame(data))
    result = qed_concordant_att(frame)
    assert result.estimand == "QED_concordant_ATT"
    assert result.n_treated == n1 and result.n_control == n4
    assert result.n == n1 + n4
    assert "mathSel" not in result.diagnostics["covariates"]
    assert "mathPost" not in result.diagnostics["covariates"]
    assert "age" not in result.diagnostics["covariates"] and "logAge" in result.diagnostics["covariates"]
    assert result.diagnostics["seed"] == 20261026
    assert result.diagnostics["n_folds"] == 5
    assert result.diagnostics["rows_deleted_for_overlap"] == 0
    assert np.isfinite(result.estimate) and result.ci_low <= result.ci_high
