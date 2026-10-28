"""Synthetic checks for the secondary full-sample ATE analysis.

Nothing in this file is a WSC empirical result. TEST_FIXTURE_NOT_RESEARCH_EVIDENCE.
"""

import json

import numpy as np
import pandas as pd
import pytest

from p4_causal_validation.data.wsc import COVARIATE_CANDIDATES
from p4_causal_validation.estimators.aipw import EmptyFoldError, FitFailure, _assign_crossfit_folds, ate_influence_scores
from p4_causal_validation.estimators.unadjusted import difference_in_means
from p4_causal_validation.experiments.wsc_secondary_ate import validate_secondary_decision
from p4_causal_validation.inference.bootstrap import stratified_primary_contrast_bootstrap
from p4_causal_validation.inference.secondary_ate_bootstrap import (
    secondary_zero_failure_rule_met,
    stratified_secondary_ate_bootstrap,
)


def test_ate_score_matches_the_hand_calculation():
    scores = ate_influence_scores(
        [1.0, 3.0, 0.0, 4.0],
        [1, 1, 0, 0],
        [2.0, 2.0, 2.0, 2.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5, 0.5],
    )
    assert np.allclose(scores, [-1.0, 3.0, 3.0, -5.0])
    assert scores.mean() == 0.0


def test_rct_ate_uses_observed_arm_means_without_reweighting():
    result = difference_in_means(
        [2.0, 4.0, 1.0, 3.0],
        [1, 1, 0, 0],
        estimand="RCT_ATE_full",
        outcome="mathPost",
        target_population="full",
        role="neyman",
    )
    assert result.estimate == 1.0
    assert result.n_treated == 2 and result.n_control == 2
    assert result.se == pytest.approx(np.sqrt(2.0))


def test_duplicate_row_ids_stay_in_one_fold():
    ids = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    folds = _assign_crossfit_folds(ids, 5, 20261026)
    for identifier in ids:
        assert len(set(folds[np.array(ids) == identifier].tolist())) == 1


def test_secondary_zero_failure_rule_requires_all_2000():
    assert secondary_zero_failure_rule_met(2000, 2000, 0) is True
    assert secondary_zero_failure_rule_met(2000, 1999, 1) is False
    assert secondary_zero_failure_rule_met(2, 2, 0) is False


def _synthetic_frame(seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cells = [(1, 1, 120), (1, 0, 90), (0, 1, 90), (0, 0, 180)]
    rows = []
    row_id = 0
    for preference, assignment, count in cells:
        for _ in range(count):
            row = {
                "row_id": row_id,
                "mathSel": preference,
                "mathGrp": assignment,
                "mathPost": float(assignment) + rng.normal(scale=2.0),
                "vocabPost": 0.0,
            }
            for candidate in COVARIATE_CANDIDATES:
                name = candidate[0] if isinstance(candidate, tuple) else candidate
                row[name] = float(rng.normal())
            rows.append(row)
            row_id += 1
    return pd.DataFrame(rows)


def test_shared_cells_and_g3_placement(monkeypatch):
    frame = _synthetic_frame()
    seen = {"rct": [], "qed": []}

    def fake_rct(data):
        seen["rct"].append(data["row_id"].tolist())
        return difference_in_means(
            data["mathPost"], data["mathGrp"], estimand="RCT_ATE_full", outcome="mathPost", target_population="t", role="t"
        )

    def fake_qed(data):
        seen["qed"].append(data["row_id"].tolist())
        result = difference_in_means(
            data["mathPost"], data["mathGrp"], estimand="QED_concordant_ATE", outcome="mathPost", target_population="t", role="t"
        )
        result.estimator = "crossfit_aipw_ate_linear"
        result.diagnostics = {
            "weight_type": "ate",
            "treated_outcome_model_used": True,
            "numerical_floor": 1e-6,
            "rows_deleted_for_overlap": 0,
            "n_folds": 5,
            "seed": 20261026,
            "fold_assignment": "distinct_participant_id",
            "n_probabilities_moved_by_floor": 0,
            "scientific_trimming": "none",
            "covariates": ["mathPre"],
            "overlap_report": None,
            "ate_arm_weights": None,
            "out_of_fold_propensity": None,
            "fold_summaries": None,
        }
        return result

    monkeypatch.setattr("p4_causal_validation.inference.secondary_ate_bootstrap._rct_estimate", fake_rct)
    monkeypatch.setattr("p4_causal_validation.inference.secondary_ate_bootstrap._qed_estimate", fake_qed)
    result = stratified_secondary_ate_bootstrap(frame, n_replicates=2, seed=11)
    g3 = set(frame.loc[(frame["mathSel"] == 0) & (frame["mathGrp"] == 1), "row_id"])
    n1 = int(((frame["mathSel"] == 1) & (frame["mathGrp"] == 1)).sum())
    n4 = int(((frame["mathSel"] == 0) & (frame["mathGrp"] == 0)).sum())
    for rct_ids, qed_ids in zip(seen["rct"][1:], seen["qed"][1:]):
        assert set(qed_ids).isdisjoint(g3)
        assert g3.intersection(rct_ids)
        assert rct_ids[:n1] == qed_ids[:n1]
        assert qed_ids[n1:] == rct_ids[-n4:]
    assert result["secondary_zero_failure_rule_met"] is False
    assert result["interval_status"] == "diagnostic_only_secondary_zero_failure_rule_not_met"
    assert result["g3_in_randomized_ate"] is True and result["g3_in_observational_ate"] is False


def test_failure_is_retained_and_seed_repeats(monkeypatch):
    frame = _synthetic_frame()
    calls = {"n": 0}

    def fake_rct(data):
        return difference_in_means(
            data["mathPost"], data["mathGrp"], estimand="RCT_ATE_full", outcome="mathPost", target_population="t", role="t"
        )

    def fake_qed(data):
        calls["n"] += 1
        if calls["n"] == 2:
            raise FitFailure("injected separation")
        result = difference_in_means(
            data["mathPost"], data["mathGrp"], estimand="QED_concordant_ATE", outcome="mathPost", target_population="t", role="t"
        )
        result.diagnostics = {
            "weight_type": "ate",
            "treated_outcome_model_used": True,
            "numerical_floor": 1e-6,
            "rows_deleted_for_overlap": 0,
            "n_folds": 5,
            "seed": 20261026,
            "fold_assignment": "distinct_participant_id",
        }
        return result

    monkeypatch.setattr("p4_causal_validation.inference.secondary_ate_bootstrap._rct_estimate", fake_rct)
    monkeypatch.setattr("p4_causal_validation.inference.secondary_ate_bootstrap._qed_estimate", fake_qed)
    result = stratified_secondary_ate_bootstrap(frame, n_replicates=2, seed=3)
    assert result["n_failed"] == 1 and result["n_success"] == 1
    assert result["failures"][0]["error_type"] == "FitFailure"
    assert result["failures_not_redrawn"] is True
    assert result["percentile_ci_low"] is None


def test_primary_bootstrap_still_rejects_the_secondary_pair():
    frame = _synthetic_frame()
    frame.attrs["source"] = "synthetic_fixture"
    with pytest.raises(Exception):
        stratified_primary_contrast_bootstrap(frame, n_replicates=2, seed=1, randomized_estimand="RCT_ATE_full")


def test_real_secondary_fit_refits_and_excludes_g3():
    frame = _synthetic_frame(7)
    result = stratified_secondary_ate_bootstrap(frame, n_replicates=2, seed=4)
    assert result["n_success"] == 2 and result["n_failed"] == 0
    assert result["refits_ate_nuisance_each_replicate"] is True
    assert result["secondary_zero_failure_rule_met"] is False
    assert result["qed_point"]["n"] == 300
    assert result["rct_point"]["n"] == 480
    again = stratified_secondary_ate_bootstrap(frame, n_replicates=2, seed=4)
    assert again["difference_qed_minus_rct"] == result["difference_qed_minus_rct"]
    assert again["successful_replicates"] == result["successful_replicates"]


def _record(protocol, spec, data, **overrides):
    payload = {
        "approver": "Hanqing Yang",
        "protocol_sha256": protocol,
        "specification_sha256": spec,
        "source_sha256": data,
        "scope": "RCT_ATE_full_versus_QED_concordant_ATE",
        "contrast": "QED_concordant_ATE - RCT_ATE_full",
        "rct_estimand": "RCT_ATE_full",
        "qed_estimand": "QED_concordant_ATE",
        "rct_estimator": "unadjusted_difference_in_means",
        "qed_estimator": "crossfit_aipw_ate_linear",
        "reweight_to_one_half": False,
        "n_replicates": 2000,
        "seed": 20261026,
        "failure_rule": "all_2000_requested_replicates_succeed",
        "l3_authorized": False,
        "vocabPost_authorized": False,
        "this_choice_was_not_uniquely_specified_by_the_frozen_protocol": True,
    }
    payload.update(overrides)
    return payload


def test_hash_and_scope_mismatches_are_refused(tmp_path):
    protocol = tmp_path / "protocol.yaml"
    spec = tmp_path / "spec.md"
    data = tmp_path / "data.bin"
    protocol.write_bytes(b"protocol")
    spec.write_bytes(b"spec")
    data.write_bytes(b"data")
    approval = {"approver": "Hanqing Yang", "decision": "approved"}
    from p4_causal_validation.experiments.wsc_secondary_ate import _sha256

    good = _record(_sha256(protocol), _sha256(spec), _sha256(data))
    validate_secondary_decision(protocol, spec, data, good, approval, enforce_known_digests=False)
    with pytest.raises(SystemExit, match="hash"):
        validate_secondary_decision(protocol, spec, data, _record("0" * 64, _sha256(spec), _sha256(data)), approval, enforce_known_digests=False)
    with pytest.raises(SystemExit, match="hash"):
        validate_secondary_decision(protocol, spec, data, _record(_sha256(protocol), "0" * 64, _sha256(data)), approval, enforce_known_digests=False)
    with pytest.raises(SystemExit, match="hash"):
        validate_secondary_decision(protocol, spec, data, _record(_sha256(protocol), _sha256(spec), "0" * 64), approval, enforce_known_digests=False)
    with pytest.raises(SystemExit, match="scope"):
        validate_secondary_decision(
            protocol, spec, data, _record(_sha256(protocol), _sha256(spec), _sha256(data), scope="L3"), approval, enforce_known_digests=False
        )
    with pytest.raises(SystemExit, match="protocol or source hash"):
        validate_secondary_decision(protocol, spec, data, good, approval, enforce_known_digests=True)


def test_crossfit_keeps_duplicate_participants_out_of_their_training_fold():
    ids = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    with pytest.raises(EmptyFoldError):
        _assign_crossfit_folds(ids[:4], 5, 1)
