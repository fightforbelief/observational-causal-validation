"""Synthetic checks for outcome-blind readiness diagnostics."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p4_causal_validation.data.wsc import COVARIATE_CANDIDATES
from p4_causal_validation.diagnostics.overlap import effective_sample_size
from p4_causal_validation.estimators.aipw import _assign_crossfit_folds
from p4_causal_validation.experiments.primary_specification import PRIMARY_FOLDS, PRIMARY_SEED
from p4_causal_validation.experiments.wsc_outcome_blind_diagnostics import (
    att_control_weight_diagnostics,
    audit_integrity,
    fit_outcome_blind_propensity,
    parse_protocol,
    protocol_matches_code,
    run_outcome_blind_diagnostics,
    window_counts,
)

ROOT = Path(__file__).resolve().parents[1]
MODULE = (
    ROOT
    / "src"
    / "p4_causal_validation"
    / "experiments"
    / "wsc_outcome_blind_diagnostics.py"
)


def covariate_names():
    return [candidate if isinstance(candidate, str) else candidate[0] for candidate in COVARIATE_CANDIDATES]


def add_covariates(frame, rng, column_names=None):
    for name in column_names or covariate_names():
        frame[name] = rng.normal(size=len(frame))
    return frame


def design_frame(rng):
    counts = ((1, 1, 288), (1, 0, 319), (0, 1, 776), (0, 0, 817))
    selection = np.concatenate([np.full(count, sel) for sel, _grp, count in counts])
    assignment = np.concatenate([np.full(count, grp) for _sel, grp, count in counts])
    frame = pd.DataFrame({"mathSel": selection, "mathGrp": assignment})
    return add_covariates(frame, rng)


def protocol_text():
    return (ROOT / "configs" / "protocol_candidate_v2.yaml").read_text(encoding="utf-8")


def called_names(source: str) -> set[str]:
    tree = ast.parse(source)
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            names.add(func.id)
        elif isinstance(func, ast.Attribute):
            names.add(func.attr)
    return names


def test_protocol_constants_match_candidate_and_code():
    spec = parse_protocol(protocol_text())
    assert spec["n_folds"] == PRIMARY_FOLDS == 5
    assert spec["seed"] == PRIMARY_SEED == 20261026
    assert protocol_matches_code(spec) == []


def test_module_does_not_estimate_effects_or_stratify_folds():
    source = MODULE.read_text(encoding="utf-8")
    forbidden = {
        "qed_concordant_att",
        "crossfit_aipw",
        "stratified_primary_contrast_bootstrap",
        "difference_in_means",
        "prepare_wsc",
        "design_matrices",
        "randomized_math_preference_benchmark",
        "StratifiedKFold",
    }
    assert called_names(source).isdisjoint(forbidden)
    assert "import sklearn" not in source
    assert "mathPost" not in source.split("POST_TREATMENT", 1)[0]


def test_integrity_reports_actual_expected_and_difference():
    frame = pd.DataFrame(
        {
            "row_id": [0, 1, 2, 3],
            "mathSel": [1, 1, 0, 0],
            "mathGrp": [1, 0, 1, 0],
        }
    )
    result = audit_integrity(frame)
    by_name = {item["check"]: item for item in result["checks"]}
    assert by_name["n_rows"] == {"check": "n_rows", "actual": 4, "expected": 2200, "difference": -2196}
    assert by_name["G1"]["actual"] == 1
    assert by_name["G1"]["difference"] == 1 - 288
    assert by_name["G4"]["actual"] == 1
    assert result["matches_design"] is False


def test_design_mismatch_does_not_fit_or_drop_rows(monkeypatch):
    frame = pd.DataFrame({"mathSel": [1, 1, 0, 0], "mathGrp": [1, 0, 1, 0], "mathPost": [9.0, 8.0, 7.0, 6.0]})

    def boom(*_args, **_kwargs):
        raise AssertionError("propensity fit was called")

    monkeypatch.setattr(
        "p4_causal_validation.experiments.wsc_outcome_blind_diagnostics._logit_predict",
        boom,
    )
    result = run_outcome_blind_diagnostics(frame, protocol_text())
    assert result["status"] == "DATA_READINESS_BLOCKED"
    assert result["propensity"] is None
    assert result["weights"] is None
    assert result["causal_effect_estimated"] is False
    assert result["rows_deleted"] == 0
    assert len(frame) == 4
    assert result["outcome_columns_dropped_without_summary"] == ["mathPost"]


def test_missing_covariate_blocks_without_deleting_or_fitting(monkeypatch):
    rng = np.random.default_rng(20261026)
    frame = design_frame(rng)
    frame.loc[0, "logAge"] = np.nan
    frame["mathPost"] = rng.normal(size=len(frame))

    def boom(*_args, **_kwargs):
        raise AssertionError("propensity fit was called")

    monkeypatch.setattr(
        "p4_causal_validation.experiments.wsc_outcome_blind_diagnostics._logit_predict",
        boom,
    )
    result = run_outcome_blind_diagnostics(frame, protocol_text())
    assert result["integrity"]["matches_design"] is True
    assert result["status"] == "DATA_READINESS_BLOCKED"
    assert result["propensity"] is None
    assert any("logAge" in reason and "G1" in reason for reason in result["covariates"]["blocking_reasons"])
    assert len(frame) == 2200
    assert frame["logAge"].isna().sum() == 1


def test_control_weights_are_odds_and_are_not_floored():
    propensity = np.array([0.4, 0.5, 1 / 3, 0.2, 0.999])
    treatment = np.array([1, 0, 0, 0, 0])
    result = att_control_weight_diagnostics(propensity, treatment)
    expected = np.array([1.0, 0.5, 0.25, 0.999 / 0.001])
    assert result["weights_truncated"] is False
    assert result["numerical_floor_applied"] is False
    assert result["n_treated_weight_one"] == 1
    assert result["ess"] == pytest.approx(effective_sample_size(expected))
    assert result["ess"] == pytest.approx(np.sum(expected) ** 2 / np.sum(expected**2))
    assert result["control_weight_distribution"]["max"] == pytest.approx(999.0)
    assert result["ess_over_n_control"] == pytest.approx(result["ess"] / 4)


def test_nonfinite_control_weight_leaves_ess_undefined():
    result = att_control_weight_diagnostics(np.array([0.4, 1.0]), np.array([1, 0]))
    assert result["n_nonfinite_or_negative_control_weights"] == 1
    assert result["ess"] is None
    assert result["rows_deleted"] == 0


def test_window_counts_do_not_delete_rows():
    counts = window_counts(np.array([0.05, 0.5, 0.95, 0.2]))
    assert counts["n_below_0.1"] == 1
    assert counts["n_above_0.9"] == 1
    assert counts["n_outside_0.1_0.9"] == 2
    assert counts["share_outside_0.1_0.9"] == pytest.approx(0.5)
    assert counts["rows_deleted_for_overlap"] == 0


def test_fold_assignment_matches_existing_helper_and_ignores_outcome():
    rng = np.random.default_rng(7)
    n_treated, n_control = 100, 150
    treatment = np.array([1] * n_treated + [0] * n_control)
    features = rng.normal(size=(treatment.size, 3))
    identifiers = list(range(treatment.size))
    first = fit_outcome_blind_propensity(
        features, treatment, identifiers, n_folds=5, seed=20261026
    )
    assert first["model_succeeded"]
    assert first["outcome_model_fit"] is False
    assert first["stratified_kfold_used"] is False
    assert first["fold_id"] == _assign_crossfit_folds(identifiers, 5, 20261026).tolist()
    assert {row["status"] for row in first["fold_summaries"]} == {"converged"}
    second_features = features.copy()
    second = fit_outcome_blind_propensity(
        second_features, treatment, identifiers, n_folds=5, seed=20261026
    )
    assert np.allclose(first["propensity"], second["propensity"])


def test_outcome_column_does_not_enter_or_change_propensity():
    rng = np.random.default_rng(11)
    n_treated, n_control = 80, 120
    frame = pd.DataFrame(
        {
            "mathSel": [1] * n_treated + [0] * n_control,
            "mathGrp": [1] * n_treated + [0] * n_control,
            "row_id": np.arange(n_treated + n_control),
        }
    )
    add_covariates(frame, rng)
    frame["mathPost"] = frame["mathGrp"] * 100
    from p4_causal_validation.experiments.wsc_outcome_blind_diagnostics import (
        audit_covariates,
        without_outcomes,
    )

    blind, dropped = without_outcomes(frame)
    assert dropped == ["mathPost"]
    resolved = audit_covariates(blind)["resolved"]
    assert "mathPost" not in resolved
    fitted = fit_outcome_blind_propensity(
        blind[resolved].to_numpy(dtype=float),
        blind["mathGrp"].to_numpy(dtype=int),
        blind["row_id"].tolist(),
        n_folds=5,
        seed=20261026,
    )
    frame["mathPost"] = -9999.0
    refit_blind, _dropped = without_outcomes(frame)
    refitted = fit_outcome_blind_propensity(
        refit_blind[resolved].to_numpy(dtype=float),
        refit_blind["mathGrp"].to_numpy(dtype=int),
        refit_blind["row_id"].tolist(),
        n_folds=5,
        seed=20261026,
    )
    assert fitted["model_succeeded"] and refitted["model_succeeded"]
    assert np.allclose(fitted["propensity"], refitted["propensity"])


def test_separation_is_a_fold_failure_and_does_not_return_weights():
    treatment = np.array([1] * 30 + [0] * 30)
    features = np.column_stack([treatment.astype(float), np.zeros(treatment.size)])
    fitted = fit_outcome_blind_propensity(
        features, treatment, list(range(treatment.size)), n_folds=5, seed=20261026
    )
    assert fitted["model_succeeded"] is False
    assert fitted["propensity"] is None
    assert fitted["failures"]
    assert any("separation" in item.lower() or "fit failed" in item.lower() for item in fitted["failures"])
