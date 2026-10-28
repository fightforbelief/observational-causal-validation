"""Synthetic checks for outcome-blind ATT covariate balance."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from p4_causal_validation.data.wsc import COVARIATE_CANDIDATES
from p4_causal_validation.diagnostics.overlap import effective_sample_size
from p4_causal_validation.experiments.primary_specification import PRIMARY_FOLDS, PRIMARY_SEED
from p4_causal_validation.experiments.wsc_outcome_blind_balance import (
    att_control_weight_diagnostics,
    balance_summary,
    covariate_balance,
    control_weights,
    parse_prior_weight_summary,
    same_reported_number,
    write_balance_plot,
)

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "p4_causal_validation" / "experiments" / "wsc_outcome_blind_balance.py"


def protocol_names():
    return [candidate if isinstance(candidate, str) else candidate[0] for candidate in COVARIATE_CANDIDATES]


def called_names(source: str) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            names.add(func.id)
        elif isinstance(func, ast.Attribute):
            names.add(func.attr)
    return names


def test_smd_matches_hand_calculation_and_keeps_one_denominator():
    g1 = np.array([[0.0], [2.0]])
    g4 = np.array([[1.0], [3.0]])
    weights = np.array([1.0, 3.0])
    sd = (5.0 / 3.0) ** 0.5
    row = covariate_balance(["x"], g1, g4, weights)[0]
    assert row["mean_g1"] == pytest.approx(1.0)
    assert row["mean_g4_unweighted"] == pytest.approx(2.0)
    assert row["mean_g4_weighted"] == pytest.approx(2.5)
    assert row["sd_pooled_unweighted"] == pytest.approx(sd)
    assert row["smd_unweighted"] == pytest.approx((1.0 - 2.0) / sd)
    assert row["smd_weighted"] == pytest.approx((1.0 - 2.5) / sd)
    assert row["abs_smd_change"] == pytest.approx(abs(row["smd_weighted"]) - abs(row["smd_unweighted"]))
    other = covariate_balance(["x"], g1, g4, np.array([1.0, 1.0]))[0]
    assert other["sd_pooled_unweighted"] == pytest.approx(row["sd_pooled_unweighted"])
    assert other["smd_unweighted"] == pytest.approx(row["smd_unweighted"])
    assert other["mean_g1"] == pytest.approx(row["mean_g1"])
    assert other["smd_weighted"] != pytest.approx(row["smd_weighted"])


def test_zero_variance_missing_and_illegal_weights_stay_in_the_output():
    constant = covariate_balance(["flat"], np.array([[1.0], [1.0]]), np.array([[1.0], [1.0]]), np.array([1.0, 1.0]))[0]
    assert constant["status"] == "not_computable"
    assert "zero" in constant["reason"]
    assert constant["covariate"] == "flat"
    missing = covariate_balance(["x"], np.array([[1.0], [np.nan]]), np.array([[0.0], [1.0]]), np.array([1.0, 1.0]))[0]
    assert missing["status"] == "not_computable"
    assert missing["smd_unweighted"] is None
    illegal = covariate_balance(["x"], np.array([[0.0], [2.0]]), np.array([[1.0], [3.0]]), np.array([1.0, -1.0]))[0]
    assert illegal["status"] == "weighted_not_computable"
    assert illegal["smd_unweighted"] is not None
    assert illegal["smd_weighted"] is None
    assert "negative" in illegal["reason"]


def test_every_protocol_covariate_is_reported_and_outcomes_are_not_required():
    names = protocol_names()
    assert len(names) == 25
    g1 = np.zeros((4, 25))
    g4 = np.zeros((6, 25))
    g1[:, 1] = [0, 1, 0, 1]
    g4[:, 1] = [1, 0, 1, 0, 1, 0]
    rows = covariate_balance(names, g1, g4, np.ones(6))
    assert [row["covariate"] for row in rows] == names
    assert rows[0]["status"] == "not_computable"
    assert rows[1]["status"] == "computed"
    summary = balance_summary(rows)
    assert summary["n_covariates"] == 25
    assert summary["n_not_computable"] == 24
    assert summary["thresholds_are_acceptance_criteria"] is False


def test_control_weights_match_existing_att_definition():
    propensity = np.array([0.25, 0.2, 0.5])
    treatment = np.array([1, 0, 0])
    weights = control_weights(propensity, treatment)
    summary = att_control_weight_diagnostics(propensity, treatment)
    assert weights == pytest.approx([0.2 / 0.8, 1.0])
    assert summary["numerical_floor_applied"] is False
    assert effective_sample_size(weights) == pytest.approx(summary["ess"])
    assert summary["n_treated_weight_one"] == 1


def test_direct_fold_helper_is_called_with_protocol_constants(monkeypatch):
    captured = {}

    def fake(features, treatment, participant_ids, *, n_folds, seed):
        captured["n_folds"] = n_folds
        captured["seed"] = seed
        return {"model_succeeded": True, "propensity": np.full(len(list(participant_ids)), 0.3), "failures": []}

    monkeypatch.setattr(
        "p4_causal_validation.experiments.wsc_outcome_blind_balance.fit_outcome_blind_propensity",
        fake,
    )
    from p4_causal_validation.experiments import wsc_outcome_blind_balance as balance

    balance.fit_outcome_blind_propensity(
        np.ones((3, 1)),
        np.array([1, 0, 0]),
        [0, 1, 2],
        n_folds=PRIMARY_FOLDS,
        seed=PRIMARY_SEED,
    )
    assert captured == {"n_folds": PRIMARY_FOLDS, "seed": PRIMARY_SEED}
    source = MODULE.read_text(encoding="utf-8")
    assert "n_folds=PRIMARY_FOLDS" in source
    assert "seed=PRIMARY_SEED" in source
    forbidden = {
        "qed_concordant_att",
        "crossfit_aipw",
        "stratified_primary_contrast_bootstrap",
        "difference_in_means",
        "StratifiedKFold",
        "prepare_wsc",
        "design_matrices",
    }
    assert called_names(source).isdisjoint(forbidden)


def test_prior_report_parser_uses_printed_precision():
    text = "\n".join(
        [
            "- ESS：145.9182",
            "- 权重总和：313.92421",
            "- 权重平方和：675.36753",
            "| G4 | 817 | 358 | 0 | 358 | 0.43818849 | 0 |",
        ]
    )
    parsed = parse_prior_weight_summary(text)
    assert parsed["ess"] == "145.9182"
    assert parsed["g4_below_0.1"] == 358
    assert same_reported_number(145.9182, parsed["ess"])
    assert same_reported_number(313.92421, parsed["sum_control_weights"])


def test_plot_writes_png(tmp_path):
    rows = [
        {"covariate": "a", "smd_unweighted": 0.4, "smd_weighted": 0.05, "status": "computed"},
        {"covariate": "b", "smd_unweighted": None, "smd_weighted": None, "status": "not_computable"},
    ]
    path = tmp_path / "balance.png"
    write_balance_plot(rows, path)
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
