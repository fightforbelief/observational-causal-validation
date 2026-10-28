"""Outcome-blind readiness diagnostics for the pre-specified WSC comparison.

This module checks the real Keller frame and the G1∪G4 propensity overlap
without reading outcome values and without estimating a treatment effect.
It does not approve the protocol and it does not authorize confirmatory
analysis.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from p4_causal_validation.data.estimands import subset_for_estimand
from p4_causal_validation.data.wsc import (
    COVARIATE_CANDIDATES,
    POST_TREATMENT,
    RAW_DUPLICATES_OF_LOGS,
    load_rda_frame,
    resolve_covariate_columns,
)
from p4_causal_validation.diagnostics.overlap import effective_sample_size, overlap_summary
from p4_causal_validation.estimators.aipw import (
    OVERLAP_REPORT_HIGH,
    OVERLAP_REPORT_LOW,
    EmptyFoldError,
    FitFailure,
    _assign_crossfit_folds,
    _logit_predict,
)
from p4_causal_validation.estimators.common import DataContractError
from p4_causal_validation.experiments.primary_specification import (
    PRIMARY_FOLDS,
    PRIMARY_OBSERVATIONAL,
    PRIMARY_RANDOMIZED,
    PRIMARY_SEED,
)

EXPECTED_SHA256 = "b47be7d8f2c0d54af9cc71f3602183e61307a964e701a2cdcf3944bcef85286b"
EXPECTED_BYTES = 33689
EXPECTED_N = 2200
EXPECTED_CELLS = {
    "G1": {"mathSel": 1, "mathGrp": 1, "n": 288},
    "G2": {"mathSel": 1, "mathGrp": 0, "n": 319},
    "G3": {"mathSel": 0, "mathGrp": 1, "n": 776},
    "G4": {"mathSel": 0, "mathGrp": 0, "n": 817},
}
EXPECTED_BRANCH = "p4/preapproval-inference-failure-policy"
EXPECTED_HEAD = "2a17d712d5cdfcaae81f624dd2c061684c869859"
BANNER = "OUTCOME-BLIND / NON-CONFIRMATORY / NO CAUSAL EFFECT ESTIMATED"
STATUS_COMPLETED = "DATA_READINESS_DIAGNOSTICS_COMPLETED"
STATUS_BLOCKED = "DATA_READINESS_BLOCKED"
QUANTILES = (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)
QUANTILE_LABELS = ("min", "p01", "p05", "p50", "p95", "p99", "max")
ALLOWED_UNTRACKED = {
    "src/p4_causal_validation/experiments/wsc_outcome_blind_diagnostics.py",
    "tests/test_wsc_outcome_blind_diagnostics.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_protocol(text: str) -> dict:
    """Read the flat candidate fields this diagnostic is allowed to use."""
    covariates: list[str] = []
    in_covariates = False
    spec = {
        "n_folds": None,
        "seed": None,
        "nuisance_models": None,
        "numerical_propensity_floor": None,
        "scientific_trimming": None,
        "covariates": covariates,
    }
    for raw in text.splitlines():
        line = raw.strip()
        if raw.startswith("covariates:"):
            in_covariates = True
            continue
        if in_covariates:
            if raw.startswith("  - "):
                covariates.append(raw[4:].strip())
                continue
            if raw and not raw.startswith(" "):
                in_covariates = False
        if raw.startswith("n_folds:"):
            spec["n_folds"] = int(line.split(":", 1)[1])
        elif raw.startswith("seed:"):
            spec["seed"] = int(line.split(":", 1)[1])
        elif raw.startswith("nuisance_models:"):
            spec["nuisance_models"] = line.split(":", 1)[1].strip()
        elif raw.startswith("numerical_propensity_floor:"):
            spec["numerical_propensity_floor"] = float(line.split(":", 1)[1])
        elif raw.startswith("scientific_trimming:"):
            spec["scientific_trimming"] = line.split(":", 1)[1].strip()
    return spec


def protocol_matches_code(spec: dict) -> list[str]:
    """Return mismatches between the candidate file and the implementation constants."""
    mismatches = []
    if spec["n_folds"] != PRIMARY_FOLDS:
        mismatches.append(f"n_folds yaml={spec['n_folds']} code={PRIMARY_FOLDS}")
    if spec["seed"] != PRIMARY_SEED:
        mismatches.append(f"seed yaml={spec['seed']} code={PRIMARY_SEED}")
    if spec["numerical_propensity_floor"] != 1e-6:
        mismatches.append(f"numerical floor yaml={spec['numerical_propensity_floor']} expected=1e-6")
    if spec["scientific_trimming"] != "none":
        mismatches.append(f"scientific_trimming yaml={spec['scientific_trimming']} expected=none")
    if len(spec["covariates"]) != len(COVARIATE_CANDIDATES):
        mismatches.append(
            f"covariate count yaml={len(spec['covariates'])} code={len(COVARIATE_CANDIDATES)}"
        )
    else:
        for token, candidate in zip(spec["covariates"], COVARIATE_CANDIDATES):
            expected = candidate if isinstance(candidate, str) else "_or_".join(candidate)
            if token != expected:
                mismatches.append(f"covariate token yaml={token} code={expected}")
    nuisance = spec["nuisance_models"] or ""
    if "logistic" not in nuisance or "main-effects" not in nuisance:
        mismatches.append(f"nuisance_models text is not the main-effects logistic specification: {nuisance}")
    return mismatches


def without_outcomes(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop outcome columns by name. Their values are not summarized."""
    dropped = []
    for column in frame.columns:
        name = str(column)
        if column in POST_TREATMENT or name.lower().endswith("post"):
            dropped.append(name)
    return frame.drop(columns=dropped), dropped


def _binary_codes(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def assign_analysis_row_id(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Use a source row_id when it exists. Otherwise follow prepare_wsc and assign 0..n-1.

    The assigned identifier is the existing adapter rule. It does not add or
    delete participants, and it does not recode the four cells.
    """
    work = frame.copy()
    source_present = "row_id" in work.columns
    if source_present:
        identifiers = work["row_id"]
        missing = int(identifiers.isna().sum())
        unique = int(identifiers.nunique(dropna=True))
    else:
        work.insert(0, "row_id", np.arange(len(work)))
        missing = 0
        unique = int(len(work))
    return work, {
        "source_column_present": source_present,
        "n_missing": missing,
        "n_unique": unique,
        "n_rows": int(len(work)),
        "assignment": "source column" if source_present else "prepare_wsc rule: 0..n-1 in file order",
    }


def audit_integrity(frame: pd.DataFrame) -> dict:
    if "mathSel" not in frame.columns or "mathGrp" not in frame.columns:
        missing = [name for name in ("mathSel", "mathGrp") if name not in frame.columns]
        return {
            "matches_design": False,
            "math_columns_missing": missing,
            "checks": [{"check": "mathSel_and_mathGrp", "actual": missing, "expected": "both present", "difference": "missing"}],
            "cells": {},
        }
    selection = _binary_codes(frame["mathSel"])
    assignment = _binary_codes(frame["mathGrp"])
    valid = selection.isin([0, 1]) & assignment.isin([0, 1])
    n_invalid = int((~valid).sum())
    checks = [
        {"check": "n_rows", "actual": int(len(frame)), "expected": EXPECTED_N, "difference": int(len(frame)) - EXPECTED_N},
        {
            "check": "mathSel_mathGrp_invalid_rows",
            "actual": n_invalid,
            "expected": 0,
            "difference": n_invalid,
        },
    ]
    cells = {}
    for name, expected in EXPECTED_CELLS.items():
        mask = valid & selection.eq(expected["mathSel"]) & assignment.eq(expected["mathGrp"])
        actual = int(mask.sum())
        cells[name] = {
            "mathSel": expected["mathSel"],
            "mathGrp": expected["mathGrp"],
            "actual": actual,
            "expected": expected["n"],
            "difference": actual - expected["n"],
        }
        checks.append(
            {
                "check": name,
                "actual": actual,
                "expected": expected["n"],
                "difference": actual - expected["n"],
            }
        )
    classified = sum(item["actual"] for item in cells.values())
    checks.append(
        {
            "check": "rows_outside_G1_G2_G3_G4",
            "actual": int(len(frame)) - classified,
            "expected": 0,
            "difference": int(len(frame)) - classified,
        }
    )
    derived = {
        "RCT_effect_prefer_math_G1_union_G2": cells["G1"]["actual"] + cells["G2"]["actual"],
        "QED_concordant_ATT_G1_union_G4": cells["G1"]["actual"] + cells["G4"]["actual"],
    }
    checks.append(
        {
            "check": "RCT_G1_union_G2",
            "actual": derived["RCT_effect_prefer_math_G1_union_G2"],
            "expected": 607,
            "difference": derived["RCT_effect_prefer_math_G1_union_G2"] - 607,
        }
    )
    checks.append(
        {
            "check": "QED_G1_union_G4",
            "actual": derived["QED_concordant_ATT_G1_union_G4"],
            "expected": 1105,
            "difference": derived["QED_concordant_ATT_G1_union_G4"] - 1105,
        }
    )
    row_id = {
        "n_missing": int(frame["row_id"].isna().sum()) if "row_id" in frame.columns else None,
        "n_unique": int(frame["row_id"].nunique(dropna=True)) if "row_id" in frame.columns else None,
    }
    checks.append(
        {
            "check": "row_id_missing",
            "actual": row_id["n_missing"],
            "expected": 0,
            "difference": None if row_id["n_missing"] is None else row_id["n_missing"],
        }
    )
    checks.append(
        {
            "check": "row_id_unique_equals_n",
            "actual": row_id["n_unique"],
            "expected": int(len(frame)),
            "difference": None if row_id["n_unique"] is None else row_id["n_unique"] - int(len(frame)),
        }
    )
    matches = all(item["difference"] == 0 for item in checks)
    return {
        "matches_design": matches,
        "math_columns_missing": [],
        "checks": checks,
        "cells": cells,
        "derived_primary_samples": derived,
        "row_id": row_id,
        "n_invalid_codes": n_invalid,
    }


def _cell_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    selection = _binary_codes(frame["mathSel"])
    assignment = _binary_codes(frame["mathGrp"])
    masks = {}
    for name, expected in EXPECTED_CELLS.items():
        masks[name] = selection.eq(expected["mathSel"]) & assignment.eq(expected["mathGrp"])
    return masks


def _column_problems(values: pd.Series) -> tuple[int, int]:
    numeric = pd.to_numeric(values, errors="coerce")
    non_numeric = int((numeric.isna() & values.notna()).sum())
    missing = int(values.isna().sum()) + non_numeric
    finite = numeric.notna()
    non_finite = int((~np.isfinite(numeric.to_numpy(dtype=float)) & finite.to_numpy()).sum()) + non_numeric
    return missing, non_finite


def audit_covariates(frame: pd.DataFrame) -> dict:
    try:
        resolved = resolve_covariate_columns(frame)
        resolution_error = None
    except DataContractError as exc:
        return {
            "resolved": [],
            "resolution_error": str(exc),
            "rows": [],
            "blocks_propensity": True,
            "blocking_reasons": [str(exc)],
            "transform_rows": [],
        }
    masks = _cell_masks(frame) if {"mathSel", "mathGrp"}.issubset(frame.columns) else {}
    rows = []
    blocking = []
    for name in resolved:
        record = {"covariate": name}
        total_missing, total_non_finite = _column_problems(frame[name])
        record["n_missing_all_rows"] = total_missing
        record["n_non_finite_all_rows"] = total_non_finite
        for cell, mask in masks.items():
            missing, non_finite = _column_problems(frame.loc[mask, name])
            record[f"{cell}_missing"] = missing
            record[f"{cell}_non_finite"] = non_finite
            if cell in {"G1", "G4"} and (missing or non_finite):
                blocking.append(f"{name} has missing={missing} non_finite={non_finite} in {cell}")
        rows.append(record)
    transform_rows = []
    for log_name, raw_name in (("logAge", "age"), ("logBooks", "books"), ("logBDI", "BDI")):
        present = log_name in frame.columns
        raw_present = raw_name in frame.columns
        if present:
            missing, non_finite = _column_problems(frame[log_name])
        else:
            missing, non_finite = None, None
            blocking.append(f"{log_name} is absent; the raw column was not transformed to replace it")
        transform_rows.append(
            {
                "log_column": log_name,
                "present": present,
                "raw_column": raw_name,
                "raw_present_but_excluded": raw_present,
                "n_missing": missing,
                "n_non_finite": non_finite,
                "transform_recomputed": False,
            }
        )
    blocked_names = [name for name in resolved if name in set(POST_TREATMENT).union(RAW_DUPLICATES_OF_LOGS).union({"mathGrp", "mathSel", "row_id"})]
    if blocked_names:
        blocking.append(f"resolved covariate list contains blocked columns: {blocked_names}")
    return {
        "resolved": resolved,
        "resolution_error": resolution_error,
        "rows": rows,
        "blocks_propensity": bool(blocking),
        "blocking_reasons": blocking,
        "transform_rows": transform_rows,
        "raw_duplicates_excluded": [name for name in RAW_DUPLICATES_OF_LOGS if name in frame.columns],
    }


def distribution(values) -> dict:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        return {label: None for label in QUANTILE_LABELS}
    quantiles = np.quantile(array, QUANTILES)
    return {label: float(value) for label, value in zip(QUANTILE_LABELS, quantiles)}


def window_counts(values) -> dict:
    array = np.asarray(values, dtype=float)
    below = int(np.sum(array < OVERLAP_REPORT_LOW))
    above = int(np.sum(array > OVERLAP_REPORT_HIGH))
    outside = below + above
    return {
        "n": int(array.size),
        "n_below_0.1": below,
        "n_above_0.9": above,
        "n_outside_0.1_0.9": outside,
        "share_outside_0.1_0.9": float(outside / array.size) if array.size else None,
        "rows_deleted_for_overlap": 0,
    }


def fit_outcome_blind_propensity(features, treatment, participant_ids, *, n_folds: int, seed: int) -> dict:
    """Fit the pre-specified main-effects logit by the existing cross-fit rule.

    The function calls ``_assign_crossfit_folds`` and ``_logit_predict`` only.
    It does not fit an outcome model and does not form an AIPW score.
    """
    matrix = np.asarray(features, dtype=float)
    assignment = np.asarray(treatment, dtype=int)
    identifiers = list(participant_ids)
    failures = []
    if matrix.ndim != 2 or matrix.shape[0] != assignment.shape[0] or len(identifiers) != matrix.shape[0]:
        raise ValueError("Propensity inputs must have the same number of rows. No row was deleted.")
    if not np.isfinite(matrix).all():
        raise ValueError("Propensity covariates contain a non-finite value. No row was deleted.")
    if set(np.unique(assignment).tolist()) != {0, 1}:
        failures.append("Treatment in the diagnostic sample is not coded 0/1.")
    fold_id = None
    summaries = []
    propensity = np.full(matrix.shape[0], np.nan)
    if not failures:
        try:
            fold_id = _assign_crossfit_folds(identifiers, n_folds, seed)
        except (EmptyFoldError, ValueError) as exc:
            failures.append(str(exc))
    if fold_id is not None:
        for fold in range(n_folds):
            test_index = np.flatnonzero(fold_id == fold)
            train_index = np.flatnonzero(fold_id != fold)
            train_ids = {identifiers[int(index)] for index in train_index}
            test_ids = {identifiers[int(index)] for index in test_index}
            if train_ids.intersection(test_ids):
                raise RuntimeError(f"Fold {fold} placed one participant in training and testing.")
            train_assignment = assignment[train_index]
            test_assignment = assignment[test_index]
            summary = {
                "fold": fold,
                "n_train": int(train_index.size),
                "n_test": int(test_index.size),
                "n_treated_train": int(np.sum(train_assignment == 1)),
                "n_control_train": int(np.sum(train_assignment == 0)),
                "n_treated_test": int(np.sum(test_assignment == 1)),
                "n_control_test": int(np.sum(test_assignment == 0)),
                "n_unique_participants_train": len(train_ids),
                "n_unique_participants_test": len(test_ids),
                "participant_ids_disjoint": True,
                "seed": seed,
                "status": "not_fit",
                "failure_reason": None,
            }
            if summary["n_treated_train"] < 2 or summary["n_control_train"] < 2:
                summary["status"] = "failed_empty_training_arm"
                summary["failure_reason"] = "training fold has fewer than two rows in an arm"
                failures.append(f"fold {fold}: {summary['failure_reason']}")
                summaries.append(summary)
                continue
            try:
                predicted = _logit_predict(matrix[train_index], train_assignment, matrix[test_index])
            except (FitFailure, EmptyFoldError) as exc:
                summary["status"] = "failed"
                summary["failure_reason"] = str(exc)
                failures.append(f"fold {fold}: {exc}")
                summaries.append(summary)
                continue
            predicted = np.asarray(predicted, dtype=float)
            if predicted.shape != (test_index.size,) or not np.isfinite(predicted).all():
                summary["status"] = "failed"
                summary["failure_reason"] = "non-finite propensity prediction"
                failures.append(f"fold {fold}: non-finite propensity prediction")
                summaries.append(summary)
                continue
            propensity[test_index] = predicted
            summary["status"] = "converged"
            summaries.append(summary)
    succeeded = not failures and fold_id is not None and np.isfinite(propensity).all()
    return {
        "model_succeeded": bool(succeeded),
        "propensity": propensity if succeeded else None,
        "fold_id": None if fold_id is None else [int(value) for value in fold_id.tolist()],
        "fold_summaries": summaries,
        "failures": failures,
        "n_folds": n_folds,
        "seed": seed,
        "fold_assignment": "distinct_participant_id",
        "stratified_kfold_used": False,
        "rows_deleted_for_overlap": 0,
        "numerical_floor_applied": False,
        "outcome_model_fit": False,
        "n": int(matrix.shape[0]),
        "n_treated": int(np.sum(assignment == 1)),
        "n_control": int(np.sum(assignment == 0)),
    }


def att_control_weight_diagnostics(propensity, treatment) -> dict:
    """ATT weights from out-of-fold propensity. Treated weight is 1. Control weight is e/(1-e).

    The numerical floor used by the AIPW score is not applied here.
    """
    probability = np.asarray(propensity, dtype=float)
    assignment = np.asarray(treatment, dtype=int)
    if probability.shape != assignment.shape:
        raise ValueError("Propensity and treatment lengths differ.")
    control = assignment == 0
    treated = assignment == 1
    control_probability = probability[control]
    with np.errstate(divide="ignore", invalid="ignore"):
        control_weights = control_probability / (1.0 - control_probability)
    finite = np.isfinite(control_weights) & (control_weights >= 0)
    result = {
        "n_treated_weight_one": int(treated.sum()),
        "treated_weight": 1,
        "n_control": int(control.sum()),
        "weights_truncated": False,
        "numerical_floor_applied": False,
        "control_weight_distribution": distribution(control_weights[finite]) if finite.any() else {label: None for label in QUANTILE_LABELS},
        "n_nonfinite_or_negative_control_weights": int((~finite).sum()),
        "control_propensity_distribution": distribution(control_probability),
        "rows_deleted": 0,
        "ess": None,
        "ess_over_n_control": None,
        "sum_control_weights": None,
        "sum_squared_control_weights": None,
        "largest_one_percent_weight_share": None,
        "max_over_median": None,
    }
    if not finite.all() or control_weights.size == 0:
        result["ess_status"] = "undefined_nonfinite_or_empty_control_weights"
        return result
    weights = np.asarray(control_weights, dtype=float)
    ess = effective_sample_size(weights)
    result["ess"] = ess
    result["ess_over_n_control"] = float(ess / weights.size)
    result["sum_control_weights"] = float(np.sum(weights))
    result["sum_squared_control_weights"] = float(np.sum(weights**2))
    result["ess_status"] = "computed_on_raw_control_weights"
    order = np.sort(weights)
    median = float(np.quantile(weights, 0.5))
    result["max_over_median"] = None if median == 0 else float(order[-1] / median)
    top_n = max(1, int(np.ceil(weights.size * 0.01)))
    result["largest_one_percent_count"] = top_n
    result["largest_one_percent_weight_share"] = float(np.sum(order[-top_n:]) / np.sum(weights))
    return result


def overlap_tables(propensity, treatment) -> dict:
    probability = np.asarray(propensity, dtype=float)
    assignment = np.asarray(treatment, dtype=int)
    grouped = {
        "G1_union_G4": probability,
        "G1": probability[assignment == 1],
        "G4": probability[assignment == 0],
    }
    return {
        "distribution": {name: distribution(values) for name, values in grouped.items()},
        "window": {name: window_counts(values) for name, values in grouped.items()},
        "pooled_overlap_summary": overlap_summary(probability, assignment, OVERLAP_REPORT_LOW, OVERLAP_REPORT_HIGH),
        "rows_deleted_for_overlap": 0,
        "reporting_window": [OVERLAP_REPORT_LOW, OVERLAP_REPORT_HIGH],
    }


def run_outcome_blind_diagnostics(frame: pd.DataFrame, protocol_text: str) -> dict:
    """Audit one outcome-blind frame. A design or model failure does not drop rows."""
    blind, dropped_outcomes = without_outcomes(frame)
    spec = parse_protocol(protocol_text)
    mismatches = protocol_matches_code(spec)
    identified, row_id_meta = assign_analysis_row_id(blind)
    integrity = audit_integrity(identified)
    covariates = audit_covariates(identified)
    failures = []
    if mismatches:
        failures.extend(mismatches)
    if not integrity["matches_design"]:
        failures.append("Cell counts, coding, or row_id do not match the design audit.")
    failures.extend(covariates["blocking_reasons"])
    result = {
        "status": STATUS_BLOCKED,
        "banner": BANNER,
        "causal_effect_estimated": False,
        "outcome_columns_dropped_without_summary": dropped_outcomes,
        "row_id": row_id_meta,
        "protocol": {
            "n_folds": spec["n_folds"],
            "seed": spec["seed"],
            "nuisance_models": spec["nuisance_models"],
            "covariates": spec["covariates"],
            "mismatches_with_code": mismatches,
            "primary_randomized": PRIMARY_RANDOMIZED,
            "primary_observational": PRIMARY_OBSERVATIONAL,
        },
        "integrity": integrity,
        "covariates": covariates,
        "propensity": None,
        "overlap": None,
        "weights": None,
        "failures": failures,
        "rows_deleted": 0,
    }
    if failures:
        return result
    subset = subset_for_estimand(identified, PRIMARY_OBSERVATIONAL).reset_index(drop=True)
    if not subset["mathGrp"].eq(subset["mathSel"]).all():
        result["failures"].append("Concordant subset does not have mathGrp equal to mathSel.")
        return result
    features = subset[covariates["resolved"]].to_numpy(dtype=float)
    propensity = fit_outcome_blind_propensity(
        features,
        subset["mathGrp"].to_numpy(dtype=int),
        subset["row_id"].tolist(),
        n_folds=PRIMARY_FOLDS,
        seed=PRIMARY_SEED,
    )
    propensity["covariates"] = list(covariates["resolved"])
    result["propensity"] = propensity
    if not propensity["model_succeeded"]:
        result["failures"].extend(propensity["failures"])
        return result
    result["overlap"] = overlap_tables(propensity["propensity"], subset["mathGrp"].to_numpy(dtype=int))
    result["weights"] = att_control_weight_diagnostics(
        propensity["propensity"], subset["mathGrp"].to_numpy(dtype=int)
    )
    if result["weights"]["ess"] is None:
        result["failures"].append(result["weights"]["ess_status"])
    result["status"] = STATUS_COMPLETED
    return result


def _fmt(value) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def _table(headers: list[str], rows: list[list]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(_fmt(value) for value in row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def render_report(payload: dict) -> str:
    context = payload["context"]
    diagnostic = payload["diagnostic"]
    lines = [
        "# P4 outcome-blind data readiness",
        "",
        BANNER,
        "",
        "本报告只记录真实 WSC 数据的完整性和 G1∪G4 样本外倾向得分重叠。没有读取结果变量的汇总值，没有拟合结果模型，没有估计 RCT 均值差、ATT、`QED ATT − RCT`、标准误、置信区间或 bootstrap。",
        "",
        "本报告不代表协议批准，也不授权确认性分析。`OWNER_APPROVAL.json` 没有创建。`REAL_WSC_EXECUTION_ENABLED` 没有打开。",
        "",
        f"判断：`{diagnostic['status']}`",
        "",
        "## 1. 执行环境",
        "",
        f"- 日期：{context['date']}",
        f"- 机器：{context['machine']}",
        f"- 分支：{context['branch']}",
        f"- HEAD：`{context['head']}`",
        f"- 诊断耗时（秒）：{_fmt(context.get('propensity_seconds'))}",
        f"- 哈希相符的文件：{context.get('matched_paths')}",
        "",
        "## 2. 数据文件",
        "",
        f"- 用于诊断的文件：`{context['data_path']}`",
        f"- 字节数：{_fmt(context['data_bytes'])}（预期 {EXPECTED_BYTES}）",
        f"- 字节差异：{_fmt(None if context['data_bytes'] is None else context['data_bytes'] - EXPECTED_BYTES)}",
        f"- SHA-256：`{context['data_sha256']}`",
        f"- 预期 SHA-256：`{EXPECTED_SHA256}`",
        f"- 哈希相符：{context['hash_matches']}",
        "",
        "搜索到的 `WSCdata.rda`：",
        "",
    ]
    found_rows = [
        [item["path"], item["bytes"], item["sha256"], item["sha256"] == EXPECTED_SHA256]
        for item in context["files_found"]
    ]
    lines.append(_table(["路径", "字节", "SHA-256", "哈希相符"], found_rows) if found_rows else "没有找到名为 WSCdata.rda 的文件。")
    lines.extend(["", "原始文件没有被修改，没有复制进 Git 跟踪目录，也没有提交。", ""])
    if not context["hash_matches"]:
        lines.extend(["数据哈希不符，后续诊断没有执行。", "", f"`{STATUS_BLOCKED}`", ""])
        return "\n".join(lines)
    lines.extend(
        [
            "## 3. 数据完整性和四组人数",
            "",
            f"- 结果列只按列名识别后丢弃，没有做汇总：{diagnostic['outcome_columns_dropped_without_summary']}",
            f"- `row_id` 来源：{diagnostic['row_id']['assignment']}",
            f"- 源文件是否自带 `row_id`：{diagnostic['row_id']['source_column_present']}",
            f"- 分析用 `row_id` 缺失数：{diagnostic['row_id']['n_missing']}",
            f"- 分析用 `row_id` 唯一数：{diagnostic['row_id']['n_unique']}",
            f"- 行数：{diagnostic['row_id']['n_rows']}",
            "",
        ]
    )
    integrity = diagnostic["integrity"]
    if integrity.get("math_columns_missing"):
        lines.append(f"`mathSel` / `mathGrp` 缺失：{integrity['math_columns_missing']}")
    else:
        lines.append(
            _table(
                ["检查", "实际", "预期", "差异"],
                [[item["check"], item["actual"], item["expected"], item["difference"]] for item in integrity["checks"]],
            )
        )
    lines.extend(["", f"与设计审计一致：{integrity.get('matches_design')}", ""])
    lines.extend(["## 4. 协变量缺失和对数列", ""])
    covariates = diagnostic["covariates"]
    lines.append(f"- 协议与代码不一致：{diagnostic['protocol']['mismatches_with_code'] or '无'}")
    lines.append(f"- 解析后的协变量（{len(covariates['resolved'])} 个）：{covariates['resolved']}")
    lines.append(f"- 排除的原始列：{covariates.get('raw_duplicates_excluded')}")
    lines.append(f"- 解析错误：{covariates['resolution_error']}")
    lines.append("")
    if covariates["rows"]:
        cov_rows = []
        for row in covariates["rows"]:
            cov_rows.append(
                [
                    row["covariate"],
                    row.get("G1_missing"),
                    row.get("G1_non_finite"),
                    row.get("G2_missing"),
                    row.get("G2_non_finite"),
                    row.get("G3_missing"),
                    row.get("G3_non_finite"),
                    row.get("G4_missing"),
                    row.get("G4_non_finite"),
                ]
            )
        lines.append(
            _table(
                ["协变量", "G1缺失", "G1非有限", "G2缺失", "G2非有限", "G3缺失", "G3非有限", "G4缺失", "G4非有限"],
                cov_rows,
            )
        )
        lines.append("")
    if covariates["transform_rows"]:
        lines.append(
            _table(
                ["对数列", "存在", "对应原始列", "原始列存在但未使用", "缺失", "非有限", "是否重新计算对数"],
                [
                    [
                        row["log_column"],
                        row["present"],
                        row["raw_column"],
                        row["raw_present_but_excluded"],
                        row["n_missing"],
                        row["n_non_finite"],
                        row["transform_recomputed"],
                    ]
                    for row in covariates["transform_rows"]
                ],
            )
        )
        lines.append("")
    lines.extend(["对数协变量沿用文件中的既有列。本诊断没有重新计算对数，也没有用原始 `age`、`books`、`BDI` 替换它们。", ""])
    lines.extend(["## 5. 五折倾向评分", ""])
    propensity = diagnostic["propensity"]
    if propensity is None:
        lines.append("未计算。前置核验未通过，没有拟合倾向评分，也没有更换模型、折数或种子。")
        lines.append("")
    else:
        lines.extend(
            [
                f"- 样本：`{PRIMARY_OBSERVATIONAL}`，即 G1∪G4。",
                f"- 折数：{propensity['n_folds']}；种子：{propensity['seed']}；分折：`{propensity['fold_assignment']}`。",
                f"- 使用 `StratifiedKFold`：{propensity['stratified_kfold_used']}。",
                "- 调用的是现有 `_assign_crossfit_folds` 和 `_logit_predict`。没有调用 `crossfit_aipw`，没有拟合结果模型。",
                f"- 模型成功：{propensity['model_succeeded']}",
                f"- 因重叠删除的行：{propensity['rows_deleted_for_overlap']}",
                f"- 诊断权重使用数值下限：{propensity['numerical_floor_applied']}",
                "",
            ]
        )
        if propensity["fold_summaries"]:
            lines.append(
                _table(
                    ["折", "训练", "测试", "训练处理", "训练对照", "测试处理", "测试对照", "状态", "失败原因"],
                    [
                        [
                            row["fold"],
                            row["n_train"],
                            row["n_test"],
                            row["n_treated_train"],
                            row["n_control_train"],
                            row["n_treated_test"],
                            row["n_control_test"],
                            row["status"],
                            row["failure_reason"],
                        ]
                        for row in propensity["fold_summaries"]
                    ],
                )
            )
            lines.append("")
    lines.extend(["## 6. G1/G4 重叠", ""])
    overlap = diagnostic["overlap"]
    if overlap is None:
        lines.append("未计算。没有完整的五折样本外倾向得分。`[0.1, 0.9]` 没有被用来删行或改卡尺。")
        lines.append("")
    else:
        lines.append("`[0.1, 0.9]` 只用于下面的计数，不是删除规则。")
        lines.append("")
        dist_rows = []
        for name, values in overlap["distribution"].items():
            dist_rows.append([name, *[values[label] for label in QUANTILE_LABELS]])
        lines.append(_table(["组别", *QUANTILE_LABELS], dist_rows))
        lines.append("")
        window_rows = []
        for name, values in overlap["window"].items():
            window_rows.append(
                [
                    name,
                    values["n"],
                    values["n_below_0.1"],
                    values["n_above_0.9"],
                    values["n_outside_0.1_0.9"],
                    values["share_outside_0.1_0.9"],
                    values["rows_deleted_for_overlap"],
                ]
            )
        lines.append(_table(["组别", "人数", "<0.1", ">0.9", "窗口外", "窗口外比例", "删行"], window_rows))
        lines.append("")
    lines.extend(["## 7. ATT 对照组权重和 ESS", ""])
    weights = diagnostic["weights"]
    if weights is None:
        lines.append("未计算。ESS 依赖于完整的样本外倾向得分，本次没有在失败的模型上继续计算。")
        lines.append("")
    else:
        distribution_values = weights["control_weight_distribution"]
        lines.extend(
            [
                f"- G1 人数（权重固定为 1）：{weights['n_treated_weight_one']}",
                f"- G4 人数：{weights['n_control']}",
                "- G4 权重：`e/(1-e)`。没有截断，没有套用 1e-6 数值下限。",
                f"- 非有限或负的 G4 权重数：{weights['n_nonfinite_or_negative_control_weights']}",
                f"- ESS：{_fmt(weights['ess'])}",
                f"- ESS / G4 人数：{_fmt(weights['ess_over_n_control'])}",
                f"- 权重总和：{_fmt(weights['sum_control_weights'])}",
                f"- 权重平方和：{_fmt(weights['sum_squared_control_weights'])}",
                f"- 最大权重 / 中位权重：{_fmt(weights['max_over_median'])}",
                f"- 最大的 1% 对照组单位所占权重份额：{_fmt(weights['largest_one_percent_weight_share'])}",
                "- 这里没有设定最低 ESS 门槛。ESS 和重叠都是描述性诊断，不是因果识别、无混杂或确认性区间的证据。",
                "",
                _table(["统计量", *QUANTILE_LABELS], [["G4 权重", *[distribution_values[label] for label in QUANTILE_LABELS]]]),
                "",
            ]
        )
    lines.extend(["## 8. 失败、警告和未完成的检查", ""])
    if diagnostic["failures"]:
        lines.extend(f"- {item}" for item in diagnostic["failures"])
    else:
        lines.append("- 无。五折主效应 logit 均返回有限的样本外概率，权重也全部有限。")
    lines.extend(
        [
            "",
            "普通数值警告如果没有导致不收敛、分离、奇异矩阵或非有限预测，现有 `_logit_predict` 不会把它记为拟合失败。本诊断没有在成功折之外另行收集被该函数消费掉的警告。",
            "",
            "## 9. 判断",
            "",
            BANNER,
            "",
            f"`{diagnostic['status']}`",
            "",
            "重叠和 ESS 描述的是预先指定的倾向评分模型在 G1∪G4 上的外推权重，不是处理效应，也不是协议批准。确认性估计、真实 bootstrap 和 L3 都没有运行。",
            "",
        ]
    )
    code_rows = [[name, context["code_hashes"][name]] for name in context["code_hashes"]]
    lines.extend(["## 10. 被复用但未修改的代码哈希", "", _table(["文件", "SHA-256"], code_rows), ""])
    return "\n".join(lines)


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def git_state(root: Path) -> dict:
    def read(args: list[str]) -> str:
        return subprocess.check_output(args, cwd=root, text=True).strip()

    branch = read(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    head = read(["git", "rev-parse", "HEAD"])
    porcelain = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    unexpected = []
    for line in porcelain.splitlines():
        path = line[3:].replace("\\", "/")
        if line.startswith("?? ") and path in ALLOWED_UNTRACKED:
            continue
        unexpected.append(line)
    return {
        "branch": branch,
        "head": head,
        "unexpected": unexpected,
        "matches_expected": branch == EXPECTED_BRANCH and head == EXPECTED_HEAD and not unexpected,
    }


def find_wsc_files(roots: list[Path]) -> list[dict]:
    found = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("WSCdata.rda"):
            if ".venv" in path.parts or "__pycache__" in path.parts:
                continue
            found.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return sorted(found, key=lambda item: item["path"].lower())


def load_outcome_blind_wsc(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load the RDA and drop outcome columns before any summary or model.

    The parser has to materialize the file. Outcome values are then discarded
    and are not counted, modeled, plotted, or written to the report.
    """
    raw = load_rda_frame(path)
    blind, dropped = without_outcomes(raw)
    del raw
    return blind, dropped


def main() -> int:
    root = project_root()
    state = git_state(root)
    if not state["matches_expected"]:
        print(f"STOP branch={state['branch']} head={state['head']} unexpected={state['unexpected']}")
        return 2
    roots = [root, Path(r"C:\Users\hanqi\Desktop\projects\_p4_external_sources")]
    found = find_wsc_files(roots)
    matches = [item for item in found if item["sha256"] == EXPECTED_SHA256 and item["bytes"] == EXPECTED_BYTES]
    code_paths = {
        "configs/protocol_candidate_v2.yaml": root / "configs" / "protocol_candidate_v2.yaml",
        "src/p4_causal_validation/data/wsc.py": root / "src" / "p4_causal_validation" / "data" / "wsc.py",
        "src/p4_causal_validation/estimators/aipw.py": root / "src" / "p4_causal_validation" / "estimators" / "aipw.py",
        "src/p4_causal_validation/experiments/primary_specification.py": root
        / "src"
        / "p4_causal_validation"
        / "experiments"
        / "primary_specification.py",
        "src/p4_causal_validation/experiments/wsc_outcome_blind_diagnostics.py": Path(__file__).resolve(),
    }
    protocol_text = code_paths["configs/protocol_candidate_v2.yaml"].read_text(encoding="utf-8")
    context = {
        "date": date.today().isoformat(),
        "machine": platform.node(),
        "branch": state["branch"],
        "head": state["head"],
        "files_found": found,
        "hash_matches": bool(matches),
        "matched_paths": [item["path"] for item in matches],
        "data_path": matches[0]["path"] if matches else None,
        "data_bytes": matches[0]["bytes"] if matches else None,
        "data_sha256": matches[0]["sha256"] if matches else None,
        "propensity_seconds": None,
        "code_hashes": {name: sha256_file(path) for name, path in code_paths.items()},
    }
    if not matches:
        diagnostic = {
            "status": STATUS_BLOCKED,
            "banner": BANNER,
            "causal_effect_estimated": False,
            "outcome_columns_dropped_without_summary": [],
            "row_id": {"assignment": "not reached", "source_column_present": None, "n_missing": None, "n_unique": None, "n_rows": None},
            "protocol": {"mismatches_with_code": []},
            "integrity": {"matches_design": False, "math_columns_missing": [], "checks": []},
            "covariates": {"resolved": [], "resolution_error": None, "rows": [], "transform_rows": [], "raw_duplicates_excluded": []},
            "propensity": None,
            "overlap": None,
            "weights": None,
            "failures": ["No WSCdata.rda matched the expected size and SHA-256."],
        }
    else:
        import time

        frame, dropped_outcomes = load_outcome_blind_wsc(Path(matches[0]["path"]))
        started = time.perf_counter()
        diagnostic = run_outcome_blind_diagnostics(frame, protocol_text)
        context["propensity_seconds"] = time.perf_counter() - started
        diagnostic["outcome_columns_dropped_without_summary"] = list(
            dict.fromkeys([*dropped_outcomes, *diagnostic["outcome_columns_dropped_without_summary"]])
        )
    report = render_report({"context": context, "diagnostic": diagnostic})
    destination = root / "reports" / "P4_OUTCOME_BLIND_DATA_READINESS.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8", newline="\n")
    print(f"STATUS={diagnostic['status']}")
    print(f"REPORT={destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
