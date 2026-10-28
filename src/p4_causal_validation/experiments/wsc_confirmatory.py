"""Confirmatory WSC analysis.

A matching owner-approval file is required. It is not sufficient. Real fitting
also requires the separate owner decision record. REAL_WSC_EXECUTION_ENABLED
stays false and cannot open this gate.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import yaml

from p4_causal_validation.experiments.approval import (
    assert_owner_approved,
    assert_real_execution_authorized,
    protocol_sha256,
)

REAL_WSC_EXECUTION_ENABLED = False

EXPECTED_BRANCH = "p4/preapproval-inference-failure-policy"
EXPECTED_HEAD = "2a17d712d5cdfcaae81f624dd2c061684c869859"
EXPECTED_PROTOCOL_SHA256 = "583ff75e5e848e4c1831e6dbc6084041afec44c5a18a4806d31db6f3d5564e59"
EXPECTED_DATA_SHA256 = "b47be7d8f2c0d54af9cc71f3602183e61307a964e701a2cdcf3944bcef85286b"
ANALYSIS_DATA_PATH = Path(
    r"C:\Users\hanqi\Desktop\projects\_p4_external_sources\cran\WSCdata_0.1.2_extract\WSCdata\data\WSCdata.rda"
)
DUPLICATE_DATA_PATH = Path(r"C:\Users\hanqi\Desktop\projects\_p4_external_sources\wscdata\data\WSCdata.rda")
EXPECTED_N = 2200
EXPECTED_CELLS = {"G1": 288, "G2": 319, "G3": 776, "G4": 817}
APPROVER = "Hanqing Yang"

HASHED_IMPLEMENTATION_FILES = (
    "src/p4_causal_validation/inference/bootstrap.py",
    "src/p4_causal_validation/estimators/aipw.py",
    "src/p4_causal_validation/experiments/primary_specification.py",
    "src/p4_causal_validation/experiments/approval.py",
)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_hashes(root: Path) -> dict[str, str]:
    return {relative: _sha256(root / relative) for relative in HASHED_IMPLEMENTATION_FILES}


def assert_implementation_hashes(root: Path, record: dict) -> None:
    if record.get("implementation_sha256") != implementation_hashes(root):
        raise SystemExit(
            "Refusing confirmatory WSC analysis: implementation hash does not match the owner decision record."
        )


def _git(root: Path, args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"git {' '.join(args)} failed")
    return completed.stdout.strip()


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_json_ready(payload), ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fmt(value) -> str:
    if isinstance(value, float):
        return format(value, ".12g")
    if value is None:
        return "null"
    return str(value)


def _cell_counts(frame) -> dict[str, int]:
    preference = frame["mathSel"].to_numpy(dtype=int)
    assignment = frame["mathGrp"].to_numpy(dtype=int)
    return {
        "G1": int(((preference == 1) & (assignment == 1)).sum()),
        "G2": int(((preference == 1) & (assignment == 0)).sum()),
        "G3": int(((preference == 0) & (assignment == 1)).sum()),
        "G4": int(((preference == 0) & (assignment == 0)).sum()),
    }


def _balance_context(root: Path) -> dict:
    path = root / "reports" / "P4_OUTCOME_BLIND_BALANCE.csv"
    if not path.exists():
        return {
            "available": False,
            "source": "reports/P4_OUTCOME_BLIND_BALANCE.csv",
            "note": "The earlier outcome-blind balance file was not found. Residual measured-covariate imbalance remains a limitation.",
        }
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    computed = []
    for row in rows:
        if row.get("status") != "computed":
            continue
        computed.append(
            {
                "covariate": row["covariate"],
                "smd_unweighted": float(row["smd_unweighted"]),
                "smd_weighted": float(row["smd_weighted"]),
                "abs_smd_unweighted": float(row["abs_smd_unweighted"]),
                "abs_smd_weighted": float(row["abs_smd_weighted"]),
            }
        )
    above = [row for row in computed if row["abs_smd_weighted"] > 0.1]
    return {
        "available": True,
        "source": "reports/P4_OUTCOME_BLIND_BALANCE.csv",
        "n_covariates": len(computed),
        "max_abs_smd_unweighted": max(computed, key=lambda row: row["abs_smd_unweighted"]),
        "max_abs_smd_weighted": max(computed, key=lambda row: row["abs_smd_weighted"]),
        "n_abs_weighted_above_0_1": len(above),
        "abs_weighted_above_0_1": above,
        "n_abs_unweighted_above_0_1": sum(row["abs_smd_unweighted"] > 0.1 for row in computed),
        "n_abs_unweighted_above_0_2": sum(row["abs_smd_unweighted"] > 0.2 for row in computed),
        "n_abs_weighted_above_0_2": sum(row["abs_smd_weighted"] > 0.2 for row in computed),
        "note": (
            "These standardized differences were computed in the earlier outcome-blind diagnostic "
            "and were not refit for this report. The values 0.1 and 0.2 are descriptive references, "
            "not protocol acceptance criteria. Balance does not establish unconfoundedness."
        ),
    }


def _pytest_text(root: Path) -> str:
    path = root / "reports" / "P4_CONFIRMATORY_PYTEST.txt"
    if not path.exists():
        return "Pytest output file was not present when this report was written."
    return path.read_text(encoding="utf-8", errors="replace")


def _failure_lines(failures: list[dict]) -> list[str]:
    if not failures:
        return ["失败账本为空。没有复制被替换，也没有复制被静默删除。"]
    lines = ["失败账本："]
    for item in failures:
        lines.append(
            f"- replicate {item.get('replicate')}: {item.get('error_type')}: {item.get('message')}"
        )
    return lines


def _render_success(payload: dict, pytest_text: str) -> str:
    result = payload["result"]
    qed = result["qed_point"]
    rct = result["rct_point"]
    overlap = qed.get("overlap_report") or {}
    weights = qed.get("att_control_weights") or {}
    balance = payload["balance_context"]
    confirmatory = bool(result["confirmatory_interval_authorized"])
    lines = [
        "# P4 确认性主分析",
        "",
        "状态：`CONFIRMATORY_PRIMARY_COMPLETED`。" if confirmatory else "状态：`CONFIRMATORY_PRIMARY_COMPLETED_INTERVAL_DIAGNOSTIC_ONLY`。",
        "",
        "## 批准",
        "",
        f"- 批准人：{payload['approver']}",
        "- 协议决定：approved",
        "- 联合参与人 bootstrap：approved",
        "- 零失败规则：approved",
        "- 本次主分析执行：authorized",
        f"- `REAL_WSC_EXECUTION_ENABLED`：{payload['real_wsc_execution_enabled']}",
        "- 执行开关保持为 false。打开分析的是单独的 owner decision record，不是这个布尔值。",
        "- 冻结协议里仍有一句把执行开关写成需要为 true 的停止规则。本轮没有为了这句话修改协议。",
        "",
        "## 协议与校验",
        "",
        f"- 分支：`{payload['git']['branch']}`",
        f"- HEAD：`{payload['git']['head']}`",
        f"- 冻结协议 SHA-256：`{payload['protocol_sha256']}`",
        f"- 批准文件校验：{payload['approval_validation']}",
        f"- 决策记录校验：{payload['decision_record_validation']}",
        f"- 实现文件哈希校验：{payload['implementation_hash_validation']}",
        "",
        "## 数据完整性",
        "",
        f"- 分析源：`{payload['integrity']['analysis_path']}`",
        f"- 分析源 SHA-256：`{payload['integrity']['analysis_sha256']}`",
        f"- 副本 SHA-256：`{payload['integrity']['duplicate_sha256']}`",
        f"- 行数：{payload['integrity']['n']}",
        f"- 单元格：{payload['integrity']['cells']}",
        f"- `row_id`：源文件没有 `row_id`；`prepare_wsc` 按文件顺序赋予 0 到 {payload['integrity']['n'] - 1}，唯一且无缺失。",
        f"- 协变量数量：{payload['integrity']['n_covariates']}",
        f"- `mathSel` 进入 QED 协变量：{payload['integrity']['mathSel_in_covariates']}",
        "",
        "## 主估计",
        "",
        "RCT 估计是 `mathSel = 1` 内 G1 对 G2 的 Neyman 均值差。它有抽样不确定性，不是无噪声真值。",
        "",
        f"- RCT 估计：{_fmt(result['rct_estimate'])}",
        f"- RCT Neyman 标准误：{_fmt(result['rct_neyman_se'])}",
        f"- RCT 正态区间：[{_fmt(rct['normal_interval_low'])}, {_fmt(rct['normal_interval_high'])}]",
        f"- RCT 样本量：{rct['n']}（处理 {rct['n_treated']}，对照 {rct['n_control']}）",
        "",
        "QED 估计是 G1∪G4 上的五折线性交叉拟合 ATT AIPW。分数标准误把处理比例视为固定，不是 bootstrap 区间。",
        "",
        f"- QED ATT：{_fmt(result['qed_estimate'])}",
        f"- QED 分数标准误：{_fmt(result['qed_score_se_treats_share_as_fixed'])}",
        f"- QED 正态区间：[{_fmt(qed['normal_interval_low'])}, {_fmt(qed['normal_interval_high'])}]",
        f"- QED 样本量：{qed['n']}（处理 {qed['n_treated']}，对照 {qed['n_control']}）",
        f"- 科学修剪：{qed.get('scientific_trimming')}",
        f"- 因重叠删除的行：{qed.get('rows_deleted_for_overlap')}",
        f"- 数值下限移动的概率数：{qed.get('n_probabilities_moved_by_floor')}",
        f"- 重叠报告：{json.dumps(_json_ready(overlap), ensure_ascii=False)}",
        f"- ATT 对照权重：{json.dumps(_json_ready(weights), ensure_ascii=False)}",
        f"- 折摘要：{json.dumps(_json_ready(qed.get('fold_summaries')), ensure_ascii=False)}",
        "",
        "## 配对差与联合 bootstrap",
        "",
        "差值是同一次计算里的 QED ATT 减 RCT。它不是自动意义上的混杂偏误。",
        "",
        f"- 原始样本差值：{_fmt(result['difference_qed_minus_rct'])}",
        f"- bootstrap 种子：{result['seed']}",
        f"- 请求复制数：{result['n_replicates']}",
        f"- 成功：{result['n_success']}",
        f"- 失败：{result['n_failed']}",
        f"- bootstrap 标准误：{_fmt(result['bootstrap_se'])}",
        f"- 百分位区间：[{_fmt(result['percentile_ci_low'])}, {_fmt(result['percentile_ci_high'])}]",
        f"- 成功复制中 RCT 与 QED 的协方差：{_fmt(result['bootstrap_covariance'])}",
        f"- 区间状态：`{result['interval_status']}`",
        f"- 零失败规则允许确认性区间：{confirmatory}",
        "",
        "百分位区间使用成功复制差值的 2.5% 和 97.5% 分位，不是 BCa，也不是等效性界限。点估计来自原始样本，不是复制均值。",
        "",
        *_failure_lines(result["failures"]),
        "",
        "## 局限",
        "",
        "重叠窗口 [0.1, 0.9] 只用于报告。有效样本量下降意味着部分对照单位承担了较大的 ATT 权重，精确度受此限制。",
        "",
        "测量协变量的加权平衡来自先前的 outcome-blind 诊断，本轮没有重算，也没有用它更改协变量、折数、种子或修剪。",
        "",
        f"- 平衡来源可用：{balance.get('available')}",
        f"- 平衡摘要：{json.dumps(_json_ready(balance), ensure_ascii=False)}",
        "",
        "加权后仍可能留下测量协变量不平衡。未测量混杂没有被双重稳健或 bootstrap 消除。处理版本一致性也没有被本设计证明：随机化臂与观察臂的处理经历不必相同。",
        "",
        "零失败规则是预先规定的报告规则。它不是 bootstrap 覆盖率或因果识别的证明。",
        "",
        "L1 仍是 partial reproduction。本轮没有重跑 L1，也没有把它写成完整复现。合成数据测试不是 WSC 证据。",
        "",
        "## 测试输出",
        "",
        "```",
        pytest_text.rstrip(),
        "```",
        "",
        "## 范围",
        "",
        "L3 和次要估计量没有运行。本报告到主分析为止。",
        "",
        f"机器可读结果：`reports/P4_CONFIRMATORY_PRIMARY.json`，SHA-256 `{payload['json_sha256']}`。",
        "",
    ]
    return "\n".join(lines)


def _render_failure(payload: dict) -> str:
    return "\n".join(
        [
            "# P4 确认性主分析",
            "",
            f"状态：`{payload['status']}`。主分析没有完成。",
            "",
            f"- 批准人：{payload.get('approver')}",
            f"- 原因：{payload.get('reason')}",
            f"- 协议 SHA-256：{payload.get('protocol_sha256')}",
            "",
            "没有为了让估计成功而修改估计量、样本规则、复制数或种子。",
            "",
            "```",
            json.dumps(_json_ready(payload.get("details", {})), ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )


def _emit(root: Path, payload: dict, success: bool) -> None:
    json_path = root / "reports" / "P4_CONFIRMATORY_PRIMARY.json"
    md_path = root / "reports" / "P4_CONFIRMATORY_PRIMARY.md"
    payload["json_sha256"] = _write_json(json_path, payload)
    if success:
        text = _render_success(payload, _pytest_text(root))
    else:
        text = _render_failure(payload)
    md_path.write_text(text, encoding="utf-8")


def _stop(root: Path, approval: dict, status: str, reason: str, details: dict) -> None:
    payload = {
        "status": status,
        "approver": approval.get("approver"),
        "reason": reason,
        "protocol_sha256": details.get("protocol_sha256"),
        "details": details,
        "completed": False,
    }
    _emit(root, payload, success=False)
    print(f"STATUS {status}", flush=True)
    raise SystemExit(1)


def _check_config(config: dict) -> None:
    expected = {
        "primary_outcome": "mathPost",
        "primary_estimand": "RCT_effect_prefer_math",
        "seed": 20261026,
        "n_folds": 5,
        "bootstrap_replicates": 2000,
        "scientific_trimming": "none",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"Config {key} is {config.get(key)!r}; the prespecified value is {value!r}.")
    if float(config.get("numerical_propensity_floor")) != 1e-6:
        raise RuntimeError("Config changed the numerical propensity floor.")
    if config.get("expected_data_sha256") != EXPECTED_DATA_SHA256:
        raise RuntimeError("Config changed the expected WSC source hash.")
    if Path(config.get("data_path", "")) != ANALYSIS_DATA_PATH:
        raise RuntimeError("Config data path is not the prespecified WSC source.")


def _file_hash(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False, "sha256": None, "bytes": None}
    return {"path": str(path), "exists": True, "sha256": _sha256(path), "bytes": path.stat().st_size}


def _execute(root: Path, config: dict, approval: dict, record: dict) -> None:
    from p4_causal_validation.data.wsc import prepare_wsc, resolve_covariate_columns
    from p4_causal_validation.experiments.primary_specification import (
        AUTHORIZED_REAL_SOURCE,
        reject_blocked_covariates,
    )
    from p4_causal_validation.inference.bootstrap import stratified_primary_contrast_bootstrap

    protocol = root / "configs" / "frozen_protocol.yaml"
    digest = protocol_sha256(protocol)
    branch = _git(root, ["branch", "--show-current"])
    head = _git(root, ["rev-parse", "HEAD"])
    common = {
        "protocol_sha256": digest,
        "git": {"branch": branch, "head": head, "status": _git(root, ["status", "--short"])},
    }
    if digest != EXPECTED_PROTOCOL_SHA256:
        _stop(root, approval, "PROTOCOL_HASH_MISMATCH", "Frozen protocol hash differs from the expected value.", common)
    if branch != EXPECTED_BRANCH or head != EXPECTED_HEAD:
        _stop(root, approval, "GIT_BASELINE_MISMATCH", "Branch or HEAD differs from the expected baseline.", common)
    try:
        _check_config(config)
    except RuntimeError as exc:
        _stop(root, approval, "CONFIG_MISMATCH", str(exc), common)

    analysis = _file_hash(ANALYSIS_DATA_PATH)
    duplicate = _file_hash(DUPLICATE_DATA_PATH)
    if analysis["sha256"] != EXPECTED_DATA_SHA256 or duplicate["sha256"] != EXPECTED_DATA_SHA256:
        _stop(
            root,
            approval,
            "DATA_HASH_MISMATCH",
            "WSC source hash does not match the expected source.",
            {**common, "analysis": analysis, "duplicate": duplicate},
        )

    frame, metadata = prepare_wsc(ANALYSIS_DATA_PATH)
    counts = _cell_counts(frame)
    covariates = resolve_covariate_columns(frame)
    reject_blocked_covariates(covariates)
    row_ids = frame["row_id"].tolist()
    math_post = frame["mathPost"].to_numpy(dtype=float)
    integrity = {
        "analysis_path": str(ANALYSIS_DATA_PATH),
        "analysis_sha256": analysis["sha256"],
        "analysis_bytes": analysis["bytes"],
        "duplicate_sha256": duplicate["sha256"],
        "n": int(len(frame)),
        "cells": counts,
        "n_covariates": len(covariates),
        "covariates": covariates,
        "mathSel_in_covariates": "mathSel" in covariates,
        "metadata": metadata,
        "mathPost_finite": bool(np.isfinite(math_post).all()),
        "primary_fields_missing": int(frame[["mathGrp", "mathSel", "mathPost", *covariates]].isna().any(axis=1).sum()),
        "row_id_unique": not frame["row_id"].duplicated().any() and not frame["row_id"].isna().any(),
        "row_id_is_file_order": row_ids == list(range(len(frame))),
    }
    problems = []
    if integrity["n"] != EXPECTED_N:
        problems.append(f"n is {integrity['n']}")
    if counts != EXPECTED_CELLS:
        problems.append(f"cells are {counts}")
    if sum(counts.values()) != integrity["n"]:
        problems.append("cell counts do not add up to n")
    if not integrity["row_id_unique"] or not integrity["row_id_is_file_order"]:
        problems.append("row_id is not the expected file-order index")
    if integrity["n_covariates"] != 25 or integrity["mathSel_in_covariates"]:
        problems.append("covariate contract failed")
    if not integrity["mathPost_finite"] or integrity["primary_fields_missing"]:
        problems.append("primary fields contain missing or non-finite values")
    if problems:
        _stop(
            root,
            approval,
            "INTEGRITY_CHECK_FAILED",
            "; ".join(problems),
            {**common, "integrity": integrity},
        )

    frame.attrs["source"] = AUTHORIZED_REAL_SOURCE
    frame.attrs["real_execution_authorized"] = True
    started = time.perf_counter()
    caught = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = stratified_primary_contrast_bootstrap(
            frame,
            n_replicates=2000,
            seed=20261026,
            authorized_real_run=True,
            failure_rule_approved=True,
            observer=lambda info: print(f"REPLICATE {info['replicate']}", flush=True)
            if info["replicate"] % 100 == 0
            else None,
        )
    elapsed = time.perf_counter() - started
    warning_rows = [
        {"category": item.category.__name__, "message": str(item.message)}
        for item in caught
        if "pyparsing" not in item.category.__name__.lower() and "pyparsing" not in str(item.message).lower()
    ]
    payload = {
        "status": (
            "CONFIRMATORY_PRIMARY_COMPLETED"
            if result["confirmatory_interval_authorized"]
            else "CONFIRMATORY_PRIMARY_COMPLETED_INTERVAL_DIAGNOSTIC_ONLY"
        ),
        "completed": True,
        "approver": approval["approver"],
        "approval_decision": approval["decision"],
        "protocol_sha256": digest,
        "approval_validation": "assert_owner_approved passed",
        "decision_record_validation": "assert_real_execution_authorized passed",
        "implementation_hash_validation": "matched owner decision record",
        "real_wsc_execution_enabled": REAL_WSC_EXECUTION_ENABLED,
        "joint_bootstrap": record["joint_bootstrap"],
        "zero_failure_rule": record["zero_failure_rule"],
        "real_primary_execution": record["real_primary_execution"],
        "git": common["git"],
        "integrity": integrity,
        "balance_context": _balance_context(root),
        "elapsed_seconds": elapsed,
        "warnings": warning_rows,
        "l1_status": "partial reproduction; not rerun in this analysis",
        "l3_run": False,
        "result": result,
    }
    _emit(root, payload, success=True)
    print(f"RCT {result['rct_estimate']}", flush=True)
    print(f"QED {result['qed_estimate']}", flush=True)
    print(f"DIFF {result['difference_qed_minus_rct']}", flush=True)
    print(f"BOOT_SE {result['bootstrap_se']}", flush=True)
    print(f"CI {result['percentile_ci_low']} {result['percentile_ci_high']}", flush=True)
    print(f"SUCCESS {result['n_success']} FAILED {result['n_failed']}", flush=True)
    print(f"CONFIRMATORY {result['confirmatory_interval_authorized']}", flush=True)
    print("STATUS COMPLETE", flush=True)


def run_authorized_primary(root: Path, config: dict, approval: dict, record: dict) -> None:
    if REAL_WSC_EXECUTION_ENABLED:
        raise SystemExit(
            "REAL_WSC_EXECUTION_ENABLED must stay false. Execution requires the owner decision record."
        )
    assert_implementation_hashes(root, record)
    try:
        _execute(root, config, approval, record)
    except SystemExit:
        raise
    except Exception as exc:
        _emit(
            root,
            {
                "status": "PRIMARY_ESTIMATE_FAILED",
                "completed": False,
                "approver": approval.get("approver"),
                "reason": f"{type(exc).__name__}: {exc}",
                "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
                "details": {"error_type": type(exc).__name__, "message": str(exc)},
            },
            success=False,
        )
        print(f"STATUS FAILED {type(exc).__name__}", flush=True)
        raise SystemExit(1) from exc


def record_owner_approval(root: Path) -> None:
    branch = _git(root, ["branch", "--show-current"])
    head = _git(root, ["rev-parse", "HEAD"])
    if branch != EXPECTED_BRANCH or head != EXPECTED_HEAD:
        print(f"GIT MISMATCH branch={branch} head={head}")
        raise SystemExit(2)
    protocol = root / "configs" / "frozen_protocol.yaml"
    digest = protocol_sha256(protocol)
    if digest != EXPECTED_PROTOCOL_SHA256:
        print(f"PROTOCOL HASH MISMATCH {digest}")
        raise SystemExit(2)
    approval_path = root / "configs" / "OWNER_APPROVAL.json"
    if approval_path.exists():
        existing = json.loads(approval_path.read_text(encoding="utf-8"))
        if existing.get("approver") != APPROVER:
            print("APPROVAL APPROVER MISMATCH")
            raise SystemExit(2)
    approval = {
        "approver": APPROVER,
        "decision": "approved",
        "protocol_sha256": digest,
    }
    record = {
        "approver": APPROVER,
        "decision_date": "2026-10-27",
        "protocol_sha256": digest,
        "frozen_protocol": "configs/frozen_protocol.yaml",
        "joint_bootstrap": "approved",
        "zero_failure_rule": "approved",
        "real_primary_execution": "authorized",
        "execution_scope": (
            "primary RCT_effect_prefer_math and QED_concordant_ATT on mathPost, "
            "plus the prespecified 2000-replicate joint bootstrap"
        ),
        "l3_authorized": False,
        "secondary_estimands_authorized": False,
        "zero_failure_rule_does_not_authorize_real_wsc": True,
        "implementation_sha256": implementation_hashes(root),
    }
    approval_path.write_text(json.dumps(approval, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    record_path = root / "configs" / "OWNER_DECISION_RECORD.json"
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checked = assert_owner_approved(protocol, approval_path)
    assert_real_execution_authorized(protocol, record_path, checked)
    assert_implementation_hashes(root, record)
    print("APPROVAL RECORDED")
    print(digest)


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] != "--config":
        raise SystemExit(
            "Usage: python -m p4_causal_validation.experiments.wsc_confirmatory "
            "--config configs/confirmatory_template.yaml"
        )
    config = yaml.safe_load(Path(sys.argv[2]).read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[3]
    protocol = _resolve(root, config["protocol_path"])
    approval = assert_owner_approved(protocol, _resolve(root, config["approval_path"]))
    record_value = config.get("decision_record_path")
    if not record_value:
        raise SystemExit(
            "Refusing real WSC confirmatory execution. Approval was checked, but this build does not fit mathPost."
        )
    record = assert_real_execution_authorized(protocol, _resolve(root, record_value), approval)
    if REAL_WSC_EXECUTION_ENABLED:
        raise SystemExit(
            "REAL_WSC_EXECUTION_ENABLED must stay false. Execution requires the owner decision record."
        )
    run_authorized_primary(root, config, approval, record)


if __name__ == "__main__":
    main()
