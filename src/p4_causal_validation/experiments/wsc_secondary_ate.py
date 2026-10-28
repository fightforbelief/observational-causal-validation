"""Isolated entry point for the owner-selected full-sample ATE secondary analysis.

The primary confirmatory gate is not used and is not weakened. REAL_WSC_EXECUTION_ENABLED
stays false in the primary module. This module runs only RCT_ATE_full versus
QED_concordant_ATE, and only when the secondary decision record matches the
frozen protocol, the secondary specification, and the WSC source file.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from p4_causal_validation.experiments.approval import protocol_sha256
from p4_causal_validation.inference.secondary_ate_bootstrap import (
    SECONDARY_REPLICATES,
    SECONDARY_SEED,
    stratified_secondary_ate_bootstrap,
)

EXPECTED_PROTOCOL_SHA256 = "583ff75e5e848e4c1831e6dbc6084041afec44c5a18a4806d31db6f3d5564e59"
EXPECTED_DATA_SHA256 = "b47be7d8f2c0d54af9cc71f3602183e61307a964e701a2cdcf3944bcef85286b"
EXPECTED_HEAD = "2a17d712d5cdfcaae81f624dd2c061684c869859"
EXPECTED_BRANCH = "p4/preapproval-inference-failure-policy"
ANALYSIS_DATA_PATH = Path(
    r"C:\Users\hanqi\Desktop\projects\_p4_external_sources\cran\WSCdata_0.1.2_extract\WSCdata\data\WSCdata.rda"
)
AUTHORIZED_SCOPE = "RCT_ATE_full_versus_QED_concordant_ATE"
AUTHORIZED_CONTRAST = "QED_concordant_ATE - RCT_ATE_full"
APPROVER = "Hanqing Yang"
EXPECTED_CELLS = {"G1": 288, "G2": 319, "G3": 776, "G4": 817}
EXPECTED_TREATED = 1064
EXPECTED_CONTROL = 1136


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, args: list[str]) -> str:
    completed = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"git {' '.join(args)} failed")
    return completed.stdout.strip()


def validate_secondary_decision(
    protocol_path: Path,
    spec_path: Path,
    data_path: Path,
    record: dict,
    approval: dict,
    *,
    enforce_known_digests: bool = True,
) -> None:
    """Refuse unless this record authorizes exactly the owner-selected ATE pair."""
    if not protocol_path.exists() or not spec_path.exists() or not data_path.exists():
        raise SystemExit("Refusing secondary ATE analysis: a required file is absent.")
    actual = {
        "protocol_sha256": protocol_sha256(protocol_path),
        "specification_sha256": _sha256(spec_path),
        "source_sha256": _sha256(data_path),
    }
    if any(record.get(key) != value for key, value in actual.items()):
        raise SystemExit("Refusing secondary ATE analysis: a decision-record hash does not match the file.")
    if enforce_known_digests and (
        actual["protocol_sha256"] != EXPECTED_PROTOCOL_SHA256 or actual["source_sha256"] != EXPECTED_DATA_SHA256
    ):
        raise SystemExit("Refusing secondary ATE analysis: protocol or source hash differs from the expected value.")
    if approval.get("approver") != APPROVER or record.get("approver") != APPROVER:
        raise SystemExit("Refusing secondary ATE analysis: approver is not the named primary approver.")
    if record.get("approver") != approval.get("approver"):
        raise SystemExit("Refusing secondary ATE analysis: approver is not the named primary approver.")
    if record.get("scope") != AUTHORIZED_SCOPE or record.get("contrast") != AUTHORIZED_CONTRAST:
        raise SystemExit("Refusing secondary ATE analysis: scope is not the authorized RCT_ATE_full versus QED_concordant_ATE pairing.")
    if record.get("rct_estimand") != "RCT_ATE_full" or record.get("qed_estimand") != "QED_concordant_ATE":
        raise SystemExit("Refusing secondary ATE analysis: scope is not the authorized RCT_ATE_full versus QED_concordant_ATE pairing.")
    if record.get("rct_estimator") != "unadjusted_difference_in_means" or record.get("qed_estimator") != "crossfit_aipw_ate_linear":
        raise SystemExit("Refusing secondary ATE analysis: estimator choice does not match the owner decision.")
    if record.get("reweight_to_one_half") is not False:
        raise SystemExit("Refusing secondary ATE analysis: reweighting to one half was not authorized.")
    if int(record.get("n_replicates", 0)) != SECONDARY_REPLICATES or int(record.get("seed", 0)) != SECONDARY_SEED:
        raise SystemExit("Refusing secondary ATE analysis: replicate count or seed does not match the owner decision.")
    if record.get("failure_rule") != "all_2000_requested_replicates_succeed":
        raise SystemExit("Refusing secondary ATE analysis: failure rule does not match the owner decision.")
    if record.get("l3_authorized") is not False or record.get("vocabPost_authorized") is not False:
        raise SystemExit("Refusing secondary ATE analysis: scope is not the authorized RCT_ATE_full versus QED_concordant_ATE pairing.")
    if record.get("this_choice_was_not_uniquely_specified_by_the_frozen_protocol") is not True:
        raise SystemExit("Refusing secondary ATE analysis: the record must say the estimator choice is a new owner decision.")


def _cell_counts(frame: pd.DataFrame) -> dict[str, int]:
    preference = frame["mathSel"].to_numpy(dtype=int)
    assignment = frame["mathGrp"].to_numpy(dtype=int)
    return {
        "G1": int(((preference == 1) & (assignment == 1)).sum()),
        "G2": int(((preference == 1) & (assignment == 0)).sum()),
        "G3": int(((preference == 0) & (assignment == 1)).sum()),
        "G4": int(((preference == 0) & (assignment == 0)).sum()),
    }


def covariate_balance(qed_frame: pd.DataFrame, propensity: list[float], covariates: list[str]) -> dict:
    """ATE-IPW covariate SMDs on the concordant sample.

    The denominator is the unweighted pooled standard deviation. Treated weights
    are 1/e and control weights are 1/(1-e). Thresholds 0.1 and 0.2 are descriptive.
    """
    probability = np.asarray(propensity, dtype=float)
    rows = []
    for name in covariates:
        values = qed_frame[name].to_numpy(dtype=float)
        treated = qed_frame["mathGrp"].to_numpy(dtype=int) == 1
        pooled = float(np.std(values, ddof=1))
        raw_treated = float(values[treated].mean())
        raw_control = float(values[~treated].mean())
        treated_weights = 1.0 / probability[treated]
        control_weights = 1.0 / (1.0 - probability[~treated])
        weighted_treated = float(np.sum(treated_weights * values[treated]) / np.sum(treated_weights))
        weighted_control = float(np.sum(control_weights * values[~treated]) / np.sum(control_weights))
        if not np.isfinite(pooled) or pooled == 0:
            rows.append({"covariate": name, "status": "not_computable"})
            continue
        unweighted = (raw_treated - raw_control) / pooled
        weighted = (weighted_treated - weighted_control) / pooled
        rows.append(
            {
                "covariate": name,
                "status": "computed",
                "smd_unweighted": unweighted,
                "smd_weighted": weighted,
                "abs_smd_unweighted": abs(unweighted),
                "abs_smd_weighted": abs(weighted),
            }
        )
    computed = [row for row in rows if row["status"] == "computed"]
    above = [row for row in computed if row["abs_smd_weighted"] > 0.1]
    return {
        "sample": "QED concordant subset",
        "denominator": "unweighted pooled SD of G1 and G4, ddof=1",
        "weights": "1/e for G1 and 1/(1-e) for G4, floored cross-fit propensity",
        "thresholds_are_descriptive_only": True,
        "n_covariates": len(computed),
        "max_abs_smd_unweighted": max(computed, key=lambda row: row["abs_smd_unweighted"]),
        "max_abs_smd_weighted": max(computed, key=lambda row: row["abs_smd_weighted"]),
        "n_abs_weighted_above_0_1": len(above),
        "abs_weighted_above_0_1": above,
        "n_abs_unweighted_above_0_1": sum(row["abs_smd_unweighted"] > 0.1 for row in computed),
        "n_abs_unweighted_above_0_2": sum(row["abs_smd_unweighted"] > 0.2 for row in computed),
        "n_abs_weighted_above_0_2": sum(row["abs_smd_weighted"] > 0.2 for row in computed),
        "rows": rows,
    }


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


def _fmt(value) -> str:
    if isinstance(value, float):
        return format(value, ".16g")
    if value is None:
        return "null"
    return str(value)


def _write_reports(root: Path, payload: dict) -> None:
    json_path = root / "reports" / "P4_SECONDARY_FULL_SAMPLE_ATE.json"
    md_path = root / "reports" / "P4_SECONDARY_FULL_SAMPLE_ATE.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_json_ready(payload), ensure_ascii=False, indent=2) + "\n"
    json_path.write_text(text, encoding="utf-8")
    payload["json_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    md_path.write_text(_render(payload), encoding="utf-8")


def _render(payload: dict) -> str:
    result = payload["result"]
    qed = result["qed_point"]
    rct = result["rct_point"]
    balance = payload["balance"]
    lines = [
        "# 全样本 ATE 次要分析",
        "",
        "状态：`SECONDARY_FULL_SAMPLE_ATE_COMPLETED`。这是次要分析，不替换已经完成的主分析。",
        "",
        "估计器选择是 Hanqing Yang 在主结果已知、次要估计尚未查看时作出的新决定。冻结协议本身没有唯一指定这些估计器。整个研究因此不是完全预先注册的。",
        "",
        f"- 规格 SHA-256：`{payload['specification_sha256']}`",
        f"- 协议 SHA-256：`{payload['protocol_sha256']}`",
        f"- 数据 SHA-256：`{payload['source_sha256']}`",
        f"- `P(Z=1)=1/2` 独立核实：{payload['p_z_one_half_independently_verified']}",
        "- 两个 ATE 估计量没有被说成同一个参数。",
        "",
        "## 样本",
        "",
        f"- 全样本 {payload['integrity']['n']} 人，数学分配 {payload['integrity']['n_treated']}，词汇分配 {payload['integrity']['n_control']}。",
        f"- 单元格：{payload['integrity']['cells']}。",
        f"- 协和子集 {payload['integrity']['n_concordant']} 人。G3 进入随机化 ATE，不进入观察性 ATE。",
        "- 随机化估计没有把实际人数重加权到 1/2。",
        "",
        "## 估计",
        "",
        f"- RCT_ATE_full：{_fmt(result['rct_estimate'])}",
        f"- Neyman 标准误：{_fmt(result['rct_neyman_se'])}",
        f"- RCT 正态区间：[{_fmt(rct['normal_interval_low'])}, {_fmt(rct['normal_interval_high'])}]",
        f"- RCT 样本量：{rct['n']}（处理 {rct['n_treated']}，对照 {rct['n_control']}）",
        "",
        f"- QED_concordant_ATE：{_fmt(result['qed_estimate'])}",
        f"- 影响函数标准误：{_fmt(result['qed_influence_se'])}",
        f"- QED 正态区间：[{_fmt(qed['normal_interval_low'])}, {_fmt(qed['normal_interval_high'])}]",
        f"- QED 样本量：{qed['n']}（处理 {qed['n_treated']}，对照 {qed['n_control']}）",
        f"- 因重叠删除的行：{qed.get('rows_deleted_for_overlap')}",
        f"- 数值下限移动的概率数：{qed.get('n_probabilities_moved_by_floor')}",
        f"- 重叠：{json.dumps(_json_ready(qed.get('overlap_report')), ensure_ascii=False)}",
        f"- ATE 权重与 ESS：{json.dumps(_json_ready(qed.get('ate_arm_weights')), ensure_ascii=False)}",
        "",
        "原始样本配对差是 QED ATE 减 RCT ATE，不是复制均值，也不是纯粹混杂偏误。",
        "",
        f"- 差值：{_fmt(result['difference_qed_minus_rct'])}",
        f"- bootstrap 成功 / 失败：{result['n_success']} / {result['n_failed']}",
        f"- bootstrap 标准误：{_fmt(result['bootstrap_se'])}",
        f"- 百分位区间：[{_fmt(result['percentile_ci_low'])}, {_fmt(result['percentile_ci_high'])}]",
        f"- 协方差：{_fmt(result['bootstrap_covariance'])}",
        f"- 区间状态：`{result['interval_status']}`",
        f"- 次要零失败规则满足：{result['secondary_zero_failure_rule_met']}",
        "",
        "失败账本：" if result["failures"] else "失败账本为空。没有复制被替换或静默删除。",
    ]
    for item in result["failures"]:
        lines.append(f"- replicate {item['replicate']}: {item['error_type']}: {item['message']}")
    lines.extend(
        [
            "",
            "## 协变量平衡",
            "",
            "标准化差值使用未加权的合并标准差。加权均值使用本次 ATE 拟合的交叉拟合倾向评分。0.1 和 0.2 只是描述性参照。平衡变好不证明无混杂。",
            "",
            f"- 平衡摘要：{json.dumps(_json_ready({key: value for key, value in balance.items() if key != 'rows'}), ensure_ascii=False)}",
            "",
            "## 局限",
            "",
            "WSCdata 文档写明 mathGrp 是随机分配，并给出四臂编码；它没有独立写出分配概率 1/2。冻结协议把 1/2 作为 Keller 等人设计证明里的条件。实际人数是数学 1064、词汇 1136。因此这次不声称两个 ATE 是同一个参数。",
            "",
            "未测量混杂仍然没有解决。双重稳健和 bootstrap 都不建立识别。处理版本一致性没有被证明。Bootstrap 以观察到的单元格人数为条件，不是随机化推断。次要零失败规则不把这次分析变成主确认性分析，也不证明覆盖率。",
            "",
            "点估计接近不能写成等效或已经恢复随机化因果效应。主分析与这次估计之间的差别同时包含目标参数、样本和估计器的差别，不能自动写成混杂偏误。",
            "",
            "L3、vocabPost 和其他估计量没有运行。",
            "",
            f"机器可读结果 SHA-256：`{payload['json_sha256']}`。",
            "",
        ]
    )
    return "\n".join(lines)


def run_authorized_secondary(root: Path) -> None:
    from p4_causal_validation.data.wsc import prepare_wsc, resolve_covariate_columns

    protocol = root / "configs" / "frozen_protocol.yaml"
    spec = root / "reports" / "P4_SECONDARY_FULL_SAMPLE_ATE_SPEC.md"
    record_path = root / "configs" / "SECONDARY_ATE_DECISION_RECORD.json"
    approval_path = root / "configs" / "OWNER_APPROVAL.json"
    if not record_path.exists() or not approval_path.exists():
        raise SystemExit("Refusing secondary ATE analysis: a required file is absent.")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    validate_secondary_decision(protocol, spec, ANALYSIS_DATA_PATH, record, approval)
    branch = _git(root, ["branch", "--show-current"])
    head = _git(root, ["rev-parse", "HEAD"])
    if branch != EXPECTED_BRANCH or head != EXPECTED_HEAD:
        raise SystemExit("Refusing secondary ATE analysis: branch or HEAD differs from the expected baseline.")
    frame, metadata = prepare_wsc(ANALYSIS_DATA_PATH)
    counts = _cell_counts(frame)
    assignment = frame["mathGrp"].to_numpy(dtype=int)
    concordant = frame["mathSel"].to_numpy(dtype=int) == assignment
    covariates = resolve_covariate_columns(frame)
    problems = []
    if len(frame) != 2200 or counts != EXPECTED_CELLS:
        problems.append(f"cells {counts} n {len(frame)}")
    if int((assignment == 1).sum()) != EXPECTED_TREATED or int((assignment == 0).sum()) != EXPECTED_CONTROL:
        problems.append("arm sizes differ from 1064 and 1136")
    if int(concordant.sum()) != 1105:
        problems.append("concordant sample is not 1105")
    if len(covariates) != 25 or "mathSel" in covariates:
        problems.append("covariate contract failed")
    if frame["row_id"].tolist() != list(range(len(frame))):
        problems.append("row_id is not file order")
    if not np.isfinite(frame["mathPost"].to_numpy(dtype=float)).all():
        problems.append("mathPost is not finite")
    if problems:
        raise SystemExit("Refusing secondary ATE analysis: " + "; ".join(problems))
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = stratified_secondary_ate_bootstrap(
            frame,
            n_replicates=SECONDARY_REPLICATES,
            seed=SECONDARY_SEED,
            observer=lambda info: print(f"REPLICATE {info['replicate']}", flush=True) if info["replicate"] % 100 == 0 else None,
        )
    from p4_causal_validation.data.estimands import subset_for_estimand

    qed_rows = frame.loc[concordant].reset_index(drop=True)
    aligned = subset_for_estimand(frame, "QED_concordant_ATE").reset_index(drop=True)
    if aligned["row_id"].tolist() != qed_rows["row_id"].tolist():
        raise SystemExit("Refusing secondary ATE analysis: balance rows are not aligned with the ATE fit.")
    propensity = result["qed_point"]["out_of_fold_propensity"]
    balance = covariate_balance(qed_rows, propensity, covariates)
    warning_rows = [
        {"category": item.category.__name__, "message": str(item.message)}
        for item in caught
        if "pyparsing" not in item.category.__name__.lower() and "pyparsing" not in str(item.message).lower()
    ]
    payload = {
        "status": "SECONDARY_FULL_SAMPLE_ATE_COMPLETED",
        "approver": APPROVER,
        "method_choices_are_a_new_owner_decision": True,
        "not_fully_preregistered": True,
        "replaces_primary_analysis": False,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "specification_sha256": _sha256(spec),
        "source_sha256": EXPECTED_DATA_SHA256,
        "p_z_one_half_independently_verified": False,
        "same_parameter_claim": False,
        "git": {"branch": branch, "head": head},
        "integrity": {
            "n": int(len(frame)),
            "n_treated": EXPECTED_TREATED,
            "n_control": EXPECTED_CONTROL,
            "n_concordant": int(concordant.sum()),
            "cells": counts,
            "covariates": covariates,
            "metadata_n": metadata["n"],
        },
        "balance": balance,
        "elapsed_seconds": time.perf_counter() - started,
        "warnings": warning_rows,
        "l3_run": False,
        "result": result,
    }
    _write_reports(root, payload)
    print(f"RCT {result['rct_estimate']}", flush=True)
    print(f"QED {result['qed_estimate']}", flush=True)
    print(f"DIFF {result['difference_qed_minus_rct']}", flush=True)
    print(f"BOOT_SE {result['bootstrap_se']}", flush=True)
    print(f"CI {result['percentile_ci_low']} {result['percentile_ci_high']}", flush=True)
    print(f"SUCCESS {result['n_success']} FAILED {result['n_failed']}", flush=True)
    print(f"RULE {result['secondary_zero_failure_rule_met']}", flush=True)
    print("STATUS COMPLETE", flush=True)


def main() -> None:
    run_authorized_secondary(Path(__file__).resolve().parents[3])


if __name__ == "__main__":
    main()
