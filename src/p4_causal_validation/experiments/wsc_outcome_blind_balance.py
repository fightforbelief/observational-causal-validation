"""Outcome-blind ATT covariate balance for G1 versus G4.

The propensity scores and covariate list come from the existing readiness
diagnostic. This module does not estimate a treatment effect.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from p4_causal_validation.data.estimands import subset_for_estimand
from p4_causal_validation.diagnostics.overlap import effective_sample_size
from p4_causal_validation.experiments.primary_specification import PRIMARY_FOLDS, PRIMARY_OBSERVATIONAL, PRIMARY_SEED
from p4_causal_validation.experiments.wsc_outcome_blind_diagnostics import (
    BANNER,
    EXPECTED_BRANCH,
    EXPECTED_BYTES,
    EXPECTED_HEAD,
    EXPECTED_SHA256,
    STATUS_COMPLETED,
    assign_analysis_row_id,
    att_control_weight_diagnostics,
    fit_outcome_blind_propensity,
    load_outcome_blind_wsc,
    parse_protocol,
    protocol_matches_code,
    run_outcome_blind_diagnostics,
    sha256_file,
    without_outcomes,
)

DATA_PATH = Path(r"C:\Users\hanqi\Desktop\projects\_p4_external_sources\cran\WSCdata_0.1.2_extract\WSCdata\data\WSCdata.rda")
STATUS_BALANCE = "OUTCOME_BLIND_BALANCE_COMPLETED"
STATUS_STOPPED = "OUTCOME_BLIND_BALANCE_STOPPED"
KNOWN_UNTRACKED = {
    "src/p4_causal_validation/experiments/wsc_outcome_blind_diagnostics.py",
    "src/p4_causal_validation/experiments/wsc_outcome_blind_balance.py",
    "tests/test_wsc_outcome_blind_diagnostics.py",
    "tests/test_wsc_outcome_blind_balance.py",
}


def estimation_subset(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Rebuild the G1∪G4 matrix in the same order as the readiness diagnostic."""
    blind, _dropped = without_outcomes(frame)
    identified, _row_meta = assign_analysis_row_id(blind)
    from p4_causal_validation.data.wsc import resolve_covariate_columns

    covariates = resolve_covariate_columns(identified)
    subset = subset_for_estimand(identified, PRIMARY_OBSERVATIONAL).reset_index(drop=True)
    if not subset["mathGrp"].eq(subset["mathSel"]).all():
        raise ValueError("Concordant subset does not have mathGrp equal to mathSel.")
    return subset, covariates


def control_weights(propensity, treatment) -> np.ndarray:
    """G4 ATT weights e/(1-e). No floor and no truncation."""
    probability = np.asarray(propensity, dtype=float)
    assignment = np.asarray(treatment, dtype=int)
    with np.errstate(divide="ignore", invalid="ignore"):
        return probability[assignment == 0] / (1.0 - probability[assignment == 0])


def weight_problem(weights) -> str | None:
    values = np.asarray(weights, dtype=float)
    if values.size == 0:
        return "control weight vector is empty"
    if not np.isfinite(values).all():
        return "non-finite control weight"
    if np.any(values < 0):
        return "negative control weight"
    if float(np.sum(values)) == 0:
        return "control weights sum to zero"
    return None


def covariate_balance(names: list[str], g1, g4, g4_weights) -> list[dict]:
    """SMD before and after ATT weighting, with one unweighted pooled SD.

    The denominator is the sample standard deviation of the concatenated
    unweighted G1 and G4 values, with divisor n-1. The weighted SMD changes
    only the G4 mean. A variable whose denominator is zero or whose values are
    not finite stays in the output and is marked not computable.
    """
    treated = np.asarray(g1, dtype=float)
    control = np.asarray(g4, dtype=float)
    weights = np.asarray(g4_weights, dtype=float)
    if treated.ndim == 1:
        treated = treated.reshape(-1, 1)
    if control.ndim == 1:
        control = control.reshape(-1, 1)
    if treated.shape[1] != len(names) or control.shape[1] != len(names):
        raise ValueError("Each protocol covariate must be one column. No covariate was dropped.")
    if weights.shape != (control.shape[0],):
        raise ValueError("G4 weights must have one value per G4 row.")
    shared_weight_problem = weight_problem(weights)
    rows = []
    for index, name in enumerate(names):
        treated_values = treated[:, index]
        control_values = control[:, index]
        row = {
            "covariate": name,
            "n_g1": int(treated_values.size),
            "n_g4": int(control_values.size),
            "mean_g1": None,
            "mean_g4_unweighted": None,
            "mean_g4_weighted": None,
            "sd_pooled_unweighted": None,
            "smd_unweighted": None,
            "smd_weighted": None,
            "abs_smd_unweighted": None,
            "abs_smd_weighted": None,
            "abs_smd_change": None,
            "status": "not_computable",
            "reason": "",
        }
        if not np.isfinite(treated_values).all() or not np.isfinite(control_values).all():
            row["reason"] = "missing or non-finite covariate value; rows were not deleted"
            rows.append(row)
            continue
        combined = np.concatenate([treated_values, control_values])
        row["mean_g1"] = float(np.mean(treated_values))
        row["mean_g4_unweighted"] = float(np.mean(control_values))
        if combined.size < 2:
            row["reason"] = "fewer than two combined observations"
            rows.append(row)
            continue
        deviation = float(np.std(combined, ddof=1))
        row["sd_pooled_unweighted"] = deviation
        if not np.isfinite(deviation) or deviation == 0:
            row["reason"] = "unweighted pooled sample standard deviation is zero"
            rows.append(row)
            continue
        unweighted = (row["mean_g1"] - row["mean_g4_unweighted"]) / deviation
        row["smd_unweighted"] = float(unweighted)
        row["abs_smd_unweighted"] = float(abs(unweighted))
        if shared_weight_problem is not None:
            row["status"] = "weighted_not_computable"
            row["reason"] = shared_weight_problem
            rows.append(row)
            continue
        weighted_mean = float(np.sum(weights * control_values) / np.sum(weights))
        weighted = (row["mean_g1"] - weighted_mean) / deviation
        row["mean_g4_weighted"] = weighted_mean
        row["smd_weighted"] = float(weighted)
        row["abs_smd_weighted"] = float(abs(weighted))
        row["abs_smd_change"] = float(abs(weighted) - abs(unweighted))
        row["status"] = "computed"
        rows.append(row)
    return rows


def balance_summary(rows: list[dict]) -> dict:
    computed = [row for row in rows if row["status"] == "computed"]

    def count_above(field: str, threshold: float) -> int:
        return sum(1 for row in computed if row[field] > threshold)

    def maximum(field: str):
        values = [row[field] for row in computed]
        return None if not values else float(max(values))

    return {
        "n_covariates": len(rows),
        "n_computed": len(computed),
        "n_not_computable": len(rows) - len(computed),
        "max_abs_smd_unweighted": maximum("abs_smd_unweighted"),
        "max_abs_smd_weighted": maximum("abs_smd_weighted"),
        "n_abs_smd_unweighted_gt_0.1": count_above("abs_smd_unweighted", 0.1),
        "n_abs_smd_weighted_gt_0.1": count_above("abs_smd_weighted", 0.1),
        "n_abs_smd_unweighted_gt_0.2": count_above("abs_smd_unweighted", 0.2),
        "n_abs_smd_weighted_gt_0.2": count_above("abs_smd_weighted", 0.2),
        "n_abs_smd_decreased": sum(1 for row in computed if row["abs_smd_change"] < 0),
        "n_abs_smd_increased": sum(1 for row in computed if row["abs_smd_change"] > 0),
        "reference_thresholds": (0.1, 0.2),
        "thresholds_are_acceptance_criteria": False,
    }


def write_balance_plot(rows: list[dict], path: Path) -> None:
    plotted = [row for row in rows if row["status"] == "computed"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not plotted:
        figure, axis = plt.subplots(figsize=(8, 3))
        axis.text(0.5, 0.5, "No computable SMD", ha="center", va="center")
        axis.axis("off")
        figure.savefig(path, dpi=150)
        plt.close(figure)
        return
    labels = [row["covariate"] for row in plotted]
    before = [row["smd_unweighted"] for row in plotted]
    after = [row["smd_weighted"] for row in plotted]
    positions = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(8, 10))
    for position, left, right in zip(positions, before, after):
        axis.plot([left, right], [position, position], color="#b0b0b0", linewidth=1, zorder=1)
    axis.scatter(before, positions, marker="o", s=28, color="#4C78A8", label="Unweighted", zorder=2)
    axis.scatter(after, positions, marker="s", s=28, color="#F58518", label="ATT-weighted", zorder=2)
    axis.axvline(0, color="black", linewidth=1)
    axis.axvline(-0.1, color="#666666", linewidth=1, linestyle="--")
    axis.axvline(0.1, color="#666666", linewidth=1, linestyle="--")
    axis.set_yticks(positions)
    axis.set_yticklabels(labels)
    axis.invert_yaxis()
    axis.set_xlabel("SMD = (mean G1 - mean G4) / unweighted pooled SD")
    axis.set_title("Covariate SMD before and after ATT weights")
    axis.legend(frameon=False, loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def parse_prior_weight_summary(text: str) -> dict:
    import re

    def grab(pattern: str):
        match = re.search(pattern, text)
        return None if match is None else match.group(1)

    g4 = re.search(r"\| G4 \| (\d+) \| (\d+) \| (\d+) \| (\d+) \| ([0-9.]+) \|", text)
    return {
        "ess": grab(r"(?m)^- ESS：([0-9.eE+-]+)\s*$"),
        "sum_control_weights": grab(r"(?m)^- 权重总和：([0-9.eE+-]+)\s*$"),
        "sum_squared_control_weights": grab(r"(?m)^- 权重平方和：([0-9.eE+-]+)\s*$"),
        "g4_n": None if g4 is None else int(g4.group(1)),
        "g4_below_0.1": None if g4 is None else int(g4.group(2)),
        "g4_above_0.9": None if g4 is None else int(g4.group(3)),
        "g4_outside": None if g4 is None else int(g4.group(4)),
        "g4_outside_share": None if g4 is None else g4.group(5),
    }


def same_reported_number(value: float, reported: str | None) -> bool:
    return reported is not None and f"{value:.8g}" == reported


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def git_allows_this_stage(root: Path) -> dict:
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root, text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    porcelain = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    unexpected = []
    for line in porcelain.splitlines():
        path = line[3:].replace("\\", "/")
        if line.startswith("?? ") and path in KNOWN_UNTRACKED:
            continue
        unexpected.append(line)
    return {
        "branch": branch,
        "head": head,
        "unexpected": unexpected,
        "matches": branch == EXPECTED_BRANCH and head == EXPECTED_HEAD and not unexpected,
    }


def _fmt(value) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def render_report(payload: dict) -> str:
    rows = payload["rows"]
    summary = payload["summary"]
    prior = payload["prior"]
    lines = [
        "# P4 outcome-blind ATT covariate balance",
        "",
        BANNER,
        "",
        "本报告只比较 G1 与 G4 在预先指定 ATT 权重下的已观测协变量平衡。没有估计处理效应，没有计算 RCT 均值差、ATT、标准误、置信区间或 bootstrap。",
        "",
        "0.1 和 0.2 只是描述性参考线，不是协议里的通过标准。本报告不宣布主分析通过或失败，也不代表协议批准。",
        "",
        f"判断：`{payload['status']}`",
        "",
        "## 规格",
        "",
        f"- 日期：{payload['date']}",
        f"- 机器：{payload['machine']}",
        f"- 分支：{payload['branch']}",
        f"- HEAD：`{payload['head']}`",
        f"- 数据：`{payload['data_path']}`",
        f"- 数据字节：{payload['data_bytes']}；SHA-256：`{payload['data_sha256']}`",
        f"- 协变量来源：`configs/protocol_candidate_v2.yaml`，SHA-256 `{payload['protocol_sha256']}`",
        f"- 协变量解析顺序与代码一致：{payload['protocol_aligned']}",
        f"- 估计样本：`{PRIMARY_OBSERVATIONAL}`，G1 对 G4",
        f"- 折数：{PRIMARY_FOLDS}；种子：{PRIMARY_SEED}；分折：distinct participant `row_id`",
        "- 倾向得分沿用现有 `fit_outcome_blind_propensity`，即 `_assign_crossfit_folds` 与 `_logit_predict`。没有使用 `StratifiedKFold`，没有拟合结果模型。",
        "- G1 权重为 1，因此 G1 均值是未加权均值。G4 权重为 `e/(1-e)`，没有截断，也没有套用 1e-6 下限。",
        "- 标准化分母是加权前 G1 与 G4 合并样本的标准差，除数是 n−1。加权后的 SMD 只把 G4 均值换成加权均值，分母不变。",
        f"- 协变量（{len(payload['covariates'])} 个）：{payload['covariates']}",
        "",
        "## 样本、权重和与前一份诊断的对照",
        "",
        f"- G1 人数：{payload['n_g1']}",
        f"- G4 人数：{payload['n_g4']}",
        f"- 非有限或负的 G4 权重数：{payload['n_bad_weights']}",
        f"- 本次 ESS：{_fmt(payload['ess'])}",
        f"- 本次权重总和：{_fmt(payload['sum_weights'])}",
        f"- 本次权重平方和：{_fmt(payload['sum_squared_weights'])}",
        f"- 前一份报告 ESS：{prior.get('ess')}",
        f"- 前一份报告权重总和：{prior.get('sum_control_weights')}",
        f"- 前一份报告权重平方和：{prior.get('sum_squared_control_weights')}",
        f"- 与前一份报告在 `.8g` 记录下一致：{payload['matches_prior_weights']}",
        f"- 与现有诊断函数重新拟合的倾向得分一致：{payload['matches_refit']}",
        f"- 前一份报告中的 G4 重叠，本次没有重做重叠审计：{prior.get('g4_below_0.1')}/{prior.get('g4_n')} 低于 0.1，{prior.get('g4_above_0.9')} 高于 0.9，窗口外比例 {prior.get('g4_outside_share')}",
        "",
        "## 汇总",
        "",
        f"- 进入输出的协变量：{summary['n_covariates']}",
        f"- 可计算：{summary['n_computed']}",
        f"- 不可计算：{summary['n_not_computable']}",
        f"- 加权前最大绝对 SMD：{_fmt(summary['max_abs_smd_unweighted'])}",
        f"- 加权后最大绝对 SMD：{_fmt(summary['max_abs_smd_weighted'])}",
        f"- 加权前绝对 SMD 大于 0.1：{summary['n_abs_smd_unweighted_gt_0.1']}",
        f"- 加权后绝对 SMD 大于 0.1：{summary['n_abs_smd_weighted_gt_0.1']}",
        f"- 加权前绝对 SMD 大于 0.2：{summary['n_abs_smd_unweighted_gt_0.2']}",
        f"- 加权后绝对 SMD 大于 0.2：{summary['n_abs_smd_weighted_gt_0.2']}",
        f"- 绝对 SMD 下降的协变量：{summary['n_abs_smd_decreased']}",
        f"- 绝对 SMD 上升的协变量：{summary['n_abs_smd_increased']}",
        "",
        "这些计数不是通过规则。",
        "",
        "## 逐变量结果",
        "",
        "| 协变量 | G1均值 | G4均值 | G4加权均值 | 合并SD | 加权前SMD | 加权后SMD | 绝对SMD变化 | 状态 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["covariate"],
                    _fmt(row["mean_g1"]),
                    _fmt(row["mean_g4_unweighted"]),
                    _fmt(row["mean_g4_weighted"]),
                    _fmt(row["sd_pooled_unweighted"]),
                    _fmt(row["smd_unweighted"]),
                    _fmt(row["smd_weighted"]),
                    _fmt(row["abs_smd_change"]),
                    row["status"],
                ]
            )
            + " |"
        )
    not_computable = [row for row in rows if row["status"] != "computed"]
    lines.extend(["", "## 不可计算或权重问题", ""])
    if not_computable or payload["failures"]:
        for row in not_computable:
            lines.append(f"- `{row['covariate']}`：{row['reason']}")
        for item in payload["failures"]:
            lines.append(f"- {item}")
    else:
        lines.append("- 无。25 个协变量都有有限的加权前后 SMD，G4 权重全部有限。")
    lines.extend(
        [
            "",
            "## 图",
            "",
            "比较图：`reports/P4_OUTCOME_BLIND_BALANCE.png`",
            "",
            "纵轴是协议顺序中的协变量，横轴是 SMD。圆点是加权前，方点是 ATT 加权后。竖线是 0 和 ±0.1，只作参照。",
            "",
            "逐变量表：`reports/P4_OUTCOME_BLIND_BALANCE.csv`",
            "",
            "## 解释边界",
            "",
            "倾向得分重叠、ATT 权重的 ESS、已观测协变量的加权平衡，回答的是三件不同的事。前一份诊断已经记录 G4 有相当一部分样本外倾向得分低于 0.1，对照组 ESS 约为 146。本次加权平衡不改变那个权重分布，因此弱重叠和较小的 ESS 仍然限制估计精度。",
            "",
            "加权后的已观测协变量更接近，只说明这组权重在这些已测变量上的平衡情况。它不证明没有未测混杂，不证明处理版本一致，也不证明条件可忽略性或重叠足以识别因果效应。",
            "",
            BANNER,
            "",
        ]
    )
    return "\n".join(lines)


def rows_to_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> int:
    from datetime import date
    import platform

    root = project_root()
    state = git_allows_this_stage(root)
    if not state["matches"]:
        print(f"STOP branch={state['branch']} head={state['head']} unexpected={state['unexpected']}")
        return 2
    branch = state["branch"]
    head = state["head"]
    data_bytes = DATA_PATH.stat().st_size if DATA_PATH.exists() else None
    data_sha = sha256_file(DATA_PATH) if DATA_PATH.exists() else None
    if data_sha != EXPECTED_SHA256 or data_bytes != EXPECTED_BYTES:
        print(f"STOP hash={data_sha} bytes={data_bytes}")
        return 2
    protocol_path = root / "configs" / "protocol_candidate_v2.yaml"
    protocol_text = protocol_path.read_text(encoding="utf-8")
    spec = parse_protocol(protocol_text)
    mismatches = protocol_matches_code(spec)
    if mismatches:
        print(f"STOP protocol {mismatches}")
        return 2
    frame, _dropped = load_outcome_blind_wsc(DATA_PATH)
    diagnostic = run_outcome_blind_diagnostics(frame, protocol_text)
    if diagnostic["status"] != STATUS_COMPLETED or diagnostic["propensity"]["propensity"] is None:
        print(f"STOP diagnostic={diagnostic['status']} failures={diagnostic['failures']}")
        return 2
    subset, covariates = estimation_subset(frame)
    treatment = subset["mathGrp"].to_numpy(dtype=int)
    refit = fit_outcome_blind_propensity(
        subset[covariates].to_numpy(dtype=float),
        treatment,
        subset["row_id"].tolist(),
        n_folds=PRIMARY_FOLDS,
        seed=PRIMARY_SEED,
    )
    verified = np.asarray(diagnostic["propensity"]["propensity"], dtype=float)
    refit_values = None if refit["propensity"] is None else np.asarray(refit["propensity"], dtype=float)
    max_abs_diff = None
    if refit_values is not None and refit_values.shape == verified.shape:
        max_abs_diff = float(np.max(np.abs(refit_values - verified)))
    matches_refit = bool(refit["model_succeeded"] and max_abs_diff == 0)
    if list(diagnostic["propensity"]["covariates"]) != covariates or not matches_refit:
        print(f"STOP propensity mismatch max_abs_diff={max_abs_diff} refit_ok={refit['model_succeeded']}")
        return 2
    g1 = treatment == 1
    g4 = treatment == 0
    weights = control_weights(verified, treatment)
    weight_summary = att_control_weight_diagnostics(verified, treatment)
    if weight_summary["ess"] is None or not np.allclose(effective_sample_size(weights), weight_summary["ess"], rtol=0, atol=0):
        print("STOP ESS from the weight vector does not match att_control_weight_diagnostics")
        return 2
    rows = covariate_balance(covariates, subset.loc[g1, covariates].to_numpy(dtype=float), subset.loc[g4, covariates].to_numpy(dtype=float), weights)
    summary = balance_summary(rows)
    prior_path = root / "reports" / "P4_OUTCOME_BLIND_DATA_READINESS.md"
    prior = parse_prior_weight_summary(prior_path.read_text(encoding="utf-8"))
    matches_prior = all(
        [
            same_reported_number(weight_summary["ess"], prior["ess"]),
            same_reported_number(weight_summary["sum_control_weights"], prior["sum_control_weights"]),
            same_reported_number(weight_summary["sum_squared_control_weights"], prior["sum_squared_control_weights"]),
        ]
    )
    payload = {
        "status": STATUS_BALANCE,
        "date": date.today().isoformat(),
        "machine": platform.node(),
        "branch": branch,
        "head": head,
        "data_path": str(DATA_PATH),
        "data_bytes": data_bytes,
        "data_sha256": data_sha,
        "protocol_sha256": sha256_file(protocol_path),
        "protocol_aligned": mismatches == [],
        "covariates": covariates,
        "n_g1": int(g1.sum()),
        "n_g4": int(g4.sum()),
        "n_bad_weights": int(weight_summary["n_nonfinite_or_negative_control_weights"]),
        "ess": weight_summary["ess"],
        "sum_weights": weight_summary["sum_control_weights"],
        "sum_squared_weights": weight_summary["sum_squared_control_weights"],
        "matches_prior_weights": matches_prior,
        "matches_refit": matches_refit,
        "prior": prior,
        "rows": rows,
        "summary": summary,
        "failures": [] if matches_prior else ["Recomputed ESS or weight sums do not match the readiness report at .8g."],
    }
    report_path = root / "reports" / "P4_OUTCOME_BLIND_BALANCE.md"
    csv_path = root / "reports" / "P4_OUTCOME_BLIND_BALANCE.csv"
    png_path = root / "reports" / "P4_OUTCOME_BLIND_BALANCE.png"
    rows_to_csv(rows, csv_path)
    write_balance_plot(rows, png_path)
    report_path.write_text(render_report(payload), encoding="utf-8", newline="\n")
    print(f"STATUS={payload['status']}")
    print(f"MATCHES_PRIOR={matches_prior}")
    print(f"N_G1={payload['n_g1']}")
    print(f"N_G4={payload['n_g4']}")
    print(f"MAX_BEFORE={summary['max_abs_smd_unweighted']}")
    print(f"MAX_AFTER={summary['max_abs_smd_weighted']}")
    print(f"GT01_BEFORE={summary['n_abs_smd_unweighted_gt_0.1']}")
    print(f"GT01_AFTER={summary['n_abs_smd_weighted_gt_0.1']}")
    print(f"GT02_BEFORE={summary['n_abs_smd_unweighted_gt_0.2']}")
    print(f"GT02_AFTER={summary['n_abs_smd_weighted_gt_0.2']}")
    print(f"REPORT={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
