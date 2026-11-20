# vocabPost 次要结局：分析规格

日期：2026-11-20。本文件在计算任何 `vocabPost` 估计之前写定。写定时 `mathPost` 的全部结果已经公开在手稿里，所以这不是预注册。它的作用是把 `vocabPost` 分析的每个选择都钉在 `mathPost` 已用的规格上，不留看完结果再选的余地。

依据：`configs/frozen_protocol.yaml` 把 `vocabPost` 列为次要结局，处理指示写为 `1 - mathGrp`，并禁止任何保留 `mathGrp` 为处理的 `vocabPost` 对比。研究协议草稿第 3 节和 H1 也写了两个结局分开分析。

## 编码

- 结果 `Y = vocabPost`。
- 处理 `D_v = 1 - mathGrp`（分到词汇训练）。
- 偏好 `S_v = 1 - mathSel`（偏好词汇训练）。
- 新编码下的单元格：G1v = 原 G4（偏好词汇、分到词汇，817 人）；G2v = 原 G3（偏好词汇、分到数学，776 人）；G3v = 原 G2（319 人）；G4v = 原 G1（288 人）。
- concordant 集合不变，仍是原 G1 ∪ G4 的 1,105 人。

## 估计量（全部沿用 mathPost）

- 随机化：未调整均值差。`τ1v` 用 G1v 对 G2v（1,593 人）；`τ0v` 用 G3v 对 G4v（607 人）；全样本均值差用 2,200 人。
- 观察性：G1v ∪ G4v 上 5 折交叉拟合 AIPW，处理为 `D_v`。倾向分为无惩罚 logistic（newton-cholesky，tol 1e-12），结果模型为分组 OLS，截断 [1e-6, 1-1e-6]，不修剪。fold 种子 20261026。
- 协变量：与 mathPost 相同的 25 个，`S_v` 不进入。`vocabPre` 和 `mathPre` 都保留。
- 对齐配对：ATT 对 `τ1v`；ATU 对 `τ0v`；ATE 对 `τC_v = π_c,v τ1v + (1-π_c,v) τ0v`；分解同 mathPost 的恒等式。
- 实现：`exploratory/wsc_reanalysis/wsc_lib.py`，不改动。脚本只在数据上重编码列。

## 不确定性

1. ATT 对比的配对 bootstrap：2,000 次，在 G1v、G2v、G4v 内按原人数有放回重抽，G3v 不进入，QED 和 RCT 用同一个 G1v 抽样；随机数 `default_rng([20261026, 1, b])`。报告 2.5% 与 97.5% 百分位。这是对 mathPost 主区间程序的结构性镜像，不是同一个实现。
2. 四格联合 bootstrap：2,000 次，所有单元格按原人数重抽；随机数 `default_rng([20261026, 2, b])`；用于 ATU、ATE、分解各项。失败的复制保留并计数。
3. 200 个 fold 种子（1–200），报告中位数、分位数和 Chernozhukov 等（2018）的中位数分割 SE。

## 解释规则

- 不设等价界，不做等价结论。区间重叠不是等价。
- 不把 ATT 或 ATU 的点差写成已确认的混杂或已确认的估计失败。
- 不根据 vocabPost 结果回头修改 mathPost 的任何分析或表述的主次地位。
- 两个结局不合并、不做多重比较校正后的“联合”结论；手稿把 vocabPost 写成次要结局的复制。
- 如果某个 fold 丢掉一个臂，按协议停止，不改折数。
