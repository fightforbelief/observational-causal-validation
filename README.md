# Observational causal validation

Independent reproduction and validation study. This repository does not implement a new causal estimator.

The revised paper is [`paper.pdf`](paper.pdf). Source is in `manuscript/revision_20261030/`. Author: Hanqing Yang.

The prespecified primary comparison on `mathPost` pairs the randomized effect among mathematics-preferring participants with a cross-fitted AIPW ATT in the concordant subset. The estimates are 0.794681731104145 and 0.7942342079185714. Their difference is -0.00044752318557361725. The prespecified 2,000-replicate percentile interval is [-0.9491600721638183, 0.6638487257328824]. The point estimates are close. The interval leaves room for large discrepancies in either direction, so this is not evidence that the observational procedure recovered the randomized effect.

A later decomposition of the full-sample observational-minus-randomized discrepancy, 0.315 points, assigns 0.288 to the gap between the observational average treatment effect and the concordant benchmark, -0.00016 to estimand composition, and 0.027 to the difference between a post-stratified randomized benchmark and the unadjusted full-sample difference in means. The concordant benchmark can be constructed for any common assignment probability strictly between 0 and 1. It equals the full-population average treatment effect if and only if that probability is one half or the two preference-group effects are equal. The composition term is small here because its preference-weight factor is fixed and small. The gap from the concordant benchmark is not a pure measure of confounding. Concordant ATE and the decomposition are subsequent analyses. Cross-fitting seeds, Lin adjustment, covariate subsets, sensitivity bounds, and trimming are exploratory. The study is not fully preregistered. L3 was not run.

## Run

From this directory, with `PYTHONPATH=src`:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m p4_causal_validation.experiments.reproduce_young --config configs/reproduction.yaml
```

Exploratory reanalysis scripts are in `exploratory/wsc_reanalysis/`. Raw WSC data and bootstrap dumps are not in this repository.
