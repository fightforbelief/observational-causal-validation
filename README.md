# Observational causal validation

Independent reproduction and validation study. This repository does not implement a new causal estimator.

The study is closed for the completed comparisons. The paper is [`paper.pdf`](paper.pdf). Decision recorded on 2026-10-28: the finished primary comparison and one full-sample ATE secondary comparison can support a bounded manuscript. They do not answer how far estimand alignment brings an observational analysis back to the randomized benchmark in this preference-based within-study comparison.

Pinned primary results on `mathPost`: the G1∪G2 randomized mean difference is 0.794681731104145 and the G1∪G4 linear cross-fit ATT is 0.7942342079185714. Their difference is -0.00044752318557361725. The 2,000-replicate joint bootstrap interval is [-0.9491600721638183, 0.6638487257328824]. The secondary full-sample Neyman ATE is 0.7603449645239859 and the concordant-subset linear AIPW ATE is 1.0753974159386432. Their difference is 0.3150524514146573, with bootstrap interval [-0.6249039850074877, 1.1444295170363872]. Both bootstrap runs completed 2,000 replicates with zero failures. Close point estimates are not equivalence, and the two comparisons do not estimate one shared confounding bias. L3, `vocabPost`, and ATU were not run.

## Run

From this directory, with `PYTHONPATH=src`:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m p4_causal_validation.experiments.reproduce_young --config configs/reproduction.yaml
```

Confirmatory entry points live in `src/p4_causal_validation/experiments/`. Raw WSC data and run artifacts are not in this repository.
