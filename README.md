# Observational causal validation

Independent reproduction and validation study. This repository does not implement a new causal estimator.

Fresh L1 target: randomized difference in means for Young et al.'s continuous outcome on the published `Outcome_Continuous <= 44` sample. Cached doubly robust files are recorded separately and are not fresh estimates.

Confirmatory WSC analysis is locked until `configs/OWNER_APPROVAL.json` matches `configs/frozen_protocol.yaml`. The project does not create that approval file.

Run on the remote machine from this directory:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m p4_causal_validation.experiments.reproduce_young --config configs/reproduction.yaml
```

Set `PYTHONPATH` to `src` before those commands.
