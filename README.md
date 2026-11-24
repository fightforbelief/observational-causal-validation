# Checking Counterfactual Predictions in a Four-Arm Within-Study Comparison

Yunjie Guo and Hanqing Yang · Independent Researcher, Suzhou, China

[Article](paper.pdf) · [Supplement](manuscript/submission/supplementary_material.pdf) · [LaTeX source](manuscript/submission) · [Vocabulary analysis specification](exploratory/wsc_reanalysis/VOCAB_SPEC.md)

The new analysis uses participants assigned opposite to preference to check counterfactual predictions fitted in the concordant cells. For mathematics ATT the original gap of −0.00045 points splits into +0.086 points of signed prediction error and −0.087 points from the difference in covariate predictions across the two target-stratum cells. Among 319 validation participants, the high-minus-low pre-test residual difference is 0.660 points. Its eight-way adjusted interval is [0.093, 1.227] for the fitted rule and [0.022, 1.298] when allowing the source slopes to vary. A linear change of 0.405 points RMS on the target concordant sample suffices to erase the observed gradient at exactly the same aggregate effect. These are exploratory checks on a previously analyzed trial.

[Analysis specification](exploratory/discordant_validation/README.md) · [Empirical estimates and simulation results](exploratory/discordant_validation/published)

Using the public four-arm mathematics and vocabulary trial of Keller et al., this study constructs a randomized benchmark for participants who received their preferred session. A finite-sample decomposition measures how much of an observational–experimental discrepancy comes from differences in the target populations’ preference composition.

In the prespecified mathematics comparison, the observational and randomized treated-effect estimates were 0.794 and 0.795 points. The 95% paired-bootstrap interval for their difference was [−0.949, 0.664]. Preference composition accounted for −0.00016 points of a 0.315-point full-sample gap. For vocabulary, the largest aligned discrepancy was 0.913 points among mathematics-preferring participants. Its marginal 95% interval was [0.093, 1.896] and its interval adjusted for six exploratory contrasts was [−0.193, 2.413]. The vocabulary specification was written after the mathematics results were known.

## Reproduce the paper exhibits

The checked environment uses Python 3.12.10 on Windows and the pinned dependencies in `requirements-dev.txt`. From the repository root, create a project virtual environment on first use, download the original R data file, and choose a fresh output directory. The script verifies the input SHA-256 before reading it. The linked author-repository commit contains the same R data bytes as CRAN WSCdata 0.1.2.

```powershell
py -3.12 -m venv .venv
$py = '.\.venv\Scripts\python.exe'
& $py -m pip install -r requirements-dev.txt
$env:PYTHONPATH = 'src'
& $py -m pytest -q
Invoke-WebRequest 'https://raw.githubusercontent.com/jzangela/WSCdata/6360a5247567c0ff632cd12aa2f6b3cfff92b2b6/data/WSCdata.rda' -OutFile WSCdata.rda
& $py scripts/reproduce_manuscript.py WSCdata.rda outputs/paper-reproduction
```

The reproduction runs scripts 01–12 and a 15,000-trial known-truth simulation on a new copy of the analysis files. It writes tables and replicate-level outputs under `results/` and five figures under `figures/` within the output directory. It checks the six observed contrasts and their marginal intervals against the archived table, all six adjusted intervals, the number of successful redraws, and the decomposition in every joint-bootstrap replicate for both outcomes. It also verifies each signed error decomposition against the original contrast, all eight fixed-rule intervals, all four source-inclusive intervals, the sharp shape bounds, and the known-truth results. Some bootstrap draws have extreme weights, so the reported standard deviations and percentile intervals describe different aspects of the distribution.

The original prespecified mathematics bootstrap used three design cells. The first script rebuilds a later four-cell bootstrap. For the primary interval in the main figure, script 08 instead takes percentiles of the 2,000 archived paired contrasts. The archive contains one row per draw with the two estimates and their difference. The original summary provides the source report and protocol hashes. The full outcome and covariate data come from the upstream authors.

[Original study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12959902/) · [Archived numerical outputs](exploratory/wsc_reanalysis/published) · [Frozen protocol](configs/frozen_protocol.yaml)

## Build the documents

From `manuscript/submission/`, compile `main.tex` and `supplementary_material.tex` separately with Tectonic or LaTeX and BibTeX. The journal style file is distributed in the [Observational Studies author template](https://obs.scholasticahq.com/for-authors).

The prior paper sources remain in `manuscript/revision_20261030/`. The separate Young reproduction entry point is `python -m p4_causal_validation.experiments.reproduce_young --config configs/reproduction.yaml`.
