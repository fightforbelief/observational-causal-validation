"""Descriptive multiplicity check over the six exploratory aligned contrasts (two outcomes x ATT/ATU/ATE).
Bonferroni-level percentile intervals from the joint bootstraps. Writes results/multiplicity.json."""
import json
import pandas as pd
a = 0.05 / 6
out = {}
d = pd.read_csv("data/WSCdata.csv")
out["sd_mathPost"] = float(d.mathPost.std()); out["sd_vocabPost"] = float(d.vocabPost.std())
for outcome, path in [("mathPost", "results/boot.csv"), ("vocabPost", "results/vocab/boot_joint.csv")]:
    b = pd.read_csv(path)
    for k in ["d_att", "d_atu", "d_ate_aligned"]:
        x = b[k].dropna()
        out[f"{outcome}_{k}"] = dict(lo_bonf=float(x.quantile(a / 2)), hi_bonf=float(x.quantile(1 - a / 2)),
                                     share_le_0=float((x <= 0).mean()))
json.dump(out, open("results/multiplicity.json", "w"), indent=1)
print(json.dumps(out, indent=1))
