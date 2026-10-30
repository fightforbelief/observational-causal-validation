"""Task C (exploratory): which covariates remove selection bias? Nested sets + leave-one-out.
Each estimate is the median over 50 fold seeds, to separate covariate effects from split noise."""
import warnings, json
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import wsc_lib as L
warnings.filterwarnings("ignore")
d = L.load(); P = json.load(open("results/point.json"))
SEEDS = range(1, 51)
DEMO = ["female", "white", "black", "asian", "hisp", "married", "logAge", "income",
        "collegeS", "collegeM", "collegeD"]
PRE = ["mathPre", "vocabPre"]
MOTIV = ["mathLike", "calc", "AMAS", "MCS"]
PSYCH = ["big5O", "big5C", "big5E", "big5A", "big5N", "logBDI", "GSES", "logBooks"]
SETS = {"0 none (raw diff)": [], "1 demographics": DEMO, "2 +pretests": DEMO + PRE,
        "3 +math motivation": DEMO + PRE + MOTIV, "4 full (+psych)": DEMO + PRE + MOTIV + PSYCH}
assert sorted(SETS["4 full (+psych)"]) == sorted(L.COVS_FULL)


def med(covs):
    if not covs:
        c = L.qed_subset(d); t = c.mathGrp.values; y = c.mathPost.values
        diff = y[t == 1].mean() - y[t == 0].mean()
        return dict(att=diff, atu=diff, ate=diff)
    r = pd.DataFrame([L.qed_estimates(d, covs=covs, seed=s) for s in SEEDS])
    return dict(att=r.att.median(), atu=r.atu.median(), ate=r.ate.median())


rows = Parallel(n_jobs=-1)(delayed(med)(c) for c in SETS.values())
nested = pd.DataFrame(rows, index=list(SETS))
nested["bias_att"] = nested.att - P["rct_tau1"]
nested["bias_atu"] = nested.atu - P["rct_tau0"]
nested["bias_ate"] = nested.ate - P["rct_tauC"]
nested.to_csv("results/covsets_nested.csv")
print(nested.round(3).to_string())

loo_rows = Parallel(n_jobs=-1)(delayed(med)([c for c in L.COVS_FULL if c != v]) for v in L.COVS_FULL)
loo = pd.DataFrame(loo_rows, index=L.COVS_FULL)
full = nested.loc["4 full (+psych)"]
for k in ["att", "atu", "ate"]:
    loo[f"shift_{k}"] = loo[k] - full[k]
loo.to_csv("results/covsets_loo.csv")
print(loo.sort_values("shift_atu", key=abs, ascending=False).round(3).head(10).to_string())
