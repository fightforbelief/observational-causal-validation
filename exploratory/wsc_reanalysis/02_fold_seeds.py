"""Task B: sensitivity of cross-fitted AIPW to the random fold split (original data, 200 seeds)."""
import warnings, json
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import wsc_lib as L
warnings.filterwarnings("ignore")
d = L.load()
seeds = list(range(1, 201))
res = Parallel(n_jobs=-1)(delayed(L.qed_estimates)(d, seed=s) for s in seeds)
df = pd.DataFrame(res)[["att", "atu", "ate", "se_att_paper", "se_atu", "se_ate"]]
df.insert(0, "seed", seeds)
df.to_csv("results/fold_seeds.csv", index=False)
summ = {k: dict(median=df[k].median(), min=df[k].min(), max=df[k].max(), sd=df[k].std(),
                p05=df[k].quantile(.05), p95=df[k].quantile(.95)) for k in ["att", "atu", "ate"]}
# Chernozhukov et al. (2018) median-over-splits variance: median of (se^2 + (est - median)^2)
for k, s in [("att", "se_att_paper"), ("atu", "se_atu"), ("ate", "se_ate")]:
    summ[k]["median_split_se"] = float(np.sqrt(np.median(df[s] ** 2 + (df[k] - df[k].median()) ** 2)))
json.dump(summ, open("results/fold_seeds_summary.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
