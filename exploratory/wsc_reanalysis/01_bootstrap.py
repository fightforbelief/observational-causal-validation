"""Task A + B: all point estimates and a single joint, cell-stratified bootstrap.

Every replicate resamples participants with replacement within G1..G4 (cell sizes fixed),
recomputes RCT pieces, refits cross-fitted AIPW (copies of one participant share a fold),
and records all estimates and contrasts. Outputs: results/point.json, results/boot.csv
"""
import json, os, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import wsc_lib as L

warnings.filterwarnings("ignore")
B = 2000
os.makedirs("results", exist_ok=True)
d = L.load()
point = L.point_all(d)
json.dump({k: float(v) for k, v in point.items()}, open("results/point.json", "w"), indent=1)


def one(b):
    rng = np.random.default_rng([L.SEED, b])
    idx = np.concatenate([rng.choice(np.flatnonzero(d.cell.values == g), (d.cell == g).sum(), replace=True)
                          for g in ["G1", "G2", "G3", "G4"]])
    db = d.iloc[idx].reset_index(drop=True)
    try:
        return L.point_all(db, groups_col="pid")
    except Exception as e:  # recorded, never silently replaced
        return {"error": repr(e)}


reps = Parallel(n_jobs=-1)(delayed(one)(b) for b in range(B))
boot = pd.DataFrame(reps)
boot.to_csv("results/boot.csv", index=False)
print("failures:", boot["error"].notna().sum() if "error" in boot else 0)
