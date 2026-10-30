"""Task E: overlap diagnostics for ATT vs ATU weights, and aligned trimming.
The trimming region R is a fixed function of X (propensity model fit once on the full concordant
subset, applied to all 2,200 people). The QED estimand and the RCT benchmark are both restricted to R,
so the two sides still target the same population after trimming."""
import warnings, json
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
import wsc_lib as L
warnings.filterwarnings("ignore")
d = L.load(); P = json.load(open("results/point.json"))
c = L.qed_subset(d)
X, t, y = c[L.COVS_FULL].values, c.mathGrp.values, c.mathPost.values.astype(float)

# --- 1. weight diagnostics: ATT (control weights) vs ATU (treated weights), cross-fitted e
e, _, _ = L.crossfit_nuisance(X, t, y)
ess = lambda w: w.sum() ** 2 / (w ** 2).sum()
w_att = (e / (1 - e))[t == 0]; w_atu = ((1 - e) / e)[t == 1]
pooled_sd = c[L.COVS_FULL].std(ddof=1)


def wsmd(wt_ref_mask, w_other, other_mask):
    """SMD between the target group (unweighted) and the reweighted other group."""
    ref = c[L.COVS_FULL].values[wt_ref_mask].mean(0)
    oth = np.average(c[L.COVS_FULL].values[other_mask], axis=0, weights=w_other)
    return pd.Series((oth - ref) / pooled_sd.values, index=L.COVS_FULL)


smd_att = wsmd(t == 1, w_att, t == 0)   # reweighted controls vs treated
smd_atu = wsmd(t == 0, w_atu, t == 1)   # reweighted treated vs controls
diag = dict(
    att=dict(n_reweighted=int((t == 0).sum()), ess=ess(w_att), max_w=w_att.max(),
             top10_weight_share=np.sort(w_att)[::-1][:10].sum() / w_att.sum(),
             max_abs_smd=smd_att.abs().max(), n_smd_gt_0_1=int((smd_att.abs() > .1).sum())),
    atu=dict(n_reweighted=int((t == 1).sum()), ess=ess(w_atu), max_w=w_atu.max(),
             top10_weight_share=np.sort(w_atu)[::-1][:10].sum() / w_atu.sum(),
             max_abs_smd=smd_atu.abs().max(), n_smd_gt_0_1=int((smd_atu.abs() > .1).sum())))
pd.DataFrame({"smd_att_weighted": smd_att, "smd_atu_weighted": smd_atu}).to_csv("results/balance_att_vs_atu.csv")
print(json.dumps(diag, indent=1, default=float))
print("ATU weighted SMD > 0.1:\n", smd_atu[smd_atu.abs() > .1].round(3).to_string())

# --- 2. aligned trimming
ps_full = LogisticRegression(C=np.inf, solver="newton-cholesky", max_iter=100000, tol=1e-12).fit(X, t)
d["e_full"] = ps_full.predict_proba(d[L.COVS_FULL].values)[:, 1]
# Crump et al. (2009) alpha, computed on the concordant subset
g = 1 / (d.loc[c.index, "e_full"] * (1 - d.loc[c.index, "e_full"]))
alpha = 0.1
for a in np.linspace(0.001, 0.499, 4000):
    thr = 1 / (a * (1 - a))
    if thr <= 2 * g[g <= thr].mean():
        alpha = a; break
REGIONS = {"none": (0.0, 1.0), "[0.05,0.95]": (0.05, 0.95), "[0.1,0.9]": (0.1, 0.9),
           f"Crump α={alpha:.3f}": (alpha, 1 - alpha)}


def trimmed_point(dd, lo, hi, groups_col=None):
    r = dd[(dd.e_full >= lo) & (dd.e_full <= hi)]
    rp = L.rct_parts(r)
    cc = L.qed_subset(r)
    gg = cc[groups_col].values if groups_col else None
    q = L.aipw_all(cc[L.COVS_FULL].values, cc.mathGrp.values, cc[L.Y].values, groups=gg)
    return dict(n_total=len(r), n_G1=int((r.cell == "G1").sum()), n_G4=int((r.cell == "G4").sum()),
                rct_tau1=rp["tau1"], rct_tau0=rp["tau0"], rct_tauC=rp["tau_C"],
                qed_att=q["att"], qed_atu=q["atu"], qed_ate=q["ate"],
                d_att=q["att"] - rp["tau1"], d_atu=q["atu"] - rp["tau0"], d_ate=q["ate"] - rp["tau_C"])


def boot_region(lo, hi, b):
    rng = np.random.default_rng([L.SEED, 11, b])
    idx = np.concatenate([rng.choice(np.flatnonzero(d.cell.values == g), (d.cell == g).sum(), replace=True)
                          for g in ["G1", "G2", "G3", "G4"]])
    return trimmed_point(d.iloc[idx].reset_index(drop=True), lo, hi, groups_col="pid")


rows = []
for name, (lo, hi) in REGIONS.items():
    p = trimmed_point(d, lo, hi)
    bt = pd.DataFrame(Parallel(n_jobs=-1)(delayed(boot_region)(lo, hi, b) for b in range(1000)))
    for k in ["d_att", "d_atu", "d_ate"]:
        p[k + "_lo"] = bt[k].quantile(.025); p[k + "_hi"] = bt[k].quantile(.975)
    p["region"] = name
    rows.append(p)
tab = pd.DataFrame(rows).set_index("region")
tab.to_csv("results/trimming.csv")
json.dump(dict(diag=diag, crump_alpha=alpha), open("results/overlap_summary.json", "w"), indent=1, default=float)
print(tab.round(3).T.to_string())
