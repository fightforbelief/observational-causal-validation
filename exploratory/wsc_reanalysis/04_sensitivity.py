"""Task D: marginal sensitivity model (Tan 2006; Zhao, Small & Bhattacharya 2019) bounds for
Hajek-IPW ATT and ATU in the concordant subset, with percentile-bootstrap intervals for the bounds.
Under the MSM the true odds of treatment differ from the measured-covariate odds by at most a factor Λ,
so each control weight e/(1-e) (ATT) or treated weight (1-e)/e (ATU) can move within [w/Λ, wΛ]."""
import warnings, json
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import wsc_lib as L
warnings.filterwarnings("ignore")
LAMS = [1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
d = L.load(); P = json.load(open("results/point.json"))


def hajek_bounds(y, w, lam):
    """min and max of sum(h w y)/sum(h w) over h in [1/lam, lam] (closed form via sorting)."""
    o = np.argsort(y); y, w = y[o], w[o]
    lo_w, hi_w = w / lam, w * lam
    n = len(y)
    # max: low weights on the k smallest y, high weights on the rest; scan k
    cl_lo, cl_hi = np.concatenate([[0], np.cumsum(lo_w)]), np.concatenate([[0], np.cumsum(hi_w)])
    cy_lo = np.concatenate([[0], np.cumsum(lo_w * y)]); cy_hi = np.concatenate([[0], np.cumsum(hi_w * y)])
    k = np.arange(n + 1)
    num_max = cy_lo[k] + (cy_hi[-1] - cy_hi[k]); den_max = cl_lo[k] + (cl_hi[-1] - cl_hi[k])
    num_min = cy_hi[k] + (cy_lo[-1] - cy_lo[k]); den_min = cl_hi[k] + (cl_lo[-1] - cl_lo[k])
    return (num_min / den_min).min(), (num_max / den_max).max()


def bounds(dd, groups_col=None):
    c = L.qed_subset(dd)
    X, t, y = c[L.COVS_FULL].values, c.mathGrp.values, c.mathPost.values.astype(float)
    g = c[groups_col].values if groups_col else None
    e, _, _ = L.crossfit_nuisance(X, t, y, groups=g)
    out = {}
    for lam in LAMS:
        m0lo, m0hi = hajek_bounds(y[t == 0], (e / (1 - e))[t == 0], lam)
        m1lo, m1hi = hajek_bounds(y[t == 1], ((1 - e) / e)[t == 1], lam)
        out[f"att_lo_{lam}"] = y[t == 1].mean() - m0hi; out[f"att_hi_{lam}"] = y[t == 1].mean() - m0lo
        out[f"atu_lo_{lam}"] = m1lo - y[t == 0].mean(); out[f"atu_hi_{lam}"] = m1hi - y[t == 0].mean()
    return out


pt = bounds(d)


def one(b):
    rng = np.random.default_rng([L.SEED, 7, b])
    idx = np.concatenate([rng.choice(np.flatnonzero(d.cell.values == g), (d.cell == g).sum(), replace=True)
                          for g in ["G1", "G2", "G3", "G4"]])
    return bounds(d.iloc[idx].reset_index(drop=True), groups_col="pid")


bt = pd.DataFrame(Parallel(n_jobs=-1)(delayed(one)(b) for b in range(1000)))
rows = []
for lam in LAMS:
    for est in ["att", "atu"]:
        rows.append(dict(Lambda=lam, estimand=est.upper(), bound_lo=pt[f"{est}_lo_{lam}"], bound_hi=pt[f"{est}_hi_{lam}"],
                         ci_lo=bt[f"{est}_lo_{lam}"].quantile(.025), ci_hi=bt[f"{est}_hi_{lam}"].quantile(.975),
                         rct=P["rct_tau1"] if est == "att" else P["rct_tau0"]))
tab = pd.DataFrame(rows)
tab["rct_inside_bounds"] = (tab.bound_lo <= tab.rct) & (tab.rct <= tab.bound_hi)
tab.to_csv("results/sensitivity.csv", index=False)
print(tab.round(3).to_string())

# smallest Λ at which the IPW ATU bounds reach the RCT tau0 (fine grid, point level)
grid = np.round(np.arange(1.0, 3.001, 0.01), 2)
c = L.qed_subset(d); X, t, y = c[L.COVS_FULL].values, c.mathGrp.values, c.mathPost.values.astype(float)
e, _, _ = L.crossfit_nuisance(X, t, y)
lam_star = None
for lam in grid:
    m1lo, _ = hajek_bounds(y[t == 1], ((1 - e) / e)[t == 1], lam)
    if m1lo - y[t == 0].mean() <= P["rct_tau0"]:
        lam_star = float(lam); break
json.dump(dict(lambda_star_atu=lam_star, ipw_att=pt["att_lo_1.0"], ipw_atu=pt["atu_lo_1.0"]),
          open("results/sensitivity_summary.json", "w"), indent=1)
print("Lambda* (ATU bounds reach RCT tau0):", lam_star, "| IPW ATT/ATU at Λ=1:", pt["att_lo_1.0"], pt["atu_lo_1.0"])
