"""Core estimators for the four-arm WSC reanalysis (mathPost outcome).

Cells: G1 = (Sel=1, Grp=1), G2 = (Sel=1, Grp=0), G3 = (Sel=0, Grp=1), G4 = (Sel=0, Grp=0).
Concordant QED subset C = G1 ∪ G4 (treatment = mathGrp = mathSel).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import KFold

SEED = 20261026
Y = "mathPost"
COVS_FULL = ["female", "white", "black", "asian", "hisp", "married", "logAge", "income",
             "collegeS", "collegeM", "collegeD", "calc", "logBooks", "mathLike",
             "big5O", "big5C", "big5E", "big5A", "big5N", "AMAS", "logBDI", "MCS", "GSES",
             "vocabPre", "mathPre"]
CLIP = 1e-6


def load(path="data/WSCdata.csv"):
    d = pd.read_csv(path)
    cell = np.select(
        [(d.mathSel == 1) & (d.mathGrp == 1), (d.mathSel == 1) & (d.mathGrp == 0),
         (d.mathSel == 0) & (d.mathGrp == 1), (d.mathSel == 0) & (d.mathGrp == 0)],
        ["G1", "G2", "G3", "G4"], default="NA")
    d["cell"] = cell
    d["pid"] = np.arange(len(d))
    return d


# ---------------------------------------------------------------- RCT pieces
def diff_means(y, t):
    y = np.asarray(y, float); t = np.asarray(t, int)
    y1, y0 = y[t == 1], y[t == 0]
    est = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    return est, se


def rct_parts(d):
    """tau1 (S=1), tau0 (S=0), full-sample ATE, pi_c, aligned concordant ATE benchmark."""
    s1, s0 = d[d.mathSel == 1], d[d.mathSel == 0]
    tau1, _ = diff_means(s1[Y], s1.mathGrp)
    tau0, _ = diff_means(s0[Y], s0.mathGrp)
    full, _ = diff_means(d[Y], d.mathGrp)
    n = d.cell.value_counts()
    pi_c = n.get("G1", 0) / (n.get("G1", 0) + n.get("G4", 0))
    pi = (d.mathSel == 1).mean()
    return dict(tau1=tau1, tau0=tau0, full=full, pi_c=pi_c, pi=pi,
                tau_C=pi_c * tau1 + (1 - pi_c) * tau0)


# ---------------------------------------------------------------- AIPW
def _folds(n, seed, groups=None, k=5):
    """Fold id per row. If groups given (bootstrap pid), copies of one person share a fold."""
    if groups is None:
        f = np.empty(n, int)
        for j, (_, te) in enumerate(KFold(k, shuffle=True, random_state=seed).split(np.zeros(n))):
            f[te] = j
        return f
    ug, inv = np.unique(groups, return_inverse=True)
    fu = np.empty(len(ug), int)
    for j, (_, te) in enumerate(KFold(k, shuffle=True, random_state=seed).split(np.zeros(len(ug)))):
        fu[te] = j
    return fu[inv]


def crossfit_nuisance(X, t, y, seed=SEED, groups=None, k=5, ps_penalty=None):
    n = len(y)
    f = _folds(n, seed, groups, k)
    e = np.empty(n); m1 = np.empty(n); m0 = np.empty(n)
    for j in range(k):
        tr, te = f != j, f == j
        lr = LogisticRegression(C=np.inf if ps_penalty is None else 1.0, solver="newton-cholesky", max_iter=100000, tol=1e-12)
        lr.fit(X[tr], t[tr])
        e[te] = lr.predict_proba(X[te])[:, 1]
        o1 = LinearRegression().fit(X[tr & (t == 1)], y[tr & (t == 1)])
        o0 = LinearRegression().fit(X[tr & (t == 0)], y[tr & (t == 0)])
        m1[te] = o1.predict(X[te]); m0[te] = o0.predict(X[te])
    return np.clip(e, CLIP, 1 - CLIP), m1, m0


def aipw_all(X, t, y, seed=SEED, groups=None, ps_penalty=None, return_nuis=False):
    """Cross-fitted AIPW ATE, ATT, ATU with influence-function SEs."""
    X = np.asarray(X, float); t = np.asarray(t, int); y = np.asarray(y, float)
    n = len(y)
    e, m1, m0 = crossfit_nuisance(X, t, y, seed, groups, ps_penalty=ps_penalty)
    p = t.mean()
    # ATE
    psi_ate = m1 - m0 + t * (y - m1) / e - (1 - t) * (y - m0) / (1 - e)
    ate = psi_ate.mean()
    # ATT (paper eq. 7)
    psi_att = t / p * (y - m0) - (1 - t) * e / (p * (1 - e)) * (y - m0)
    att = psi_att.mean()
    # ATU (mirror image: uses mu1 for treated -> control extrapolation)
    q = 1 - p
    psi_atu = (1 - t) / q * (m1 - y) + t * (1 - e) / (q * e) * (y - m1)
    atu = psi_atu.mean()
    # IF-based SEs (centered scores; for ATT/ATU include the p-hat term)
    se_ate = psi_ate.std(ddof=1) / np.sqrt(n)
    if_att = psi_att - t / p * att
    if_atu = psi_atu - (1 - t) / q * atu
    se_att = if_att.std(ddof=1) / np.sqrt(n)
    se_att_paper = psi_att.std(ddof=1) / np.sqrt(n)  # paper's convention (raw score sd)
    se_atu = if_atu.std(ddof=1) / np.sqrt(n)
    out = dict(ate=ate, att=att, atu=atu, se_ate=se_ate, se_att=se_att,
               se_att_paper=se_att_paper, se_atu=se_atu)
    if return_nuis:
        out.update(e=e, m1=m1, m0=m0)
    return out


def qed_subset(d):
    return d[d.cell.isin(["G1", "G4"])]


def qed_estimates(d, covs=COVS_FULL, seed=SEED, groups_col=None, **kw):
    c = qed_subset(d)
    g = c[groups_col].values if groups_col else None
    return aipw_all(c[covs].values, c.mathGrp.values, c[Y].values, seed=seed, groups=g, **kw)


def lin_adjusted(y, t, X):
    """Lin (2013) regression-adjusted difference in means with HC2 SE.
    Regress y on t, centered X, and t * centered X."""
    y = np.asarray(y, float); t = np.asarray(t, float); X = np.asarray(X, float)
    Xc = X - X.mean(0)
    Z = np.column_stack([np.ones(len(y)), t, Xc, t[:, None] * Xc])
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    r = y - Z @ beta
    ZtZi = np.linalg.pinv(Z.T @ Z)
    h = np.einsum("ij,jk,ik->i", Z, ZtZi, Z)
    meat = (Z * (r ** 2 / (1 - h))[:, None]).T @ Z
    V = ZtZi @ meat @ ZtZi
    return beta[1], np.sqrt(V[1, 1])


def point_all(d, covs=COVS_FULL, seed=SEED, groups_col=None):
    """Every point estimate used in the reanalysis, on one (possibly resampled) dataset."""
    r = rct_parts(d)
    q = qed_estimates(d, covs=covs, seed=seed, groups_col=groups_col)
    s1, s0 = d[d.mathSel == 1], d[d.mathSel == 0]
    r["tau1_lin"] = lin_adjusted(s1[Y], s1.mathGrp, s1[covs].values)[0]
    r["tau0_lin"] = lin_adjusted(s0[Y], s0.mathGrp, s0[covs].values)[0]
    r["tauC_lin"] = r["pi_c"] * r["tau1_lin"] + (1 - r["pi_c"]) * r["tau0_lin"]
    out = dict(rct_tau1=r["tau1"], rct_tau0=r["tau0"], rct_full=r["full"], rct_tauC=r["tau_C"],
               pi=r["pi"], pi_c=r["pi_c"],
               rct_tau1_lin=r["tau1_lin"], rct_tau0_lin=r["tau0_lin"], rct_tauC_lin=r["tauC_lin"],
               qed_att=q["att"], qed_atu=q["atu"], qed_ate=q["ate"])
    # contrasts
    out["d_att"] = out["qed_att"] - out["rct_tau1"]              # primary (paper)
    out["d_atu"] = out["qed_atu"] - out["rct_tau0"]              # new
    out["d_ate_aligned"] = out["qed_ate"] - out["rct_tauC"]      # new aligned
    out["d_ate_misaligned"] = out["qed_ate"] - out["rct_full"]   # paper secondary
    out["rct_ps"] = r["pi"] * r["tau1"] + (1 - r["pi"]) * r["tau0"]   # post-stratified full ATE
    # exact decomposition: d_ate_misaligned = qed_error + estimand_misalignment + chance_S_imbalance
    out["qed_error"] = out["qed_ate"] - out["rct_tauC"]
    out["estimand_misalignment"] = out["rct_tauC"] - out["rct_ps"]
    out["chance_S_imbalance"] = out["rct_ps"] - out["rct_full"]
    out["misalign_formula"] = (out["pi_c"] - out["pi"]) * (out["rct_tau1"] - out["rct_tau0"])
    return out
