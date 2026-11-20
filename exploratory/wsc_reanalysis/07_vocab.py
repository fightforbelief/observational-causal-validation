"""Secondary outcome vocabPost, analysed exactly as mathPost after recoding (see VOCAB_SPEC.md).

Recoding: treatment D_v = 1 - mathGrp, preference S_v = 1 - mathSel, outcome vocabPost.
wsc_lib is used unchanged; the recoded frame stores D_v and S_v in the mathGrp/mathSel
columns so every wsc_lib function applies without edits, and Y is set inside each worker.

Outputs (results/vocab/): point.json, boot_paired.csv, boot_joint.csv, fold_seeds.csv,
summary.json, run_meta.json
"""
import json, os, sys, time, platform, hashlib, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import sklearn
import wsc_lib as L

warnings.filterwarnings("ignore")
B = 2000
OUT = "results/vocab"
os.makedirs(OUT, exist_ok=True)


def recode(d):
    v = d.copy()
    v["mathGrp"] = 1 - d["mathGrp"]
    v["mathSel"] = 1 - d["mathSel"]
    v["cell"] = np.select(
        [(v.mathSel == 1) & (v.mathGrp == 1), (v.mathSel == 1) & (v.mathGrp == 0),
         (v.mathSel == 0) & (v.mathGrp == 1), (v.mathSel == 0) & (v.mathGrp == 0)],
        ["G1", "G2", "G3", "G4"], default="NA")
    return v


def setY():
    L.Y = "vocabPost"


def paired_one(b, d):
    """Mirror of the prespecified primary bootstrap: redraw G1, G2, G4 (recoded), hold out G3."""
    setY()
    rng = np.random.default_rng([L.SEED, 1, b])
    try:
        g1 = d[d.cell == "G1"]; g2 = d[d.cell == "G2"]; g4 = d[d.cell == "G4"]
        b1 = g1.iloc[rng.choice(len(g1), len(g1), replace=True)]
        b2 = g2.iloc[rng.choice(len(g2), len(g2), replace=True)]
        b4 = g4.iloc[rng.choice(len(g4), len(g4), replace=True)]
        rct = L.diff_means(pd.concat([b1, b2])[L.Y], pd.concat([b1, b2]).mathGrp)[0]
        c = pd.concat([b1, b4])
        q = L.aipw_all(c[L.COVS_FULL].values, c.mathGrp.values, c[L.Y].values, groups=c.pid.values)
        return dict(rct_tau1=rct, qed_att=q["att"], d_att=q["att"] - rct)
    except Exception as e:
        return {"error": repr(e)}


def joint_one(b, d):
    setY()
    rng = np.random.default_rng([L.SEED, 2, b])
    idx = np.concatenate([rng.choice(np.flatnonzero(d.cell.values == g), (d.cell == g).sum(), replace=True)
                          for g in ["G1", "G2", "G3", "G4"]])
    db = d.iloc[idx].reset_index(drop=True)
    try:
        return L.point_all(db, groups_col="pid")
    except Exception as e:
        return {"error": repr(e)}


def seed_one(s, d):
    setY()
    return L.qed_estimates(d, seed=s)


def summ(x):
    x = pd.Series(x).dropna()
    return dict(n=int(len(x)), mean=float(x.mean()), sd=float(x.std()),
                p025=float(x.quantile(.025)), p975=float(x.quantile(.975)),
                p05=float(x.quantile(.05)), p95=float(x.quantile(.95)),
                median=float(x.median()), iqr_sd=float((x.quantile(.75) - x.quantile(.25)) / 1.349),
                min=float(x.min()), max=float(x.max()), n_neg=int((x < 0).sum()),
                excess_kurtosis=float(x.kurt()))


if __name__ == "__main__":
    t0 = time.time()
    setY()
    raw = L.load()
    d = recode(raw)
    counts = d.cell.value_counts().to_dict()
    assert counts == {"G1": 817, "G2": 776, "G3": 319, "G4": 288}, counts
    # sanity: recoded concordant set is the original one
    assert set(d.pid[d.cell.isin(["G1", "G4"])]) == set(raw.pid[raw.cell.isin(["G1", "G4"])])

    point = L.point_all(d)
    # analytic SEs and overlap diagnostics on the point fit
    s1, s0 = d[d.mathSel == 1], d[d.mathSel == 0]
    point["rct_tau1_se"] = L.diff_means(s1[L.Y], s1.mathGrp)[1]
    point["rct_tau0_se"] = L.diff_means(s0[L.Y], s0.mathGrp)[1]
    point["rct_full_se"] = L.diff_means(d[L.Y], d.mathGrp)[1]
    c = L.qed_subset(d)
    q = L.aipw_all(c[L.COVS_FULL].values, c.mathGrp.values, c[L.Y].values, return_nuis=True)
    e, t = q["e"], c.mathGrp.values
    point["qed_att_se_score"] = q["se_att_paper"]; point["qed_att_se_if"] = q["se_att"]
    point["qed_atu_se_if"] = q["se_atu"]; point["qed_ate_se"] = q["se_ate"]
    w_att = (e / (1 - e))[t == 0]; w_atu = ((1 - e) / e)[t == 1]
    ess = lambda w: float(w.sum() ** 2 / (w ** 2).sum())
    point.update(ess_att_controls=ess(w_att), n_att_controls=int((t == 0).sum()),
                 ess_atu_treated=ess(w_atu), n_atu_treated=int((t == 1).sum()),
                 max_w_att=float(w_att.max()), max_w_atu=float(w_atu.max()),
                 top10_share_att=float(np.sort(w_att)[-10:].sum() / w_att.sum()),
                 top10_share_atu=float(np.sort(w_atu)[-10:].sum() / w_atu.sum()),
                 ps_min=float(e.min()), ps_max=float(e.max()),
                 n_ps_outside_01_09=int(((e < .1) | (e > .9)).sum()), n_concordant=int(len(e)))
    json.dump({k: float(v) for k, v in point.items()}, open(f"{OUT}/point.json", "w"), indent=1)
    print(json.dumps({k: round(float(v), 6) for k, v in point.items()}, indent=1), flush=True)

    paired = pd.DataFrame(Parallel(n_jobs=-1)(delayed(paired_one)(b, d) for b in range(B)))
    paired.to_csv(f"{OUT}/boot_paired.csv", index=False)
    joint = pd.DataFrame(Parallel(n_jobs=-1)(delayed(joint_one)(b, d) for b in range(B)))
    joint.to_csv(f"{OUT}/boot_joint.csv", index=False)
    seeds = list(range(1, 201))
    fs = pd.DataFrame(Parallel(n_jobs=-1)(delayed(seed_one)(s, d) for s in seeds))
    fs = fs[["att", "atu", "ate", "se_att_paper", "se_atu", "se_ate"]]
    fs.insert(0, "seed", seeds)
    fs.to_csv(f"{OUT}/fold_seeds.csv", index=False)

    S = {"paired_failures": int(paired["error"].notna().sum()) if "error" in paired else 0,
         "joint_failures": int(joint["error"].notna().sum()) if "error" in joint else 0}
    S["paired"] = {k: summ(paired[k]) for k in ["rct_tau1", "qed_att", "d_att"]}
    S["joint"] = {k: summ(joint[k]) for k in ["rct_tau1", "rct_tau0", "rct_tauC", "rct_full", "qed_att", "qed_atu",
                                             "qed_ate", "d_att", "d_atu", "d_ate_aligned", "d_ate_misaligned",
                                             "qed_error", "estimand_misalignment", "chance_S_imbalance"]}
    ident = (joint["qed_error"] + joint["estimand_misalignment"] + joint["chance_S_imbalance"]
             - joint["d_ate_misaligned"]).abs().max()
    S["joint_identity_max_abs_err"] = float(ident)
    S["fold_seeds"] = {}
    for k, s in [("att", "se_att_paper"), ("atu", "se_atu"), ("ate", "se_ate")]:
        x = fs[k]
        S["fold_seeds"][k] = dict(median=float(x.median()), p05=float(x.quantile(.05)), p95=float(x.quantile(.95)),
                                  min=float(x.min()), max=float(x.max()), sd=float(x.std()),
                                  median_split_se=float(np.sqrt(np.median(fs[s] ** 2 + (x - x.median()) ** 2))))
    S["fold_seeds"]["share_abs_att_contrast_lt_0.01"] = float(((fs.att - point["rct_tau1"]).abs() < .01).mean())
    S["fold_seeds"]["prespecified_seed_att_percentile"] = float((fs.att < point["qed_att"]).mean())
    json.dump(S, open(f"{OUT}/summary.json", "w"), indent=1)

    meta = dict(python=sys.version, platform=platform.platform(), sklearn=sklearn.__version__,
                numpy=np.__version__, pandas=pd.__version__, B=B, seed=L.SEED,
                data_sha256=hashlib.sha256(open("data/WSCdata.csv", "rb").read()).hexdigest(),
                wsc_lib_sha256=hashlib.sha256(open("wsc_lib.py", "rb").read()).hexdigest(),
                script_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
                seconds=time.time() - t0)
    json.dump(meta, open(f"{OUT}/run_meta.json", "w"), indent=1)
    print(json.dumps(S, indent=1))
    print("done", meta["seconds"])
