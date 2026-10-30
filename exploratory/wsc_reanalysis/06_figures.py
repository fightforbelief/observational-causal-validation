"""Figures for the reanalysis (static PNG + PDF for the manuscript)."""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "legend.frameon": False, "savefig.dpi": 200})
P = json.load(open("results/point.json")); B = pd.read_csv("results/boot.csv")
q = lambda k: (B[k].quantile(.025), B[k].quantile(.975))


def save(fig, name):
    fig.tight_layout(); fig.savefig(f"figures/{name}.png"); fig.savefig(f"figures/{name}.pdf"); plt.close(fig)


# Fig 1: aligned RCT vs QED estimates
pairs = [("ATT  (S = 1)", "rct_tau1", "qed_att"), ("ATU  (S = 0)", "rct_tau0", "qed_atu"),
         ("ATE  (concordant pop.)", "rct_tauC", "qed_ate")]
fig, ax = plt.subplots(figsize=(6.4, 3.2))
for i, (lab, r, qq) in enumerate(pairs):
    yv = len(pairs) - 1 - i
    for off, k, col, name in [(0.13, r, BLUE, "RCT benchmark"), (-0.13, qq, ORANGE, "QED (cross-fitted AIPW)")]:
        lo, hi = q(k)
        ax.plot([lo, hi], [yv + off] * 2, color=col, lw=2, solid_capstyle="round")
        ax.plot(P[k], yv + off, "o", ms=8, color=col, mec="white", mew=1.5, label=name if i == 0 else None)
        ax.text(hi + 0.05, yv + off, f"{P[k]:.2f}", va="center", fontsize=8.5, color=INK)
ax.set_yticks(range(len(pairs))); ax.set_yticklabels([p[0] for p in pairs][::-1], color=INK)
ax.set_xlabel("Effect on mathPost (points; 95% joint-bootstrap percentile interval)")
ax.grid(axis="y", visible=False); ax.set_xlim(-0.2, 2.6)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9)
save(fig, "fig1_aligned_estimates")

# Fig 2: fold-seed sensitivity
F = pd.read_csv("results/fold_seeds.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
for ax, (k, rk, title) in zip(axes, [("att", "rct_tau1", "QED ATT"), ("atu", "rct_tau0", "QED ATU")]):
    ax.hist(F[k], bins=30, color=ORANGE, edgecolor="white", linewidth=0.8)
    ax.axvline(P[rk], color=BLUE, lw=2); ax.text(P[rk], ax.get_ylim()[1] * 0.95, " RCT", color=INK, va="top", fontsize=8.5)
    ax.axvline(P["qed_" + k], color=INK, lw=1, ls="--")
    ax.text(P["qed_" + k], ax.get_ylim()[1] * 0.80, " paper's seed", color=INK, va="top", fontsize=8.5)
    ax.set_title(f"{title} across 200 fold splits", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("Estimate"); ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Number of seeds")
save(fig, "fig2_fold_seeds")

# Fig 3: covariate sets
N = pd.read_csv("results/covsets_nested.csv", index_col=0)
labels = ["None", "+ Demo-\ngraphics", "+ Pretests", "+ Math\nmotivation", "+ Personality\n/psych"]
fig, ax = plt.subplots(figsize=(6.4, 3.2))
x = np.arange(len(N))
for k, col, name in [("bias_att", BLUE, "ATT"), ("bias_atu", ORANGE, "ATU"), ("bias_ate", AQUA, "ATE")]:
    ax.plot(x, N[k], color=col, lw=2, marker="o", ms=7, mec="white", mew=1.5, label=name)
    ax.text(x[-1] + 0.1, N[k].iloc[-1], f"{name} {N[k].iloc[-1]:+.2f}", va="center", fontsize=8.5, color=INK)
ax.axhline(0, color=INK, lw=1)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5, color=INK); ax.set_xlim(-0.3, len(N) - 0.3 + 0.9)
ax.set_ylabel("QED − RCT (points)"); ax.grid(axis="x", visible=False)
ax.legend(loc="upper right", fontsize=9)
save(fig, "fig3_covariate_sets")

# Fig 4: sensitivity bounds
S = pd.read_csv("results/sensitivity.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
for ax, est in zip(axes, ["ATT", "ATU"]):
    s = S[S.estimand == est]
    ax.fill_between(s.Lambda, s.ci_lo, s.ci_hi, color=ORANGE, alpha=0.18, lw=0, label="95% CI for bounds")
    ax.fill_between(s.Lambda, s.bound_lo, s.bound_hi, color=ORANGE, alpha=0.45, lw=0, label="IPW bounds")
    ax.axhline(s.rct.iloc[0], color=BLUE, lw=2, label="RCT benchmark"); ax.axhline(0, color=INK, lw=0.8)
    ax.set_title(f"QED {est}", fontsize=10, color=INK, loc="left"); ax.set_xlabel("Λ (sensitivity parameter)")
axes[0].set_ylabel("Effect on mathPost"); axes[1].legend(fontsize=8, loc="upper left")
save(fig, "fig4_sensitivity")
print("ok")
