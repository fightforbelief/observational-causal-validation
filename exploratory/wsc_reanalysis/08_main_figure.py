"""Main-text figure: aligned observational-minus-randomized contrasts for both outcomes.

Reads results/point.json, results/boot.csv (mathPost) and results/vocab/{point.json,boot_joint.csv,
boot_paired.csv}. Writes figures/fig_main_contrasts.pdf/.png and results/main_contrasts.csv.
"""
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Recover the prespecified interval from the original paired redraws.
math_primary = pd.read_csv("published/math_primary_bootstrap.csv")
math_primary_interval = math_primary["difference_qed_minus_rct"].quantile([0.025, 0.975])

rows = []
for outcome, pp, bj, bp in [("mathPost", "results/point.json", "results/boot.csv", None),
                            ("vocabPost", "results/vocab/point.json", "results/vocab/boot_joint.csv",
                             "results/vocab/boot_paired.csv")]:
    p = json.load(open(pp)); b = pd.read_csv(bj)
    for est, key in [("ATT", "d_att"), ("ATU", "d_atu"), ("Concordant ATE", "d_ate_aligned")]:
        x = b[key].dropna()
        r = dict(outcome=outcome, estimand=est, point=p[key], joint_lo=x.quantile(.025), joint_hi=x.quantile(.975),
                 joint_sd=x.std(), joint_n=len(x))
        if est == "ATT":
            if bp is None:
                r["paired_lo"], r["paired_hi"] = math_primary_interval.to_numpy()
            else:
                y = pd.read_csv(bp)["d_att"].dropna()
                r["paired_lo"], r["paired_hi"] = y.quantile(.025), y.quantile(.975)
        rows.append(r)
df = pd.DataFrame(rows)
df.to_csv("results/main_contrasts.csv", index=False)
print(df.to_string())

plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharex=True)
ests = ["ATT", "ATU", "Concordant ATE"]
for ax, outcome in zip(axes, ["mathPost", "vocabPost"]):
    sub = df[df.outcome == outcome].set_index("estimand")
    for i, e in enumerate(ests):
        y = len(ests) - 1 - i
        r = sub.loc[e]
        ax.plot([r.joint_lo, r.joint_hi], [y, y], color="0.25", lw=1.4)
        ax.plot(r.point, y, "o", color="0.1", ms=4.5, zorder=3)
        if e == "ATT":
            ax.plot([r.paired_lo, r.paired_hi], [y - 0.22, y - 0.22], color="0.55", lw=1.4, ls=(0, (3, 1.5)))
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_yticks(range(len(ests)))
    ax.set_yticklabels(ests[::-1])
    ax.set_ylim(-0.6, len(ests) - 0.5)
    ax.set_title(outcome, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
axes[1].tick_params(labelleft=False)
fig.supxlabel("Observational minus randomized estimate (test points)", fontsize=9, y=0.02)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig("figures/fig_main_contrasts.pdf")
fig.savefig("figures/fig_main_contrasts.png", dpi=200)
