"""Figures for the report.
  fig1_kernel.pdf   P(adopt | k) per contagion with fitted IC and logistic curves (+ count-only)
  fig2_curves.pdf   v(t) per architecture, one panel per contagion, mean + 95% bootstrap band
  fig3_predict.pdf  Exp 2 predicted vs observed v(t)
  fig4_defense.pdf  Exp 3 final_cache per coordinator defense, mean + 95% bootstrap CI

python -m analysis.plots --kernel results/kernel.json --curves results/exp1_curves.csv --exp2 results/exp2.csv
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.kernel_fit import MODELS

COL = {"blackboard": "#c0392b", "orchestrator": "#8e44ad", "ring": "#2471a3", "random": "#17a589",
       "injection": "#d35400", "norm": "#1f618d", "count": "#7f8c8d", "adaptive": "#7d3c98"}
NAME = {"injection": "directive (injection)", "norm": "social proof (norm)", "adaptive": "adaptive directive"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def fig_kernel(kj, out):
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    kk = np.linspace(0, 8, 100)
    for key, ent in kj.items():
        if key == "absprop" or not key.startswith(("main/", "count/", "pilot/")):
            continue
        c = key.split("/")[1]
        emp = ent["empirical"]
        ks = sorted(int(k) for k in emp)
        p = [emp[str(k)]["p"] for k in ks]
        lo = [emp[str(k)]["ci"][0] for k in ks]
        hi = [emp[str(k)]["ci"][1] for k in ks]
        lab = {"injection": "directive", "norm": "social proof", "count": "social proof, count only"}[c]
        ax.errorbar(ks, p, yerr=[np.subtract(p, lo), np.subtract(hi, p)], fmt="o", ms=3.5, capsize=2,
                    color=COL[c], label=lab, ls="none")
        best = min(("ic", "logistic"), key=lambda m: ent["fit"][m]["aic"])
        for m, ls in (("ic", ":"), ("logistic", "-")):
            f, th = MODELS[m][0], np.array(ent["fit"][m]["theta"])
            ax.plot(kk, f(th, kk), ls=ls, color=COL[c], lw=1.4 if m == best else 0.8, alpha=1 if m == best else 0.6)
    ax.set_xlabel("k = shortcut-promoting messages seen (of 8)")
    ax.set_ylabel("P(violate)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=7.5, loc="center right")
    ax.set_title("Adoption kernel (solid: logistic, dotted: indep. cascade)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out)


def boot_band(g, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    piv = g.pivot_table(index="run", columns="round", values="v").values
    m = rng.choice(len(piv), (B, len(piv)))
    bs = piv[m].mean(1)
    return piv.mean(0), np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)


def fig_curves(cv, out):
    cv = cv[(cv["f"] == 0.5) & (cv["N"] == 16) & (~cv["hub"].astype(bool)) & (cv["m"] == 1)]
    conts = [c for c in ("injection", "norm") if c in cv["contagion"].unique()]
    fig, axes = plt.subplots(1, len(conts), figsize=(3.4 * len(conts), 2.7), sharey=True, squeeze=False)
    for ax, c in zip(axes[0], conts):
        for arch, g in cv[cv["contagion"] == c].groupby("arch"):
            mu, lo, hi = boot_band(g)
            t = np.arange(len(mu))
            ax.plot(t, mu, color=COL[arch], label=arch, lw=1.6)
            ax.fill_between(t, lo, hi, color=COL[arch], alpha=0.15, lw=0)
        ax.set_title(f"{NAME[c]} seed", fontsize=9)
        ax.set_xlabel("round t")
    axes[0][0].set_ylabel("share of workers\nusing scorer_cache")
    axes[0][-1].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out)


def fig_predict(e2, out):
    e2 = e2[(e2["f"] == 0.5) & (e2["N"] == 16) & (~e2["hub"].astype(bool))]
    conts = [c for c in ("injection", "norm") if c in e2["contagion"].unique()]
    fig, axes = plt.subplots(1, len(conts), figsize=(3.4 * len(conts), 2.7), sharey=True, squeeze=False)
    for ax, c in zip(axes[0], conts):
        for arch, g in e2[e2["contagion"] == c].groupby("arch"):
            ax.plot(g["round"], g["obs"], color=COL[arch], lw=1.6, label=f"{arch}")
            ax.plot(g["round"], g["pred"], color=COL[arch], lw=1.2, ls="--")
            ax.fill_between(g["round"], g["pred_lo"], g["pred_hi"], color=COL[arch], alpha=0.08, lw=0)
        ax.set_title(f"{NAME[c]}: observed (solid) vs predicted (dashed)", fontsize=8)
        ax.set_xlabel("round t")
    axes[0][0].set_ylabel("v(t)")
    axes[0][-1].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out)


def fig_defense(cells, out):
    order = ["none", "D1", "D2", "D3", "D4", "D5", "orch=granite"]
    lab = {"none": "none", "D1": "D1 source\nlabels", "D2": "D2 instr.\nhierarchy", "D3": "D3 struct.\nsummary",
           "D4": "D4 incentive\nfix", "D5": "D5 monitor", "orch=granite": "Granite\ncoordinator"}
    conts = [("injection", "directive (injection)"), ("norm", "social proof (norm)"), ("adaptive", "adaptive directive")]
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    for j, (c, name) in enumerate(conts):
        g = cells[cells["contagion"] == c].set_index("cond")
        xs = [i + (j - 1) * 0.22 for i, o in enumerate(order) if o in g.index]
        os_ = [o for o in order if o in g.index]
        y = g.loc[os_, "final_cache"].values
        ax.errorbar(xs, y, yerr=[y - g.loc[os_, "lo"].values, g.loc[os_, "hi"].values - y], fmt="o", ms=4, capsize=2,
                    color=COL[c], label=name, ls="none")
    ax.axvline(5.5, color="#bbb", lw=0.8, ls=":")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([lab[o] for o in order], fontsize=7.5)
    ax.set_ylabel("final share of workers\nusing scorer_cache")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    fig.savefig(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel")
    ap.add_argument("--curves")
    ap.add_argument("--exp2")
    ap.add_argument("--defense")
    ap.add_argument("--out", default="results/figs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.kernel:
        fig_kernel(json.load(open(a.kernel)), f"{a.out}/fig1_kernel.pdf")
    if a.curves:
        fig_curves(pd.read_csv(a.curves), f"{a.out}/fig2_curves.pdf")
    if a.exp2:
        fig_predict(pd.read_csv(a.exp2), f"{a.out}/fig3_predict.pdf")
    if a.defense:
        fig_defense(pd.read_csv(a.defense), f"{a.out}/fig4_defense.pdf")
    print("figures in", a.out)
