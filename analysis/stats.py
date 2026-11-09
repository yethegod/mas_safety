"""Exp 1 statistics: per-cell summaries, pairwise architecture tests, arch x contagion GLM.

python -m analysis.stats 'runs/exp1/*.jsonl' --labels results/labels.json --out results/
"""
import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from analysis.load import curves, load_runs, run_metrics


def boot_ci(x, B=5000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (B, len(x))).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def pairwise(rm, contagion, y="final"):
    d = rm[(rm["contagion"] == contagion) & (rm["f"] == 0.5) & (rm["N"] == 16) & (~rm["hub"]) & (rm["m"] == 1)]
    rows = []
    for a, b in itertools.combinations(sorted(d["arch"].unique()), 2):
        x, z = d[d["arch"] == a][y], d[d["arch"] == b][y]
        if len(x) < 2 or len(z) < 2:
            continue
        u, p = mannwhitneyu(x, z, alternative="two-sided")
        rbc = 1 - 2 * u / (len(x) * len(z))  # rank-biserial (U is for a); >0 means b > a
        rows.append({"metric": y, "contagion": contagion, "a": a, "b": b, "U": u, "p": p, "rank_biserial_b_gt_a": rbc})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = multipletests(out["p"], method="holm")[1]
    return out


def glm(rm, y="final_count"):
    d = rm[(rm["f"] == 0.5) & (rm["N"] == 16) & (~rm["hub"]) & (rm["m"] == 1) & rm["contagion"].isin(["injection", "norm"])].copy()
    if d["arch"].nunique() < 2 or d["contagion"].nunique() < 2:
        return None
    d["fail"] = d["n_agents"] - d[y]
    d["arch"] = pd.Categorical(d["arch"], categories=["random"] + [a for a in sorted(d["arch"].unique()) if a != "random"])
    endog = np.column_stack([d[y], d["fail"]])
    X = pd.get_dummies(d[["arch", "contagion"]], drop_first=True).astype(float)
    for c in [c for c in X.columns if c.startswith("arch_")]:
        X[f"{c}:norm"] = X[c] * X.get("contagion_norm", 0)
    X = sm.add_constant(X)
    try:
        res = sm.GLM(endog, X, family=sm.families.Binomial()).fit(cov_type="HC0")
    except Exception as e:
        print("GLM failed:", e)
        return None
    ci = res.conf_int()
    return pd.DataFrame({"coef": res.params, "lo": ci[0], "hi": ci[1], "p": res.pvalues})


def _main_cells(rm):
    d = rm[(rm["f"] == 0.5) & (rm["N"] == 16) & (~rm["hub"]) & (rm["m"] == 1) & rm["contagion"].isin(["injection", "norm"])].copy()
    d["arch"] = pd.Categorical(d["arch"], categories=["random"] + [a for a in sorted(d["arch"].unique()) if a != "random"])
    return d


def design(d):
    X = pd.get_dummies(d[["arch", "contagion"]], drop_first=True).astype(float)
    for c in [c for c in X.columns if c.startswith("arch_")]:
        X[f"{c}:norm"] = X[c] * X.get("contagion_norm", 0)
    return sm.add_constant(X)


def glm_quasi(rm, y="final_cache_count"):
    """Run-level binomial GLM with the dispersion estimated from Pearson chi2 (quasibinomial):
    agents in one run share a channel, so they are not independent trials."""
    d = _main_cells(rm)
    endog = np.column_stack([d[y], d["n_agents"] - d[y]])
    phi = float(sm.GLM(endog, design(d), family=sm.families.Binomial()).fit(scale="X2").scale)
    res = sm.GLM(endog, design(d), family=sm.families.Binomial()).fit(scale=max(1.0, phi))  # never narrower than binomial
    ci = res.conf_int()
    out = pd.DataFrame({"coef": res.params, "lo": ci[0], "hi": ci[1], "p": res.pvalues})
    out.attrs["dispersion"] = phi
    return out


def glmm(df, rm):
    """Agent-level logit (ever used scorer_cache) with a random intercept per run (variational Bayes)."""
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    keep = set(_main_cells(rm)["run"])
    a = df[df["run"].isin(keep)].groupby(["run", "agent"]).agg(y=("viol_cache", "any"), arch=("arch", "first"),
                                                               contagion=("contagion", "first")).reset_index()
    a["y"] = a["y"].astype(float)
    a["arch"] = pd.Categorical(a["arch"], categories=["random", "blackboard", "orchestrator", "ring"])
    m = BinomialBayesMixedGLM.from_formula("y ~ C(arch) * C(contagion)", {"run": "0 + C(run)"}, a)
    r = m.fit_vb()
    names = m.exog_names
    sd = np.asarray(r.fe_sd)
    from scipy.stats import norm
    z = r.fe_mean / sd
    out = pd.DataFrame({"coef": r.fe_mean, "lo": r.fe_mean - 1.96 * sd, "hi": r.fe_mean + 1.96 * sd,
                        "p": 2 * norm.sf(np.abs(z))}, index=names)
    out.attrs["run_sd"] = float(np.exp(r.vcp_mean[0]))
    return out


def perm_interaction(rm, y="final_cache", B=10000, seed=0):
    """Interaction contrast on run-level means: (arch - random | injection) - (arch - random | norm).
    Null: within each contagion, arch labels are exchangeable between the two architectures compared."""
    d = _main_cells(rm)
    d["arch"] = d["arch"].astype(str)
    rng = np.random.default_rng(seed)
    rows = []
    for a in ("blackboard", "orchestrator", "ring"):
        sub = {c: d[(d["contagion"] == c) & d["arch"].isin([a, "random"])] for c in ("injection", "norm")}

        def stat(lab):
            diff = {}
            for c, g in sub.items():
                v, l = g[y].values, lab[c]
                diff[c] = v[l == a].mean() - v[l == "random"].mean()
            return diff["injection"] - diff["norm"]

        obs = stat({c: g["arch"].values for c, g in sub.items()})
        null = np.array([stat({c: rng.permutation(g["arch"].values) for c, g in sub.items()}) for _ in range(B)])
        rows.append({"arch": a, "contrast_inj_minus_norm": obs, "p_perm": float((np.abs(null) >= abs(obs) - 1e-12).mean())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern")
    ap.add_argument("--labels", default="none", help="LLM label file (strict); default regex labels")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    metas, df = load_runs(a.pattern, a.labels)
    rm = run_metrics(df)
    rm.to_csv(f"{a.out}/exp1_runs.csv", index=False)
    curves(df, "viol_cache").to_csv(f"{a.out}/exp1_curves.csv", index=False)  # adoption = scorer_cache, as final_cache
    cell = []
    for key, g in rm.groupby(["arch", "contagion", "m", "f", "N", "hub"]):
        lo, hi = boot_ci(g["final"])
        clo, chi = boot_ci(g["final_cache"])
        cell.append(dict(zip(["arch", "contagion", "m", "f", "N", "hub"], key), runs=len(g),
                         final_cache=g["final_cache"].mean(), c_lo=clo, c_hi=chi, final=g["final"].mean(), lo=lo, hi=hi,
                         T50=g["T50"].mean(), spreader=g["spreader"].mean(), viol_solvable=g["viol_solvable"].mean(),
                         honest_report=g["honest_report"].mean(), parse_fail=g["parse_fail"].mean(), aware=g["aware"].mean()))
    cell = pd.DataFrame(cell)
    cell.to_csv(f"{a.out}/exp1_cells.csv", index=False)
    print(cell.round(3).to_string(index=False))
    pw = pd.concat([pairwise(rm, c, y) for y in ("final_cache", "final") for c in ("injection", "norm")], ignore_index=True)
    pw.to_csv(f"{a.out}/exp1_pairwise.csv", index=False)
    print("\n", pw.round(4).to_string(index=False))
    for y in ("final_cache_count", "final_count"):
        g = glm(rm, y)
        if g is not None:
            g.to_csv(f"{a.out}/exp1_glm_{y}.csv")
            print(f"\nGLM on {y}\n", g.round(3).to_string())
    # run-level inference (primary): agents within a run are not independent
    q = glm_quasi(rm)
    q.to_csv(f"{a.out}/exp1_glm_quasi.csv")
    print(f"\nquasibinomial GLM on final_cache_count (Pearson dispersion {q.attrs['dispersion']:.2f}, floored at 1)\n", q.round(3).to_string())
    from analysis.load import add_cache
    g = glmm(add_cache(df), rm)
    g.to_csv(f"{a.out}/exp1_glmm.csv")
    print(f"\nGLMM, run random intercept (sd {g.attrs['run_sd']:.2f})\n", g.round(3).to_string())
    pm = perm_interaction(rm)
    pm.to_csv(f"{a.out}/exp1_perm.csv", index=False)
    print("\npermutation test of arch x contagion (final_cache, vs random)\n", pm.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
