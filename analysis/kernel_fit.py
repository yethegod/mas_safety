"""Exp 0: fit the adoption kernel per contagion type and compare models by AIC.

  independent cascade  p(k) = 1 - (1 - p0)(1 - p)^k          (simple contagion)
  logistic             p(k) = sigmoid(a + b k)                (complex if flat at k=0,1 then steep)

python -m analysis.kernel_fit runs/exp0/main.jsonl [runs/exp0/count.jsonl ...] --out results/kernel.json

--metric cache (default): adoption = the seeded shortcut (scorer_cache in code, or SUBMIT citing it), the same
definition as Exp 1's final_cache. --metric any also counts spontaneous hardcoding.
"""
import argparse
import json
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import beta


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [json.loads(l) for l in f if l.strip()]
    return [r for r in rows if r.get("parse_ok", True)]


def nll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


def ic(theta, k):
    p0, p = expit(theta)
    return 1 - (1 - p0) * (1 - p) ** k


def lg(theta, k):
    return expit(theta[0] + theta[1] * k)


MODELS = {"ic": (ic, np.array([-3.0, -1.0])), "logistic": (lg, np.array([-2.0, 0.5]))}


def fit(k, y):
    out = {}
    for name, (f, x0) in MODELS.items():
        best = None
        for jitter in (0, 1, -1, 2):
            r = minimize(lambda th: nll(f(th, k), y), x0 + jitter, method="Nelder-Mead", options={"maxiter": 4000})
            if best is None or r.fun < best.fun:
                best = r
        out[name] = {"theta": best.x.tolist(), "nll": float(best.fun), "aic": float(2 * len(x0) + 2 * best.fun)}
    out["delta_aic_ic_minus_logistic"] = out["ic"]["aic"] - out["logistic"]["aic"]
    return out


def boot(k, y, B=500, seed=0):
    rng = np.random.default_rng(seed)
    th = {m: [] for m in MODELS}
    for _ in range(B):
        idx = rng.integers(0, len(k), len(k))
        f = fit(k[idx], y[idx])
        for m in MODELS:
            th[m].append(f[m]["theta"])
    return {m: {"lo": np.percentile(v, 2.5, axis=0).tolist(), "hi": np.percentile(v, 97.5, axis=0).tolist()} for m, v in th.items()}


def clopper(x, n):
    lo = beta.ppf(0.025, x, n - x + 1) if x > 0 else 0.0
    hi = beta.ppf(0.975, x + 1, n - x) if x < n else 1.0
    return float(lo), float(hi)


def adopted(r, metric):
    return float(r["v_cache"] or r.get("v_intent", False)) if metric == "cache" else float(r["violate"])


def summarize(rows, B, metric="cache"):
    res = {}
    groups = defaultdict(list)
    for r in rows:
        if r["part"] in ("main", "count", "solvable", "pilot"):
            groups[(r["part"], r["contagion"])].append(r)
    for (part, c), rs in groups.items():
        k = np.array([r["k"] for r in rs], float)
        y = np.array([adopted(r, metric) for r in rs])
        emp = {}
        for kk in sorted(set(k.tolist())):
            m = k == kk
            x, n = int(y[m].sum()), int(m.sum())
            report = sum(r["action"] == "REPORT_IMPOSSIBLE" for r, mm in zip(rs, m) if mm) / n
            emp[int(kk)] = {"p": x / n, "x": x, "ci": clopper(x, n), "n": n, "report": report}
        f = fit(k, y)
        if B:
            ci = boot(k, y, B)
            for m in MODELS:
                f[m]["ci"] = ci[m]
        res[f"{part}/{c}"] = {"empirical": emp, "fit": f}
    ap = [r for r in rows if r["part"] == "absprop"]
    if ap:
        tab = defaultdict(list)
        for r in ap:
            tab[(r["contagion"], r["n"])].append(adopted(r, metric))
        res["absprop"] = {f"{c}/n={n}": {"p": float(np.mean(v)), "n": len(v), "ci": clopper(int(sum(v)), len(v))}
                          for (c, n), v in sorted(tab.items())}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default="results/kernel.json")
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--metric", default="cache", choices=["cache", "any"])
    a = ap.parse_args()
    res = summarize(load(a.paths), a.boot, a.metric)
    import os
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    for key, v in res.items():
        if key == "absprop":
            print("absprop", {kk: round(x["p"], 2) for kk, x in v.items()})
            continue
        emp = " ".join(f"k{k}={e['p']:.2f}" for k, e in v["empirical"].items())
        print(f"{key:24s} {emp}  dAIC(ic-logit)={v['fit']['delta_aic_ic_minus_logistic']:+.1f}")
