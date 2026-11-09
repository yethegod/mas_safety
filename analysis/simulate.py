"""Exp 2: predict team-level spread from the single-agent kernel, compare with Exp 1.

Each round every not-yet-violated worker violates with p_kernel(k), where k is the number
of shortcut-promoting messages it sees under its architecture:
  blackboard  promoting posts among the last K posts
  ring/random promoting posts from neighbours in the previous round
  orchestrator 1 if the coordinator summary relays the shortcut (relay prob fitted from Exp 1 logs
              as a logistic in the number of promoting inputs), else 0
Who promotes: seeds always; workers with probability q_viol / q_clean (from Exp 1 logs).
k above the kernel's grid (n=8) is clamped to 8.
Adoption = the seeded shortcut (viol_cache), matching Exp 0's --metric cache kernel.
Prediction bands include kernel uncertainty: each simulated team draws its own kernel, p_k* = Binomial(n_k, p_k)/n_k
(a bootstrap of the Exp 0 trials within each k). The orchestrator relay curve and q_viol/q_clean come from the
Exp 1 logs themselves, so orchestrator predictions are only half out-of-sample.

python -m analysis.simulate --kernel results/kernel.json --runs 'runs/exp1/*.jsonl' --out results/exp2.csv
"""
import argparse
import json

import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import expit

from analysis.load import add_cache, curves, load_runs
from analysis.kernel_fit import MODELS


def kernel_table(kjson, contagion, solvable):
    part = "solvable" if solvable else "main"
    key = f"{part}/{contagion}"
    if key not in kjson:
        key = f"main/{contagion}" if f"main/{contagion}" in kjson else f"pilot/{contagion}"
    ent = kjson[key]
    ks = sorted(int(k) for k in ent["empirical"])
    e = [ent["empirical"][str(k)] for k in ks]
    return ent, np.array(ks), np.array([x["p"] for x in e]), np.array([x["n"] for x in e])


def kernel_fn(kjson, contagion, solvable, use="empirical", rng=None):
    """rng given -> one bootstrap draw of the empirical kernel."""
    ent, ks, ps, ns = kernel_table(kjson, contagion, solvable)
    if use == "empirical":
        if rng is not None:
            ps = rng.binomial(ns, ps) / ns
        return lambda k: float(np.interp(min(k, ks[-1]), ks, ps))
    best = min(("ic", "logistic"), key=lambda m: ent["fit"][m]["aic"])
    f, th = MODELS[best][0], np.array(ent["fit"][best]["theta"])
    return lambda k: float(f(th, np.array(min(k, 8.0))))


def calibrate(df, metas, col="viol_cache"):
    """Promotion probabilities and orchestrator relay curve, per contagion."""
    cal = {}
    for c, d in df.groupby("contagion"):
        d = d.sort_values("round")
        d["viol_before"] = d.groupby(["run", "agent"])[col].transform(lambda s: s.cummax())
        qv = d[d["viol_before"]]["promote"].mean() if d["viol_before"].any() else 0.5
        qc = d[~d["viol_before"]]["promote"].mean()
        # orchestrator relay: did round-t summary promote, vs # promoting inputs in round t
        xs, ys = [], []
        for run, m in metas.items():
            if m["cfg"]["arch"] != "orchestrator" or m["cfg"]["contagion"] != c or not m.get("orch_log"):
                continue
            dr = df[df["run"] == run]
            for e in m["orch_log"]:
                t = e["round"]
                n_prom = len(m["seeds"]) if (t == 0 or m["cfg"]["seed_rounds"] == "all") else 0
                n_prom += int(dr[(dr["round"] == t)]["promote"].sum())
                nxt = dr[dr["round"] == t + 1]
                if len(nxt):
                    xs.append(n_prom)
                    ys.append(float(nxt["inbox_promote"].iloc[0] > 0))
        relay = (-2.0, 0.5)
        if len(set(ys)) > 1:
            from scipy.optimize import minimize
            X, Y = np.array(xs, float), np.array(ys)
            r = minimize(lambda th: -np.sum(Y * np.log(expit(th[0] + th[1] * X) + 1e-9) + (1 - Y) * np.log(1 - expit(th[0] + th[1] * X) + 1e-9)), [-1, 0.5])
            relay = tuple(r.x)
        elif ys:
            relay = (6.0 if ys[0] else -6.0, 0.0)
        cal[c] = {"q_viol": float(qv), "q_clean": float(qc), "relay": list(relay), "relay_n": len(xs)}
    return cal


def sim_once(rng, arch, N, m, T, K, d, f, kern_u, kern_s, cal, seed_rounds="all", hub=False, mode="memoryless"):
    """mode: memoryless = fresh draw each round; threshold = one latent u_i per agent, violate once p(k) > u_i."""
    seeds = list(range(m)) if not hub else []
    workers = [i for i in range(N) if i not in seeds]
    uns = set(rng.choice(workers, round(f * len(workers)), replace=False).tolist())
    G = None
    if arch == "ring":
        G = nx.watts_strogatz_graph(N, d, 0.0, seed=int(rng.integers(1e9)))
    elif arch == "random":
        G = nx.random_regular_graph(d, N, seed=int(rng.integers(1e9)))
    viol = {i: False for i in workers}
    u = {i: rng.random() for i in workers}
    board = []  # list of (round, promoting)
    prev = {}   # agent -> promoting in previous round
    summary_prom = hub
    out = []

    def seed_post(t):
        return seed_rounds == "all" or t == 0

    for s in seeds:
        prev[s] = True
        board.append((0, True))
    if arch == "orchestrator" and not hub:
        a, b = cal["relay"]
        summary_prom = rng.random() < expit(a + b * len(seeds))
    for t in range(1, T + 1):
        new = {}
        for i in workers:
            if arch == "blackboard":
                k = sum(p for _, p in board[-K:])
            elif arch == "orchestrator":
                k = int(summary_prom)
            else:
                k = sum(prev.get(j, False) for j in G.neighbors(i))
            if not viol[i]:
                p = (kern_u if i in uns else kern_s)(k)
                if (u[i] < p) if mode == "threshold" else (rng.random() < p):
                    viol[i] = True
            new[i] = rng.random() < (cal["q_viol"] if viol[i] else cal["q_clean"])
        for s in seeds:
            new[s] = seed_post(t)
        posts = [(t, new[i]) for i in rng.permutation(list(new)).tolist()]
        board.extend(posts)
        prev = new
        if arch == "orchestrator":
            n_prom = sum(new.values())
            a, b = cal["relay"]
            summary_prom = hub or (rng.random() < expit(a + b * n_prom))
        out.append(sum(viol.values()) / len(workers))
    return [0.0] + out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default="results/kernel.json")
    ap.add_argument("--runs", default="runs/exp1/*.jsonl")
    ap.add_argument("--labels", default="none", help="LLM label file (strict); default regex labels")
    ap.add_argument("--use", default="empirical", choices=["empirical", "fit"])
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--no_kboot", action="store_true", help="fixed kernel (old behaviour)")
    ap.add_argument("--metric", default="viol_cache", choices=["viol_cache", "violate"])
    ap.add_argument("--out", default="results/exp2.csv")
    a = ap.parse_args()
    kj = json.load(open(a.kernel))
    metas, df = load_runs(a.runs, a.labels)
    df = add_cache(df)
    cal = calibrate(df, metas, a.metric)
    json.dump(cal, open(a.out.replace(".csv", "_calibration.json"), "w"), indent=1)
    obs = curves(df, a.metric)
    rows = []
    rng = np.random.default_rng(0)
    for (arch, c, mm, f, N, hub), g in obs.groupby(["arch", "contagion", "m", "f", "N", "hub"]):
        if c not in ("injection", "norm") or (mm == 0 and not hub):
            continue  # controls have no seeded shortcut to propagate
        cfg = metas[g["run"].iloc[0]]["cfg"]
        T, K, d, m = cfg["T"], cfg["K"], cfg["d"], cfg["m"]
        boot = a.use == "empirical" and not a.no_kboot

        def one(md):
            kr = rng if boot else None
            ku, ks = kernel_fn(kj, c, False, a.use, kr), kernel_fn(kj, c, True, a.use, kr)
            return sim_once(rng, arch, N, m, T, K, d, f, ku, ks, cal[c], cfg["seed_rounds"], hub, md)

        ob = g.groupby("round")["v"]
        sims = {md: np.array([one(md) for _ in range(a.sims)]) for md in ("memoryless", "threshold")}
        for t in range(T + 1):
            row = {"arch": arch, "contagion": c, "f": f, "N": N, "hub": hub, "round": t,
                   "obs": ob.mean().get(t, np.nan), "obs_n": ob.count().get(t, 0)}
            for md, sm in sims.items():
                suf = "" if md == "memoryless" else "_thr"
                row.update({f"pred{suf}": sm[:, t].mean(), f"pred{suf}_lo": np.percentile(sm[:, t], 2.5),
                            f"pred{suf}_hi": np.percentile(sm[:, t], 97.5)})
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(a.out, index=False)
    fin = res[res["round"] == res.groupby(["arch", "contagion", "f", "N", "hub"])["round"].transform("max")]
    print(fin[["arch", "contagion", "f", "N", "hub", "pred", "pred_lo", "pred_hi", "pred_thr", "pred_thr_lo", "pred_thr_hi", "obs"]].round(3).to_string(index=False))
    fin = fin.assign(hit=(fin["obs"] >= fin["pred_lo"]) & (fin["obs"] <= fin["pred_hi"]))
    for orch, h in fin.groupby(fin["arch"] == "orchestrator"):
        print(("orchestrator" if orch else "blackboard+peer"), f"coverage {int(h['hit'].sum())}/{len(h)}")
    print("calibration", json.dumps(cal))


if __name__ == "__main__":
    main()
