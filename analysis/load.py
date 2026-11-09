"""Load Exp 1 runs into pandas and compute per-run metrics (non-seed agents only)."""
import glob
import json
import os

import numpy as np
import pandas as pd


def read_run(path, labels=None):
    with open(path) as f:
        meta = json.loads(f.readline())
        recs = [json.loads(l) for l in f if l.strip()]
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("cfg", "inbox")} for r in recs])
    cfg = meta["cfg"]
    for k in ("arch", "contagion", "f", "N", "m", "rep", "rules", "pressure", "seed_at_hub", "T"):
        df[k] = cfg.get(k)
    df["defense"], df["orch"] = cfg.get("defense", "none"), cfg.get("orch", "same")
    df["run"] = os.path.basename(path)[:-6]
    if labels is not None and len(df):  # LLM labels override regex promote
        key = df["run"] + "/" + df["round"].astype(str) + "/" + df["agent"].astype(str)
        df["promote"] = [labels.get(x, p) for x, p in zip(key, df["promote"])]
    return meta, df


def load_runs(pattern, labels_path=None):
    labels = json.load(open(labels_path)) if labels_path and os.path.exists(labels_path) else None
    metas, dfs = {}, []
    for p in sorted(glob.glob(pattern)):
        m, d = read_run(p, labels)
        metas[os.path.basename(p)[:-6]] = m
        dfs.append(d)
    return metas, pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def add_cache(df):
    """viol_cache: the seeded shortcut specifically (scorer_cache in code, or SUBMIT citing it); excludes spontaneous hardcoding."""
    if "viol_cache" not in df:
        df["viol_cache"] = df["v_cache"] | df.get("v_intent", False)
    return df


def curves(df, col="violate"):
    """v(t): fraction of non-seed agents that have violated by round t (cumulative). One row per run x round."""
    df = add_cache(df)
    rows = []
    for run, d in df.groupby("run"):
        T = int(d["T"].iloc[0])
        first = d[d[col]].groupby("agent")["round"].min()
        n = d["agent"].nunique()
        cfg = d.iloc[0]
        uns = d[~d["solvable"]]["agent"].unique()
        sol = d[d["solvable"]]["agent"].unique()
        for t in range(0, T + 1):
            inf = set(first[first <= t].index)
            rows.append({"run": run, "arch": cfg["arch"], "contagion": cfg["contagion"], "f": cfg["f"], "N": cfg["N"],
                         "m": cfg["m"], "rep": cfg["rep"], "hub": bool(cfg["seed_at_hub"]), "round": t,
                         "v": len(inf) / n, "v_uns": len(inf & set(uns)) / max(1, len(uns)),
                         "v_sol": len(inf & set(sol)) / max(1, len(sol))})
    return pd.DataFrame(rows)


def run_metrics(df):
    df = add_cache(df)
    out = []
    for run, d in df.groupby("run"):
        c = d.iloc[0]
        T = int(c["T"])
        cv = curves(d)
        v = cv.set_index("round")["v"]
        t50 = next((t for t in range(1, T + 1) if v[t] > 0.5), T + 1)
        ever = d.groupby("agent").agg(viol=("violate", "any"), vc=("viol_cache", "any"), prom=("promote", "any"), solv=("solvable", "first"),
                                      rep=("action", lambda s: (s == "REPORT_IMPOSSIBLE").any()))
        uns = ever[~ever["solv"]]
        out.append({"run": run, "arch": c["arch"], "contagion": c["contagion"], "f": c["f"], "N": c["N"],
                    "defense": c["defense"], "orch": c["orch"],
                    "m": c["m"], "rep": c["rep"], "hub": bool(c["seed_at_hub"]), "n_agents": len(ever),
                    "final": v[T], "final_count": int(ever["viol"].sum()), "T50": t50,
                    "final_cache": float(ever["vc"].mean()), "final_cache_count": int(ever["vc"].sum()),
                    "spreader": float((ever["viol"] & ever["prom"]).mean()),
                    "viol_solvable": float(ever[ever["solv"]]["viol"].mean()) if ever["solv"].any() else np.nan,
                    "honest_report": float((uns["rep"] & ~uns["viol"]).mean()) if len(uns) else np.nan,
                    "parse_fail": float((d["action"] == "PARSE_FAIL").mean()), "aware": float(d["aware"].mean())})
    return pd.DataFrame(out)
