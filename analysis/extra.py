"""Two follow-up checks.

info   Exp 0: does knowing where scorer_cache is (one info-only message, no recommendation) already raise adoption,
       compared with k=0 and with one injection / norm message (k=1)?
scale  Exp 1 blackboard at N=32: is the drop from N=16 a dilution of the seed (m/N) or of the board window (K=20)?
       N=16 m=1 K=20 | N=32 m=1 K=20 | N=32 m=2 K=20 (m/N fixed) | N=32 m=1 K=40 (window scaled)

python -m analysis.extra
"""
import json

import pandas as pd

from analysis.kernel_fit import clopper
from analysis.load import load_runs, run_metrics
from analysis.stats import boot_ci


def info():
    rows = [json.loads(l) for p in ("runs/exp0/main.jsonl", "runs/exp0/info.jsonl") for l in open(p)]
    d = pd.DataFrame(rows)
    d = d[((d["part"] == "main") & (d["k"] <= 1)) | (d["part"] == "info")]
    d["y"] = d["v_cache"] | d["v_intent"]
    d["cond"] = d["contagion"] + "/k=" + d["k"].astype(str)
    d.loc[d["k"] == 0, "cond"] = "none/k=0"
    out = []
    for c, g in d.groupby("cond"):
        x, n = int(g["y"].sum()), len(g)
        lo, hi = clopper(x, n)
        out.append({"cond": c, "adopt": x / n, "lo": lo, "hi": hi, "n": n,
                    "report": (g["action"] == "REPORT_IMPOSSIBLE").mean()})
    return pd.DataFrame(out)


def scale():
    _, a = load_runs("runs/exp1/blackboard_*_f0.5_N*_m1_strong_medium_r*.jsonl")
    _, b = load_runs("runs/exp3_scale/*.jsonl")
    df = pd.concat([a, b], ignore_index=True)
    df = df[df["contagion"].isin(["injection", "norm"]) & df["N"].isin([16, 32])]
    rm = run_metrics(df)
    rm["K"] = rm["run"].str.extract(r"_K(\d+)_")[0].fillna(20).astype(int)
    out = []
    for (c, N, m, K), g in rm.groupby(["contagion", "N", "m", "K"]):
        lo, hi = boot_ci(g["final_cache"])
        out.append({"contagion": c, "N": N, "m": m, "K": K, "runs": len(g), "final_cache": g["final_cache"].mean(), "lo": lo, "hi": hi})
    return pd.DataFrame(out)


if __name__ == "__main__":
    i = info()
    i.to_csv("results/exp0_info.csv", index=False)
    print(i.round(3).to_string(index=False))
    s = scale()
    s.to_csv("results/exp1_scale_ratio.csv", index=False)
    print("\n", s.round(3).to_string(index=False))
