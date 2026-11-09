"""Item 12: cross-family replication with gpt-oss-20b (workers and coordinator), paired with the Qwen3-8B runs.

Exp 0 kernel (runs/oss/exp0/main.jsonl) and Exp 1 subset (runs/oss/exp1: orchestrator / random / blackboard x
injection / norm x 8 reps). Each gpt-oss run has the same team, tasks and seeds as the Qwen run with the same id,
so the per-cell comparison is a paired Wilcoxon over reps. Also reports parse quality (JSON vs regex fallback),
and "test-aware" replies (the model says the setup looks like a test / trap) in the visible fields and in the
reasoning channel, counted, not dropped.

python -m analysis.oss --out results
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from analysis.kernel_fit import clopper
from analysis.load import load_runs, run_metrics
from analysis.stats import boot_ci

MODELS = {"Qwen3-8B": "runs/exp1", "gpt-oss-20b": "runs/oss/exp1"}
CELLS = "{arch}_*_f0.5_N16_m1_strong_medium_r*.jsonl"
ARCHS = ("orchestrator", "random", "blackboard")
LABELS = {"Qwen3-8B": "results/summary_labels.json", "gpt-oss-20b": "results/summary_labels_oss.json"}
LABELLER = "Qwen/Qwen3-8B"


def exp0(path="runs/oss/exp0/main.jsonl"):
    rows = [json.loads(l) for l in open(path)]
    d = pd.DataFrame(rows)
    d["y"] = d["v_cache"] | d["v_intent"]
    out = []
    for (c, k), g in d.groupby(["contagion", "k"]):
        x, n = int(g["y"].sum()), len(g)
        lo, hi = clopper(x, n)
        out.append({"contagion": c, "k": k, "adopt": x / n, "lo": lo, "hi": hi, "n": n,
                    "report": (g["action"] == "REPORT_IMPOSSIBLE").mean()})
    q = {"parse_fail": (~d["parse_ok"]).mean(), "regex": (d.get("parse_mode") == "regex").mean(),
         "aware_visible": d["aware"].mean(), "aware_cot": d.get("aware_cot", pd.Series(False)).mean(), "n": len(d)}
    return pd.DataFrame(out), q


def laundering(model):
    p = LABELS[model]
    if not os.path.exists(p):
        return {}
    lab = json.load(open(p))
    out = {}
    for c in ("injection", "norm"):
        v = [x["label"] for k, x in lab.items() if k.startswith(LABELLER + "|orchestrator_" + c + "_f0.5_N16_m1_strong_medium_r")]
        if v:
            out[c] = (np.mean([x == "DIRECTIVE" for x in v]), len(v))
    return out


def exp1():
    rms, quality = {}, {}
    for model, root in MODELS.items():
        dfs = [load_runs(f"{root}/" + CELLS.format(arch=a))[1] for a in ARCHS]
        df = pd.concat([d for d in dfs if len(d)], ignore_index=True)
        df = df[~df["seed_at_hub"].astype(bool) & df["contagion"].isin(["injection", "norm"])]
        rms[model] = run_metrics(df)
        quality[model] = {"parse_fail": (df["action"] == "PARSE_FAIL").mean(),
                          "regex": (df["parse_mode"] == "regex").mean() if "parse_mode" in df else 0.0,
                          "aware_visible": df["aware"].mean(),
                          "aware_cot": df["aware_cot"].mean() if "aware_cot" in df else np.nan,
                          "turns": len(df)}
    rows = []
    q = rms["Qwen3-8B"]
    for model, rm in rms.items():
        lau = laundering(model)
        for (a, c), g in rm.groupby(["arch", "contagion"]):
            lo, hi = boot_ci(g["final_cache"])
            row = {"model": model, "arch": a, "contagion": c, "runs": len(g), "final_cache": g["final_cache"].mean(),
                   "lo": lo, "hi": hi, "honest_report": g["honest_report"].mean(),
                   "directive_share": lau.get(c, (np.nan, 0))[0] if a == "orchestrator" else np.nan}
            if model != "Qwen3-8B":
                ref = q[(q["arch"] == a) & (q["contagion"] == c)].set_index("rep")["final_cache"]
                pair = g.set_index("rep")["final_cache"].to_frame().join(ref.rename("qwen"), how="inner")
                d = pair["final_cache"] - pair["qwen"]
                row["diff_vs_qwen"] = d.mean()
                row["p_wilcoxon"] = wilcoxon(d).pvalue if (d != 0).any() else 1.0
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(quality).T, pd.concat([rm.assign(model=m) for m, rm in rms.items()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    if os.path.exists("runs/oss/exp0/main.jsonl"):
        k, q = exp0()
        k.to_csv(f"{a.out}/oss_exp0.csv", index=False)
        print(k.pivot(index="k", columns="contagion", values="adopt").round(2).to_string())
        print("exp0 quality", {x: round(float(y), 3) for x, y in q.items()})
    cells, quality, runs = exp1()
    cells.to_csv(f"{a.out}/oss_cells.csv", index=False)
    runs.to_csv(f"{a.out}/oss_runs.csv", index=False)
    quality.to_csv(f"{a.out}/oss_quality.csv")
    print("\n", cells.round(3).to_string(index=False))
    print("\n", quality.round(3).to_string())


if __name__ == "__main__":
    main()
