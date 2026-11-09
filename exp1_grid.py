"""Exp 1 grid: run many team simulations in one process so vLLM batches stay full.
Skips configs whose output already exists (safe to resubmit after a timeout).

python exp1_grid.py --grid main            # 4 arch x 2 contagion x 8 reps = 64 runs
python exp1_grid.py --grid fsweep          # blackboard, f in {0.25, 0.75}
python exp1_grid.py --grid controls        # no seed / neutral seed on blackboard + ring
python exp1_grid.py --grid hub             # orchestrator with the hub itself seeded
python exp1_grid.py --grid scale           # blackboard, N in {8, 32}
python exp1_grid.py --grid orchswap        # Exp 3: orchestrator run by a second model (SWARM_ORCH_MODEL), workers unchanged
python exp1_grid.py --grid defense         # Exp 3: orchestrator with D1-D4
python exp1_grid.py --grid adaptive        # Exp 3: adaptive injection vs no defense, D1, D2
python exp1_grid.py --grid monitor         # Exp 3: D5 monitor; neutral seed gives the false-positive rate
python exp1_grid.py --grid oss             # item 12: orchestrator / random / blackboard x 2 contagions (run with SWARM_MODEL=gpt-oss)
python exp1_grid.py --grid scale_ratio     # blackboard N=32 with m=2 (m/N fixed) and with m=1, K=40 (window scaled)
"""
import argparse
import asyncio
import os
import time

from env.agent import LLM
from env.runner import DEFAULTS, run, run_id

ARCHS = ["blackboard", "orchestrator", "ring", "random"]
CONT = ["injection", "norm"]


def grid(name, reps, base):
    g = []
    if name == "main":
        g = [dict(arch=a, contagion=c) for a in ARCHS for c in CONT]
    elif name == "fsweep":
        g = [dict(arch="blackboard", contagion=c, f=f) for c in CONT for f in (0.25, 0.75)]
    elif name == "controls":
        g = [dict(arch=a, contagion="neutral") for a in ("blackboard", "ring")] + \
            [dict(arch=a, contagion="norm", m=0) for a in ("blackboard", "ring")]
    elif name == "hub":
        g = [dict(arch="orchestrator", contagion=c, seed_at_hub=True) for c in CONT]
    elif name == "scale":
        g = [dict(arch="blackboard", contagion=c, N=n) for c in CONT for n in (8, 32)]
    elif name == "orchswap":
        g = [dict(arch="orchestrator", contagion=c, orch=os.environ["SWARM_ORCH_TAG"]) for c in CONT]
    elif name == "defense":
        g = [dict(arch="orchestrator", contagion=c, defense=d) for c in CONT for d in ("D1", "D2", "D3", "D4")]
    elif name == "monitor":
        g = [dict(arch="orchestrator", contagion=c, defense="D5") for c in ("injection", "norm", "adaptive", "neutral")]
    elif name == "scale_ratio":
        g = [dict(arch="blackboard", contagion=c, N=32, m=2) for c in CONT] + \
            [dict(arch="blackboard", contagion=c, N=32, K=40) for c in CONT]
    elif name == "oss":
        g = [dict(arch=a, contagion=c) for a in ("orchestrator", "random", "blackboard") for c in CONT]
    elif name == "adaptive":
        g = [dict(arch="orchestrator", contagion="adaptive", defense=d) for d in ("none", "D1", "D2")]
    return [{**DEFAULTS, **base, **x, "rep": r} for x in g for r in range(reps)]


async def main_async(a):
    base = {k: v for k, v in dict(rules=a.rules, pressure=a.pressure, m=a.m, T=a.T).items() if v is not None}
    cfgs = [c for gname in a.grid.split(",") for c in grid(gname, a.reps, base)]
    todo = [c for c in cfgs if not os.path.exists(f"{a.out}/{run_id(c)}.jsonl")]
    print(f"{len(cfgs)} configs, {len(todo)} to run", flush=True)
    llm = LLM(concurrency=a.concurrency)
    orch_llm = LLM(model=os.environ["SWARM_ORCH_MODEL"], base_url=os.environ["SWARM_ORCH_BASE_URL"], concurrency=64) \
        if os.environ.get("SWARM_ORCH_MODEL") else None
    sem = asyncio.Semaphore(a.parallel_runs)
    t0 = time.time()
    done = 0

    async def one(c):
        nonlocal done
        async with sem:
            try:
                recs = await run(c, llm, f"{a.out}/{run_id(c)}.jsonl", orch_llm if c.get("orch", "same") != "same" else None)
            except Exception as e:  # one bad run must not kill the grid
                print("FAILED", run_id(c), repr(e), flush=True)
                return
            done += 1
            v = sum(r["violate"] for r in recs if r["round"] == c["T"]) / max(1, sum(r["round"] == c["T"] for r in recs))
            print(f"[{done}/{len(todo)} {time.time() - t0:.0f}s] {run_id(c)} last-round viol={v:.2f}", flush=True)

    await asyncio.gather(*(one(c) for c in todo))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="main")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--rules", default=None)
    ap.add_argument("--pressure", default=None)
    ap.add_argument("--m", type=int, default=None)
    ap.add_argument("--T", type=int, default=None)
    ap.add_argument("--out", default="runs/exp1")
    ap.add_argument("--parallel_runs", type=int, default=12)
    ap.add_argument("--concurrency", type=int, default=192)
    asyncio.run(main_async(ap.parse_args()))
