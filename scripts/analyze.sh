#!/bin/bash
# Run every analysis step on finished runs (CPU only; label.py needs the vLLM server, so it runs inside job.sh).
set -euo pipefail
cd /u/zsun9/mas_safety
P=/work/nvme/bhta/zsun9/swarm/venv/bin/python
$P -m analysis.kernel_fit runs/exp0/main.jsonl --out results/kernel.json  # adoption = scorer_cache, as Exp 1 final_cache
$P -m analysis.kernel_fit runs/exp0/main.jsonl --metric any --boot 0 --out results/kernel_any.json
$P -m analysis.stats 'runs/exp1/*.jsonl' --out results
# Exp 2 primary: exposure = explicit recommendation (LLM labels)
$P -m analysis.simulate --kernel results/kernel.json --runs 'runs/exp1/*.jsonl' --labels results/labels.json --out results/exp2.csv
$P -m analysis.plots --kernel results/kernel.json --curves results/exp1_curves.csv --exp2 results/exp2.csv --out results/figs
# sensitivity: exposure = any non-warning mention of the shortcut (regex)
$P -m analysis.simulate --kernel results/kernel.json --runs 'runs/exp1/*.jsonl' --out results/exp2_regex.csv
# Exp 3 (coordinator swap, defenses, monitor) and follow-ups (info control, fixed-ratio scale)
$P -m analysis.exp3 --out results
$P -m analysis.extra
$P -m analysis.kernel_fit runs/exp0/granite_main.jsonl --out results/kernel_granite.json
$P -m analysis.label_summaries 'runs/exp1/orchestrator*.jsonl' 'runs/exp3/*.jsonl' --out results/summary_labels.json  # tables from cached labels
$P -m analysis.plots --defense results/exp3_cells.csv --out results/figs
# item 12: cross-family replication (gpt-oss-20b workers + coordinator; summary labels cached by job swarm-oss)
$P -m analysis.kernel_fit runs/oss/exp0/main.jsonl --out results/kernel_oss.json
$P -m analysis.label_summaries 'runs/oss/exp1/orchestrator*.jsonl' --out results/summary_labels_oss.json
$P -m analysis.oss --out results
cp results/figs/fig*.pdf report/ && /work/nvme/bhta/zsun9/swarm/tex/bin/tectonic report/main.tex
