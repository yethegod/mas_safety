#!/bin/bash
# Two vLLM servers on two GPUs: workers (SWARM_MODEL, port 8000) and coordinator (SWARM_ORCH_MODEL, port 8001).
# SWARM_ORCH_MODEL=ibm-granite/granite-3.3-8b-instruct SWARM_ORCH_TAG=granite sbatch scripts/job2.sh "python exp1_grid.py --grid orchswap --out runs/exp3"
#SBATCH --account=bhta-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --job-name=swarm2
#SBATCH --output=/u/zsun9/mas_safety/logs/%x-%j.out

set -euo pipefail
W=/work/nvme/bhta/zsun9/swarm
export HF_HOME=$W/hf HF_HUB_OFFLINE=1
export PATH=$W/venv/bin:$PATH
MODEL=${SWARM_MODEL:-Qwen/Qwen3-8B}
OMODEL=${SWARM_ORCH_MODEL:?set SWARM_ORCH_MODEL}
cd /u/zsun9/mas_safety
J=${SLURM_JOB_ID:-local}

CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL" --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.90 \
  --max-num-seqs 256 --enable-prefix-caching --disable-log-requests ${VLLM_EXTRA:-} > logs/vllm-$J-w.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=1 vllm serve "$OMODEL" --port 8001 --max-model-len 8192 --gpu-memory-utilization 0.90 \
  --max-num-seqs 128 --enable-prefix-caching --disable-log-requests > logs/vllm-$J-o.log 2>&1 &
P2=$!
trap 'kill $P1 $P2 2>/dev/null' EXIT
for port in 8000 8001; do
  for i in $(seq 1 180); do
    curl -sf localhost:$port/v1/models >/dev/null && break
    kill -0 $P1 2>/dev/null && kill -0 $P2 2>/dev/null || { echo "vllm died"; tail -30 logs/vllm-$J-*.log; exit 1; }
    sleep 5
  done
done
echo "both servers up on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
export SWARM_MODEL=$MODEL SWARM_BASE_URL=http://localhost:8000/v1 SWARM_ORCH_MODEL=$OMODEL SWARM_ORCH_BASE_URL=http://localhost:8001/v1
for cmd in "$@"; do echo ">>> $cmd"; bash -c "$cmd"; done
