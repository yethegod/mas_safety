#!/bin/bash
# Start a local vLLM OpenAI server on the allocated GPU, wait for it, then run "$@" from the repo root.
# sbatch scripts/job.sh python exp0.py --part pilot
#SBATCH --account=bhta-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=swarm
#SBATCH --output=/u/zsun9/mas_safety/logs/%x-%j.out

set -euo pipefail
W=/work/nvme/bhta/zsun9/swarm
export HF_HOME=$W/hf HF_HUB_OFFLINE=1
export PATH=$W/venv/bin:$PATH
MODEL=${SWARM_MODEL:-Qwen/Qwen3-8B}
PORT=${PORT:-8000}
cd /u/zsun9/mas_safety

vllm serve "$MODEL" --port $PORT --max-model-len 8192 --gpu-memory-utilization 0.90 \
  --max-num-seqs 256 --enable-prefix-caching --disable-log-requests ${VLLM_EXTRA:-} > logs/vllm-${SLURM_JOB_ID:-local}.log 2>&1 &
VPID=$!
trap 'kill $VPID 2>/dev/null' EXIT
for i in $(seq 1 180); do
  curl -sf localhost:$PORT/v1/models >/dev/null && break
  kill -0 $VPID 2>/dev/null || { echo "vllm died"; tail -50 logs/vllm-${SLURM_JOB_ID:-local}.log; exit 1; }
  sleep 5
done
echo "vllm up after $((i*5))s on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
export SWARM_MODEL=$MODEL SWARM_BASE_URL=http://localhost:$PORT/v1
for cmd in "$@"; do echo ">>> $cmd"; bash -c "$cmd"; done
