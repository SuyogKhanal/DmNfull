#!/bin/bash
#SBATCH --job-name=pool_rl_robo_smoke
#SBATCH --partition=gpu-large
#SBATCH --qos=batch-short
#SBATCH --constraint=gpu-h100|gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=3
#SBATCH --mem=160G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=03:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s226137394@deakin.edu.au
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# FULL-pipeline GPU smoke gate (sbatch submit_smoke.sh). Serves text Qwen3-32B
# (GPU0) + vision Qwen3-VL-32B (GPU1); torch+render GPU2. Validates, in order:
#   1) the 5 experts load+play (no LLM)
#   2) token budget + a live TEXT round-trip
#   3) the LIVE full prescribe-and-load P4 on FetchPickAndPlace (render -> VLM
#      analysis -> prescription -> strict loadable+solvable check -> expert demo)
# Must pass before submit_5seeds_5jobs.sh.
set -eo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then SCRIPT_DIR="${SLURM_SUBMIT_DIR}"; else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; fi
cd "${SCRIPT_DIR}"
mkdir -p slurm_logs results logs

PYBIN="${PYBIN:-/home/s226137394/.conda/envs/pool_rl_robo/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-/home/s226137394/.conda/envs/vllm_embed/bin/python}"
LLM_MODEL_PATH="${LLM_MODEL_PATH:-/weka/s226137394/models/Qwen3-32B}"
LLM_MODEL_NAME="${LLM_MODEL_NAME:-qwen3-32b}"
VLM_MODEL_PATH="${VLM_MODEL_PATH:-/weka/s226137394/models/Qwen3-VL-32B}"
VLM_MODEL_NAME="${VLM_MODEL_NAME:-qwen3-vl-32b}"

if command -v module >/dev/null 2>&1; then module purge || true; module load Anaconda3 || true; fi
[ -f "${HOME}/.bashrc" ] && source "${HOME}/.bashrc"
if command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)"; conda activate pool_rl_robo || true; fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

echo "===== SMOKE 1/3: experts (5 must print checkmark) ====="
"${PYBIN}" smoke_test.py

TAG="${SLURM_JOB_ID:-$$}"; _PS="${SLURM_JOB_ID:-$$}"
LLM_PORT="${LLM_PORT:-$(( 20000 + (_PS % 12000) * 3 ))}"; VLM_PORT="${VLM_PORT:-$(( LLM_PORT + 1 ))}"
LLM_LOG="slurm_logs/vllm_llm_smoke_${TAG}.log"; VLM_LOG="slurm_logs/vllm_vlm_smoke_${TAG}.log"
echo "===== launching text vLLM (GPU0:${LLM_PORT}) + vision vLLM (GPU1:${VLM_PORT}) ====="
CUDA_VISIBLE_DEVICES=0 "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${LLM_MODEL_PATH}" --served-model-name "${LLM_MODEL_NAME}" \
    --quantization bitsandbytes --load-format bitsandbytes --dtype auto \
    --max-model-len 40960 --gpu-memory-utilization 0.92 --enforce-eager \
    --trust-remote-code --port "${LLM_PORT}" --host 127.0.0.1 > "${LLM_LOG}" 2>&1 &
LLM_PID=$!
CUDA_VISIBLE_DEVICES=1 "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${VLM_MODEL_PATH}" --served-model-name "${VLM_MODEL_NAME}" \
    --quantization bitsandbytes --load-format bitsandbytes --dtype auto \
    --max-model-len 40960 --gpu-memory-utilization 0.92 --enforce-eager \
    --trust-remote-code --limit-mm-per-prompt '{"image": 8}' \
    --port "${VLM_PORT}" --host 127.0.0.1 > "${VLM_LOG}" 2>&1 &
VLM_PID=$!
cleanup() { for p in "${LLM_PID}" "${VLM_PID}"; do kill "$p" 2>/dev/null || true; done; sleep 3; for p in "${LLM_PID}" "${VLM_PID}"; do kill -9 "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM
wait_url() { local u="$1" what="$2" pid="$3" t=0 to="${VLLM_READY_TIMEOUT_SEC:-1200}"
    until curl -sf -o /dev/null --max-time 5 "$u"; do
        if ! kill -0 "$pid" 2>/dev/null; then echo "[smoke] ERROR $what died (see log)" >&2; exit 3; fi
        if [ "$t" -ge "$to" ]; then echo "[smoke] ERROR $what not ready in ${to}s" >&2; exit 4; fi
        sleep 5; t=$((t+5)); done; echo "[smoke] $what ready after ${t}s"; }
wait_url "http://127.0.0.1:${LLM_PORT}/v1/models" "text-LLM" "${LLM_PID}"
wait_url "http://127.0.0.1:${VLM_PORT}/v1/models" "VLM" "${VLM_PID}"

export OPENAI_BASE_URL="http://127.0.0.1:${LLM_PORT}/v1"
export VLM_BASE_URL="http://127.0.0.1:${VLM_PORT}/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local-qwen}"
export LLM_MODEL_NAME VLM_MODEL_NAME
export OAI_SDK_TIMEOUT="${OAI_SDK_TIMEOUT:-900}" OAI_SDK_MAX_RETRIES="${OAI_SDK_MAX_RETRIES:-2}"

echo "===== SMOKE 2/3: token budget + live TEXT round-trip ====="
"${PYBIN}" smoke_llm.py --live

echo "===== SMOKE 3/3: LIVE full prescribe-and-load P4 (FetchPickAndPlace, --smoke) ====="
CUDA_VISIBLE_DEVICES=2 "${PYBIN}" run_experiment.py \
    --env FetchPickAndPlace-v4 --methods p4_llm --smoke 2>&1 | tee "logs/smoke_fullp4.log"

echo "SMOKE_ALL_DONE $(date)"
