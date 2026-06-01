#!/bin/bash
#SBATCH --job-name=pool_rl_robo_smoke2
#SBATCH --partition=gpu-large
#SBATCH --qos=batch-short
#SBATCH --constraint=gpu-h100|gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=3
#SBATCH --mem=160G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s226137394@deakin.edu.au
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# SMOKE 2: P4-LLM ON, ONE env (FetchPickAndPlace), SHORT. Serves text Qwen3-32B
# (GPU0) + vision Qwen3-VL-32B (GPU1); diffusion+render on GPU2. Validates the
# LIVE full prescribe-and-load path: render -> VLM analysis -> prescription ->
# strict loadable+solvable check -> expert demo, on the diffusion backbone.
#   sbatch submit_smoke2.sh   (override env: SMOKE2_ENV=FetchReach-v4 sbatch ...)
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
SMOKE2_ENV="${SMOKE2_ENV:-FetchPickAndPlace-v4}"

if command -v module >/dev/null 2>&1; then module purge || true; module load Anaconda3 || true; fi
[ -f "${HOME}/.bashrc" ] && source "${HOME}/.bashrc"
if command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)"; conda activate pool_rl_robo || true; fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

TAG="${SLURM_JOB_ID:-$$}"; _PS="${SLURM_JOB_ID:-$$}"
LLM_PORT="${LLM_PORT:-$(( 20000 + (_PS % 12000) * 3 ))}"; VLM_PORT="${VLM_PORT:-$(( LLM_PORT + 1 ))}"
echo "[smoke2] $(date) env=${SMOKE2_ENV} LLM:${LLM_PORT} VLM:${VLM_PORT}"
CUDA_VISIBLE_DEVICES=0 "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${LLM_MODEL_PATH}" --served-model-name "${LLM_MODEL_NAME}" \
    --quantization bitsandbytes --load-format bitsandbytes --dtype auto \
    --max-model-len 40960 --gpu-memory-utilization 0.92 --enforce-eager --trust-remote-code \
    --port "${LLM_PORT}" --host 127.0.0.1 > "slurm_logs/vllm_llm_smoke2_${TAG}.log" 2>&1 &
LLM_PID=$!
CUDA_VISIBLE_DEVICES=1 "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${VLM_MODEL_PATH}" --served-model-name "${VLM_MODEL_NAME}" \
    --quantization bitsandbytes --load-format bitsandbytes --dtype auto \
    --max-model-len 40960 --gpu-memory-utilization 0.92 --enforce-eager --trust-remote-code \
    --limit-mm-per-prompt '{"image": 8}' \
    --port "${VLM_PORT}" --host 127.0.0.1 > "slurm_logs/vllm_vlm_smoke2_${TAG}.log" 2>&1 &
VLM_PID=$!
cleanup() { for p in "${LLM_PID}" "${VLM_PID}"; do kill "$p" 2>/dev/null || true; done; sleep 3; for p in "${LLM_PID}" "${VLM_PID}"; do kill -9 "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM
wait_url() { local u="$1" what="$2" pid="$3" t=0 to="${VLLM_READY_TIMEOUT_SEC:-1200}"
    until curl -sf -o /dev/null --max-time 5 "$u"; do
        if ! kill -0 "$pid" 2>/dev/null; then echo "[smoke2] ERROR $what died" >&2; exit 3; fi
        if [ "$t" -ge "$to" ]; then echo "[smoke2] ERROR $what not ready in ${to}s" >&2; exit 4; fi
        sleep 5; t=$((t+5)); done; echo "[smoke2] $what ready after ${t}s"; }
wait_url "http://127.0.0.1:${LLM_PORT}/v1/models" "text-LLM" "${LLM_PID}"
wait_url "http://127.0.0.1:${VLM_PORT}/v1/models" "VLM" "${VLM_PID}"

export OPENAI_BASE_URL="http://127.0.0.1:${LLM_PORT}/v1"
export VLM_BASE_URL="http://127.0.0.1:${VLM_PORT}/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local-qwen}"
export LLM_MODEL_NAME VLM_MODEL_NAME
export OAI_SDK_TIMEOUT="${OAI_SDK_TIMEOUT:-900}" OAI_SDK_MAX_RETRIES="${OAI_SDK_MAX_RETRIES:-2}"

echo "===== SMOKE 2: live full-P4 on ${SMOKE2_ENV} (diffusion + VLM) ====="
CUDA_VISIBLE_DEVICES=2 "${PYBIN}" -u run_experiment.py \
    --env "${SMOKE2_ENV}" --methods p4_llm --config config_smoke2.yaml \
    2>&1 | tee "logs/smoke2_${SMOKE2_ENV}.log"
echo "[smoke2] $(date) DONE"
