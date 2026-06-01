#!/bin/bash
#SBATCH --job-name=pool_rl_robo_qwen
#SBATCH --partition=gpu-large
#SBATCH --qos=batch-long
#SBATCH --constraint=gpu-h100|gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --mem=128G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=48:00:00
#SBATCH --array=0-4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s226137394@deakin.edu.au
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#
# Main production launcher: one array task per environment. Each task launches a
# text-only Qwen3-32B vLLM (GPU 0) then runs run_experiment.py (P4-LLM + the 5
# IIL baselines; torch on GPU 1). Submit via submit_all.sh (or directly:
#   mkdir -p slurm_logs results logs && sbatch submit_one_qwen.sh).
# NOTE --nodes=1 --gpus-per-node=2 (NEVER --gpus=2). ~/.bashrc prepends maze/bin
# to PATH, so we use EXPLICIT interpreters below.
set -eo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then SCRIPT_DIR="${SLURM_SUBMIT_DIR}"; else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; fi
cd "${SCRIPT_DIR}"
mkdir -p slurm_logs results logs

PYBIN="${PYBIN:-/home/s226137394/.conda/envs/pool_rl_robo/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-/home/s226137394/.conda/envs/vllm_embed/bin/python}"
LLM_MODEL_PATH="${LLM_MODEL_PATH:-/weka/s226137394/models/Qwen3-32B}"
LLM_MODEL_NAME="${LLM_MODEL_NAME:-qwen3-32b}"

if command -v module >/dev/null 2>&1; then module purge || true; module load Anaconda3 || true; fi
[ -f "${HOME}/.bashrc" ] && source "${HOME}/.bashrc"
if command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)"; conda activate pool_rl_robo || true; fi

ENVS=(HalfCheetah-v4 Hopper-v4 Walker2d-v4 FetchReach-v4 FetchPickAndPlace-v4)
ENV="${ENVS[${SLURM_ARRAY_TASK_ID:-0}]}"
TAG="${SLURM_JOB_ID:-$$}"
echo "[pool_rl_robo] $(date) task=${SLURM_ARRAY_TASK_ID:-0} ENV=${ENV} job=${TAG}"

_PORT_SEED="${SLURM_JOB_ID:-$$}"
LLM_PORT="${LLM_PORT:-$(( 20000 + (_PORT_SEED % 12000) * 3 ))}"
VLLM_LOG="slurm_logs/vllm_${TAG}_${SLURM_ARRAY_TASK_ID:-0}.log"
echo "[pool_rl_robo] launching vLLM ${LLM_MODEL_NAME} GPU0 port ${LLM_PORT} -> ${VLLM_LOG}"
CUDA_VISIBLE_DEVICES=0 "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${LLM_MODEL_PATH}" --served-model-name "${LLM_MODEL_NAME}" \
    --quantization bitsandbytes --load-format bitsandbytes --dtype auto \
    --max-model-len 40960 --gpu-memory-utilization 0.92 --enforce-eager \
    --port "${LLM_PORT}" --host 127.0.0.1 > "${VLLM_LOG}" 2>&1 &
LLM_PID=$!
cleanup() { echo "[pool_rl_robo] cleanup vLLM ${LLM_PID}"; kill "${LLM_PID}" 2>/dev/null || true; sleep 3; kill -9 "${LLM_PID}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

READY_TIMEOUT="${VLLM_READY_TIMEOUT_SEC:-900}"; elapsed=0
until curl -sf -o /dev/null --max-time 5 "http://127.0.0.1:${LLM_PORT}/v1/models"; do
    if ! kill -0 "${LLM_PID}" 2>/dev/null; then echo "[pool_rl_robo] ERROR vLLM died (see ${VLLM_LOG})" >&2; exit 3; fi
    if [ "${elapsed}" -ge "${READY_TIMEOUT}" ]; then echo "[pool_rl_robo] ERROR vLLM not ready in ${READY_TIMEOUT}s" >&2; exit 4; fi
    sleep 5; elapsed=$((elapsed + 5))
done
echo "[pool_rl_robo] vLLM ready after ${elapsed}s"

export OPENAI_BASE_URL="http://127.0.0.1:${LLM_PORT}/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local-qwen}"
export LLM_MODEL_NAME
export OAI_SDK_TIMEOUT="${OAI_SDK_TIMEOUT:-900}"
export OAI_SDK_MAX_RETRIES="${OAI_SDK_MAX_RETRIES:-2}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

CUDA_VISIBLE_DEVICES=1 "${PYBIN}" -u run_experiment.py \
    --env "${ENV}" --seed "${SEED:-42}" --config config.yaml \
    2>&1 | tee "logs/${ENV}_run0.log"
echo "[pool_rl_robo] $(date) ENV=${ENV} DONE"
