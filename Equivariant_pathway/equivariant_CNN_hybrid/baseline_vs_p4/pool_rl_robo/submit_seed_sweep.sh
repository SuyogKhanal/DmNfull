#!/bin/bash
#SBATCH --job-name=pool_rl_robo_sweep
#SBATCH --partition=gpu-large
#SBATCH --qos=batch-long
#SBATCH --constraint=gpu-h100|gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --mem=128G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=48:00:00
#SBATCH --array=0-24
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s226137394@deakin.edu.au
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#
# Multi-seed production sweep: 25 tasks = 5 envs x 5 seeds (42..46). Task id
# decodes to (env_idx = id%5, seed_idx = id//5); seed = 42+seed_idx, run_id =
# seed_idx (so results/{ENV}/run_{0..4}/ and {ENV}_run{0..4}.json don't collide).
# Each task launches its own text Qwen3-32B vLLM (GPU0) + runs all 6 methods
# (torch GPU1). Submit via submit_seeds_all.sh.
set -eo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then SCRIPT_DIR="${SLURM_SUBMIT_DIR}"; else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; fi
cd "${SCRIPT_DIR}"
mkdir -p slurm_logs results logs

PYBIN="${PYBIN:-/home/s226137394/.conda/envs/pool_rl_robo/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-/home/s226137394/.conda/envs/vllm_embed/bin/python}"
LLM_MODEL_PATH="${LLM_MODEL_PATH:-/weka/s226137394/models/Qwen3-32B}"
LLM_MODEL_NAME="${LLM_MODEL_NAME:-qwen3-32b}"
SEED_BASE="${SEED_BASE:-42}"
N_ENVS=5

if command -v module >/dev/null 2>&1; then module purge || true; module load Anaconda3 || true; fi
[ -f "${HOME}/.bashrc" ] && source "${HOME}/.bashrc"
if command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)"; conda activate pool_rl_robo || true; fi

ENVS=(HalfCheetah-v4 Hopper-v4 Walker2d-v4 FetchReach-v4 FetchPickAndPlace-v4)
TASK="${SLURM_ARRAY_TASK_ID:-0}"
ENV_IDX=$(( TASK % N_ENVS ))
SEED_IDX=$(( TASK / N_ENVS ))
ENV="${ENVS[$ENV_IDX]}"
SEED=$(( SEED_BASE + SEED_IDX ))
RUN_ID="${SEED_IDX}"
TAG="${SLURM_JOB_ID:-$$}"
echo "[sweep] $(date) task=${TASK} ENV=${ENV} seed=${SEED} run_id=${RUN_ID} job=${TAG}"

_PORT_SEED="${SLURM_JOB_ID:-$$}"
LLM_PORT="${LLM_PORT:-$(( 20000 + (_PORT_SEED % 12000) * 3 ))}"
VLLM_LOG="slurm_logs/vllm_${TAG}_${TASK}.log"
echo "[sweep] launching vLLM ${LLM_MODEL_NAME} GPU0 port ${LLM_PORT} -> ${VLLM_LOG}"
CUDA_VISIBLE_DEVICES=0 "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${LLM_MODEL_PATH}" --served-model-name "${LLM_MODEL_NAME}" \
    --quantization bitsandbytes --load-format bitsandbytes --dtype auto \
    --max-model-len 40960 --gpu-memory-utilization 0.92 --enforce-eager \
    --port "${LLM_PORT}" --host 127.0.0.1 > "${VLLM_LOG}" 2>&1 &
LLM_PID=$!
cleanup() { echo "[sweep] cleanup vLLM ${LLM_PID}"; kill "${LLM_PID}" 2>/dev/null || true; sleep 3; kill -9 "${LLM_PID}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

READY_TIMEOUT="${VLLM_READY_TIMEOUT_SEC:-900}"; elapsed=0
until curl -sf -o /dev/null --max-time 5 "http://127.0.0.1:${LLM_PORT}/v1/models"; do
    if ! kill -0 "${LLM_PID}" 2>/dev/null; then echo "[sweep] ERROR vLLM died (see ${VLLM_LOG})" >&2; exit 3; fi
    if [ "${elapsed}" -ge "${READY_TIMEOUT}" ]; then echo "[sweep] ERROR vLLM not ready in ${READY_TIMEOUT}s" >&2; exit 4; fi
    sleep 5; elapsed=$((elapsed + 5))
done
echo "[sweep] vLLM ready after ${elapsed}s"

export OPENAI_BASE_URL="http://127.0.0.1:${LLM_PORT}/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local-qwen}"
export LLM_MODEL_NAME
export OAI_SDK_TIMEOUT="${OAI_SDK_TIMEOUT:-900}"
export OAI_SDK_MAX_RETRIES="${OAI_SDK_MAX_RETRIES:-2}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

CUDA_VISIBLE_DEVICES=1 "${PYBIN}" -u run_experiment.py \
    --env "${ENV}" --seed "${SEED}" --run_id "${RUN_ID}" --config config.yaml \
    2>&1 | tee "logs/${ENV}_run${RUN_ID}.log"
echo "[sweep] $(date) ENV=${ENV} seed=${SEED} DONE"
