#!/bin/bash
#SBATCH --job-name=pool_rl_robo_seed
#SBATCH --partition=gpu-large
#SBATCH --qos=batch-long
#SBATCH --constraint=gpu-h100|gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=3
#SBATCH --mem=160G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s226137394@deakin.edu.au
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# ONE job = ONE seed, all 5 envs SEQUENTIAL (no array). Serves a text Qwen3-32B
# (GPU0) AND a vision Qwen3-VL-32B (GPU1) — the full prescribe-and-load P4 on the
# Fetch envs sends rendered frames to the VLM. torch + MuJoCo EGL rendering run
# on GPU2. QwenClient talks to the two servers directly (no proxy).
#   sbatch submit_one_seed.sh                                # seed 42 -> run_0
#   sbatch --export=ALL,SEED=43,RUN_ID=1 submit_one_seed.sh  # any seed
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
SEED="${SEED:-42}"
RUN_ID="${RUN_ID:-$(( SEED - 42 ))}"

if command -v module >/dev/null 2>&1; then module purge || true; module load Anaconda3 || true; fi
[ -f "${HOME}/.bashrc" ] && source "${HOME}/.bashrc"
if command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)"; conda activate pool_rl_robo || true; fi

ENVS=(HalfCheetah-v4 Hopper-v4 Walker2d-v4 FetchReach-v4 FetchPickAndPlace-v4)
TAG="${SLURM_JOB_ID:-$$}"
_PORT_SEED="${SLURM_JOB_ID:-$$}"
LLM_PORT="${LLM_PORT:-$(( 20000 + (_PORT_SEED % 12000) * 3 ))}"
VLM_PORT="${VLM_PORT:-$(( LLM_PORT + 1 ))}"
echo "[seed_job] $(date) seed=${SEED} run_id=${RUN_ID} job=${TAG} LLM:${LLM_PORT} VLM:${VLM_PORT}"

LLM_LOG="slurm_logs/vllm_llm_${TAG}.log"; VLM_LOG="slurm_logs/vllm_vlm_${TAG}.log"
echo "[seed_job] launching text vLLM (GPU0) + vision vLLM (GPU1)"
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
cleanup() { echo "[seed_job] cleanup vLLM"; for p in "${LLM_PID}" "${VLM_PID}"; do kill "$p" 2>/dev/null || true; done; sleep 3; for p in "${LLM_PID}" "${VLM_PID}"; do kill -9 "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

wait_url() { local u="$1" what="$2" pid="$3" t=0 to="${VLLM_READY_TIMEOUT_SEC:-1200}"
    until curl -sf -o /dev/null --max-time 5 "$u"; do
        if ! kill -0 "$pid" 2>/dev/null; then echo "[seed_job] ERROR $what died" >&2; return 1; fi
        if [ "$t" -ge "$to" ]; then echo "[seed_job] ERROR $what not ready in ${to}s" >&2; return 1; fi
        sleep 5; t=$((t+5)); done; echo "[seed_job] $what ready after ${t}s"; }
wait_url "http://127.0.0.1:${LLM_PORT}/v1/models" "text-LLM" "${LLM_PID}"
wait_url "http://127.0.0.1:${VLM_PORT}/v1/models" "VLM" "${VLM_PID}"

export OPENAI_BASE_URL="http://127.0.0.1:${LLM_PORT}/v1"
export VLM_BASE_URL="http://127.0.0.1:${VLM_PORT}/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local-qwen}"
export LLM_MODEL_NAME VLM_MODEL_NAME
export OAI_SDK_TIMEOUT="${OAI_SDK_TIMEOUT:-900}"
export OAI_SDK_MAX_RETRIES="${OAI_SDK_MAX_RETRIES:-2}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

for ENV in "${ENVS[@]}"; do
    echo "================= ENV=${ENV} seed=${SEED} run_id=${RUN_ID} ================="
    CUDA_VISIBLE_DEVICES=2 "${PYBIN}" -u run_experiment.py \
        --env "${ENV}" --seed "${SEED}" --run_id "${RUN_ID}" --config config.yaml \
        2>&1 | tee "logs/${ENV}_run${RUN_ID}.log" || echo "[seed_job] ENV=${ENV} FAILED (continuing)"
done

"${PYBIN}" -u -m \
    Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo.aggregation.aggregate \
    || echo "[seed_job] aggregate failed (continuing)"
echo "[seed_job] $(date) seed=${SEED} DONE"
