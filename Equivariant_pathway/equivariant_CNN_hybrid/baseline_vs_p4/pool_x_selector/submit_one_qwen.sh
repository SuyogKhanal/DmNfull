#!/bin/bash
#SBATCH --job-name=bvp_qwen
#SBATCH --partition=gpu-large
#SBATCH --qos=batch-long
#SBATCH --constraint=gpu-h100|gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --mem=128G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s226137394@deakin.edu.au
#SBATCH --output=slurm_logs/bvp_qwen_%j.out
#SBATCH --error=slurm_logs/bvp_qwen_%j.err
#
# One SLURM job = one run_id, running on the open-source Qwen cluster.
#
# Side-by-side vLLM servers on GPU 0 (Qwen3-VL-32B, port 8001) and GPU 1
# (Qwen3-32B, port 8000), plus a tiny FastAPI proxy on port 8002 that
# routes /v1/chat/completions to the right backend by ``body["model"]``.
# OPENAI_BASE_URL points at the proxy so the existing pipeline code is
# byte-identical to the OpenAI path.
#
# Submit ten of these with:
#   SUBMIT_SCRIPT=submit_one_qwen.sh bash submit_all.sh
# or by hand:
#   for i in {1..10}; do
#       sbatch --export=ALL,RUN_ID=$i submit_one_qwen.sh
#   done

set -eo pipefail

# Under SLURM the script is copied to a spool dir; prefer SLURM_SUBMIT_DIR.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

# Conda env activation (mirrors submit_one.sh).
if command -v module >/dev/null 2>&1; then
    module purge || true
    module load Anaconda3 || true
fi
if [ -f "${HOME}/.bashrc" ]; then
    # shellcheck disable=SC1091
    source "${HOME}/.bashrc"
fi
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${QWEN_CONDA_ENV:-maze}" || true
fi

mkdir -p "${SCRIPT_DIR}/slurm_logs" "${SCRIPT_DIR}/logs"

if [ -z "${RUN_ID:-}" ]; then
    echo "[submit_one_qwen] ERROR: RUN_ID not set." >&2
    exit 2
fi

# ----------------------------- knobs (mirror submit_one.sh) -----------------------------
CONFIG="${CONFIG:-${SCRIPT_DIR}/config.yaml}"
BUDGET="${BUDGET:-9999}"
TARGET_SR="${TARGET_SR:-1.0}"
CORRECTION_N="${CORRECTION_N:-}"
HELDOUT_N="${HELDOUT_N:-}"
INITIAL_DEMOS="${INITIAL_DEMOS:-}"
INITIAL_EPOCHS="${INITIAL_EPOCHS:-}"
ROUND_EPOCHS="${ROUND_EPOCHS:-600}"
BASELINE_ROUND_EPOCHS="${BASELINE_ROUND_EPOCHS:-90}"
MAX_ROUNDS="${MAX_ROUNDS:-}"
MAX_STEPS="${MAX_STEPS:-}"
SEED="${SEED:-}"
LR="${LR:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
WEIGHT_DECAY="${WEIGHT_DECAY:-}"
REPLAY_MIX="${REPLAY_MIX:-}"
REPLAY_MIX_FLOOR="${REPLAY_MIX_FLOOR:-}"
CORRIDOR_BLOCKING="${CORRIDOR_BLOCKING:-}"
METHODS="${METHODS:-}"
# submit_{smoke,all}.sh encodes commas as '+' to survive sbatch --export
# (which uses commas as the variable separator). Decode here.
METHODS="${METHODS//+/,}"

# ----------------------------- Qwen / vLLM knobs ----------------------------------------
LLM_MODEL_PATH="${LLM_MODEL_PATH:-/weka/s226137394/models/Qwen3-32B}"
VLM_MODEL_PATH="${VLM_MODEL_PATH:-/weka/s226137394/models/Qwen3-VL-32B}"
LLM_MODEL_NAME="${LLM_MODEL_NAME:-qwen3-32b}"        # served name; pipeline asks for this
VLM_MODEL_NAME="${VLM_MODEL_NAME:-qwen3-vl-32b}"
# Ports must be unique per job: a gpu-large node has 8 GPUs and each job
# takes 2, so up to 4 jobs co-locate on one node. Hardcoded 8000/8001/8002
# collide (OSError: [Errno 98] Address already in use) — and also clash
# with orphaned vLLM servers left by cancelled jobs. Derive a per-job port
# block from the unique SLURM job id (job ids are monotonic & unique, so
# co-located jobs and prior dead jobs never share a block). Still
# overridable via env for manual single-node runs.
_PORT_SEED="${SLURM_JOB_ID:-$$}"
_PORT_BASE=$(( 20000 + (_PORT_SEED % 12000) * 3 ))
VLLM_LLM_PORT="${VLLM_LLM_PORT:-${_PORT_BASE}}"
VLLM_VLM_PORT="${VLLM_VLM_PORT:-$(( _PORT_BASE + 1 ))}"
PROXY_PORT="${PROXY_PORT:-$(( _PORT_BASE + 2 ))}"
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-bitsandbytes}"
VLLM_LOAD_FORMAT="${VLLM_LOAD_FORMAT:-bitsandbytes}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.92}"
VLLM_DTYPE="${VLLM_DTYPE:-auto}"
VLLM_READY_TIMEOUT_SEC="${VLLM_READY_TIMEOUT_SEC:-600}"
VLLM_EXTRA_FLAGS="${VLLM_EXTRA_FLAGS:---enforce-eager}"
# Context length. Qwen3-32B's max_position_embeddings is 40960 — vLLM
# REFUSES to start if max_model_len exceeds it (RoPE → NaN risk), so this
# is a hard ceiling: don't set it higher (65536 failed with "User-specified
# max_model_len (65536) is greater than the derived max_model_len
# (max_position_embeddings=40960)"). We pair this with a proxy-side cap on
# max_output_tokens (8192) so the aggregator's large input (~16k at
# correction_n=50) still leaves ~32k of input room within 40960. KV cache
# at 40960 is ~10 GiB, well under the ~49 GiB free after model weights.
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-40960}"
# Proxy caps each request's max_output_tokens (the pipeline asks for 16384,
# which would overflow the 40960 ctx). Export BEFORE the proxy launches so
# the proxy process inherits it. Default 8192 matches the proxy's own.
export PROXY_MAX_OUTPUT_TOKENS="${PROXY_MAX_OUTPUT_TOKENS:-8192}"

# vLLM lives in a separate conda env (the maze env has no vllm). The two
# vLLM servers below launch with an explicit interpreter; the proxy and
# orchestrator keep using whatever env this script activated (maze).
VLLM_PYTHON="${VLLM_PYTHON:-/home/s226137394/.conda/envs/vllm_embed/bin/python}"
if [ ! -x "${VLLM_PYTHON}" ]; then
    echo "[submit_one_qwen] ERROR: VLLM_PYTHON not executable: ${VLLM_PYTHON}" >&2
    exit 3
fi
if ! "${VLLM_PYTHON}" -c "import vllm" >/dev/null 2>&1; then
    echo "[submit_one_qwen] ERROR: ${VLLM_PYTHON} cannot import vllm" >&2
    exit 3
fi

# Build pass-through CLI flags for the python orchestrator (same scheme
# as submit_one.sh — only emit a flag when its env var is set).
CLI=()
[ -n "${METHODS}" ]               && CLI+=(--methods "${METHODS}")
[ -n "${BUDGET}" ]                && CLI+=(--budget "${BUDGET}")
[ -n "${TARGET_SR}" ]             && CLI+=(--target_sr "${TARGET_SR}")
[ -n "${CORRECTION_N}" ]          && CLI+=(--correction_n "${CORRECTION_N}")
[ -n "${HELDOUT_N}" ]             && CLI+=(--heldout_n "${HELDOUT_N}")
[ -n "${INITIAL_DEMOS}" ]         && CLI+=(--initial_demos "${INITIAL_DEMOS}")
[ -n "${INITIAL_EPOCHS}" ]        && CLI+=(--initial_epochs "${INITIAL_EPOCHS}")
[ -n "${ROUND_EPOCHS}" ]          && CLI+=(--p4_finetune_epochs "${ROUND_EPOCHS}")
[ -n "${BASELINE_ROUND_EPOCHS}" ] && CLI+=(--baseline_finetune_epochs "${BASELINE_ROUND_EPOCHS}")
[ -n "${MAX_ROUNDS}" ]            && CLI+=(--max_rounds "${MAX_ROUNDS}")
[ -n "${MAX_STEPS}" ]             && CLI+=(--max_steps "${MAX_STEPS}")
[ -n "${SEED}" ]                  && CLI+=(--seed "${SEED}")
[ -n "${LR}" ]                    && CLI+=(--lr "${LR}")
[ -n "${BATCH_SIZE}" ]            && CLI+=(--batch_size "${BATCH_SIZE}")
[ -n "${WEIGHT_DECAY}" ]          && CLI+=(--weight_decay "${WEIGHT_DECAY}")
[ -n "${REPLAY_MIX}" ]            && CLI+=(--replay_mix "${REPLAY_MIX}")
[ -n "${REPLAY_MIX_FLOOR}" ]      && CLI+=(--replay_mix_floor "${REPLAY_MIX_FLOOR}")
[ -n "${CORRIDOR_BLOCKING}" ]     && CLI+=(--corridor_blocking "${CORRIDOR_BLOCKING}")

PER_RUN_LOG="${SCRIPT_DIR}/logs/run_$(printf '%02d' "${RUN_ID}").log"
PROXY_LOG="${SCRIPT_DIR}/slurm_logs/proxy_${SLURM_JOB_ID:-local}.log"
LLM_LOG="${SCRIPT_DIR}/slurm_logs/vllm_llm_${SLURM_JOB_ID:-local}.log"
VLM_LOG="${SCRIPT_DIR}/slurm_logs/vllm_vlm_${SLURM_JOB_ID:-local}.log"

echo "[submit_one_qwen] $(date)  RUN_ID=${RUN_ID}  job=${SLURM_JOB_ID:-?}"
echo "[submit_one_qwen] LLM_MODEL_PATH=${LLM_MODEL_PATH}  served_name=${LLM_MODEL_NAME}"
echo "[submit_one_qwen] VLM_MODEL_PATH=${VLM_MODEL_PATH}  served_name=${VLM_MODEL_NAME}"
echo "[submit_one_qwen] ports: llm=${VLLM_LLM_PORT}  vlm=${VLLM_VLM_PORT}  proxy=${PROXY_PORT}"
echo "[submit_one_qwen] logs:  proxy=${PROXY_LOG}  llm=${LLM_LOG}  vlm=${VLM_LOG}"
echo "[submit_one_qwen] VLLM_PYTHON=${VLLM_PYTHON}  VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN}"
echo "[submit_one_qwen] CLI overrides: ${CLI[*]:-<none>}"

# ----- diagnose GPU allocation ----------------------------------------------------------
echo "[submit_one_qwen] SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>&1 | sed 's/^/[submit_one_qwen] nvidia-smi: /'
else
    echo "[submit_one_qwen] nvidia-smi: not in PATH"
fi
# Sanity-check: both vLLM servers need their own GPU on this node.
NUM_LOCAL_GPUS=$(python3 -c "
import os
cvd=os.environ.get('CUDA_VISIBLE_DEVICES','')
print(len([x for x in cvd.split(',') if x.strip()]) if cvd else 0)
")
if [ "${NUM_LOCAL_GPUS}" -lt 2 ]; then
    echo "[submit_one_qwen] ERROR: need 2 GPUs on this node, but CUDA_VISIBLE_DEVICES has ${NUM_LOCAL_GPUS} (='${CUDA_VISIBLE_DEVICES:-}')." >&2
    echo "[submit_one_qwen] Check #SBATCH directives — must use --nodes=1 --gpus-per-node=2 (not --gpus=2 which can spread across nodes)." >&2
    exit 5
fi

# ----- start vLLM LLM server on GPU 1 ---------------------------------------------------
CUDA_VISIBLE_DEVICES=1 "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${LLM_MODEL_PATH}" \
    --served-model-name "${LLM_MODEL_NAME}" \
    --quantization "${VLLM_QUANTIZATION}" \
    --load-format "${VLLM_LOAD_FORMAT}" \
    --dtype "${VLLM_DTYPE}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}" \
    --port "${VLLM_LLM_PORT}" --host 127.0.0.1 \
    ${VLLM_EXTRA_FLAGS} \
    > "${LLM_LOG}" 2>&1 &
LLM_PID=$!
echo "[submit_one_qwen] launched vLLM LLM  pid=${LLM_PID}  gpu=1  port=${VLLM_LLM_PORT}"

# ----- start vLLM VLM server on GPU 0 ---------------------------------------------------
CUDA_VISIBLE_DEVICES=0 "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${VLM_MODEL_PATH}" \
    --served-model-name "${VLM_MODEL_NAME}" \
    --quantization "${VLLM_QUANTIZATION}" \
    --load-format "${VLLM_LOAD_FORMAT}" \
    --dtype "${VLLM_DTYPE}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}" \
    --port "${VLLM_VLM_PORT}" --host 127.0.0.1 \
    ${VLLM_EXTRA_FLAGS} \
    > "${VLM_LOG}" 2>&1 &
VLM_PID=$!
echo "[submit_one_qwen] launched vLLM VLM  pid=${VLM_PID}  gpu=0  port=${VLLM_VLM_PORT}"

# ----- start the routing proxy on CPU --------------------------------------------------
python -u -m \
    Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_x_selector.qwen.proxy \
    --port "${PROXY_PORT}" \
    --llm_port "${VLLM_LLM_PORT}" \
    --vlm_port "${VLLM_VLM_PORT}" \
    --vlm_model "${VLM_MODEL_NAME}" \
    > "${PROXY_LOG}" 2>&1 &
PROXY_PID=$!
echo "[submit_one_qwen] launched proxy   pid=${PROXY_PID}  port=${PROXY_PORT}"

# Make sure children are torn down no matter how this script exits.
cleanup() {
    echo "[submit_one_qwen] cleanup: killing children"
    for pid in "${PROXY_PID:-}" "${VLM_PID:-}" "${LLM_PID:-}"; do
        if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
    # Give them a moment, then SIGKILL anything still alive.
    sleep 3
    for pid in "${PROXY_PID:-}" "${VLM_PID:-}" "${LLM_PID:-}"; do
        if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
            kill -9 "${pid}" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT INT TERM

# ----- wait for all three to come up ----------------------------------------------------
wait_for_url() {
    local url="$1"
    local what="$2"
    local timeout="${VLLM_READY_TIMEOUT_SEC}"
    local elapsed=0
    while [ "${elapsed}" -lt "${timeout}" ]; do
        if curl -sf -o /dev/null --max-time 5 "${url}"; then
            echo "[submit_one_qwen] ${what} ready (${url})  after ${elapsed}s"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    echo "[submit_one_qwen] ERROR: ${what} did not come up within ${timeout}s (${url})" >&2
    return 1
}

wait_for_url "http://127.0.0.1:${VLLM_LLM_PORT}/v1/models" "vLLM LLM (port ${VLLM_LLM_PORT})"
wait_for_url "http://127.0.0.1:${VLLM_VLM_PORT}/v1/models" "vLLM VLM (port ${VLLM_VLM_PORT})"
wait_for_url "http://127.0.0.1:${PROXY_PORT}/healthz"      "proxy (port ${PROXY_PORT})"

# ----- point the pipeline at the proxy --------------------------------------------------
export OPENAI_BASE_URL="http://127.0.0.1:${PROXY_PORT}/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local-qwen}"
export LLM_MODEL_NAME
export VLM_MODEL_NAME

# Robustness against the retry death-spiral: slow bnb+eager generation can
# exceed the 120s default SDK timeout, and timed-out requests aren't
# cancelled server-side, so retries pile up and throughput collapses. Give
# requests time to finish and keep concurrency low. These are operational
# knobs only (they don't change what the model generates). The proxy also
# caps max_output_tokens (PROXY_MAX_OUTPUT_TOKENS, default 8192).
export OAI_SDK_TIMEOUT="${OAI_SDK_TIMEOUT:-900}"
export OAI_MAX_IN_FLIGHT="${OAI_MAX_IN_FLIGHT:-2}"
export OAI_SDK_MAX_RETRIES="${OAI_SDK_MAX_RETRIES:-2}"

echo "[submit_one_qwen] env: OPENAI_BASE_URL=${OPENAI_BASE_URL}  LLM_MODEL_NAME=${LLM_MODEL_NAME}  VLM_MODEL_NAME=${VLM_MODEL_NAME}"
echo "[submit_one_qwen] starting orchestrator -> ${PER_RUN_LOG}"

# ----- run the orchestrator (same module as the OpenAI path) ----------------------------
python3 -u -m \
    Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_x_selector.orchestrator.run_one \
    --run_id "${RUN_ID}" \
    --config "${CONFIG}" \
    "${CLI[@]}" \
    2>&1 | tee "${PER_RUN_LOG}"

echo "[submit_one_qwen] $(date)  RUN_ID=${RUN_ID} DONE"
