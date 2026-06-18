#!/bin/bash
#SBATCH --job-name=pool_rl_robo
#SBATCH --partition=gpu-large
#SBATCH --qos=batch-long
#SBATCH --constraint=gpu-h100|gpu-h200
#SBATCH --array=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=3
#SBATCH --mem=24G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=s226137394@deakin.edu.au
#SBATCH --output=slurm_logs/pool_rl_%A_%a.out
#SBATCH --error=slurm_logs/pool_rl_%A_%a.err
# ===========================================================================
# run_pool_rl_robo.sh — one job runs the requested METHODS for one ManiSkill
# task on a shared diffusion-policy bootstrap (continuous protocol: initial=50,
# budget=100, target_sr=0.90, 1 demo/round). Override SBATCH directives on the
# sbatch CLI (--array, --gpus-per-node, --time) per job.
#
# METHODS selects what runs (default "all"). Only p4_top3 needs the LLM, so:
#   * p4 job (needs VLM+text+orch): 3 GPUs + vLLM + proxy
#       METHODS=p4_top3 sbatch --array=1 --gpus-per-node=3 --time=10-00:00:00 run_pool_rl_robo.sh
#   * baselines job (NO LLM): 1 GPU, no vLLM
#       METHODS=diff_dagger,safe_dagger,dropout_dagger,ensemble_dagger,thrifty_dagger,stagger \
#         sbatch --array=1 --gpus-per-node=1 --time=4-00:00:00 run_pool_rl_robo.sh
#   * smoke (all 7, 1 round):  SMOKE=1 sbatch --array=1 run_pool_rl_robo.sh
# See submit_pusht.sh for the parallel-split launcher.
# ===========================================================================
set -eo pipefail

ENVS=(StackCube-v1 PushT-v1 PickCube-v1 PlugCharger-v1)
ENV="${ENV:-${ENVS[${SLURM_ARRAY_TASK_ID:-1}]}}"
# Methods may arrive '+'-encoded (sbatch --export truncates comma values, C5).
METHODS="${METHODS:-all}"; METHODS="${METHODS//+/,}"
# Any LLM method needs vLLM + the proxy: p4_top3 / p4_subtask (prescribe) or p4_select.
if [ "${METHODS}" = "all" ] || echo ",${METHODS}," | grep -qE ",p4_top3,|,p4_select,|,p4_subtask,"; then
    NEED_LLM=1
else
    NEED_LLM=0
fi

# Suite dir (this script) and DmNfull repo root (for the dotted module path).
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SUITE_DIR="${SLURM_SUBMIT_DIR}"
else
    SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
DMN_ROOT="$(cd "${SUITE_DIR}/../../../.." && pwd)"     # …/DmNfull
FORK_ROOT="$(cd "${SUITE_DIR}/external/diff_dagger" && pwd)"
PKG="Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo"
mkdir -p "${SUITE_DIR}/slurm_logs" "${SUITE_DIR}/logs"
cd "${DMN_ROOT}"

# ── Conda + interpreters ───────────────────────────────────────────────────
if command -v module >/dev/null 2>&1; then module purge || true; module load Anaconda3 || true; fi
[ -f "${HOME}/.bashrc" ] && source "${HOME}/.bashrc"
CONDA_ENV="${CONDA_ENV:-diffdagger}"
if command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)"; conda activate "${CONDA_ENV}" || true; fi
PYTHON_BIN="${PYTHON_BIN:-/home/s226137394/.conda/envs/${CONDA_ENV}/bin/python}"
[ -x "${PYTHON_BIN}" ] || { echo "[run] ERROR: orchestrator interpreter missing: ${PYTHON_BIN}" >&2; exit 3; }

SEED="${SEED:-42}"
RUN_ID="${RUN_ID:-0}"
SMOKE_FLAG=""; [ "${SMOKE:-0}" = "1" ] && SMOKE_FLAG="--smoke"
CONFIG_FLAG=""; [ -n "${CONFIG:-}" ] && CONFIG_FLAG="--config ${CONFIG}"  # e.g. config_industry.yaml
EXTRA_ARGS="${EXTRA_ARGS:-}"

NUM_LOCAL_GPUS=$(python3 -c "import os;cvd=os.environ.get('CUDA_VISIBLE_DEVICES','');print(len([x for x in cvd.split(',') if x.strip()]) if cvd else 0)")
echo "[run] env=${ENV} methods=${METHODS} need_llm=${NEED_LLM} gpus=${NUM_LOCAL_GPUS} seed=${SEED} run_id=${RUN_ID} smoke=${SMOKE:-0}"
# Exact-bootstrap reuse (equal-to-equal): orchestrator/_common.py reads this env var and
# loads the existing init_ckpt.pth instead of rebuilding. Inherited via --export=ALL.
[ -n "${P4_REUSE_INIT_CKPT:-}" ] && { export P4_REUSE_INIT_CKPT; echo "[run] P4_REUSE_INIT_CKPT=${P4_REUSE_INIT_CKPT}"; }

NO_LLM_FLAG=""
if [ "${NEED_LLM}" = "1" ]; then
    # GPU layout: default GPU0 VLM, GPU1 text(LLM), GPU2 orchestrator (3 GPUs).
    # SKIP_VLM=1 (text-only methods, e.g. the StackCube hybrid which never renders
    # for the VLM): GPU0 LLM, GPU1 orchestrator (2 GPUs); the proxy's vlm_port is
    # pointed at the LLM so /healthz stays green without a VLM server.
    _need_gpus=3; _llm_gpu=1; _orch_gpu_default=2
    if [ "${SKIP_VLM:-0}" = "1" ]; then _need_gpus=2; _llm_gpu=0; _orch_gpu_default=1; fi
    if [ "${NUM_LOCAL_GPUS}" -lt "${_need_gpus}" ]; then
        echo "[run] ERROR: p4 needs ${_need_gpus} GPUs (have ${NUM_LOCAL_GPUS}); use --gpus-per-node=${_need_gpus}." >&2
        exit 5
    fi
    VLLM_PYTHON="${VLLM_PYTHON:-/home/s226137394/.conda/envs/vllm_embed/bin/python}"
    PROXY_PYTHON="${PROXY_PYTHON:-/home/s226137394/.conda/envs/maze/bin/python}"
    "${VLLM_PYTHON}" -c "import vllm, bitsandbytes" >/dev/null 2>&1 || { echo "[run] ERROR: VLLM_PYTHON lacks vllm+bitsandbytes" >&2; exit 3; }
    "${PROXY_PYTHON}" -c "import fastapi, uvicorn, httpx" >/dev/null 2>&1 || { echo "[run] ERROR: PROXY_PYTHON lacks fastapi/uvicorn/httpx" >&2; exit 3; }

    LLM_MODEL_PATH="${LLM_MODEL_PATH:-/weka/s226137394/models/Qwen3-32B}"
    VLM_MODEL_PATH="${VLM_MODEL_PATH:-/weka/s226137394/models/Qwen3-VL-32B}"
    export LLM_MODEL_NAME="${LLM_MODEL_NAME:-qwen3-32b}"
    export VLM_MODEL_NAME="${VLM_MODEL_NAME:-qwen3-vl-32b}"
    _PORT_SEED="${SLURM_JOB_ID:-$$}"
    _PORT_BASE=$(( 20000 + (_PORT_SEED % 12000) * 3 ))
    VLLM_LLM_PORT="${VLLM_LLM_PORT:-${_PORT_BASE}}"
    VLLM_VLM_PORT="${VLLM_VLM_PORT:-$(( _PORT_BASE + 1 ))}"
    PROXY_PORT="${PROXY_PORT:-$(( _PORT_BASE + 2 ))}"
    VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-bitsandbytes}"
    VLLM_LOAD_FORMAT="${VLLM_LOAD_FORMAT:-bitsandbytes}"
    VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.92}"
    VLLM_DTYPE="${VLLM_DTYPE:-auto}"
    VLLM_READY_TIMEOUT_SEC="${VLLM_READY_TIMEOUT_SEC:-900}"
    VLLM_EXTRA_FLAGS="${VLLM_EXTRA_FLAGS:---enforce-eager}"
    VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-40960}"
    export PROXY_MAX_OUTPUT_TOKENS="${PROXY_MAX_OUTPUT_TOKENS:-8192}"
    export OAI_SDK_TIMEOUT="${OAI_SDK_TIMEOUT:-900}"
    export OAI_SDK_MAX_RETRIES="${OAI_SDK_MAX_RETRIES:-2}"
    export OAI_MAX_ATTEMPTS="${OAI_MAX_ATTEMPTS:-3}"
    LLM_LOG="${SUITE_DIR}/slurm_logs/vllm_llm_${SLURM_JOB_ID:-local}.log"
    VLM_LOG="${SUITE_DIR}/slurm_logs/vllm_vlm_${SLURM_JOB_ID:-local}.log"
    PROXY_LOG="${SUITE_DIR}/slurm_logs/proxy_${SLURM_JOB_ID:-local}.log"

    CUDA_VISIBLE_DEVICES=${_llm_gpu} "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
        --model "${LLM_MODEL_PATH}" --served-model-name "${LLM_MODEL_NAME}" \
        --quantization "${VLLM_QUANTIZATION}" --load-format "${VLLM_LOAD_FORMAT}" \
        --dtype "${VLLM_DTYPE}" --max-model-len "${VLLM_MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}" --port "${VLLM_LLM_PORT}" \
        --host 127.0.0.1 ${VLLM_EXTRA_FLAGS} > "${LLM_LOG}" 2>&1 &
    LLM_PID=$!
    _eff_vlm_port="${VLLM_VLM_PORT}"
    if [ "${SKIP_VLM:-0}" = "1" ]; then
        _eff_vlm_port="${VLLM_LLM_PORT}"   # proxy health-checks the LLM in place of the VLM
        echo "[run] SKIP_VLM=1: no VLM server (text-only); proxy vlm_port→LLM (${_eff_vlm_port})."
    else
        CUDA_VISIBLE_DEVICES=0 "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
            --model "${VLM_MODEL_PATH}" --served-model-name "${VLM_MODEL_NAME}" \
            --quantization "${VLLM_QUANTIZATION}" --load-format "${VLLM_LOAD_FORMAT}" \
            --dtype "${VLLM_DTYPE}" --max-model-len "${VLLM_MAX_MODEL_LEN}" \
            --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}" --port "${VLLM_VLM_PORT}" \
            --host 127.0.0.1 ${VLLM_EXTRA_FLAGS} > "${VLM_LOG}" 2>&1 &
        VLM_PID=$!
    fi
    VLM_MODEL_NAME="${VLM_MODEL_NAME}" PROXY_MAX_OUTPUT_TOKENS="${PROXY_MAX_OUTPUT_TOKENS}" \
        "${PROXY_PYTHON}" -u "${FORK_ROOT}/diffdagger/qwen/proxy.py" \
        --port "${PROXY_PORT}" --llm_port "${VLLM_LLM_PORT}" \
        --vlm_port "${_eff_vlm_port}" --vlm_model "${VLM_MODEL_NAME}" > "${PROXY_LOG}" 2>&1 &
    PROXY_PID=$!

    cleanup() {
        for pid in "${PROXY_PID:-}" "${VLM_PID:-}" "${LLM_PID:-}"; do
            [ -n "${pid}" ] && kill "${pid}" 2>/dev/null || true
        done
        sleep 3
        for pid in "${PROXY_PID:-}" "${VLM_PID:-}" "${LLM_PID:-}"; do
            [ -n "${pid}" ] && kill -9 "${pid}" 2>/dev/null || true
        done
    }
    trap cleanup EXIT INT TERM

    wait_for_url() {
        local url="$1" what="$2" t=0
        while [ "${t}" -lt "${VLLM_READY_TIMEOUT_SEC}" ]; do
            curl -sf -o /dev/null --max-time 5 "${url}" && { echo "[run] ${what} ready (${t}s)"; return 0; }
            sleep 5; t=$((t + 5))
        done
        echo "[run] ERROR: ${what} not up within ${VLLM_READY_TIMEOUT_SEC}s (${url})" >&2; return 1
    }
    wait_for_url "http://127.0.0.1:${VLLM_LLM_PORT}/v1/models" "vLLM text"
    [ "${SKIP_VLM:-0}" = "1" ] || wait_for_url "http://127.0.0.1:${VLLM_VLM_PORT}/v1/models" "vLLM vision"
    wait_for_url "http://127.0.0.1:${PROXY_PORT}/healthz"      "proxy"

    export OPENAI_BASE_URL="http://127.0.0.1:${PROXY_PORT}/v1"
    export OPENAI_API_KEY="${OPENAI_API_KEY:-local-qwen}"
    export CUDA_VISIBLE_DEVICES="${ORCH_GPU:-${_orch_gpu_default}}"   # orchestrator GPU
    echo "[run] OPENAI_BASE_URL=${OPENAI_BASE_URL}; orchestrator on GPU ${CUDA_VISIBLE_DEVICES}"
else
    # ── baselines: NO LLM. 1 GPU for the orchestrator (ManiSkill + diffusion).
    if [ "${NUM_LOCAL_GPUS}" -lt 1 ]; then
        echo "[run] ERROR: need ≥1 GPU for the orchestrator; have ${NUM_LOCAL_GPUS}." >&2
        exit 5
    fi
    NO_LLM_FLAG="--no-llm"
    echo "[run] baselines-only (no vLLM); orchestrator uses the allocated GPU(s)."
fi

"${PYTHON_BIN}" -u -m "${PKG}.orchestrator.run_one" \
    --env "${ENV}" --methods "${METHODS}" --run_id "${RUN_ID}" --seed "${SEED}" \
    ${CONFIG_FLAG} ${NO_LLM_FLAG} ${SMOKE_FLAG} ${EXTRA_ARGS} 2>&1 | tee "${SUITE_DIR}/logs/${ENV}_run${RUN_ID}_${SLURM_JOB_ID:-local}.log"

echo "[run] DONE env=${ENV} methods=${METHODS} → ${SUITE_DIR}/results/${ENV}/run_${RUN_ID}/"
