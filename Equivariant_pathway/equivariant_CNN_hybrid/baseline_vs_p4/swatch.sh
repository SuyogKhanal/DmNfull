#!/bin/bash
#SBATCH --job-name=bo_vs_p4_hybrid
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --gpus=4
#SBATCH --mem=200G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s226137394@deakin.edu.au
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# Hybrid baseline_only vs p4_only — budget-constrained comparison
# across N_RUNS repetitions in parallel.
#
# Each repetition:
#   * has its own run_NN/ directory with isolated demo + checkpoint dirs
#     (no inter-run contamination of training state)
#   * shares the SAME 20 training layouts and 200 heldout layouts as
#     every other repetition (the only randomised piece is the 40-layout
#     correction pool, seeded by the run index)
#   * runs baseline_only and p4_only sequentially against an identical
#     initial model so the only difference is the demo-selection policy
#
# Parallel design:
#   * N_RUNS repetitions are launched in parallel, GPU-pinned round-robin
#     across $NUM_GPUS available cards.
#   * The OpenAI rate-limit budget is split across the parallel processes
#     via OAI_TPM_SHARE so the org-wide TPM ceiling is respected. The
#     in-process token bucket + in-flight semaphore in
#     pipeline/_oai_retry.py absorbs the rest of the pressure.
#   * P4 reasoning fans out one OpenAI chain per failure inside a single
#     process (Phase B parallelism — see pipeline/pipeline_runner.py),
#     which is what "OpenAI request in parallel for each VLM reasoning"
#     refers to and is unchanged here.
#
# Phase C aggregates the per-run learning_curve.json files into the
# mean +/- std budget plot.
#
# Usage (sbatch):
#   sbatch Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/swatch.sh
# Optional knobs:
#   N_RUNS=10  BUDGET=15  TARGET_SR=0.90  CORRECTION_N=40 \
#       sbatch Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/swatch.sh
#
# Usage (local, no sbatch):
#   N_RUNS=4 BUDGET=15 \
#       bash Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/swatch.sh

set -eo pipefail

if command -v module >/dev/null 2>&1; then
    module purge || true
    module load Anaconda3 || true
fi
if [ -f /home/s226137394/.bashrc ]; then
    source /home/s226137394/.bashrc
fi
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate maze || true
fi
mkdir -p slurm_logs

# ----------------------------- knobs ----------------------------- #
N_RUNS="${N_RUNS:-10}"
BUDGET="${BUDGET:-15}"
TARGET_SR="${TARGET_SR:-0.90}"
CORRECTION_N="${CORRECTION_N:-40}"
HELDOUT_N="${HELDOUT_N:-200}"
INITIAL_DEMOS="${INITIAL_DEMOS:-20}"
INITIAL_EPOCHS="${INITIAL_EPOCHS:-500}"
ROUND_EPOCHS="${ROUND_EPOCHS:-500}"
BASELINE_ROUND_EPOCHS="${BASELINE_ROUND_EPOCHS:-100}"
MAX_ROUNDS="${MAX_ROUNDS:-50}"
MAX_STEPS="${MAX_STEPS:-60}"
SEED="${SEED:-0}"

REPO_ROOT="$(pwd)"
SWEEP_ROOT="${REPO_ROOT}/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/runs"
LOG_ROOT="${REPO_ROOT}/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/logs"
FIG_ROOT="${REPO_ROOT}/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/figures"
mkdir -p "${SWEEP_ROOT}" "${LOG_ROOT}" "${FIG_ROOT}"

# How many GPUs are exposed to this job? If unset (local run) assume 1.
NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
if ! [[ "${NUM_GPUS}" =~ ^[0-9]+$ ]] || [ "${NUM_GPUS}" -lt 1 ]; then
    NUM_GPUS=1
fi

# Share the OAI TPM ceiling across N_RUNS parallel processes with ~10%
# headroom for the token-estimate error. _oai_retry.py reads these from
# the environment and clamps in-process to TARGET_TPM = BUDGET * SHARE.
SHARE=$(python -c "import sys; n=int(sys.argv[1]); print(f'{max(0.05, 0.90/n):.3f}')" "${N_RUNS}")
export OAI_TPM_BUDGET="${OAI_TPM_BUDGET:-200000}"
export OAI_TPM_SHARE="${OAI_TPM_SHARE:-${SHARE}}"
export OAI_MAX_IN_FLIGHT="${OAI_MAX_IN_FLIGHT:-2}"
export OAI_SDK_MAX_RETRIES="${OAI_SDK_MAX_RETRIES:-8}"
export OAI_SDK_TIMEOUT="${OAI_SDK_TIMEOUT:-120}"

echo "[$(date)] ============================================================"
echo "[$(date)] HYBRID BASELINE_ONLY vs P4_ONLY  (budget=${BUDGET}, runs=${N_RUNS})"
echo "[$(date)] ============================================================"
echo "[$(date)]   sweep root : ${SWEEP_ROOT}"
echo "[$(date)]   log root   : ${LOG_ROOT}"
echo "[$(date)]   GPUs       : ${NUM_GPUS}"
echo "[$(date)]   OAI share  : ${OAI_TPM_SHARE}"

# Pre-seed BOTH the shared training+heldout layouts AND the global
# initial-20 BFS demos + initial checkpoint. Doing this serially ONCE
# means the 10 parallel children never race to collect/train the same
# initial state — each just copies from the global cache.
python -u -m Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.run_one \
    --run_index 0 \
    --budget "${BUDGET}" \
    --target_sr "${TARGET_SR}" \
    --correction_n "${CORRECTION_N}" \
    --heldout_n "${HELDOUT_N}" \
    --initial_demos "${INITIAL_DEMOS}" \
    --initial_epochs "${INITIAL_EPOCHS}" \
    --seed "${SEED}" \
    --bootstrap_only

run_one_repetition () {
    local RUN_INDEX="$1"
    local GPU_IDX="$2"
    local LOG_FILE="${LOG_ROOT}/run_$(printf '%02d' "${RUN_INDEX}").log"
    {
        echo "[$(date)] [run=${RUN_INDEX} gpu=${GPU_IDX}] starting"
        export CUDA_VISIBLE_DEVICES="${GPU_IDX}"
        # Re-export the OAI knobs in the child so subprocess Pythons inherit them.
        export OAI_TPM_BUDGET="${OAI_TPM_BUDGET}"
        export OAI_TPM_SHARE="${OAI_TPM_SHARE}"
        export OAI_MAX_IN_FLIGHT="${OAI_MAX_IN_FLIGHT}"
        export OAI_SDK_MAX_RETRIES="${OAI_SDK_MAX_RETRIES}"
        export OAI_SDK_TIMEOUT="${OAI_SDK_TIMEOUT}"

        python -u -m Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.run_one \
            --run_index "${RUN_INDEX}" \
            --budget "${BUDGET}" \
            --target_sr "${TARGET_SR}" \
            --correction_n "${CORRECTION_N}" \
            --heldout_n "${HELDOUT_N}" \
            --initial_demos "${INITIAL_DEMOS}" \
            --initial_epochs "${INITIAL_EPOCHS}" \
            --round_epochs "${ROUND_EPOCHS}" \
            --baseline_round_epochs "${BASELINE_ROUND_EPOCHS}" \
            --max_rounds "${MAX_ROUNDS}" \
            --max_steps "${MAX_STEPS}" \
            --seed "${SEED}" \
            --force_restart
        echo "[$(date)] [run=${RUN_INDEX}] DONE"
    } >"${LOG_FILE}" 2>&1
    local rc=$?
    echo "[$(date)] [run=${RUN_INDEX} gpu=${GPU_IDX}] finished rc=${rc} (log: ${LOG_FILE})"
    return $rc
}

PIDS=()
for ((i=0; i<N_RUNS; i++)); do
    GPU_IDX=$(( i % NUM_GPUS ))
    run_one_repetition "${i}" "${GPU_IDX}" &
    PIDS+=($!)
    echo "[$(date)] launched run=${i} on GPU ${GPU_IDX} (pid=$!)"
done

FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        FAILED=$((FAILED + 1))
    fi
done
echo "[$(date)] all parallel repetitions finished. failed=${FAILED}/${#PIDS[@]}"

echo "[$(date)] === AGGREGATE CHART ==="
python -u -m Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.aggregate_chart \
    --runs_dir "${SWEEP_ROOT}" \
    --out_dir  "${FIG_ROOT}" \
    --budget "${BUDGET}" \
    --target_sr "${TARGET_SR}"

echo "[$(date)] baseline_vs_p4 sweep done. artefacts:"
ls -la "${FIG_ROOT}" || true
