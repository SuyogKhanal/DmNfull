#!/bin/bash
#SBATCH --job-name=hybrid_4way
#SBATCH --partition=gpu
#SBATCH --qos=batch-long
#SBATCH --gpus=4
#SBATCH --mem=160G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s226137394@deakin.edu.au
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# Hybrid 4-way comparison at correction_n=40:
#   baseline_only  vs  p4_only  vs  p5_only  vs  p6_only
#
# Phase A (sequential, GPU 0): baseline_only must run first because
# P4/P5/P6 all bootstrap from its initial 20 BFS demos + initial
# checkpoint. Heldout=200 baked in.
#
# Phase B (parallel, GPUs 0/1/2): p4_only / p5_only / p6_only each on
# their own GPU. Each method has an isolated ROOT under
# four_way_compare/runs/<method>/, so demos / checkpoints / results /
# RAG banks never collide.
#
# Phase C (sequential): super_chart aggregates the four
# learning_curve.json files into 3 pair charts + 1 super 4-way overlay
# + a summary table.
#
# RAG bank isolation:
#   P5 -> four_way_compare/runs/p5_only/rag_bank/
#   P6 -> four_way_compare/runs/p6_only/rag_bank/
# (sibling dirs, no overlap)
#
# TKF demo_dir (P6 only) is pinned to four_way_compare/runs/p6_only/demos
# which contains ONLY BFS demos (source-filter at bootstrap rejects
# every "hybrid_baseline_dagger_correction" demo from baseline_only).

set -eo pipefail

module purge
module load Anaconda3
source /home/s226137394/.bashrc
eval "$(conda shell.bash hook)"
conda activate maze
mkdir -p slurm_logs

POOL_SIZE="${POOL_SIZE:-40}"

REPO_ROOT="$(pwd)"
SWEEP_ROOT="${REPO_ROOT}/Equivariant_pathway/equivariant_CNN_hybrid/four_way_compare/runs"
mkdir -p "${SWEEP_ROOT}"

BO_ROOT="${SWEEP_ROOT}/baseline_only"
P4_ROOT="${SWEEP_ROOT}/p4_only"
P5_ROOT="${SWEEP_ROOT}/p5_only"
P6_ROOT="${SWEEP_ROOT}/p6_only"
mkdir -p "${BO_ROOT}" "${P4_ROOT}" "${P5_ROOT}" "${P6_ROOT}"

echo "[$(date)] ============================================================"
echo "[$(date)] HYBRID 4-WAY COMPARISON  (pool_size=${POOL_SIZE})"
echo "[$(date)] ============================================================"
echo "[$(date)]   baseline_only -> ${BO_ROOT}"
echo "[$(date)]   p4_only       -> ${P4_ROOT}"
echo "[$(date)]   p5_only       -> ${P5_ROOT}"
echo "[$(date)]   p6_only       -> ${P6_ROOT}"

# Shared OAI rate-limit env. With 4 sweep processes each ~22% of the
# 200k TPM budget collectively stays under the org ceiling with ~10%
# headroom. Per-process MAX_IN_FLIGHT=2 caps concurrent calls. See
# pipeline/_oai_retry.py for the full pacing stack.
export OAI_TPM_BUDGET="${OAI_TPM_BUDGET:-200000}"
export OAI_TPM_SHARE="${OAI_TPM_SHARE:-0.22}"
export OAI_MAX_IN_FLIGHT="${OAI_MAX_IN_FLIGHT:-2}"
export OAI_SDK_MAX_RETRIES="${OAI_SDK_MAX_RETRIES:-8}"
export OAI_SDK_TIMEOUT="${OAI_SDK_TIMEOUT:-120}"

# -----------------------------------------------------------------------------
# PHASE A — baseline_only on GPU 0 (sequential; everything below depends on it)
# -----------------------------------------------------------------------------
{
    echo "[$(date)] [baseline] starting"
    export CUDA_VISIBLE_DEVICES=0
    export BASELINE_ONLY_ROOT="${BO_ROOT}"
    python -u -m Equivariant_pathway.equivariant_CNN_hybrid.baseline_only.pipeline \
        --force_restart \
        --correction_n "${POOL_SIZE}" \
        --heldout_n 200
    echo "[$(date)] [baseline] DONE"
} >"${SWEEP_ROOT}/baseline_only.log" 2>&1
BO_RC=$?
echo "[$(date)] baseline_only finished rc=${BO_RC} (log: ${SWEEP_ROOT}/baseline_only.log)"
if [ "${BO_RC}" -ne 0 ]; then
    echo "[$(date)] baseline_only failed; aborting downstream."
    exit "${BO_RC}"
fi

# -----------------------------------------------------------------------------
# PHASE B — p4 / p5 / p6 in parallel on GPUs 0 / 1 / 2
# -----------------------------------------------------------------------------
run_method () {
    local NAME="$1"; local GPU="$2"; local METHOD_ROOT="$3"; local ENV_VAR_NAME="$4"
    local LOG="${SWEEP_ROOT}/${NAME}.log"
    {
        echo "[$(date)] [${NAME} gpu=${GPU}] starting"
        export CUDA_VISIBLE_DEVICES="${GPU}"
        export BASELINE_ONLY_ROOT="${BO_ROOT}"
        export "${ENV_VAR_NAME}=${METHOD_ROOT}"
        python -u -m "Equivariant_pathway.equivariant_CNN_hybrid.${NAME}.pipeline" \
            --force_restart
        echo "[$(date)] [${NAME}] DONE"
    } >"${LOG}" 2>&1
    local rc=$?
    echo "[$(date)] [${NAME} gpu=${GPU}] finished rc=${rc} (log: ${LOG})"
    return $rc
}

PIDS=()
run_method p4_only 0 "${P4_ROOT}" P4_ONLY_ROOT &
PIDS+=($!)
run_method p5_only 1 "${P5_ROOT}" P5_ONLY_ROOT &
PIDS+=($!)
run_method p6_only 2 "${P6_ROOT}" P6_ONLY_ROOT &
PIDS+=($!)

FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        FAILED=$((FAILED + 1))
    fi
done
echo "[$(date)] phase B finished. failed=${FAILED}/3"

# -----------------------------------------------------------------------------
# PHASE C — super-plot (4 charts + summary)
# -----------------------------------------------------------------------------
echo "[$(date)] === SUPER-PLOT ==="
python -u -m Equivariant_pathway.equivariant_CNN_hybrid.four_way_compare.super_chart \
    --sweep_dir "${SWEEP_ROOT}"

echo "[$(date)] hybrid 4-way comparison done. artefacts:"
ls -la "${SWEEP_ROOT}" || true
