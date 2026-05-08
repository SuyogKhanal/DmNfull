#!/bin/bash
#SBATCH --job-name=eq_pool_sweep
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
# Correction-pool-size sweep: 20 / 30 / 40 / 50.
#
# For each pool size N we run baseline_only THEN p4_only sequentially
# on a single dedicated GPU. The 4 pool sizes run IN PARALLEL across
# the 4 GPUs requested above — wall-clock is dominated by the slowest
# pool size, not the sum.
#
# Each (pool_size, method) lands in a fully isolated tree:
#   Equivariant_pathway/pool_sweep/runs/pool_<N>/baseline_only/
#   Equivariant_pathway/pool_sweep/runs/pool_<N>/p4_only/
# so the four parallel runs do not collide on demos / checkpoints /
# layout YAMLs / results. Isolation is provided by the BASELINE_ONLY_ROOT
# and P4_ONLY_ROOT env vars that the underlying pipelines now honour.
#
# After all 4 pool sizes finish, a super-plot is rendered to
#   Equivariant_pathway/pool_sweep/runs/super_baseline_vs_p4.png
# with one subplot per pool size + a side panel summarising
# demos-to-target for each method.
#
# Usage:
#   sbatch Equivariant_pathway/pool_sweep/run.sh
# Optional override (space-separated list of pool sizes):
#   POOL_SIZES="20 30 40 50" sbatch Equivariant_pathway/pool_sweep/run.sh

set -eo pipefail

module purge
module load Anaconda3
source /home/s226137394/.bashrc
eval "$(conda shell.bash hook)"
conda activate maze

mkdir -p slurm_logs

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
POOL_SIZES_ENV="${POOL_SIZES:-20 30 40 50}"
read -r -a POOL_SIZES <<<"$POOL_SIZES_ENV"

REPO_ROOT="$(pwd)"
SWEEP_ROOT="${REPO_ROOT}/Equivariant_pathway/pool_sweep/runs"
mkdir -p "${SWEEP_ROOT}"

NUM_GPUS=$(echo "${SLURM_GPUS:-${#POOL_SIZES[@]}}" | grep -o '[0-9]\+' | head -1)
if [ -z "${NUM_GPUS}" ] || [ "${NUM_GPUS}" -lt 1 ]; then
    NUM_GPUS=${#POOL_SIZES[@]}
fi

echo "[$(date)] ============================================================"
echo "[$(date)] CORRECTION-POOL-SIZE SWEEP"
echo "[$(date)] ============================================================"
echo "[$(date)]   pool sizes  : ${POOL_SIZES[*]}"
echo "[$(date)]   sweep root  : ${SWEEP_ROOT}"
echo "[$(date)]   GPUs        : ${NUM_GPUS}"
echo "[$(date)]   each GPU runs:  baseline_only -> p4_only  for ONE pool size."

# -----------------------------------------------------------------------------
# Per-pool-size runner. Pinned to a single GPU; runs baseline THEN p4
# back-to-back so p4 picks up baseline's initial-checkpoint snapshot.
# -----------------------------------------------------------------------------
run_one_pool () {
    local POOL="$1"
    local GPU_IDX="$2"
    local POOL_DIR="${SWEEP_ROOT}/pool_${POOL}"
    local BO_ROOT="${POOL_DIR}/baseline_only"
    local P4_ROOT="${POOL_DIR}/p4_only"
    local LOG_FILE="${POOL_DIR}/run.log"

    mkdir -p "${BO_ROOT}" "${P4_ROOT}"

    {
        echo "[$(date)] [pool=${POOL} gpu=${GPU_IDX}] starting"
        echo "[$(date)] [pool=${POOL} gpu=${GPU_IDX}]   BO_ROOT=${BO_ROOT}"
        echo "[$(date)] [pool=${POOL} gpu=${GPU_IDX}]   P4_ROOT=${P4_ROOT}"

        export CUDA_VISIBLE_DEVICES="${GPU_IDX}"
        export BASELINE_ONLY_ROOT="${BO_ROOT}"
        export P4_ONLY_ROOT="${P4_ROOT}"

        # Heldout is 200 (not 50) so the sub-percent SR differences
        # we care about are above the noise floor. Resolution per
        # success drops from ~2% to ~0.5%; eval cost grows ~4x but
        # there are no extra demos, so this is a near-free win on
        # statistical power.
        echo "[$(date)] [pool=${POOL}] === BASELINE_ONLY ==="
        python -u -m Equivariant_pathway.baseline_only.pipeline \
            --force_restart \
            --correction_n "${POOL}" \
            --heldout_n 200

        echo "[$(date)] [pool=${POOL}] === P4_ONLY ==="
        python -u -m Equivariant_pathway.p4_only.pipeline \
            --force_restart

        echo "[$(date)] [pool=${POOL}] DONE"
    } >"${LOG_FILE}" 2>&1
    local rc=$?
    echo "[$(date)] [pool=${POOL} gpu=${GPU_IDX}] finished rc=${rc} (log: ${LOG_FILE})"
    return $rc
}

# -----------------------------------------------------------------------------
# Fan out: one background job per pool size, each pinned to its own GPU.
# -----------------------------------------------------------------------------
PIDS=()
for i in "${!POOL_SIZES[@]}"; do
    POOL="${POOL_SIZES[$i]}"
    GPU_IDX="$i"
    if [ "${i}" -ge "${NUM_GPUS}" ]; then
        echo "[$(date)] WARNING: more pool sizes than GPUs — pool=${POOL} will share GPU ${GPU_IDX}."
        GPU_IDX=$((i % NUM_GPUS))
    fi
    run_one_pool "${POOL}" "${GPU_IDX}" &
    PIDS+=($!)
    echo "[$(date)] launched pool=${POOL} on GPU ${GPU_IDX} (pid=$!)"
done

# Wait for all and collect failures.
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        FAILED=$((FAILED + 1))
    fi
done

echo "[$(date)] ------------------------------------------------------------"
echo "[$(date)] all parallel pool-size runs finished. failed=${FAILED}/${#PIDS[@]}"
if [ "${FAILED}" -gt 0 ]; then
    echo "[$(date)] some pool sizes failed — super-plot will skip missing curves."
fi

# -----------------------------------------------------------------------------
# Super-plot.
# -----------------------------------------------------------------------------
echo "[$(date)] === SUPER-PLOT ==="
python -u -m Equivariant_pathway.pool_sweep.super_chart \
    --sweep_dir "${SWEEP_ROOT}" \
    --pool_sizes "${POOL_SIZES[@]}"

echo "[$(date)] sweep done. artefacts:"
ls -la "${SWEEP_ROOT}" || true
