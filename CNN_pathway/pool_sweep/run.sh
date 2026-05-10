#!/bin/bash
#SBATCH --job-name=cnn_pool_sweep
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
# CNN_pathway correction-pool-size sweep — RGBCNNPolicy version.
#
# Self-contained: does NOT depend on Equivariant_pathway/pool_sweep/run.sh
# or any module under Equivariant_pathway/. Each pool-size sub-job runs
# CNN_pathway.baseline_only.pipeline THEN CNN_pathway.p4_only.pipeline
# back-to-back on a single dedicated GPU. The 4 pool sizes run IN
# PARALLEL across the 4 GPUs requested above. Pool roots are isolated
# under CNN_pathway/pool_sweep/runs/pool_<N>/{baseline_only,p4_only}/
# via the BASELINE_ONLY_ROOT / P4_ONLY_ROOT env vars CNN_pathway's
# pipelines honour.
#
# Usage:
#   sbatch CNN_pathway/pool_sweep/run.sh
# Optional override:
#   POOL_SIZES="20 30 40 50" sbatch CNN_pathway/pool_sweep/run.sh

set -eo pipefail

module purge
module load Anaconda3
source /home/s226137394/.bashrc
eval "$(conda shell.bash hook)"
conda activate maze

mkdir -p slurm_logs

POOL_SIZES_ENV="${POOL_SIZES:-20 30 40 50}"
read -r -a POOL_SIZES <<<"$POOL_SIZES_ENV"

REPO_ROOT="$(pwd)"
SWEEP_ROOT="${REPO_ROOT}/CNN_pathway/pool_sweep/runs"
mkdir -p "${SWEEP_ROOT}"

NUM_GPUS=$(echo "${SLURM_GPUS:-${#POOL_SIZES[@]}}" | grep -o '[0-9]\+' | head -1)
if [ -z "${NUM_GPUS}" ] || [ "${NUM_GPUS}" -lt 1 ]; then
    NUM_GPUS=${#POOL_SIZES[@]}
fi

echo "[$(date)] ============================================================"
echo "[$(date)] CNN_PATHWAY CORRECTION-POOL-SIZE SWEEP (RGBCNNPolicy)"
echo "[$(date)] ============================================================"
echo "[$(date)]   pool sizes  : ${POOL_SIZES[*]}"
echo "[$(date)]   sweep root  : ${SWEEP_ROOT}"
echo "[$(date)]   GPUs        : ${NUM_GPUS}"

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
        export CUDA_VISIBLE_DEVICES="${GPU_IDX}"
        export BASELINE_ONLY_ROOT="${BO_ROOT}"
        export P4_ONLY_ROOT="${P4_ROOT}"

        # OpenAI rate-limit pacing (shared retry/bucket from pipeline/_oai_retry.py).
        export OAI_TPM_BUDGET="${OAI_TPM_BUDGET:-200000}"
        export OAI_TPM_SHARE="${OAI_TPM_SHARE:-0.22}"
        export OAI_MAX_IN_FLIGHT="${OAI_MAX_IN_FLIGHT:-2}"
        export OAI_SDK_MAX_RETRIES="${OAI_SDK_MAX_RETRIES:-8}"
        export OAI_SDK_TIMEOUT="${OAI_SDK_TIMEOUT:-120}"

        echo "[$(date)] [pool=${POOL}] === CNN BASELINE_ONLY ==="
        python -u -m CNN_pathway.baseline_only.pipeline \
            --force_restart \
            --correction_n "${POOL}" \
            --heldout_n 200

        echo "[$(date)] [pool=${POOL}] === CNN P4_ONLY ==="
        python -u -m CNN_pathway.p4_only.pipeline \
            --force_restart

        echo "[$(date)] [pool=${POOL}] DONE"
    } >"${LOG_FILE}" 2>&1
    local rc=$?
    echo "[$(date)] [pool=${POOL} gpu=${GPU_IDX}] finished rc=${rc} (log: ${LOG_FILE})"
    return $rc
}

PIDS=()
for i in "${!POOL_SIZES[@]}"; do
    POOL="${POOL_SIZES[$i]}"
    GPU_IDX="$i"
    if [ "${i}" -ge "${NUM_GPUS}" ]; then
        GPU_IDX=$((i % NUM_GPUS))
    fi
    run_one_pool "${POOL}" "${GPU_IDX}" &
    PIDS+=($!)
    echo "[$(date)] launched pool=${POOL} on GPU ${GPU_IDX} (pid=$!)"
done

FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        FAILED=$((FAILED + 1))
    fi
done

echo "[$(date)] all parallel pool-size runs finished. failed=${FAILED}/${#PIDS[@]}"

echo "[$(date)] === SUPER-PLOT ==="
python -u -m CNN_pathway.pool_sweep.super_chart \
    --sweep_dir "${SWEEP_ROOT}" \
    --pool_sizes "${POOL_SIZES[@]}"

echo "[$(date)] CNN sweep done. artefacts:"
ls -la "${SWEEP_ROOT}" || true
