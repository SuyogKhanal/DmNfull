#!/bin/bash
# variants_fixed_vs_p4top3 — single entry point.
#
# Usage (production, all 10 runs, all 3 methods):
#   bash run_all.sh
#
# Usage (smoke, 1 run, baseline only, tiny budget):
#   bash run_all.sh --smoke
#
# Override env vars to retarget:
#   N_RUNS=4 METHODS=baseline,p4_sequential bash run_all.sh
#   NUM_GPUS=2 bash run_all.sh
#
# Skip the aggregation pass (useful while iterating):
#   SKIP_AGG=1 bash run_all.sh
#
# No SLURM. Runs jobs as background processes; one job per run.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_NAME="variants_fixed_vs_p4top3"

# ----- resolve repo root from this script's location ------------------------
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

# ----- optionally activate the 'maze' conda env so subprocesses inherit it --
if [ -z "${CONDA_PREFIX:-}" ] || [[ "${CONDA_PREFIX:-}" != *"/envs/maze" ]]; then
    if [ -f "${HOME}/.bashrc" ]; then
        # shellcheck disable=SC1091
        source "${HOME}/.bashrc" || true
    fi
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate maze || true
    fi
fi

# ----- knobs (override on the command line via env vars) --------------------
SMOKE=0
for arg in "$@"; do
    case "$arg" in
        --smoke) SMOKE=1 ;;
    esac
done

# Auto-detect GPU count (override via NUM_GPUS=...).
detect_gpus() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L 2>/dev/null | grep -c "^GPU " || true
    else
        echo 0
    fi
}
DETECTED_GPUS="$(detect_gpus)"
if [[ -z "${DETECTED_GPUS}" || "${DETECTED_GPUS}" -lt 1 ]]; then
    DETECTED_GPUS=1
fi
NUM_GPUS="${NUM_GPUS:-${DETECTED_GPUS}}"

# Pull defaults from config.yaml via python so we don't need yq.
read_config() {
    /usr/bin/env python3 - <<'PY' "${SCRIPT_DIR}/config.yaml"
import sys, yaml
with open(sys.argv[1], "r") as f:
    cfg = yaml.safe_load(f) or {}
print("N_RUNS=%d"      % int(cfg.get("n_runs", 10)))
print("BUDGET=%d"      % int(cfg.get("budget", 15)))
print("METHODS=%s"     % ",".join(cfg.get("methods", ["baseline"])))
print("TARGET_SR=%s"   % cfg.get("target_sr", 0.90))
PY
}
eval "$(read_config)"

# Allow CLI / env overrides for the most common knobs.
N_RUNS="${N_RUNS:-${N_RUNS}}"
BUDGET="${BUDGET:-${BUDGET}}"
METHODS="${METHODS:-${METHODS}}"
TARGET_SR="${TARGET_SR:-${TARGET_SR}}"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/config.yaml}"
SKIP_AGG="${SKIP_AGG:-0}"

if [ "${SMOKE}" -eq 1 ]; then
    N_RUNS=1
    METHODS="baseline"
    BUDGET=2
    SMOKE_FLAG="--smoke"
    echo "[run_all] SMOKE MODE: N_RUNS=${N_RUNS} METHODS=${METHODS} BUDGET=${BUDGET}"
else
    SMOKE_FLAG=""
fi

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "[run_all] $(date)  repo=${REPO_ROOT}"
echo "[run_all] suite=${SUITE_NAME} n_runs=${N_RUNS} num_gpus=${NUM_GPUS}"
echo "[run_all] methods=${METHODS}  budget=${BUDGET}  target_sr=${TARGET_SR}"
echo "[run_all] config=${CONFIG_PATH}"
echo "[run_all] logs ->  ${LOG_DIR}/run_NN.log"

# ----- one-time upstream bootstrap (serial) ---------------------------------
# Run a dummy run_id=0 in foreground with --smoke-style minimum to populate
# the upstream init cache before any parallel child needs it. We use run_one
# in dry-bootstrap mode by invoking the upstream bootstrap step inline via
# Python so we don't burn a real method execution here.
echo "[run_all] pre-seeding upstream bootstrap…"
python3 -u -c "
import sys
from pathlib import Path
sys.path.insert(0, '${REPO_ROOT}')
from Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.variants_fixed_vs_p4top3.orchestrator import bootstrap as B
B.ensure_upstream_bootstrap(seed=0, initial_epochs=500, initial_demos=20, heldout_n=200)
" || { echo "[run_all] bootstrap failed; aborting"; exit 1; }

# ----- launch N runs in parallel --------------------------------------------
PIDS=()
for ((i=1; i<=N_RUNS; i++)); do
    GPU_IDX=$(( (i - 1) % NUM_GPUS ))
    LOG_FILE="${LOG_DIR}/run_$(printf '%02d' "${i}").log"
    {
        echo "[$(date)] [run=${i} gpu=${GPU_IDX}] starting"
        CUDA_VISIBLE_DEVICES="${GPU_IDX}" \
        python3 -u -m \
            Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.variants_fixed_vs_p4top3.orchestrator.run_one \
            --run_id "${i}" \
            --config "${CONFIG_PATH}" \
            --methods "${METHODS}" ${SMOKE_FLAG}
        echo "[$(date)] [run=${i}] DONE"
    } >"${LOG_FILE}" 2>&1 &
    PIDS+=($!)
    echo "[run_all] launched run=${i} on gpu=${GPU_IDX} pid=$! -> ${LOG_FILE}"
done

FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        FAILED=$((FAILED + 1))
    fi
done
echo "[run_all] all parallel runs finished. failed=${FAILED}/${#PIDS[@]}"

# ----- aggregation ----------------------------------------------------------
if [ "${SKIP_AGG}" -eq 1 ]; then
    echo "[run_all] SKIP_AGG=1 -> skipping aggregation"
    exit 0
fi

echo "[run_all] === AGGREGATION ==="
python3 -u -m \
    Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.variants_fixed_vs_p4top3.aggregation.aggregate \
    --budget "${BUDGET}" --target_sr "${TARGET_SR}"

echo "[run_all] DONE  ($(date))"
