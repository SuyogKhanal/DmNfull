#!/bin/bash
#SBATCH --job-name=p4sweep
#SBATCH --partition=gpu
#SBATCH --qos=batch-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --cpus-per-gpu=16
#SBATCH --time=08:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s226137394@deakin.edu.au
#SBATCH --output=slurm_logs/p4sweep_%j.out
#SBATCH --error=slurm_logs/p4sweep_%j.err
#
# One SLURM job = one CHUNK of the P4 hyperparameter sweep. Runs that chunk's
# (config x run) P4 replays IN PARALLEL via sweep_hp.py (no LLM). With RANK=1
# it instead runs the leaderboard ranking (sweep_rank.py) after all chunks.

set -eo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then SCRIPT_DIR="${SLURM_SUBMIT_DIR}";
else SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; fi
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

if command -v module >/dev/null 2>&1; then module purge || true; module load Anaconda3 || true; fi
if [ -f "${HOME}/.bashrc" ]; then source "${HOME}/.bashrc"; fi
if command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)"; conda activate "${BASELINE_CONDA_ENV:-maze}" || true; fi

mkdir -p "${SCRIPT_DIR}/slurm_logs" "${SCRIPT_DIR}/logs"

CONFIG="${CONFIG:-${SCRIPT_DIR}/config_baselines.yaml}"
GRID="${GRID:-${SCRIPT_DIR}/results_hpsweep/grid.json}"
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/results_hpsweep}"
RUNS="${RUNS:-1 6}"
RUNS="${RUNS//+/ }"   # launcher encodes spaces as '+' to survive sbatch --export
DEST_METHOD="${DEST_METHOD:-p4_top3_rotate}"
SOURCE_METHOD="${SOURCE_METHOD:-p4_top3_rotate}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-4}"
BUDGET="${BUDGET:-15}"

PKG="Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_x_selector.orchestrator"

if [ "${RANK:-0}" = "1" ]; then
    echo "[p4sweep] $(date) RANK job (runs='${RUNS}')"
    python3 -u -m "${PKG}.sweep_rank" \
        --grid "${GRID}" --out_root "${OUT_ROOT}" --runs "${RUNS}" \
        --dest_method "${DEST_METHOD}" --budget "${BUDGET}"
    echo "[p4sweep] $(date) RANK DONE"
    exit 0
fi

CHUNK_INDEX="${CHUNK_INDEX:?CHUNK_INDEX not set}"
CHUNK_SIZE="${CHUNK_SIZE:-2}"
echo "[p4sweep] $(date) chunk=${CHUNK_INDEX} size=${CHUNK_SIZE} runs='${RUNS}' out=${OUT_ROOT}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>&1 | sed 's/^/[p4sweep] /'

python3 -u -m "${PKG}.sweep_hp" \
    --grid "${GRID}" --chunk_index "${CHUNK_INDEX}" --chunk_size "${CHUNK_SIZE}" \
    --runs "${RUNS}" --out_root "${OUT_ROOT}" --config "${CONFIG}" \
    --source_method "${SOURCE_METHOD}" --dest_method "${DEST_METHOD}" \
    --max_concurrency "${MAX_CONCURRENCY}"

echo "[p4sweep] $(date) chunk=${CHUNK_INDEX} DONE"
