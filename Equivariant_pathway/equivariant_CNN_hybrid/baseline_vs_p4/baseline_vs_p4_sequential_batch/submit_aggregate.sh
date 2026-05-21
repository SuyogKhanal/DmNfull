#!/bin/bash
#SBATCH --job-name=bvp_aggregate
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --gpus=1
#SBATCH --constraint="gpu-l40s|gpu-v100"
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/bvp_aggregate_%j.out
#SBATCH --error=slurm_logs/bvp_aggregate_%j.err
#
# Cross-run aggregation + contamination check + figures.
# Run after the 10 per-run jobs finish (or queue with
# --dependency=afterok:... — submit_all.sh does this when AGGREGATE_AFTER=1).

set -eo pipefail

# Under SLURM the script is copied to a spool dir; prefer SLURM_SUBMIT_DIR.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

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
    conda activate maze || true
fi

CONFIG="${CONFIG:-${SCRIPT_DIR}/config.yaml}"
BUDGET=$(python3 -c "import yaml,sys; print(int(yaml.safe_load(open('${CONFIG}')).get('budget', 15)))")
TARGET=$(python3 -c "import yaml,sys; print(float(yaml.safe_load(open('${CONFIG}')).get('target_sr', 0.90)))")

python3 -u -m \
    Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.baseline_vs_p4_sequential_batch.aggregation.aggregate \
    --budget "${BUDGET}" --target_sr "${TARGET}"
