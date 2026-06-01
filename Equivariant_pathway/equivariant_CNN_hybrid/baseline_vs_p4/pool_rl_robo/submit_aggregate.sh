#!/bin/bash
#SBATCH --job-name=pool_rl_robo_aggregate
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# Cross-env aggregation. Submittable (sbatch submit_aggregate.sh) or runnable
# directly (bash submit_aggregate.sh) — it is a quick CPU job.
set -eo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then SCRIPT_DIR="${SLURM_SUBMIT_DIR}"; else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; fi
cd "${SCRIPT_DIR}"
mkdir -p slurm_logs
PYBIN="${PYBIN:-/home/s226137394/.conda/envs/pool_rl_robo/bin/python}"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"
"${PYBIN}" -u -m \
    Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo.aggregation.aggregate
