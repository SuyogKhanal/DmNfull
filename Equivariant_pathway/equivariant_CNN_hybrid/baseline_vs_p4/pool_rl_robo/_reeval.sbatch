#!/bin/bash
#SBATCH --job-name=sc_reeval
#SBATCH --partition=gpu,gpu-large
#SBATCH --qos=batch-short
#SBATCH --constraint=gpu-a100|gpu-h100|gpu-h200
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/reeval_%j.out
#SBATCH --error=slurm_logs/reeval_%j.err
# Tighter (400-ep) re-eval of near-final StackCube checkpoints. Eval-only, no render
# → any GPU is fine (a100 ok). Settles the noisy final head-to-head.
set -eo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then SUITE_DIR="${SLURM_SUBMIT_DIR}"; else SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; fi
DMN_ROOT="$(cd "${SUITE_DIR}/../../../.." && pwd)"
PKG="Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo"
mkdir -p "${SUITE_DIR}/slurm_logs" "${SUITE_DIR}/logs"
cd "${DMN_ROOT}"
[ -f "${HOME}/.bashrc" ] && source "${HOME}/.bashrc"
if command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)"; conda activate diffdagger || true; fi
PY="/home/s226137394/.conda/envs/diffdagger/bin/python"
echo "[reeval] $(date) gpu=${CUDA_VISIBLE_DEVICES:-?} node=$(hostname)"
"${PY}" -u -m "${PKG}.tools.reeval_final" 2>&1 | tee "${SUITE_DIR}/logs/reeval_${SLURM_JOB_ID:-local}.log"
echo "[reeval] DONE $(date)"
