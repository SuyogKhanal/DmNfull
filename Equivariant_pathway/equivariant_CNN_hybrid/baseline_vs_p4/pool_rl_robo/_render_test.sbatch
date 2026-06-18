#!/bin/bash
#SBATCH --job-name=sc_rendertest
#SBATCH --partition=gpu-large
#SBATCH --qos=batch-short
#SBATCH --constraint=gpu-h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/rendertest_%j.out
#SBATCH --error=slurm_logs/rendertest_%j.err
# 1-GPU validation of the eval-headless render fix (reproduces the full-job env set
# and renders the policy env, the point that hung). Override partition/constraint on
# the CLI to pick a100 vs h100.
set -eo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then SUITE_DIR="${SLURM_SUBMIT_DIR}"; else SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; fi
DMN_ROOT="$(cd "${SUITE_DIR}/../../../.." && pwd)"
PKG="Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo"
mkdir -p "${SUITE_DIR}/slurm_logs" "${SUITE_DIR}/logs"
cd "${DMN_ROOT}"
[ -f "${HOME}/.bashrc" ] && source "${HOME}/.bashrc"
if command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)"; conda activate diffdagger || true; fi
PY="/home/s226137394/.conda/envs/diffdagger/bin/python"
echo "[rendertest] $(date) gpu=${CUDA_VISIBLE_DEVICES:-?} node=$(hostname)"
"${PY}" -u -m "${PKG}.tools.test_render_with_eval" \
    2>&1 | tee "${SUITE_DIR}/logs/render_test_${SLURM_JOB_ID:-local}.log"
echo "[rendertest] DONE $(date)"
