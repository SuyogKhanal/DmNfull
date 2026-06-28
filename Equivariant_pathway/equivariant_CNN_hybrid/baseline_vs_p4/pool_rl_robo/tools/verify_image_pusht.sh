#!/bin/bash
# verify_image_pusht.sh — sbatch the fast GPU sanity for image-based PushT.
# 1 GPU, h100 (Vulkan-safe for the rgb render path during this critical check).
# Usage:  sbatch tools/verify_image_pusht.sh      (from the suite dir)
#SBATCH --job-name=pusht_img_verify
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu-large
#SBATCH --constraint=gpu-h100
#SBATCH --qos=batch-short
#SBATCH --time=00:40:00
#SBATCH --output=slurm_logs/img_verify_%j.out
#SBATCH --error=slurm_logs/img_verify_%j.err
set -eo pipefail
SUITE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "${SUITE_DIR}/slurm_logs"
if command -v module >/dev/null 2>&1; then module purge || true; module load Anaconda3 || true; fi
[ -f "${HOME}/.bashrc" ] && source "${HOME}/.bashrc"
CONDA_ENV="${CONDA_ENV:-diffdagger}"
if command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)"; conda activate "${CONDA_ENV}" || true; fi
PYTHON_BIN="${PYTHON_BIN:-/home/s226137394/.conda/envs/${CONDA_ENV}/bin/python}"
echo "[verify] node=$(hostname) python=${PYTHON_BIN}"
exec "${PYTHON_BIN}" "${SUITE_DIR}/tools/verify_image_pusht.py"
