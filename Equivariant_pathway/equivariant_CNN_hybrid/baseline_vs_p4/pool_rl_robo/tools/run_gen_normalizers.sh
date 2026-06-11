#!/bin/bash
#SBATCH --job-name=stack_normgen
#SBATCH --partition=gpu-large
#SBATCH --qos=batch-long
#SBATCH --constraint=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=24G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/normgen_%j.out
#SBATCH --error=slurm_logs/normgen_%j.err
# ===========================================================================
# Generate StackCube-v1 normalizers + validate the motion-planner demo path.
# 1 GPU, diffdagger env, no LLM. Run BEFORE the smoke.
#   sbatch tools/run_gen_normalizers.sh
# ===========================================================================
set -eo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then SUITE_DIR="${SLURM_SUBMIT_DIR}"; else
    SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; fi
DMN_ROOT="$(cd "${SUITE_DIR}/../../../.." && pwd)"
PKG="Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo"
mkdir -p "${SUITE_DIR}/slurm_logs"
cd "${DMN_ROOT}"

if command -v module >/dev/null 2>&1; then module purge || true; module load Anaconda3 || true; fi
[ -f "${HOME}/.bashrc" ] && source "${HOME}/.bashrc"
CONDA_ENV="${CONDA_ENV:-diffdagger}"
if command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)"; conda activate "${CONDA_ENV}" || true; fi
PYTHON_BIN="${PYTHON_BIN:-/home/s226137394/.conda/envs/${CONDA_ENV}/bin/python}"
[ -x "${PYTHON_BIN}" ] || { echo "[gen] ERROR: interpreter missing: ${PYTHON_BIN}" >&2; exit 3; }

MODULE="${MODULE:-${PKG}.tools.gen_stackcube_normalizers}"
LOGTAG="${LOGTAG:-stackcube_normgen}"
echo "[gen] node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-?} module=${MODULE}"
PYTHONPATH="${DMN_ROOT}" "${PYTHON_BIN}" -u -m "${MODULE}" \
    2>&1 | tee "${SUITE_DIR}/logs/${LOGTAG}_${SLURM_JOB_ID:-local}.log"
echo "[gen] DONE"
