#!/bin/bash
#SBATCH --job-name=prr_subtask_envcheck
#SBATCH --partition=gpu-large
#SBATCH --qos=batch-long
#SBATCH --constraint=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=24G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/envcheck_%j.out
#SBATCH --error=slurm_logs/envcheck_%j.err
set -eo pipefail
SUITE_DIR="${SLURM_SUBMIT_DIR:-/weka/s226137394/DmNfull/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_rl_robo}"
DMN_ROOT="$(cd "${SUITE_DIR}/../../../.." && pwd)"
mkdir -p "${SUITE_DIR}/slurm_logs"
cd "${DMN_ROOT}"
if command -v module >/dev/null 2>&1; then module purge || true; module load Anaconda3 || true; fi
[ -f "${HOME}/.bashrc" ] && source "${HOME}/.bashrc"
if command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)"; conda activate diffdagger || true; fi
PYTHON_BIN="/home/s226137394/.conda/envs/diffdagger/bin/python"
echo "[envcheck] host=$(hostname) gpu:"; nvidia-smi -L || true
PYTHONPATH="${DMN_ROOT}" "${PYTHON_BIN}" -u "${SUITE_DIR}/_env_sanity_subtask.py"
echo "[envcheck] DONE"
