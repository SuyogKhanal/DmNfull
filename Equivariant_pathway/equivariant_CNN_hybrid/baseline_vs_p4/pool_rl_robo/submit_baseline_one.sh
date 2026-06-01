#!/bin/bash
#SBATCH --job-name=pool_rl_robo_baselines
#SBATCH --partition=gpu
#SBATCH --qos=batch-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s226137394@deakin.edu.au
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#
# Baselines-only launcher: one array task per env, NO vLLM (the 5 IIL baselines
# never use the LLM). Submit via submit_baselines_all.sh or:
#   mkdir -p slurm_logs results logs && sbatch submit_baseline_one.sh
set -eo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then SCRIPT_DIR="${SLURM_SUBMIT_DIR}"; else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; fi
cd "${SCRIPT_DIR}"
mkdir -p slurm_logs results logs

PYBIN="${PYBIN:-/home/s226137394/.conda/envs/pool_rl_robo/bin/python}"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

if command -v module >/dev/null 2>&1; then module purge || true; module load Anaconda3 || true; fi
[ -f "${HOME}/.bashrc" ] && source "${HOME}/.bashrc"
if command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)"; conda activate pool_rl_robo || true; fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

ENVS=(HalfCheetah-v4 Hopper-v4 Walker2d-v4 FetchReach-v4 FetchPickAndPlace-v4)
ENV="${ENVS[${SLURM_ARRAY_TASK_ID:-0}]}"
echo "[pool_rl_robo_baselines] $(date) ENV=${ENV}"

cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYBIN}" -u -m \
    Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo.orchestrator.run_baselines \
    --env "${ENV}" --seed "${SEED:-42}" \
    2>&1 | tee "${SCRIPT_DIR}/logs/${ENV}_baselines.log"
echo "[pool_rl_robo_baselines] $(date) ENV=${ENV} DONE"
