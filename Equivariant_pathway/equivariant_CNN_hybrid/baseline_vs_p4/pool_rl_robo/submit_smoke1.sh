#!/bin/bash
#SBATCH --job-name=pool_rl_robo_smoke1
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=08:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s226137394@deakin.edu.au
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# SMOKE 1: verify the policy BACKBONES (Gaussian MLP for locomotion; R3M-image
# Diffusion for Fetch) genuinely train + roll out on a REGULAR gpu (NO h100/h200
# constraint), with P4-LLM OFF (the 5 IIL baselines only, no vLLM). Short budget.
#   sbatch submit_smoke1.sh
set -eo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then SCRIPT_DIR="${SLURM_SUBMIT_DIR}"; else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; fi
cd "${SCRIPT_DIR}"
mkdir -p slurm_logs results logs

PYBIN="${PYBIN:-/home/s226137394/.conda/envs/pool_rl_robo/bin/python}"
if command -v module >/dev/null 2>&1; then module purge || true; module load Anaconda3 || true; fi
[ -f "${HOME}/.bashrc" ] && source "${HOME}/.bashrc"
if command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)"; conda activate pool_rl_robo || true; fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"   # experts pre-cached
export MUJOCO_GL="${MUJOCO_GL:-egl}"           # Fetch R3M rendering

ENVS=(HalfCheetah-v4 Hopper-v4 Walker2d-v4 FetchReach-v4 FetchPickAndPlace-v4)
echo "[smoke1] $(date) backbones (P4 off): ${ENVS[*]}"
for ENV in "${ENVS[@]}"; do
    echo "================= SMOKE1 ENV=${ENV} (P4 off) ================="
    "${PYBIN}" -u run_experiment.py --env "${ENV}" --config config_smoke1.yaml \
        --methods safe_dagger dropout_dagger ensemble_dagger thrifty_dagger stagger \
        --no_llm 2>&1 | tee "logs/smoke1_${ENV}.log" || echo "[smoke1] ENV=${ENV} FAILED (continuing)"
done
echo "[smoke1] $(date) DONE"
