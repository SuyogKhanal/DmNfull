#!/bin/bash
#SBATCH --job-name=sigma_calib
# NOTE: this cluster exposes only GPU partitions (sinfo: gpu, gpu-large). The job
# requests NO GPU (--gres=none) and is a pure-CPU analysis; it just has to land on
# a node in a GPU partition because there is no cpu partition to send it to.
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=distil/slurm_logs/%x_%j.out
#SBATCH --error=distil/slurm_logs/%x_%j.err
# ===========================================================================
# run_sigma_analysis.sbatch — calibrate the Gaussian memory-kernel width sigma
# from cluster centroids in ALREADY-COMPLETED runs.  Pure CPU: no GPU, no env
# rollouts, no LLM calls, no code defaults changed.
#
#   sbatch distil/scripts/run_sigma_analysis.sbatch
# ===========================================================================
set -eo pipefail

ROOT="${DISTIL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ROOT"
mkdir -p distil/slurm_logs

if command -v module >/dev/null 2>&1; then module purge || true; module load Anaconda3 || true; fi
[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"

# openpyxl/matplotlib/numpy all present in this env (the 'diffdagger' env has no openpyxl).
PYTHON_BIN="${PYTHON_BIN:-/home/s226137394/.conda/envs/maze/bin/python}"
export DISTIL_ROOT="$ROOT"

echo "=== sigma calibration | host=$(hostname) | python=$PYTHON_BIN ==="
"$PYTHON_BIN" distil/scripts/analyze_sigma.py
echo "=== DONE (exit $?) ==="
