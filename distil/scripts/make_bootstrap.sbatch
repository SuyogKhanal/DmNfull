#!/bin/bash
#SBATCH --job-name=mkbs
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=24G
#SBATCH --output=distil/slurm_logs/%x_%j.out
#SBATCH --error=distil/slurm_logs/%x_%j.err
# make_bootstrap.sbatch — generate a shared bootstrap on a GPU node (needed for IMAGE
# modality, which renders frames per demo -> EGL, can't run on the login node).
# Params via --export=ALL,TASK=,MODALITY=,BOOTSTRAP_DIR=[,CONDA_ENV=]
set -eo pipefail
ROOT="${DISTIL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ROOT"
mkdir -p distil/slurm_logs
if command -v module >/dev/null 2>&1; then module purge || true; module load Anaconda3 || true; fi
[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"
CONDA_ENV="${CONDA_ENV:-diffdagger}"
if command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)"; conda activate "$CONDA_ENV" || true; fi
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export DISTIL_ROOT="$ROOT"
TASK="${TASK:?}"; MODALITY="${MODALITY:-state}"; BOOTSTRAP_DIR="${BOOTSTRAP_DIR:?}"
echo "=== make-bootstrap: task=$TASK modality=$MODALITY dir=$BOOTSTRAP_DIR MUJOCO_GL=$MUJOCO_GL ==="
nvidia-smi -L || true
python -m distil.run --make-bootstrap --task "$TASK" --modality "$MODALITY" --bootstrap-dir "$BOOTSTRAP_DIR"
echo "=== DONE (exit $?) ==="
