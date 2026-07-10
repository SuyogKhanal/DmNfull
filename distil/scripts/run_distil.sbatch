#!/bin/bash
#SBATCH --job-name=distil
#SBATCH --partition=gpu,gpu-large
#SBATCH --qos=batch-long
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=distil/slurm_logs/%x_%j.out
#SBATCH --error=distil/slurm_logs/%x_%j.err
# ===========================================================================
# run_distil.sbatch — ONE DISTIL run (task, modality, ablation, seed).
# The LLM is an OpenRouter API call (no local weights) -> 1 GPU only, ANY
# partition (no h100/h200 constraint). One ablation branch = one job (golden rule 5).
#
# Parameterised via env vars (pass with --export=ALL,...):
#   TASK        Lift | Wipe | Door         (required)
#   MODALITY    state | image              (default: state)
#   ABLATION    full | memory_off | ...    (default: full)
#   SEED        integer                    (default: 1)
#   BUDGET      successful demos added     (default: 20)
#   NUM_INIT    bootstrap size Ni          (default: 20)
#   SMOKE       1 => tiny plumbing test    (default: 0)
#   BOOTSTRAP_DIR / OUTPUT_DIR             (sensible defaults below)
#
# Example (smoke):
#   sbatch --job-name=distil_smoke_lift --qos=batch-short --time=01:30:00 --mem=24G \
#     --export=ALL,TASK=Lift,MODALITY=state,ABLATION=full,SEED=1,BUDGET=1,SMOKE=1 \
#     distil/scripts/run_distil.sbatch
# ===========================================================================
set -eo pipefail

# Portable across clusters: DISTIL_ROOT wins; else the dir sbatch was called from
# (submit from the repo root). No hardcoded /weka path (golden rule 8).
ROOT="${DISTIL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ROOT"
mkdir -p distil/slurm_logs

# ── env: module + conda (both generic) ──
if command -v module >/dev/null 2>&1; then module purge || true; module load Anaconda3 || true; fi
[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"
CONDA_ENV="${CONDA_ENV:-distil}"     # HPC-A overrides with CONDA_ENV=diffdagger
if command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)"; conda activate "$CONDA_ENV" || true; fi
PYTHON_BIN="${PYTHON_BIN:-python}"   # the activated env's python (portable)

export MUJOCO_GL="${MUJOCO_GL:-egl}"     # offscreen render for the VLM frames (GPU node)
export DISTIL_ROOT="$ROOT"

TASK="${TASK:?set TASK=Lift|Wipe|Door}"
MODALITY="${MODALITY:-state}"
ABLATION="${ABLATION:-full}"
SEED="${SEED:-1}"
BUDGET="${BUDGET:-20}"
NUM_INIT="${NUM_INIT:-20}"
SMOKE="${SMOKE:-0}"
BOOTSTRAP_DIR="${BOOTSTRAP_DIR:-$ROOT/distil/results/shared_bootstrap}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/distil/results/${TASK}/${MODALITY}/${ABLATION}/seed${SEED}}"

SMOKE_FLAG=""; [ "$SMOKE" = "1" ] && SMOKE_FLAG="--smoke"

echo "=== DISTIL run: task=$TASK modality=$MODALITY ablation=$ABLATION seed=$SEED budget=$BUDGET smoke=$SMOKE ==="
echo "python: $PYTHON_BIN | out: $OUTPUT_DIR"
nvidia-smi -L || true

"$PYTHON_BIN" -m distil.run \
  --task "$TASK" --modality "$MODALITY" --ablation "$ABLATION" --seed "$SEED" \
  --budget "$BUDGET" --num-init-demos "$NUM_INIT" \
  --bootstrap-dir "$BOOTSTRAP_DIR" --output-dir "$OUTPUT_DIR" $SMOKE_FLAG

echo "=== DONE (exit $?) ==="
