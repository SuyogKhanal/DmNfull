#!/bin/bash
#SBATCH --job-name=eq_p4_only
#SBATCH --partition=gpu
#SBATCH --qos=batch-long
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s226137394@deakin.edu.au
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# Minimal P4-only equivariant pipeline. Pre-requisite: baseline_only/
# has been run at least once so its training_layouts.yaml,
# heldout_layouts.yaml, demos/, and checkpoints/best_eq_policy.pth
# exist on disk. p4_only copies the initial 20 demos and the initial
# checkpoint from baseline_only/ (no fresh sampling, no fresh initial
# training) so the two methods start from byte-identical model weights
# and diverge only via P4's LLM-prescribed demos.
#
# - Single GPU run.
# - --force_restart is ALWAYS passed so every submission wipes
#   p4_only/{demos,checkpoints,results} before bootstrapping. The
#   baseline_only/ tree is NEVER modified.
# - Extra CLI flags after the script name are forwarded to pipeline.py
#   (e.g. --max_rounds 20, --round_epochs 80).

set -eo pipefail

EXTRA_ARGS=("$@")
set --

module purge
module load Anaconda3
source /home/s226137394/.bashrc
eval "$(conda shell.bash hook)"
conda activate maze

mkdir -p slurm_logs

echo "[$(date)] ============================================================"
echo "[$(date)] P4-ONLY EQUIVARIANT PIPELINE"
echo "[$(date)] ============================================================"
echo "[$(date)]   forwarding extras: ${EXTRA_ARGS[*]}"
echo "[$(date)]   --force_restart is ALWAYS on: p4_only/{demos,checkpoints,"
echo "[$(date)]   results} are wiped and re-bootstrapped from baseline_only/"
echo "[$(date)]   on every submission. baseline_only/ is read-only."

if [ ! -f "Equivariant_pathway/baseline_only/checkpoints/best_eq_policy.pth" ]; then
    echo "[$(date)] ERROR: baseline_only/ has not been run."
    echo "[$(date)] Expected: Equivariant_pathway/baseline_only/checkpoints/best_eq_policy.pth"
    echo "[$(date)] Run Equivariant_pathway/baseline_only/run.sh first."
    exit 1
fi

echo "[$(date)] sha256 of baseline_only's shared best_eq_policy.pth (this is what"
echo "[$(date)] p4_only is starting from):"
sha256sum Equivariant_pathway/baseline_only/checkpoints/best_eq_policy.pth || true

python -u -m Equivariant_pathway.p4_only.pipeline \
    --force_restart \
    "${EXTRA_ARGS[@]}"
rc=$?

echo "[$(date)] pipeline exited rc=${rc}"
if [ "${rc}" -ne 0 ]; then
    exit "${rc}"
fi

echo "[$(date)] artefacts in Equivariant_pathway/p4_only/:"
ls -la Equivariant_pathway/p4_only/ || true
echo "[$(date)]   demos/:"
ls -la Equivariant_pathway/p4_only/demos/ | head -30 || true
echo "[$(date)]   checkpoints/:"
ls -la Equivariant_pathway/p4_only/checkpoints/ || true
echo "[$(date)]   results/:"
ls -la Equivariant_pathway/p4_only/results/ || true
echo "[$(date)] done."
