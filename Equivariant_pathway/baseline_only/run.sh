#!/bin/bash
#SBATCH --job-name=eq_baseline_only
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
# Minimal baseline-DAgger-only equivariant pipeline.
# - Single GPU run.
# - --force_restart is ALWAYS passed so every submission re-samples
#   layouts, re-collects the initial 20 demos, and retrains the
#   initial checkpoint from scratch. Existing artefacts under
#   Equivariant_pathway/baseline_only/{demos,checkpoints,results} and
#   the layout YAMLs are wiped before the loop starts.
# - Extra CLI flags after the script name are forwarded to pipeline.py
#   (e.g. --max_rounds 10, --train_every_n 5).

set -eo pipefail

# Capture extras BEFORE sourcing conda activation (which would otherwise
# pick up $1 as an env name and crash). Then clear $@ for the source.
EXTRA_ARGS=("$@")
set --

module purge
module load Anaconda3
source /home/s226137394/.bashrc
eval "$(conda shell.bash hook)"
conda activate maze

mkdir -p slurm_logs

echo "[$(date)] ============================================================"
echo "[$(date)] BASELINE-DAGGER-ONLY EQUIVARIANT PIPELINE"
echo "[$(date)] ============================================================"
echo "[$(date)]   forwarding extras: ${EXTRA_ARGS[*]}"
echo "[$(date)]   --force_restart is ALWAYS on: layouts, demos, checkpoints,"
echo "[$(date)]   and results are wiped and regenerated on every submission."

python -u -m Equivariant_pathway.baseline_only.pipeline \
    --force_restart \
    "${EXTRA_ARGS[@]}"
rc=$?

echo "[$(date)] pipeline exited rc=${rc}"
if [ "${rc}" -ne 0 ]; then
    exit "${rc}"
fi

echo "[$(date)] artefacts in Equivariant_pathway/baseline_only/:"
ls -la Equivariant_pathway/baseline_only/ || true
echo "[$(date)]   demos/:"
ls -la Equivariant_pathway/baseline_only/demos/ | head -30 || true
echo "[$(date)]   checkpoints/:"
ls -la Equivariant_pathway/baseline_only/checkpoints/ || true
echo "[$(date)]   results/:"
ls -la Equivariant_pathway/baseline_only/results/ || true
echo "[$(date)] done."
