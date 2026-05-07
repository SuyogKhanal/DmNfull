#!/bin/bash
#SBATCH --job-name=eq_p6_only
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
# Minimal P6-only equivariant pipeline (full LLM stack: VLM + Reasoning +
# KAG + RAG + TKF + Cross-Episode + Plain-LLM Aggregator).
#
# Pre-requisite: baseline_only/ has been run with the new architecture so
# its training_layouts.yaml / correction_layouts.yaml / heldout_layouts.yaml
# and demos/ exist. p6_only reads all three layout YAMLs from baseline_only/
# (read-only) and copies the initial BFS demos. The post-DAgger
# best_eq_policy.pth is NEVER copied; if the immutable initial_*_eq_policy.pth
# snapshot is missing, p6_only trains its own initial model from scratch on
# the same 20 demos using the same hyperparameters baseline_only uses.
#
# Prompt strategy lives in Equivariant_pathway/p6_only/prompts/{reasoning,
# cross_episode,aggregator}.yml. Edit those YAMLs to tune the prompting
# without touching code.
#
# - Single GPU run.
# - --force_restart is ALWAYS passed so every submission wipes
#   p6_only/{demos,checkpoints,results,rag_bank} before bootstrapping.
#   baseline_only/ is NEVER modified.
# - Extra CLI flags after the script name are forwarded to pipeline.py.

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
echo "[$(date)] P6-ONLY EQUIVARIANT PIPELINE (full LLM stack)"
echo "[$(date)] ============================================================"
echo "[$(date)]   forwarding extras: ${EXTRA_ARGS[*]}"
echo "[$(date)]   --force_restart is ALWAYS on: p6_only/{demos,checkpoints,"
echo "[$(date)]   results,rag_bank} are wiped and re-bootstrapped from"
echo "[$(date)]   baseline_only/ on every submission. baseline_only/ is"
echo "[$(date)]   read-only."

if [ ! -f "Equivariant_pathway/baseline_only/heldout_layouts.yaml" ] \
   || [ ! -f "Equivariant_pathway/baseline_only/correction_layouts.yaml" ] \
   || [ ! -d "Equivariant_pathway/baseline_only/demos" ]; then
    echo "[$(date)] ERROR: baseline_only/ has not been run with the new"
    echo "[$(date)]        correction-pool architecture yet."
    echo "[$(date)] Need: Equivariant_pathway/baseline_only/heldout_layouts.yaml"
    echo "[$(date)]       Equivariant_pathway/baseline_only/correction_layouts.yaml"
    echo "[$(date)]       Equivariant_pathway/baseline_only/demos/*.json"
    echo "[$(date)] Run Equivariant_pathway/baseline_only/run.sh first."
    exit 1
fi

if [ -f "Equivariant_pathway/baseline_only/checkpoints/initial_best_eq_policy.pth" ]; then
    echo "[$(date)] initial-snapshot present — p6_only will start from byte-identical weights:"
    sha256sum Equivariant_pathway/baseline_only/checkpoints/initial_best_eq_policy.pth || true
else
    echo "[$(date)] initial_best_eq_policy.pth NOT FOUND in baseline_only/checkpoints/."
    echo "[$(date)]   p6_only will train its OWN initial model on the same 20 demos"
    echo "[$(date)]   for --initial_epochs. Re-run baseline_only/run.sh to get"
    echo "[$(date)]   byte-identical starts in the future."
fi

echo "[$(date)] prompt YAMLs (edit these to tune prompting without code changes):"
ls -la Equivariant_pathway/p6_only/prompts/ || true

python -u -m Equivariant_pathway.p6_only.pipeline \
    --force_restart \
    "${EXTRA_ARGS[@]}"
rc=$?

echo "[$(date)] pipeline exited rc=${rc}"
if [ "${rc}" -ne 0 ]; then
    exit "${rc}"
fi

echo "[$(date)] artefacts in Equivariant_pathway/p6_only/:"
ls -la Equivariant_pathway/p6_only/ || true
echo "[$(date)]   demos/:"
ls -la Equivariant_pathway/p6_only/demos/ | head -30 || true
echo "[$(date)]   checkpoints/:"
ls -la Equivariant_pathway/p6_only/checkpoints/ || true
echo "[$(date)]   results/:"
ls -la Equivariant_pathway/p6_only/results/ || true
echo "[$(date)] done."
