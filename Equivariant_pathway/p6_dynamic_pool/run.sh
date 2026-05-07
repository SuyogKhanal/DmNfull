#!/bin/bash
#SBATCH --job-name=eq_p6_dynamic_pool
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
# LLM-guided dynamic-pool-expansion pipeline (P6 stack with prescribed
# layouts appended to a growing correction pool).
#
# Pre-requisite: baseline_only/ has been run with the new architecture so
# its training_layouts.yaml / correction_layouts.yaml / heldout_layouts.yaml
# / demos/ exist. p6_dynamic_pool seeds its dynamic_pool.yaml from
# baseline_only/correction_layouts.yaml on first run, then GROWS the pool
# round by round via LLM prescriptions. The 50 held-out layouts are NEVER
# touched by any demo collector.
#
# Prompt strategy lives in p6_dynamic_pool/prompts/{reasoning,
# cross_episode,aggregator}.yml. Edit those YAMLs to tune the prompting
# without touching code.

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
echo "[$(date)] P6 DYNAMIC-POOL EXPANSION PIPELINE"
echo "[$(date)] ============================================================"
echo "[$(date)]   forwarding extras: ${EXTRA_ARGS[*]}"
echo "[$(date)]   --force_restart is ALWAYS on: p6_dynamic_pool/{demos,"
echo "[$(date)]   checkpoints,results,rag_bank,dynamic_pool.yaml} are wiped"
echo "[$(date)]   and re-bootstrapped from baseline_only/. baseline_only/"
echo "[$(date)]   is read-only."

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
    echo "[$(date)] initial-snapshot present — starting from byte-identical weights:"
    sha256sum Equivariant_pathway/baseline_only/checkpoints/initial_best_eq_policy.pth || true
else
    echo "[$(date)] initial_best_eq_policy.pth NOT FOUND in baseline_only/checkpoints/."
    echo "[$(date)]   p6_dynamic_pool will train its OWN initial model on the same 20 demos."
fi

echo "[$(date)] prompt YAMLs:"
ls -la Equivariant_pathway/p6_dynamic_pool/prompts/ || true

python -u -m Equivariant_pathway.p6_dynamic_pool.pipeline \
    --force_restart \
    "${EXTRA_ARGS[@]}"
rc=$?

echo "[$(date)] pipeline exited rc=${rc}"
if [ "${rc}" -ne 0 ]; then
    exit "${rc}"
fi

echo "[$(date)] artefacts in Equivariant_pathway/p6_dynamic_pool/:"
ls -la Equivariant_pathway/p6_dynamic_pool/ || true
echo "[$(date)]   dynamic_pool.yaml head:"
head -30 Equivariant_pathway/p6_dynamic_pool/dynamic_pool.yaml || true
echo "[$(date)]   results/:"
ls -la Equivariant_pathway/p6_dynamic_pool/results/ || true
echo "[$(date)] done."
