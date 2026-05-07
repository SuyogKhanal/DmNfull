#!/bin/bash
#SBATCH --job-name=eq_baseline_dynamic_pool
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
# Random-expansion baseline mirroring p6_dynamic_pool. Each round samples
# N random layouts (N = prior round's heldout failure count, capped by
# config.max_pool_expansion_per_round) signature-disjoint from heldout +
# training + existing pool, BFS-collects demos on them, retrains, and
# evals on heldout. The held-out 50 layouts are NEVER touched by any demo
# collector — only their failure COUNT feeds back as a budget signal.
#
# Pre-requisite: baseline_only/ has been run with the new architecture so
# its heldout / training / correction YAMLs and demos exist.

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
echo "[$(date)] BASELINE DYNAMIC-POOL EXPANSION (RANDOM)"
echo "[$(date)] ============================================================"
echo "[$(date)]   forwarding extras: ${EXTRA_ARGS[*]}"
echo "[$(date)]   --force_restart is ALWAYS on: baseline_dynamic_pool/{demos,"
echo "[$(date)]   checkpoints,results,dynamic_pool.yaml} are wiped and"
echo "[$(date)]   re-bootstrapped from baseline_only/ on every submission."
echo "[$(date)]   baseline_only/ stays read-only."

if [ ! -f "Equivariant_pathway/baseline_only/heldout_layouts.yaml" ] \
   || [ ! -f "Equivariant_pathway/baseline_only/correction_layouts.yaml" ] \
   || [ ! -d "Equivariant_pathway/baseline_only/demos" ]; then
    echo "[$(date)] ERROR: baseline_only/ has not been run with the new"
    echo "[$(date)]        correction-pool architecture yet."
    exit 1
fi

if [ -f "Equivariant_pathway/baseline_only/checkpoints/initial_best_eq_policy.pth" ]; then
    echo "[$(date)] initial-snapshot present — starting from byte-identical weights:"
    sha256sum Equivariant_pathway/baseline_only/checkpoints/initial_best_eq_policy.pth || true
else
    echo "[$(date)] initial_best_eq_policy.pth NOT FOUND in baseline_only/checkpoints/."
    echo "[$(date)]   baseline_dynamic_pool will train its own initial model on the same 20 demos."
fi

python -u -m Equivariant_pathway.baseline_dynamic_pool.pipeline \
    --force_restart \
    "${EXTRA_ARGS[@]}"
rc=$?

echo "[$(date)] pipeline exited rc=${rc}"
if [ "${rc}" -ne 0 ]; then
    exit "${rc}"
fi

echo "[$(date)] artefacts in Equivariant_pathway/baseline_dynamic_pool/:"
ls -la Equivariant_pathway/baseline_dynamic_pool/ || true
echo "[$(date)]   dynamic_pool.yaml head:"
head -30 Equivariant_pathway/baseline_dynamic_pool/dynamic_pool.yaml || true
echo "[$(date)]   results/:"
ls -la Equivariant_pathway/baseline_dynamic_pool/results/ || true
echo "[$(date)] done."
