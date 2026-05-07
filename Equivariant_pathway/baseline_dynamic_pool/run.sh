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

# Detailed precheck — print which specific artefact is missing so the
# user can self-diagnose. The previous "has not been run" message hid
# whether the issue was a missing YAML, an empty demos/, or running
# from the wrong working directory.
echo "[$(date)] precheck: looking for baseline_only/ artefacts"
echo "[$(date)]   working directory: $(pwd)"
need_run_baseline=false
required_files=(
    "Equivariant_pathway/baseline_only/heldout_layouts.yaml"
    "Equivariant_pathway/baseline_only/correction_layouts.yaml"
    "Equivariant_pathway/baseline_only/training_layouts.yaml"
)
required_dirs=(
    "Equivariant_pathway/baseline_only/demos"
)
for f in "${required_files[@]}"; do
    if [ ! -f "$f" ]; then
        echo "[$(date)]   MISSING file: $f"
        need_run_baseline=true
    else
        echo "[$(date)]   found:        $f"
    fi
done
for d in "${required_dirs[@]}"; do
    if [ ! -d "$d" ]; then
        echo "[$(date)]   MISSING dir:  $d"
        need_run_baseline=true
    else
        n=$(find "$d" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
        echo "[$(date)]   found:        $d  ($n top-level *.json files)"
        if [ "$n" -eq 0 ]; then
            echo "[$(date)]                 (directory is empty — counts as missing)"
            need_run_baseline=true
        else
            # Break the count down by source field so the user can see
            # upfront that only the initial BFS demos get copied. The
            # DAgger corrections from baseline_only's own loop are
            # filtered out by the bootstrap to prevent contamination.
            init_count=0
            dagger_count=0
            unknown_count=0
            while IFS= read -r f; do
                if grep -q '"source": *"equivariant_pathway_bfs"' "$f" 2>/dev/null; then
                    init_count=$((init_count + 1))
                elif grep -q '"source": *"baseline_dagger_correction"' "$f" 2>/dev/null; then
                    dagger_count=$((dagger_count + 1))
                else
                    unknown_count=$((unknown_count + 1))
                fi
            done < <(find "$d" -maxdepth 1 -name '*.json' 2>/dev/null)
            echo "[$(date)]                 source breakdown: $init_count initial BFS, $dagger_count DAgger correction, $unknown_count unknown"
            echo "[$(date)]                 baseline_dynamic_pool WILL COPY ONLY the $init_count initial BFS demos"
            echo "[$(date)]                 (DAgger corrections are skipped — they would contaminate the starting set)"
        fi
    fi
done

if [ "${need_run_baseline}" = true ]; then
    echo "[$(date)] ============================================================"
    echo "[$(date)] ERROR: baseline_only/ is missing one or more required artefacts."
    echo "[$(date)] Run baseline_only first to regenerate them:"
    echo "[$(date)]   sbatch Equivariant_pathway/baseline_only/run.sh"
    echo "[$(date)] (or interactively: bash Equivariant_pathway/baseline_only/run.sh)"
    echo "[$(date)] That populates: heldout_layouts.yaml, correction_layouts.yaml,"
    echo "[$(date)]   training_layouts.yaml, demos/, checkpoints/."
    echo "[$(date)] Then re-submit this job."
    echo "[$(date)] ============================================================"
    exit 1
fi
echo "[$(date)] precheck: all required baseline_only/ artefacts present."

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
