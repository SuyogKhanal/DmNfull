#!/bin/bash
#SBATCH --job-name=eq_full_cycle
#SBATCH --partition=gpu
#SBATCH --qos=batch-long
#SBATCH --gpus=4
#SBATCH --mem=80G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=68:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s226137394@deakin.edu.au
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# Equivariant_pathway entry point — runs the full cycle for the four
# methods in parallel (one method per GPU). Mirrors the layout of
# dmnpol/scripts/run_all_models.sh.
#
# Usage
# =====
#   sbatch Equivariant_pathway/run_all_methods.sh                 # default
#   sbatch Equivariant_pathway/run_all_methods.sh --force_retrain # nuke artefacts first
#   bash   Equivariant_pathway/run_all_methods.sh                 # interactive 4-GPU box
#
# Any extra CLI flags after the script name are forwarded to the Phase 1
# setup invocation, so e.g. `--force_retrain`, `--regen_training`, or
# `--regen_heldout` will trigger the appropriate fresh-sample paths.
#
# Phases
# ======
# 1. SETUP (sequential, GPU 0):
#    1A. AUTO-GENERATE layouts when missing/stale:
#          training_layouts.yaml — 20 unique random 5x5 layouts (random
#                                  start/goal/fires each), excluded
#                                  against test + heldout signatures
#          heldout_test_layouts.yaml — 50 random layouts, disjoint
#    1B. BFS-collect 20 expert demos — exactly ONE demo per training
#        layout (n_repetitions=1, dmnpol-style).
#    1C. Train the SHARED initial best_eq_policy.pth + last_eq_policy.pth
#        on those 20 demos.
#    Layout sets are persisted as JSON+PNG into
#    cycle_<ts>/{training,test,heldout}/ for audit.
#
# 2. METHODS (parallel, one per GPU): for each of
#    {baseline_dagger, p4, p5, p6}, launch run_full_cycle.py on its own
#    GPU with --skip_initial_collect --skip_initial_train --skip_charts.
#    Each method writes into its own subtree under
#    cycle_<ts>/<method>/{demos,checkpoints,rounds,...} (rag_bank only
#    for p5/p6) and runs round-by-round until heldout success rate
#    reaches 0.90 — there is NO 5-round cap.
#
# 3. CHARTS (sequential, GPU 0): once all four parallel methods finish,
#    run charts.py on the shared cycle_dir to emit the four required
#    line charts (baseline_vs_p4.png, baseline_vs_p5.png,
#    baseline_vs_p6.png, baseline_vs_all.png).

set -eo pipefail

# Pass-through flags from the caller (e.g. --force_retrain) go to Phase 1.
# IMPORTANT: capture them BEFORE sourcing conda activation scripts, because
# `source` inherits the script's positional parameters — `source activate`
# with no explicit env arg will pick up $1 (e.g. "--force_retrain") and try
# to activate it as a conda env, producing
#   "EnvironmentNameNotFound: Could not find conda environment: --force_retrain".
# After capture we `set --` to wipe $@ so the sourced scripts see no args.
EXTRA_SETUP_ARGS=("$@")
set --

module purge
module load Anaconda3
source /home/s226137394/.bashrc

# Initialize conda for this shell session, then activate the env.
# `eval "$(conda shell.bash hook)"` is the modern, $@-safe replacement
# for the legacy `source activate` step (which was needed before
# `conda activate maze` could work but happily ate $1 as an env name —
# the very bug we worked around above). The hook just registers the
# `conda` shell function and never reads positional parameters.
eval "$(conda shell.bash hook)"
conda activate maze

# ---- Cycle configuration ----
INITIAL_DEMOS=20            # 20 unique random training layouts, 1 demo each
INITIAL_EPOCHS=200
ROUND_EPOCHS=60
HELDOUT_N=50
DAGGER_EPISODES=50          # rollouts per round (same for ALL methods)
SEED=0
TARGET_SR=0.90
MAX_ROUNDS=50               # runaway guard only — there is NO 5-round cap
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CYCLE_DIR="results/equivariant_pathway/cycle_${TIMESTAMP}"
METHODS=(baseline_dagger p4 p5 p6)

mkdir -p slurm_logs "${CYCLE_DIR}"

# ---- Phase 1: setup (sequential, GPU 0) ----------------------------------
# Run only the bootstrap part of the cycle: layout generation, BFS demo
# collection, initial training. We pass --methods='' so run_full_cycle.py
# performs setup but skips every method loop. EXTRA_SETUP_ARGS are
# forwarded so the caller can opt into --force_retrain etc. without
# editing this script.
echo "[$(date)] PHASE 1 — setup on GPU 0  -> ${CYCLE_DIR}"
echo "[$(date)]   forwarding extra args: ${EXTRA_SETUP_ARGS[*]}"
CUDA_VISIBLE_DEVICES=0 python -u -m Equivariant_pathway.run_full_cycle \
    --methods "" \
    --cycle_dir "${CYCLE_DIR}" \
    --initial_demos "${INITIAL_DEMOS}" \
    --initial_epochs "${INITIAL_EPOCHS}" \
    --heldout_n "${HELDOUT_N}" \
    --seed "${SEED}" \
    --skip_charts \
    "${EXTRA_SETUP_ARGS[@]}" \
    > "${CYCLE_DIR}/setup.log" 2>&1

if [ ! -f "Equivariant_pathway/checkpoints/best_eq_policy.pth" ]; then
    echo "[$(date)] ERROR: setup phase did not produce best_eq_policy.pth"
    echo "[$(date)] tail of setup log:"
    tail -40 "${CYCLE_DIR}/setup.log" || true
    exit 1
fi
echo "[$(date)] PHASE 1 done — initial checkpoint ready"
echo "[$(date)]   files in Equivariant_pathway/demos/:"
ls -la Equivariant_pathway/demos/ | head -30 || true
echo "[$(date)]   files in Equivariant_pathway/checkpoints/:"
ls -la Equivariant_pathway/checkpoints/ | head -10 || true
echo "[$(date)]   layout snapshots: ${CYCLE_DIR}/{training,test,heldout}/"

# ---- Phase 2: parallel methods (one per GPU) -----------------------------
# Each method gets its own GPU, its own log, its own subtree under
# ${CYCLE_DIR}/<method>/. They share the SETUP artefacts (cycle_dir,
# Equivariant_pathway/demos, Equivariant_pathway/checkpoints) read-only —
# run_full_cycle.py copies the initial 20 demos + initial checkpoint
# into the per-method subtree on first invocation and writes nothing to
# the shared dirs after that.
echo "[$(date)] PHASE 2 — launching ${#METHODS[@]} methods in parallel"
PIDS=()
for i in 0 1 2 3; do
    method=${METHODS[$i]}
    method_log="${CYCLE_DIR}/${method}/method.log"
    mkdir -p "${CYCLE_DIR}/${method}"
    echo "[$(date)]   launch ${method} on GPU ${i}  -> ${method_log}"
    CUDA_VISIBLE_DEVICES=${i} python -u -m Equivariant_pathway.run_full_cycle \
        --methods "${method}" \
        --cycle_dir "${CYCLE_DIR}" \
        --round_epochs "${ROUND_EPOCHS}" \
        --max_rounds "${MAX_ROUNDS}" \
        --dagger_episodes "${DAGGER_EPISODES}" \
        --seed "${SEED}" \
        --skip_initial_collect \
        --skip_initial_train \
        --skip_charts \
        > "${method_log}" 2>&1 &
    PIDS+=($!)
done

# ---- Wait for all methods; record failures without aborting --------------
echo "[$(date)] waiting on ${#PIDS[@]} method runs..."
fail=0
for idx in "${!PIDS[@]}"; do
    pid=${PIDS[$idx]}
    method=${METHODS[$idx]}
    if wait "${pid}"; then
        echo "[$(date)] method ${method} (pid ${pid}) OK"
    else
        echo "[$(date)] method ${method} (pid ${pid}) FAILED — see ${CYCLE_DIR}/${method}/method.log"
        fail=$((fail + 1))
    fi
done

# ---- Phase 3: charts (sequential) ----------------------------------------
echo "[$(date)] PHASE 3 — charts on GPU 0"
CUDA_VISIBLE_DEVICES=0 python -u -m Equivariant_pathway.charts \
    --cycle_dir "${CYCLE_DIR}" \
    > "${CYCLE_DIR}/charts.log" 2>&1 || {
        echo "[$(date)] WARNING: charts.py exited non-zero; partial logs in ${CYCLE_DIR}/charts.log"
    }

if [ ${fail} -gt 0 ]; then
    echo "[$(date)] WARNING: ${fail} method(s) failed; charts may be partial."
    echo "[$(date)] cycle dir: ${CYCLE_DIR}"
    exit 1
fi
echo "[$(date)] all done. cycle dir: ${CYCLE_DIR}"
echo "[$(date)] charts: ${CYCLE_DIR}/charts/"
