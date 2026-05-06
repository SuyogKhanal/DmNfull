#!/bin/bash
#SBATCH --job-name=eq_full_cycle
#SBATCH --partition=gpu
#SBATCH --qos=batch-long
#SBATCH --gpus=2
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
ROUND_EPOCHS=200
HELDOUT_N=50
DAGGER_EPISODES=50        # rollouts per round (same for ALL methods)
SEED=0
TARGET_SR=0.90
MAX_ROUNDS=50               # runaway guard only — there is NO 5-round cap
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CYCLE_DIR="results/equivariant_pathway/cycle_${TIMESTAMP}"
# METHODS=(baseline_dagger p4 p5 p6)
METHODS=(baseline_dagger p4)

mkdir -p slurm_logs "${CYCLE_DIR}"

# ---- Phase 1: setup (sequential, GPU 0) ----------------------------------
# Run only the bootstrap part of the cycle: layout generation, BFS demo
# collection, initial training. We pass --methods='' so run_full_cycle.py
# performs setup but skips every method loop.
#
# We ALWAYS pass --force_retrain so every sbatch submission redoes the
# initial demo collection and initial training from scratch — otherwise
# Phase 1 would skip training whenever Equivariant_pathway/checkpoints/
# already contains best_eq_policy.pth from a previous run, and every
# per-method folder would silently inherit a stale model. With fixed seeds
# the regenerated layout YAMLs are bit-identical, so this is a no-op for
# layouts but a hard reset for demos + initial weights. EXTRA_SETUP_ARGS
# are appended after, so a caller passing --force_retrain again is still
# accepted by argparse (action="store_true" is idempotent).
echo "[$(date)] ============================================================"
echo "[$(date)] PHASE 1 — INITIAL DEMO COLLECTION + INITIAL TRAINING (GPU 0)"
echo "[$(date)] ============================================================"
echo "[$(date)]   cycle dir         : ${CYCLE_DIR}"
echo "[$(date)]   forwarding extras : ${EXTRA_SETUP_ARGS[*]}"
echo "[$(date)]   demo dir (target) : Equivariant_pathway/demos"
echo "[$(date)]   ckpt dir (target) : Equivariant_pathway/checkpoints"
echo "[$(date)]   --force_retrain is ALWAYS on: existing demos and checkpoints"
echo "[$(date)]   in Equivariant_pathway/{demos,checkpoints} are wiped and"
echo "[$(date)]   regenerated from scratch on every sbatch submission."
echo "[$(date)]   Initial training output is streamed to BOTH this slurm log"
echo "[$(date)]   AND ${CYCLE_DIR}/setup.log so the run is auditable from"
echo "[$(date)]   either place."
echo "[$(date)]   ----- BEGIN setup output -----"
# `tee` so the slurm .out captures everything (layouts, demo collection,
# training epochs, best/last checkpoint writes). `set -o pipefail` is on,
# so the rc of the python process is preserved through tee.
# Wrap the pipeline in `|| true` so `set -e` doesn't abort before we read
# PIPESTATUS[0]. We surface the python process's actual rc explicitly via
# the check below so a real setup failure still stops the script.
CUDA_VISIBLE_DEVICES=0 python -u -m Equivariant_pathway.run_full_cycle \
    --methods "" \
    --cycle_dir "${CYCLE_DIR}" \
    --initial_demos "${INITIAL_DEMOS}" \
    --initial_epochs "${INITIAL_EPOCHS}" \
    --heldout_n "${HELDOUT_N}" \
    --seed "${SEED}" \
    --skip_charts \
    --force_retrain \
    "${EXTRA_SETUP_ARGS[@]}" \
    2>&1 | tee "${CYCLE_DIR}/setup.log" || true
setup_rc=${PIPESTATUS[0]}
echo "[$(date)]   ----- END setup output (rc=${setup_rc}) -----"

if [ "${setup_rc}" -ne 0 ] || [ ! -f "Equivariant_pathway/checkpoints/best_eq_policy.pth" ]; then
    echo "[$(date)] ERROR: setup phase did not produce best_eq_policy.pth (rc=${setup_rc})"
    echo "[$(date)] tail of setup log:"
    tail -40 "${CYCLE_DIR}/setup.log" || true
    exit 1
fi
echo "[$(date)] PHASE 1 done — initial checkpoint ready"
echo "[$(date)]   files in Equivariant_pathway/demos/:"
ls -la Equivariant_pathway/demos/ | head -30 || true
echo "[$(date)]   files in Equivariant_pathway/checkpoints/:"
ls -la Equivariant_pathway/checkpoints/ | head -10 || true
# Print a sha256 of the shared starting checkpoint so the slurm log
# captures the exact hash every method copies into its own folder.
# This makes it trivial to verify in Phase 2 logs (or post-mortem) that
# every method started from the same byte-identical weights.
echo "[$(date)]   sha256 of shared best_eq_policy.pth:"
sha256sum Equivariant_pathway/checkpoints/best_eq_policy.pth || true
echo "[$(date)]   layout snapshots: ${CYCLE_DIR}/{training,test,heldout}/"

# ---- Phase 2: parallel methods (one per GPU) -----------------------------
# Each method gets its own GPU, its own log, its own subtree under
# ${CYCLE_DIR}/<method>/. They share the SETUP artefacts (cycle_dir,
# Equivariant_pathway/demos, Equivariant_pathway/checkpoints) read-only —
# run_full_cycle.py copies the initial 20 demos + initial checkpoint
# into the per-method subtree on first invocation and writes nothing to
# the shared dirs after that.
echo "[$(date)] PHASE 2 — launching ${#METHODS[@]} methods in parallel"

# Detect the number of GPUs actually available. Prefer $CUDA_VISIBLE_DEVICES
# (set by SLURM/the user); fall back to nvidia-smi. If fewer than 4 GPUs are
# present we still launch all four methods, but clamp the per-method GPU
# index with `i % N_GPUS` so we never pass an invalid index that would crash
# the worker with no clear error.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    N_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
elif command -v nvidia-smi >/dev/null 2>&1; then
    N_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
else
    N_GPUS=0
fi
if [ -z "${N_GPUS}" ] || [ "${N_GPUS}" -lt 1 ]; then
    N_GPUS=1
fi
if [ "${N_GPUS}" -lt 4 ]; then
    echo "[$(date)] WARNING: only ${N_GPUS} GPU(s) detected; ${#METHODS[@]} methods will share them via i % N_GPUS."
fi

PIDS=()
for i in 0 1; do
    method=${METHODS[$i]}
    method_log="${CYCLE_DIR}/${method}/method.log"
    mkdir -p "${CYCLE_DIR}/${method}"
    device=$(( i % N_GPUS ))
    echo "[$(date)]   launch ${method} on GPU ${device}  -> ${method_log}"
    CUDA_VISIBLE_DEVICES=${device} python -u -m Equivariant_pathway.run_full_cycle \
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
