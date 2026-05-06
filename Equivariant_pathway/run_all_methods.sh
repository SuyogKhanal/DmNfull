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
# Phases
# ======
# 1. SETUP (sequential, GPU 0): generate the 50-layout heldout YAML if
#    missing, persist train/test/heldout layouts as JSON+PNG, BFS-collect
#    the initial 10 expert demos, and train the SHARED initial
#    checkpoint. Every method must start from the same demos and the
#    same checkpoint, so this phase MUST finish before the parallel
#    methods launch.
#
# 2. METHODS (parallel, one per GPU): for each of
#    {baseline_dagger, p4, p5, p6}, launch run_full_cycle.py on its own
#    GPU with --skip_initial_collect --skip_initial_train --skip_charts
#    --methods <one>. Each method writes into its own subtree under
#    cycle_<ts>/<method>/ so the parallel runs do not contend on demos /
#    checkpoints / RAG bank.
#
# 3. CHARTS (sequential, GPU 0): once all four parallel methods finish,
#    run charts.py on the shared cycle_dir to emit the four required
#    line charts (baseline vs P4, vs P5, vs P6, combined).

set -eo pipefail

module purge
module load Anaconda3
source /home/s226137394/.bashrc
source activate
conda activate maze

# ---- Cycle configuration ----
INITIAL_DEMOS=10
INITIAL_EPOCHS=200
ROUND_EPOCHS=60
HELDOUT_N=50
SEED=0
TARGET_SR=0.90
MAX_ROUNDS=50              # runaway guard only — there is NO 5-round cap
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CYCLE_DIR="results/equivariant_pathway/cycle_${TIMESTAMP}"
METHODS=(baseline_dagger p4 p5 p6)

mkdir -p slurm_logs "${CYCLE_DIR}"

# ---- Phase 1: setup (sequential, GPU 0) ----------------------------------
# Run only the bootstrap part of the cycle: heldout YAML, layout tracking,
# initial 10 demos, initial checkpoint. We do this with --methods='' so
# run_full_cycle.py performs setup but skips every method loop.
echo "[$(date)] PHASE 1 — setup on GPU 0  -> ${CYCLE_DIR}"
CUDA_VISIBLE_DEVICES=0 python -u -m Equivariant_pathway.run_full_cycle \
    --methods "" \
    --cycle_dir "${CYCLE_DIR}" \
    --initial_demos "${INITIAL_DEMOS}" \
    --initial_epochs "${INITIAL_EPOCHS}" \
    --heldout_n "${HELDOUT_N}" \
    --seed "${SEED}" \
    --skip_charts \
    > "${CYCLE_DIR}/setup.log" 2>&1

if [ ! -f "Equivariant_pathway/checkpoints/best_eq_policy.pth" ]; then
    echo "[$(date)] ERROR: setup phase did not produce best_eq_policy.pth"
    exit 1
fi
echo "[$(date)] PHASE 1 done — initial checkpoint ready"

# ---- Phase 2: parallel methods (one per GPU) -----------------------------
# Each method gets its own GPU, its own log, its own subtree under
# ${CYCLE_DIR}/<method>/. They share the SETUP artefacts (cycle_dir,
# Equivariant_pathway/demos, Equivariant_pathway/checkpoints) read-only —
# run_full_cycle.py copies the initial demos + checkpoint into the
# per-method subtree on first invocation and writes nothing to the
# shared dirs after that.
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
        echo "[$(date)] method ${method} (pid ${pid}) FAILED"
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
