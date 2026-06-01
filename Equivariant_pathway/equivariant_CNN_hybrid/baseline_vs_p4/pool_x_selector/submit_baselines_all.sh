#!/bin/bash
# Submit N_RUNS (default 10) independent SLURM jobs, one per run_id, each
# running the 5 IIL baselines against P4's rotated correction pools. No LLM —
# each job is a single-GPU policy/ensemble/Q training job.
#
# Usage:
#   bash submit_baselines_all.sh                       # runs 1..10
#   AGGREGATE_AFTER=1 bash submit_baselines_all.sh      # + chained aggregation
#   METHODS=safe_dagger,stagger bash submit_baselines_all.sh   # subset
#   N_RUNS=2 SMOKE_FLAG=--smoke METHODS=safe_dagger+stagger \
#       SMOKE_RUN_IDS="99 100" bash submit_baselines_all.sh    # smoke
#
# Per-run seed = config.seed + run_id (the orchestrator does this), matching
# P4, so runs 1..10 read P4's run_1..run_10 shared pools.

set -eo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${SCRIPT_DIR}"

# ----------------------------- knobs ----------------------------- #
N_RUNS="${N_RUNS:-10}"
CONFIG="${CONFIG:-${SCRIPT_DIR}/config_baselines.yaml}"
SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-submit_baseline_one.sh}"
METHODS="${METHODS:-}"
BUDGET="${BUDGET:-}"
TARGET_SR="${TARGET_SR:-}"
CORRECTION_N="${CORRECTION_N:-}"
HELDOUT_N="${HELDOUT_N:-}"
INITIAL_DEMOS="${INITIAL_DEMOS:-}"
INITIAL_EPOCHS="${INITIAL_EPOCHS:-}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-}"
MAX_ROUNDS="${MAX_ROUNDS:-}"
MAX_STEPS="${MAX_STEPS:-}"
SEED="${SEED:-}"
MAX_CONSECUTIVE_EMPTY="${MAX_CONSECUTIVE_EMPTY:-}"
LR="${LR:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
WEIGHT_DECAY="${WEIGHT_DECAY:-}"
REPLAY_MIX="${REPLAY_MIX:-}"
REPLAY_MIX_FLOOR="${REPLAY_MIX_FLOOR:-}"
# baseline decision-rule knobs
TAU_SAFE="${TAU_SAFE:-}"
MC_N="${MC_N:-}"
DROPOUT_D="${DROPOUT_D:-}"
P_THRESH="${P_THRESH:-}"
TAU_DROP="${TAU_DROP:-}"
ENSEMBLE_M="${ENSEMBLE_M:-}"
TAU_ENS="${TAU_ENS:-}"
SIGMA_ENS="${SIGMA_ENS:-}"
ENSEMBLE_EVAL_MODE="${ENSEMBLE_EVAL_MODE:-}"
ALPHA_H="${ALPHA_H:-}"
GAMMA="${GAMMA:-}"
THRIFTY_RISK_AGG="${THRIFTY_RISK_AGG:-}"
Q_EPOCHS="${Q_EPOCHS:-}"
Q_LR="${Q_LR:-}"
SMOKE_FLAG="${SMOKE_FLAG:-}"
NO_PARALLEL="${NO_PARALLEL:-}"
SMOKE_RUN_IDS="${SMOKE_RUN_IDS:-}"   # space-sep ids; overrides the 1..N_RUNS loop
AGGREGATE_AFTER="${AGGREGATE_AFTER:-0}"

mkdir -p slurm_logs logs

# Build the --export list. ALL propagates env; each knob forwarded only if set.
EXPORT_BASE="ALL,CONFIG=${CONFIG}"
_append_if_set() {
    local name="$1" value="$2"
    if [ -n "${value}" ]; then
        EXPORT_BASE="${EXPORT_BASE},${name}=${value}"
    fi
}
# sbatch --export uses comma as the separator; encode METHODS commas as '+'.
_append_if_set METHODS "${METHODS//,/+}"
_append_if_set BUDGET "${BUDGET}"
_append_if_set TARGET_SR "${TARGET_SR}"
_append_if_set CORRECTION_N "${CORRECTION_N}"
_append_if_set HELDOUT_N "${HELDOUT_N}"
_append_if_set INITIAL_DEMOS "${INITIAL_DEMOS}"
_append_if_set INITIAL_EPOCHS "${INITIAL_EPOCHS}"
_append_if_set FINETUNE_EPOCHS "${FINETUNE_EPOCHS}"
_append_if_set MAX_ROUNDS "${MAX_ROUNDS}"
_append_if_set MAX_STEPS "${MAX_STEPS}"
_append_if_set SEED "${SEED}"
_append_if_set MAX_CONSECUTIVE_EMPTY "${MAX_CONSECUTIVE_EMPTY}"
_append_if_set LR "${LR}"
_append_if_set BATCH_SIZE "${BATCH_SIZE}"
_append_if_set WEIGHT_DECAY "${WEIGHT_DECAY}"
_append_if_set REPLAY_MIX "${REPLAY_MIX}"
_append_if_set REPLAY_MIX_FLOOR "${REPLAY_MIX_FLOOR}"
_append_if_set TAU_SAFE "${TAU_SAFE}"
_append_if_set MC_N "${MC_N}"
_append_if_set DROPOUT_D "${DROPOUT_D}"
_append_if_set P_THRESH "${P_THRESH}"
_append_if_set TAU_DROP "${TAU_DROP}"
_append_if_set ENSEMBLE_M "${ENSEMBLE_M}"
_append_if_set TAU_ENS "${TAU_ENS}"
_append_if_set SIGMA_ENS "${SIGMA_ENS}"
_append_if_set ENSEMBLE_EVAL_MODE "${ENSEMBLE_EVAL_MODE}"
_append_if_set ALPHA_H "${ALPHA_H}"
_append_if_set GAMMA "${GAMMA}"
_append_if_set THRIFTY_RISK_AGG "${THRIFTY_RISK_AGG}"
_append_if_set Q_EPOCHS "${Q_EPOCHS}"
_append_if_set Q_LR "${Q_LR}"
_append_if_set SMOKE_FLAG "${SMOKE_FLAG}"
_append_if_set NO_PARALLEL "${NO_PARALLEL}"

if [ -n "${SMOKE_RUN_IDS}" ]; then
    RUN_IDS="${SMOKE_RUN_IDS}"
else
    RUN_IDS="$(seq 1 "${N_RUNS}")"
fi

JOB_TIME="${JOB_TIME:-}"   # optional sbatch --time override (e.g. 02:00:00 for smoke backfill)
TIME_ARG=()
[ -n "${JOB_TIME}" ] && TIME_ARG=(--time="${JOB_TIME}")

JOB_IDS=()
for i in ${RUN_IDS}; do
    JOB_ID=$(sbatch --parsable \
        --job-name="bvp_baseline_${i}" \
        "${TIME_ARG[@]}" \
        --export="${EXPORT_BASE},RUN_ID=${i}" \
        "${SCRIPT_DIR}/${SUBMIT_SCRIPT}")
    JOB_IDS+=("${JOB_ID}")
    echo "[submit_baselines_all] run_id=${i}  sbatch_job_id=${JOB_ID}"
done

echo "[submit_baselines_all] submitted ${#JOB_IDS[@]} jobs: ${JOB_IDS[*]}"

if [ "${AGGREGATE_AFTER}" = "1" ]; then
    DEP="afterok:$(IFS=:; echo "${JOB_IDS[*]}")"
    AGG_ID=$(sbatch --parsable --dependency="${DEP}" \
        --export="ALL,CONFIG=${CONFIG}" \
        "${SCRIPT_DIR}/submit_aggregate.sh")
    echo "[submit_baselines_all] aggregation queued as job ${AGG_ID} (waits on ${DEP})"
fi
