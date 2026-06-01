#!/bin/bash
# Submit 10 (or N_RUNS) independent SLURM jobs, one per run_id.
#
# Each sbatch is a single GPU / single-process job that runs all 3
# methods sequentially for ONE run_id, writing into
# results/run_{id}/{baseline,p4_sequential,p4_batch}/.
#
# Usage:
#   bash submit_all.sh                  # submits 1..N_RUNS (default 10)
#   N_RUNS=4 bash submit_all.sh
#   METHODS=baseline bash submit_all.sh # smoke a subset of methods
#   AGGREGATE_AFTER=1 bash submit_all.sh  # also queue an aggregation job
#                                         # that runs after all 10 succeed
#
# After all jobs finish, run aggregation by hand:
#   sbatch --dependency=afterok:JOBID1,afterok:JOBID2,... submit_aggregate.sh
# or simply (interactive, single GPU not needed):
#   bash submit_aggregate.sh

set -eo pipefail

# Resolve this suite's directory. This script is a launcher (it queues
# sbatch jobs); the intended invocation is ``bash submit_all.sh`` from
# the login node. If someone runs ``sbatch submit_all.sh`` instead,
# SLURM copies the script into a spool dir before executing it, so
# ${BASH_SOURCE[0]} would resolve there. Prefer $SLURM_SUBMIT_DIR (the
# dir sbatch was invoked from) when present.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${SCRIPT_DIR}"

# ----------------------------- knobs ----------------------------- #
# Identical name-set to submit_one.sh — each one is forwarded to every
# child sbatch via --export so the per-run job sees the override.
N_RUNS="${N_RUNS:-10}"
CONFIG="${CONFIG:-${SCRIPT_DIR}/config.yaml}"
# SUBMIT_SCRIPT picks the underlying sbatch script. Defaults to the
# OpenAI-cluster path; set to ``submit_one_qwen.sh`` on the H100/H200
# cluster (vLLM + local Qwen models). The submit script must live next
# to submit_all.sh in this suite directory.
SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-submit_one.sh}"
METHODS="${METHODS:-}"
BUDGET="${BUDGET:-}"
TARGET_SR="${TARGET_SR:-}"
CORRECTION_N="${CORRECTION_N:-}"
HELDOUT_N="${HELDOUT_N:-}"
INITIAL_DEMOS="${INITIAL_DEMOS:-}"
INITIAL_EPOCHS="${INITIAL_EPOCHS:-}"
ROUND_EPOCHS="${ROUND_EPOCHS:-}"
BASELINE_ROUND_EPOCHS="${BASELINE_ROUND_EPOCHS:-}"
MAX_ROUNDS="${MAX_ROUNDS:-}"
MAX_STEPS="${MAX_STEPS:-}"
SEED="${SEED:-}"
LR="${LR:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
WEIGHT_DECAY="${WEIGHT_DECAY:-}"
REPLAY_MIX="${REPLAY_MIX:-}"
REPLAY_MIX_FLOOR="${REPLAY_MIX_FLOOR:-}"
CORRIDOR_BLOCKING="${CORRIDOR_BLOCKING:-}"
AGGREGATE_AFTER="${AGGREGATE_AFTER:-0}"

mkdir -p slurm_logs logs

# Build the --export list. `ALL` propagates PATH/HOME/etc.; each knob is
# only forwarded when it's actually set so unset values fall through to
# config.yaml on the receiving end.
EXPORT_BASE="ALL,CONFIG=${CONFIG}"
_append_if_set() {
    local name="$1" value="$2"
    if [ -n "${value}" ]; then
        EXPORT_BASE="${EXPORT_BASE},${name}=${value}"
    fi
}
# sbatch --export uses comma as the variable separator; encode commas in
# METHODS as '+' so multi-method values survive transport. The receiving
# submit_one*.sh decodes them back.
_append_if_set METHODS "${METHODS//,/+}"
_append_if_set BUDGET "${BUDGET}"
_append_if_set TARGET_SR "${TARGET_SR}"
_append_if_set CORRECTION_N "${CORRECTION_N}"
_append_if_set HELDOUT_N "${HELDOUT_N}"
_append_if_set INITIAL_DEMOS "${INITIAL_DEMOS}"
_append_if_set INITIAL_EPOCHS "${INITIAL_EPOCHS}"
_append_if_set ROUND_EPOCHS "${ROUND_EPOCHS}"
_append_if_set BASELINE_ROUND_EPOCHS "${BASELINE_ROUND_EPOCHS}"
_append_if_set MAX_ROUNDS "${MAX_ROUNDS}"
_append_if_set MAX_STEPS "${MAX_STEPS}"
_append_if_set SEED "${SEED}"
_append_if_set LR "${LR}"
_append_if_set BATCH_SIZE "${BATCH_SIZE}"
_append_if_set WEIGHT_DECAY "${WEIGHT_DECAY}"
_append_if_set REPLAY_MIX "${REPLAY_MIX}"
_append_if_set REPLAY_MIX_FLOOR "${REPLAY_MIX_FLOOR}"
_append_if_set CORRIDOR_BLOCKING "${CORRIDOR_BLOCKING}"

JOB_IDS=()
for ((i=1; i<=N_RUNS; i++)); do
    JOB_ID=$(sbatch --parsable \
        --job-name="bvp_run_${i}" \
        --export="${EXPORT_BASE},RUN_ID=${i}" \
        "${SCRIPT_DIR}/${SUBMIT_SCRIPT}")
    JOB_IDS+=("${JOB_ID}")
    echo "[submit_all] run_id=${i}  sbatch_job_id=${JOB_ID}"
done

echo "[submit_all] submitted ${#JOB_IDS[@]} jobs: ${JOB_IDS[*]}"

if [ "${AGGREGATE_AFTER}" = "1" ]; then
    DEP="afterok:$(IFS=:; echo "${JOB_IDS[*]}")"
    AGG_ID=$(sbatch --parsable --dependency="${DEP}" \
        --export="ALL,CONFIG=${CONFIG}" \
        "${SCRIPT_DIR}/submit_aggregate.sh")
    echo "[submit_all] aggregation queued as job ${AGG_ID} (waits on ${DEP})"
fi
