#!/bin/bash
#SBATCH --job-name=bvp_seqbatch
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --gpus=1
#SBATCH --constraint="gpu-l40s|gpu-v100"
#SBATCH --mem=60G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s226137394@deakin.edu.au
#SBATCH --output=slurm_logs/bvp_seqbatch_%j.out
#SBATCH --error=slurm_logs/bvp_seqbatch_%j.err
#
# One SLURM job = one run_id. Submit ten of these with:
#   bash submit_all.sh
# or by hand:
#   for i in {1..10}; do
#       sbatch --export=ALL,RUN_ID=$i submit_one.sh
#   done
#
# RUN_ID is required (passed via --export). METHODS is optional; defaults
# to whatever config.yaml says (baseline,p4_sequential,p4_batch).
# CONFIG is optional and defaults to this suite's config.yaml.

set -eo pipefail

# Resolve the suite directory.
#
# Under SLURM, this script is *copied* to a per-job spool dir
# (/var/spool/slurm/d/jobNNNN/slurm_script) before execution, so
# `${BASH_SOURCE[0]}` points at the spool copy, not the original. We
# prefer ``$SLURM_SUBMIT_DIR`` (the dir sbatch was invoked from) and
# fall back to BASH_SOURCE only for non-SLURM hand invocations.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

# Same env-init dance as swatch.sh.
if command -v module >/dev/null 2>&1; then
    module purge || true
    module load Anaconda3 || true
fi
if [ -f "${HOME}/.bashrc" ]; then
    # shellcheck disable=SC1091
    source "${HOME}/.bashrc"
fi
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate maze || true
fi

mkdir -p "${SCRIPT_DIR}/slurm_logs" "${SCRIPT_DIR}/logs"

if [ -z "${RUN_ID:-}" ]; then
    echo "[submit_one] ERROR: RUN_ID not set. Submit with --export=ALL,RUN_ID=<n>." >&2
    exit 2
fi

# ----------------------------- knobs ----------------------------- #

CONFIG="${CONFIG:-${SCRIPT_DIR}/config.yaml}"
BUDGET="${BUDGET:-}"
TARGET_SR="${TARGET_SR:-}"
CORRECTION_N="${CORRECTION_N:-}"
HELDOUT_N="${HELDOUT_N:-}"
INITIAL_DEMOS="${INITIAL_DEMOS:-}"
INITIAL_EPOCHS="${INITIAL_EPOCHS:-}"
ROUND_EPOCHS="${ROUND_EPOCHS:-500}"                  
BASELINE_ROUND_EPOCHS="${BASELINE_ROUND_EPOCHS:-100}"  
MAX_ROUNDS="${MAX_ROUNDS:-}"
MAX_STEPS="${MAX_STEPS:-}"
SEED="${SEED:-}"
LR="${LR:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
WEIGHT_DECAY="${WEIGHT_DECAY:-}"
REPLAY_MIX="${REPLAY_MIX:-}"
REPLAY_MIX_FLOOR="${REPLAY_MIX_FLOOR:-}"
CORRIDOR_BLOCKING="${CORRIDOR_BLOCKING:-}"        # true / false
METHODS="${METHODS:-}"
# submit_{smoke,all}.sh encodes commas as '+' to survive sbatch --export
# (which uses commas as the variable separator). Decode here.
METHODS="${METHODS//+/,}"

# Build the python CLI flag list — only emit a flag when its env var is set,
# so unset values fall through to config.yaml.
CLI=()
[ -n "${METHODS}" ]              && CLI+=(--methods "${METHODS}")
[ -n "${BUDGET}" ]               && CLI+=(--budget "${BUDGET}")
[ -n "${TARGET_SR}" ]            && CLI+=(--target_sr "${TARGET_SR}")
[ -n "${CORRECTION_N}" ]         && CLI+=(--correction_n "${CORRECTION_N}")
[ -n "${HELDOUT_N}" ]            && CLI+=(--heldout_n "${HELDOUT_N}")
[ -n "${INITIAL_DEMOS}" ]        && CLI+=(--initial_demos "${INITIAL_DEMOS}")
[ -n "${INITIAL_EPOCHS}" ]       && CLI+=(--initial_epochs "${INITIAL_EPOCHS}")
[ -n "${ROUND_EPOCHS}" ]         && CLI+=(--p4_finetune_epochs "${ROUND_EPOCHS}")
[ -n "${BASELINE_ROUND_EPOCHS}" ] && CLI+=(--baseline_finetune_epochs "${BASELINE_ROUND_EPOCHS}")
[ -n "${MAX_ROUNDS}" ]           && CLI+=(--max_rounds "${MAX_ROUNDS}")
[ -n "${MAX_STEPS}" ]            && CLI+=(--max_steps "${MAX_STEPS}")
[ -n "${SEED}" ]                 && CLI+=(--seed "${SEED}")
[ -n "${LR}" ]                   && CLI+=(--lr "${LR}")
[ -n "${BATCH_SIZE}" ]           && CLI+=(--batch_size "${BATCH_SIZE}")
[ -n "${WEIGHT_DECAY}" ]         && CLI+=(--weight_decay "${WEIGHT_DECAY}")
[ -n "${REPLAY_MIX}" ]           && CLI+=(--replay_mix "${REPLAY_MIX}")
[ -n "${REPLAY_MIX_FLOOR}" ]     && CLI+=(--replay_mix_floor "${REPLAY_MIX_FLOOR}")
[ -n "${CORRIDOR_BLOCKING}" ]    && CLI+=(--corridor_blocking "${CORRIDOR_BLOCKING}")

PER_RUN_LOG="${SCRIPT_DIR}/logs/run_$(printf '%02d' "${RUN_ID}").log"
echo "[submit_one] $(date)  RUN_ID=${RUN_ID}  GPU=${CUDA_VISIBLE_DEVICES:-?}  job=${SLURM_JOB_ID:-?}"
echo "[submit_one] config=${CONFIG}"
echo "[submit_one] CLI overrides: ${CLI[*]:-<none>}"
echo "[submit_one] per-run log -> ${PER_RUN_LOG}"

python3 -u -m \
    Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.baseline_vs_p4_sequential_batch.orchestrator.run_one \
    --run_id "${RUN_ID}" \
    --config "${CONFIG}" \
    "${CLI[@]}" \
    2>&1 | tee "${PER_RUN_LOG}"

echo "[submit_one] $(date)  RUN_ID=${RUN_ID} DONE"
