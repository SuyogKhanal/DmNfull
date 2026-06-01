#!/bin/bash
#SBATCH --job-name=bvp_baseline
#SBATCH --partition=gpu
#SBATCH --qos=batch-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --cpus-per-gpu=16
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s226137394@deakin.edu.au
#SBATCH --output=slurm_logs/bvp_baseline_%j.out
#SBATCH --error=slurm_logs/bvp_baseline_%j.err
#
# One SLURM job = one run_id, running the 5 IIL baselines (SafeDAgger*,
# DropoutDAgger, EnsembleDAgger, ThriftyDAgger, Stagger) against P4's rotated
# correction pools. These baselines DO NOT use the LLM — no vLLM servers, no
# proxy, no OPENAI_BASE_URL. Just one GPU for policy/ensemble/Q training.
#
# Submit ten of these with:
#   SUBMIT_SCRIPT=submit_baseline_one.sh bash submit_baselines_all.sh
# or by hand:
#   for i in {1..10}; do sbatch --export=ALL,RUN_ID=$i submit_baseline_one.sh; done

set -eo pipefail

# Under SLURM the script is copied to a spool dir; prefer SLURM_SUBMIT_DIR.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

# Conda env activation (mirrors submit_one_qwen.sh; the 'maze' env has the
# full pipeline + torch — no vllm needed for the baselines).
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
    conda activate "${BASELINE_CONDA_ENV:-maze}" || true
fi

mkdir -p "${SCRIPT_DIR}/slurm_logs" "${SCRIPT_DIR}/logs"

if [ -z "${RUN_ID:-}" ]; then
    echo "[submit_baseline_one] ERROR: RUN_ID not set." >&2
    exit 2
fi

# ----------------------------- knobs ----------------------------------------
CONFIG="${CONFIG:-${SCRIPT_DIR}/config_baselines.yaml}"
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
METHODS="${METHODS:-}"
# submit_baselines_all.sh encodes commas as '+' to survive sbatch --export.
METHODS="${METHODS//+/,}"

# ----------------------------- baseline decision-rule knobs -----------------
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

SMOKE_FLAG="${SMOKE_FLAG:-}"   # set to "--smoke" for a fast wiring check
NO_PARALLEL="${NO_PARALLEL:-}" # set to anything to run the 5 methods sequentially

# Build pass-through CLI flags (emit a flag only when its env var is set).
CLI=()
[ -n "${METHODS}" ]               && CLI+=(--methods "${METHODS}")
[ -n "${BUDGET}" ]                && CLI+=(--budget "${BUDGET}")
[ -n "${TARGET_SR}" ]             && CLI+=(--target_sr "${TARGET_SR}")
[ -n "${CORRECTION_N}" ]          && CLI+=(--correction_n "${CORRECTION_N}")
[ -n "${HELDOUT_N}" ]             && CLI+=(--heldout_n "${HELDOUT_N}")
[ -n "${INITIAL_DEMOS}" ]         && CLI+=(--initial_demos "${INITIAL_DEMOS}")
[ -n "${INITIAL_EPOCHS}" ]        && CLI+=(--initial_epochs "${INITIAL_EPOCHS}")
[ -n "${FINETUNE_EPOCHS}" ]       && CLI+=(--finetune_epochs "${FINETUNE_EPOCHS}")
[ -n "${MAX_ROUNDS}" ]            && CLI+=(--max_rounds "${MAX_ROUNDS}")
[ -n "${MAX_STEPS}" ]             && CLI+=(--max_steps "${MAX_STEPS}")
[ -n "${SEED}" ]                  && CLI+=(--seed "${SEED}")
[ -n "${MAX_CONSECUTIVE_EMPTY}" ] && CLI+=(--max_consecutive_empty "${MAX_CONSECUTIVE_EMPTY}")
[ -n "${LR}" ]                    && CLI+=(--lr "${LR}")
[ -n "${BATCH_SIZE}" ]            && CLI+=(--batch_size "${BATCH_SIZE}")
[ -n "${WEIGHT_DECAY}" ]          && CLI+=(--weight_decay "${WEIGHT_DECAY}")
[ -n "${REPLAY_MIX}" ]            && CLI+=(--replay_mix "${REPLAY_MIX}")
[ -n "${REPLAY_MIX_FLOOR}" ]      && CLI+=(--replay_mix_floor "${REPLAY_MIX_FLOOR}")
[ -n "${TAU_SAFE}" ]              && CLI+=(--tau_safe "${TAU_SAFE}")
[ -n "${MC_N}" ]                  && CLI+=(--mc_N "${MC_N}")
[ -n "${DROPOUT_D}" ]             && CLI+=(--dropout_d "${DROPOUT_D}")
[ -n "${P_THRESH}" ]              && CLI+=(--p_thresh "${P_THRESH}")
[ -n "${TAU_DROP}" ]              && CLI+=(--tau_drop "${TAU_DROP}")
[ -n "${ENSEMBLE_M}" ]            && CLI+=(--ensemble_M "${ENSEMBLE_M}")
[ -n "${TAU_ENS}" ]               && CLI+=(--tau_ens "${TAU_ENS}")
[ -n "${SIGMA_ENS}" ]             && CLI+=(--sigma_ens "${SIGMA_ENS}")
[ -n "${ENSEMBLE_EVAL_MODE}" ]    && CLI+=(--ensemble_eval_mode "${ENSEMBLE_EVAL_MODE}")
[ -n "${ALPHA_H}" ]               && CLI+=(--alpha_h "${ALPHA_H}")
[ -n "${GAMMA}" ]                 && CLI+=(--gamma "${GAMMA}")
[ -n "${THRIFTY_RISK_AGG}" ]      && CLI+=(--thrifty_risk_agg "${THRIFTY_RISK_AGG}")
[ -n "${Q_EPOCHS}" ]              && CLI+=(--q_epochs "${Q_EPOCHS}")
[ -n "${Q_LR}" ]                  && CLI+=(--q_lr "${Q_LR}")
[ -n "${SMOKE_FLAG}" ]            && CLI+=("${SMOKE_FLAG}")
[ -n "${NO_PARALLEL}" ]           && CLI+=(--no_parallel)

PER_RUN_LOG="${SCRIPT_DIR}/logs/baseline_run_$(printf '%02d' "${RUN_ID}").log"

echo "[submit_baseline_one] $(date)  RUN_ID=${RUN_ID}  job=${SLURM_JOB_ID:-?}"
echo "[submit_baseline_one] CONFIG=${CONFIG}"
echo "[submit_baseline_one] CLI overrides: ${CLI[*]:-<none>}"
echo "[submit_baseline_one] SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>&1 | sed 's/^/[submit_baseline_one] nvidia-smi: /'
fi

# ----- run the baseline orchestrator (no LLM servers needed) ----------------
python3 -u -m \
    Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_x_selector.orchestrator.run_baselines \
    --run_id "${RUN_ID}" \
    --config "${CONFIG}" \
    "${CLI[@]}" \
    2>&1 | tee "${PER_RUN_LOG}"

echo "[submit_baseline_one] $(date)  RUN_ID=${RUN_ID} DONE"
