#!/bin/bash
#SBATCH --job-name=p4replay
#SBATCH --partition=gpu
#SBATCH --qos=batch-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=48G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=08:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s226137394@deakin.edu.au
#SBATCH --output=slurm_logs/p4replay_%j.out
#SBATCH --error=slurm_logs/p4replay_%j.err
#
# One SLURM job = one run_id. Retrains P4-top3-rotated's ALREADY-PRESCRIBED
# demonstrations at EPOCHS (default 90) per round — no LLM, no rollout, no
# re-prescription. Writes results/run_{id}/<DEST_METHOD>/.
#
#   for i in {1..10}; do EPOCHS=90 sbatch --export=ALL,RUN_ID=$i submit_p4_replay_one.sh; done

set -eo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

if command -v module >/dev/null 2>&1; then module purge || true; module load Anaconda3 || true; fi
if [ -f "${HOME}/.bashrc" ]; then source "${HOME}/.bashrc"; fi
if command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)"; conda activate "${BASELINE_CONDA_ENV:-maze}" || true; fi

mkdir -p "${SCRIPT_DIR}/slurm_logs" "${SCRIPT_DIR}/logs"

if [ -z "${RUN_ID:-}" ]; then echo "[p4replay] ERROR: RUN_ID not set." >&2; exit 2; fi

CONFIG="${CONFIG:-${SCRIPT_DIR}/config_baselines.yaml}"
EPOCHS="${EPOCHS:-90}"
SOURCE_METHOD="${SOURCE_METHOD:-p4_top3_rotate}"
DEST_METHOD="${DEST_METHOD:-p4_top3_rotate_e90}"

CLI=(--run_id "${RUN_ID}" --config "${CONFIG}" --epochs "${EPOCHS}"
     --source_method "${SOURCE_METHOD}" --dest_method "${DEST_METHOD}")
[ -n "${LR:-}" ]          && CLI+=(--lr "${LR}")
[ -n "${BATCH_SIZE:-}" ]  && CLI+=(--batch_size "${BATCH_SIZE}")
[ -n "${SEED:-}" ]        && CLI+=(--seed "${SEED}")
[ -n "${MAX_STEPS:-}" ]   && CLI+=(--max_steps "${MAX_STEPS}")

PER_RUN_LOG="${SCRIPT_DIR}/logs/p4replay_run_$(printf '%02d' "${RUN_ID}").log"
echo "[p4replay] $(date) RUN_ID=${RUN_ID} EPOCHS=${EPOCHS} -> ${DEST_METHOD}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>&1 | sed 's/^/[p4replay] /'

python3 -u -m \
    Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_x_selector.orchestrator.replay_p4 \
    "${CLI[@]}" 2>&1 | tee "${PER_RUN_LOG}"

echo "[p4replay] $(date) RUN_ID=${RUN_ID} DONE"
