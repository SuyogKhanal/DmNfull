#!/bin/bash
#SBATCH --job-name=d5_pusht
#SBATCH --partition=gpu-large
#SBATCH --qos=batch-long
#SBATCH --constraint=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=distil/slurm_logs/%x_%j.out
#SBATCH --error=distil/slurm_logs/%x_%j.err
# ===========================================================================
# run_pusht_d5.sbatch — ONE ROUND of Push-T (IMAGE modality) for the D5_Compute
# table. Modelled on distil/scripts/run_distil.sbatch (same partition/qos/log
# pattern, parameterised purely by env vars).
#
# Push-T does NOT live in the distil/ module (distil.config has no Push-T task);
# it lives in the fork-backed suite at
#   Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_rl_robo
# whose LLM is a LOCAL vLLM (Qwen3-32B text + Qwen3-VL-32B vision) fronted by the
# qwen proxy — NOT OpenRouter. So this wrapper delegates to that suite's proven
# launcher (run_pool_rl_robo.sh), which starts the two vLLM servers + proxy and
# then runs the orchestrator. Nothing about the arms is re-implemented here.
#
# Env vars (pass with --export=ALL,...):
#   METHODS   p4_subtask (DISEIL) | safe_dagger (baseline)     (required)
#   SEED      integer                                          (default 1)
#   RUN_ID    results/PushT-v1/run_<RUN_ID>/                   (default 511)
#   REUSE_CKPT  init_ckpt.pth reused byte-identically by BOTH arms
#               (default: the finished image bootstrap of run_111)
#   CONFIG    suite config                     (default config_pusht_image_d5.yaml)
#   D5_TELEMETRY=1 arms the additive per-round wall-clock + token telemetry
#                  (default 1 here; the suite is inert when it is unset)
#
# Resources: the DISEIL arm needs 3 GPUs (GPU0 VLM, GPU1 LLM, GPU2 orchestrator)
# and is H100-ONLY (on h200l the extra rgb reposition env hangs during Vulkan
# render-context setup — confirmed job 102948). The SafeDAgger arm needs no LLM:
# override with --gpus-per-node=1 --constraint="gpu-h100|gpu-h200".
#
# Examples:
#   # DISEIL (Push-T image)
#   sbatch --job-name=d5_PushT_image_full \
#     --export=ALL,METHODS=p4_subtask,SEED=1,RUN_ID=511 \
#     distil/scripts/run_pusht_d5.sbatch
#   # SafeDAgger baseline (Push-T image)
#   sbatch --job-name=d5_PushT_image_safe --gpus-per-node=1 \
#     --constraint="gpu-h100|gpu-h200" \
#     --export=ALL,METHODS=safe_dagger,SEED=1,RUN_ID=511 \
#     distil/scripts/run_pusht_d5.sbatch
# ===========================================================================
set -eo pipefail

ROOT="${DISTIL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ROOT"
mkdir -p distil/slurm_logs

SUITE_DIR="${SUITE_DIR:-$ROOT/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_rl_robo}"
[ -d "$SUITE_DIR" ] || { echo "[d5] ERROR: suite dir not found: $SUITE_DIR" >&2; exit 2; }

export ENV="${ENV:-PushT-v1}"
export METHODS="${METHODS:?set METHODS=p4_subtask|safe_dagger}"
export SEED="${SEED:-1}"
export RUN_ID="${RUN_ID:-511}"
export CONFIG="${CONFIG:-$SUITE_DIR/config_pusht_image_d5.yaml}"
export SUITE_HYDRA_CFG="${SUITE_HYDRA_CFG:-$SUITE_DIR/configs/hydra/pusht_image.yaml}"
export P4_REUSE_INIT_CKPT="${REUSE_CKPT:-$SUITE_DIR/results/PushT-v1/run_111/shared_baselines/init_ckpt.pth}"
export CONDA_ENV="${CONDA_ENV:-diffdagger}"
export D5_TELEMETRY="${D5_TELEMETRY:-1}"

[ -f "$P4_REUSE_INIT_CKPT" ] || {
  echo "[d5] ERROR: shared image bootstrap missing: $P4_REUSE_INIT_CKPT" >&2
  exit 3; }

echo "=== D5 Push-T (image): methods=$METHODS seed=$SEED run_id=$RUN_ID ==="
echo "config : $CONFIG"
echo "hydra  : $SUITE_HYDRA_CFG"
echo "reuse  : $P4_REUSE_INIT_CKPT"
echo "telem  : D5_TELEMETRY=$D5_TELEMETRY -> results/PushT-v1/run_$RUN_ID/$METHODS/results/telemetry/d5_events.jsonl"
nvidia-smi -L || true

# run_pool_rl_robo.sh resolves its own suite root from SLURM_SUBMIT_DIR; we submit
# from the repo root, so point it at the suite explicitly (it is the ONLY use of
# that variable in that script).
export SLURM_SUBMIT_DIR="$SUITE_DIR"
exec bash "$SUITE_DIR/run_pool_rl_robo.sh"
