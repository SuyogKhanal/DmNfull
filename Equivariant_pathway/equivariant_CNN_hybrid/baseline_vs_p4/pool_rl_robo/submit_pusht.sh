#!/bin/bash
# Parallel-split launcher for the PushT-v1 MAIN run (bash, NOT sbatch — it calls
# sbatch internally). Two independent jobs run concurrently:
#   * p4_top3   — its own job: 3 GPUs (VLM+text+orchestrator) + vLLM, 10 days
#                 (LLM-bound: ~37 min/round failure rollout × ~100 rounds).
#   * baselines — its own job: 1 GPU, NO vLLM (diff_dagger + 5 IIL don't use the
#                 LLM), 4 days.
# Each job runs its own deterministic shared bootstrap into a separate
# shared_<tag>/ dir and writes a separate run_summary_<tag>.json (no clobber).
# Methods are '+'-encoded to survive sbatch --export comma truncation (C5).
#
#   bash submit_pusht.sh
#
set -e
SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SUITE_DIR}"
SEED="${SEED:-42}"
RUN_ID="${RUN_ID:-0}"
P4_TIME="${P4_TIME:-10-00:00:00}"
BL_TIME="${BL_TIME:-4-00:00:00}"
BASELINES="diff_dagger+safe_dagger+dropout_dagger+ensemble_dagger+thrifty_dagger+stagger"

echo "[submit_pusht] p4_top3 → own job: 3 GPU + vLLM, time=${P4_TIME}, seed=${SEED}"
METHODS=p4_top3 SEED="${SEED}" RUN_ID="${RUN_ID}" \
    sbatch --array=1 --gpus-per-node=3 --time="${P4_TIME}" \
           --job-name=prr_p4_pusht run_pool_rl_robo.sh

echo "[submit_pusht] baselines → own job: 1 GPU, no vLLM, time=${BL_TIME}, seed=${SEED}"
METHODS="${BASELINES}" SEED="${SEED}" RUN_ID="${RUN_ID}" \
    sbatch --array=1 --gpus-per-node=1 --time="${BL_TIME}" \
           --job-name=prr_bl_pusht run_pool_rl_robo.sh

echo "[submit_pusht] queued. watch: squeue --me ; tail -f slurm_logs/pool_rl_*_1.out"
