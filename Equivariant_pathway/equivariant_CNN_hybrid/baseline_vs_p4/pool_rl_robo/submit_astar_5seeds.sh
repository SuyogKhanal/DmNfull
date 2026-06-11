#!/bin/bash
# submit_astar_5seeds.sh — A* study: P4-LLM-select vs Diff-DAgger, 5 seeds.
#
# 5 ISOLATED sbatch jobs (one seed each). Each job runs BOTH methods
# (diff_dagger + p4_select) from ONE shared bootstrap (identical initial policy),
# nd_retrain=1, stopping on budget OR 100% (config_astar.yaml). Writes
# results/PushT-v1/run_<seed>/{diff_dagger,p4_select}/. 3 GPUs (p4_select needs
# the LLM selector) pinned to H100. bash launcher (calls sbatch 5x).
#
#   bash submit_astar_5seeds.sh
#   SEEDS="1 2 3 4 5" bash submit_astar_5seeds.sh   # override the seed list
#
set -e
SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SUITE_DIR}"
ENV="${ENV:-PushT-v1}"
SEEDS="${SEEDS:-1 2 3 4 5}"
TIME="${TIME:-10-00:00:00}"
echo "[astar] P4-LLM-select vs Diff-DAgger | seeds=[${SEEDS}] | shared bootstrap, nd=1, stop@budget-or-100%"
for S in ${SEEDS}; do
    echo "[astar] seed ${S} → run_${S}  (methods=diff_dagger+p4_select)"
    ENV="${ENV}" METHODS="diff_dagger+p4_select" SEED="${S}" RUN_ID="${S}" \
        CONFIG="${SUITE_DIR}/config_astar.yaml" \
        sbatch --array=1 --gpus-per-node=3 --constraint=gpu-h100 --time="${TIME}" \
               --job-name="prr_astar_s${S}" run_pool_rl_robo.sh
done
echo "[astar] queued ${SEEDS}. results → results/${ENV}/run_<seed>/  (aggregate run_1..run_5)"
