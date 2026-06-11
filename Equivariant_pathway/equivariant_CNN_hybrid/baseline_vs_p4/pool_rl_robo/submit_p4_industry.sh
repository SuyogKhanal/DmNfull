#!/bin/bash
# submit_p4_industry.sh — INDUSTRY P4 launcher (bash, calls sbatch internally).
#
# Runs p4_top3 ONLY with config_industry.yaml (nd_retrain=1 → P4 retrains after
# EVERY prescribed demo + heavier per-demo training) to demonstrate that the
# LLM-prescription method reaches the 90% success target with FEW operator
# demonstrations. Writes to results/PushT-v1/run_900/ — a separate "industry"
# namespace, so it NEVER clobbers and is NEVER mistaken for the research run_0
# (the honest A*/comparative study). 3 GPUs (VLM+text+orchestrator) + vLLM.
#
#   bash submit_p4_industry.sh
#
set -e
SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SUITE_DIR}"
ENV="${ENV:-PushT-v1}"
RUN_ID="${RUN_ID:-900}"          # industry namespace (research run is run_0)
SEED="${SEED:-42}"
TIME="${TIME:-10-00:00:00}"
echo "[industry] P4-only, config_industry.yaml (nd_retrain=1), ${ENV} run_${RUN_ID}, ${TIME}"
ENV="${ENV}" METHODS=p4_top3 SEED="${SEED}" RUN_ID="${RUN_ID}" \
    CONFIG="${SUITE_DIR}/config_industry.yaml" \
    sbatch --array=1 --gpus-per-node=3 --time="${TIME}" \
           --job-name=prr_p4_industry run_pool_rl_robo.sh
echo "[industry] queued. watch: squeue --me ; results land in results/${ENV}/run_${RUN_ID}/"
