#!/bin/bash
# Production launcher (bash, NOT sbatch — calls sbatch internally). Submits the
# 5-env array (P4-LLM + 5 IIL baselines per env, each with its own text Qwen3-32B
# vLLM). AGGREGATE_AFTER=1 chains aggregation after the array finishes.
#   bash submit_all.sh
#   AGGREGATE_AFTER=1 bash submit_all.sh
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p slurm_logs results logs

JID=$(sbatch --parsable submit_one_qwen.sh)
echo "submitted array job ${JID} (pool_rl_robo_qwen, 5 envs)"

if [ "${AGGREGATE_AFTER:-0}" = "1" ]; then
    AID=$(sbatch --parsable --dependency=afterany:"${JID}" submit_aggregate.sh)
    echo "chained aggregate job ${AID} (after ${JID})"
fi
