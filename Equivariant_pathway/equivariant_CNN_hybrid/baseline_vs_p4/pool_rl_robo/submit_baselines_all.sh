#!/bin/bash
# Baselines-only launcher (bash, NOT sbatch). Submits the 5-env array of the 5
# IIL baselines with NO vLLM. AGGREGATE_AFTER=1 chains aggregation.
#   bash submit_baselines_all.sh
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p slurm_logs results logs

JID=$(sbatch --parsable submit_baseline_one.sh)
echo "submitted array job ${JID} (pool_rl_robo_baselines, 5 envs, no vLLM)"

if [ "${AGGREGATE_AFTER:-0}" = "1" ]; then
    AID=$(sbatch --parsable --dependency=afterany:"${JID}" submit_aggregate.sh)
    echo "chained aggregate job ${AID} (after ${JID})"
fi
