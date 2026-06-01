#!/bin/bash
# Multi-seed production launcher (bash, NOT sbatch). Submits the 25-task
# env x seed sweep (5 envs x 5 seeds 42..46) and chains aggregation
# (cross-seed mean +/- std) after it finishes.
#   bash submit_seeds_all.sh
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p slurm_logs results logs

JID=$(sbatch --parsable submit_seed_sweep.sh)
echo "submitted sweep array ${JID} (pool_rl_robo_sweep, 5 envs x 5 seeds = 25 tasks)"
AID=$(sbatch --parsable --dependency=afterany:"${JID}" submit_aggregate.sh)
echo "chained aggregate ${AID} (after ${JID})"
