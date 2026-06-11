#!/bin/bash
# Minimal smoke launcher (bash, NOT sbatch — it calls sbatch internally).
# PushT-v1 only, all 7 methods, EXACTLY 1 round each, LLM live (P4 must prescribe
# a non-empty config that loads + is expert-solved). Gate before the real run.
#
#   bash submit_smoke.sh
#
set -e
SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SUITE_DIR}"
echo "[submit_smoke] launching PushT-v1 1-round 7-method smoke (SMOKE=1, array=1)"
SMOKE=1 sbatch --array=1 --time="${TIME:-04:00:00}" \
    --job-name="${JOB_NAME:-prr_smoke}" run_pool_rl_robo.sh "$@"
echo "[submit_smoke] queued. watch: squeue --me ; tail -f slurm_logs/pool_rl_*_1.out"
