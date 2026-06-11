#!/bin/bash
# submit_p4select.sh — launch p4_select (P4-LLM selection on top of SafeDAgger).
# Needs the LLM (the selector) → 3 GPUs (text+vision+orchestrator) + vLLM. Pinned
# to H100 (render/eval-safe). Writes to results/PushT-v1/run_901/ (separate from
# research run_0 and industry run_900). Compared against the completed Diff-DAgger
# curve (run_0). bash launcher (calls sbatch).
#   bash submit_p4select.sh
set -e
SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SUITE_DIR}"
ENV="${ENV:-PushT-v1}"
RUN_ID="${RUN_ID:-901}"
SEED="${SEED:-42}"
TIME="${TIME:-10-00:00:00}"
echo "[p4select] P4-LLM-select on SafeDAgger, ${ENV} run_${RUN_ID}, ${TIME}"
ENV="${ENV}" METHODS=p4_select SEED="${SEED}" RUN_ID="${RUN_ID}" \
    CONFIG="${SUITE_DIR}/config_p4select.yaml" \
    sbatch --array=1 --gpus-per-node=3 --constraint=gpu-h100 --time="${TIME}" \
           --job-name=prr_p4select run_pool_rl_robo.sh
echo "[p4select] queued. results → results/${ENV}/run_${RUN_ID}/p4_select/"
