#!/bin/bash
# submit_pusht_img_smoke.sh — FAST gate for the IMAGE-based PushT pipeline.
#
# Runs BOTH arms (diff_dagger + p4_subtask) with tiny knobs (config_pusht_image_smoke.yaml)
# + --smoke, on 3 GPUs (p4_subtask needs the VLM). Validates end-to-end:
#   image bootstrap collect+train (rgb obs), held-out image eval, on-policy +
#   bridge demo collection carrying rgb, VISUAL-embedding clustering (telemetry
#   "visual_embed": ok), VLM render+read of PushT failures (no Vulkan hang).
# No bootstrap reuse (builds its own tiny one). h100|h200 per request.
# Usage:  ./submit_pusht_img_smoke.sh
set -euo pipefail
SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="${SUITE_DIR}/config_pusht_image_smoke.yaml"
HCFG="${SUITE_DIR}/configs/hydra/pusht_image.yaml"
RID="${1:-980}"   # fresh run-id (image smoke; distinct from the prior state run_990)
echo "[submit] PushT IMAGE smoke (3 GPU, h100|h200, both arms, --smoke) RUN_ID=${RID}"
ENV=PushT-v1 METHODS=diff_dagger,p4_subtask SEED="${RID}" RUN_ID="${RID}" SMOKE=1 \
  CONFIG="${CFG}" SUITE_HYDRA_CFG="${HCFG}" \
  sbatch --job-name="pti_smoke" --array=1 --nodes=1 --gpus-per-node=3 \
         --partition=gpu-large --constraint="gpu-h100|gpu-h200" \
         --qos=batch-long --time=06:00:00 --export=ALL \
         "${SUITE_DIR}/run_pool_rl_robo.sh"
echo "[submit] smoke submitted (RUN_ID=${RID}). Watch: tail -f slurm_logs/pool_rl_*_1.out"
echo "[submit] PASS gate: image bootstrap trains, held-out eval runs, 'visual_embed':'ok'"
echo "         in run_${RID}/p4_subtask/results/telemetry/round_*.jsonl, demo collected, no Vulkan hang."
