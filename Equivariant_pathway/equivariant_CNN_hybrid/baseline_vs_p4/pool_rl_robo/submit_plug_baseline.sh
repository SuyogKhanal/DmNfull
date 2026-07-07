#!/bin/bash
# submit_plug_baseline.sh — Diff-DAgger PlugCharger baseline (seeds 1,2).
#
# 1 GPU, NO LLM (NEED_LLM=0 for diff_dagger, so run_pool_rl_robo.sh skips the
# 3-GPU/2-GPU vLLM check). Builds, per seed:
#   results/PlugCharger-v1/run_<seed>/shared_baselines/init_ckpt.pth (+ init_meta.json)
#     — the BYTE-IDENTICAL warm-start bootstrap the hybrid later reuses, and
#   results/PlugCharger-v1/run_<seed>/diff_dagger/results/learning_curve.json
#     — the baseline SR-vs-demos curve.
#
# Run FIRST (before submit_plug_hybrid.sh). Render-free eval (headless), so a100 is
# fine and usually idle. Usage:
#   ./submit_plug_baseline.sh         # seeds 1 2
#   ./submit_plug_baseline.sh 1       # one seed
set -euo pipefail
SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="${SUITE_DIR}/config_plug_hybrid.yaml"
SEEDS=("${@:-1 2}"); SEEDS=(${SEEDS[@]})

for S in "${SEEDS[@]}"; do
  echo "[submit] PlugCharger diff_dagger baseline seed ${S} (1 GPU, no LLM)"
  ENV=PlugCharger-v1 METHODS=diff_dagger SEED="${S}" RUN_ID="${S}" CONFIG="${CFG}" \
    sbatch --job-name="pc_dd_s${S}" --array=1 --nodes=1 --gpus-per-node=1 \
           --partition=gpu,gpu-large --constraint="gpu-a100|gpu-h100|gpu-h200" \
           --qos=batch-long --time=7-00:00:00 --export=ALL \
           "${SUITE_DIR}/run_pool_rl_robo.sh"
done
echo "[submit] baselines submitted. Monitor: squeue -u \$USER"
echo "[submit] When done: results/PlugCharger-v1/run_<seed>/{diff_dagger,shared_baselines}/"
