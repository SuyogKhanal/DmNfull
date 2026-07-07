#!/bin/bash
# submit_pusht_img_baseline.sh — Diff-DAgger IMAGE-based PushT baseline (seeds 1,2).
#
# 1 GPU, NO LLM. Per seed builds:
#   results/PushT-v1/run_<seed>/shared_baselines/init_ckpt.pth (+ init_meta.json)
#     — the BYTE-IDENTICAL image-policy bootstrap the hybrid later reuses, and
#   results/PushT-v1/run_<seed>/diff_dagger/results/learning_curve.json
#     — the baseline SR-vs-demos curve (image obs).
#
# Diff-DAgger renders nothing for a VLM, so it runs on ANY node (a100/h100/h200).
# SUITE_HYDRA_CFG points BOTH arms at the image Hydra config (obs_mode=rgb, R3M).
# Run FIRST (before submit_pusht_img_hybrid.sh). Usage:
#   ./submit_pusht_img_baseline.sh        # seeds 1 2
#   ./submit_pusht_img_baseline.sh 1      # one seed
set -euo pipefail
SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="${SUITE_DIR}/config_pusht_image.yaml"
HCFG="${SUITE_DIR}/configs/hydra/pusht_image.yaml"
SEEDS=("${@:-1 2}"); SEEDS=(${SEEDS[@]})

# FINETUNE image study uses run-id namespace run_11<seed> (frozen-R3M runs were
# run_10<seed>, now abandoned; state study is run_1..run_5/run_11/run_12). SEED
# stays <seed> for fairness. h100|h200 ONLY: finetuning R3M (gradients + rgb
# replay buffer) needs >40GB, so a100 is dropped to avoid a mid-run OOM. The
# diff_dagger baseline has no reposition env, so h200l is safe for it.
for S in "${SEEDS[@]}"; do
  RID=$((110 + S))
  echo "[submit] PushT IMAGE(finetune) diff_dagger baseline seed ${S} → run_${RID} (1 GPU, h100|h200, rgb)"
  ENV=PushT-v1 METHODS=diff_dagger SEED="${S}" RUN_ID="${RID}" CONFIG="${CFG}" \
    SUITE_HYDRA_CFG="${HCFG}" \
    sbatch --job-name="pti_dd_s${S}" --array=1 --nodes=1 --gpus-per-node=1 \
           --partition=gpu-large --constraint="gpu-h100|gpu-h200" \
           --qos=batch-long --time=7-00:00:00 --export=ALL \
           "${SUITE_DIR}/run_pool_rl_robo.sh"
done
echo "[submit] image(finetune) baselines submitted. Monitor: squeue -u \$USER"
echo "[submit] When run_11<seed>/shared_baselines/init_ckpt.pth exists → submit_pusht_img_hybrid.sh"
