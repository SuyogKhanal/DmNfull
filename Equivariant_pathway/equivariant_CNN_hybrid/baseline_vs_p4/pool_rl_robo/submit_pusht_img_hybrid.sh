#!/bin/bash
# submit_pusht_img_hybrid.sh — P4-LLM V3-hybrid IMAGE-based PushT (seeds 1,2).
# Run AFTER the baseline builds each seed's bootstrap AND the smoke passes.
#
# FULL P4 pipeline (VLM analyses the rendered failure frames) ⇒ 3 GPUs
# (GPU0 VLM, GPU1 LLM, GPU2 orch). p4_subtask clusters failures in the LATENT
# VISUAL space (cluster_features: visual → R3M over key-frames). Each seed reuses
# its diff_dagger baseline's BYTE-IDENTICAL image bootstrap via P4_REUSE_INIT_CKPT
# (fairness invariant). SUITE_HYDRA_CFG points the policy at the image config.
#
# NODES: P4LLM is H100-ONLY. CONFIRMED 2026-06-25 (job 102948): on h200l the p4 arm
# HANGS during rgb env construction (it builds an EXTRA reposition env with render;
# h200l can't handle the multiple Vulkan render contexts) — never reaches bootstrap,
# times out. The diff_dagger BASELINE has no reposition env, so it runs fine on h200l.
# This matches the PlugCharger precedent (submit_plug_hybrid.sh = gpu-h100).
#
# Usage:  ./submit_pusht_img_hybrid.sh        # seeds 1 2
#         ./submit_pusht_img_hybrid.sh 1      # one seed
set -euo pipefail
SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="${SUITE_DIR}/config_pusht_image.yaml"
HCFG="${SUITE_DIR}/configs/hydra/pusht_image.yaml"
SEEDS=("${@:-1 2}"); SEEDS=(${SEEDS[@]})

# Same run-id namespace as the finetune baseline: seed S → run_11S. Reuses the
# baseline's image bootstrap byte-identical.
for S in "${SEEDS[@]}"; do
  RID=$((110 + S))
  CK="${SUITE_DIR}/results/PushT-v1/run_${RID}/shared_baselines/init_ckpt.pth"
  if [ ! -f "${CK}" ]; then
    echo "ERROR seed ${S}: image baseline bootstrap missing: ${CK}" >&2
    echo "  (run submit_pusht_img_baseline.sh first and let run_${RID} build it)" >&2
    exit 2
  fi
  echo "[submit] seed ${S} → run_${RID}: P4 IMAGE V3-hybrid (3 GPU, h100-only) reuses ${CK}"
  ENV=PushT-v1 METHODS=p4_subtask SEED="${S}" RUN_ID="${RID}" \
    CONFIG="${CFG}" P4_REUSE_INIT_CKPT="${CK}" SUITE_HYDRA_CFG="${HCFG}" \
    sbatch --job-name="pti_hyb_s${S}" --array=1 --nodes=1 --gpus-per-node=3 \
           --partition=gpu-large --constraint="gpu-h100" \
           --qos=batch-long --time=7-00:00:00 --export=ALL \
           "${SUITE_DIR}/run_pool_rl_robo.sh"
done
echo "[submit] image hybrid jobs submitted. Results: results/PushT-v1/run_<seed>/p4_subtask/"
