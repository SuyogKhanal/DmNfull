#!/bin/bash
# submit_stack_hybrid.sh — launch the two FULL StackCube V3-hybrid jobs (seeds 1,2),
# each reusing its Diff-DAgger baseline's BYTE-IDENTICAL bootstrap via P4_REUSE_INIT_CKPT.
#
# Order (already done): the diff_dagger baselines (run_1/run_2) build
#   results/StackCube-v1/run_<seed>/shared_baselines/init_ckpt.pth (+ init_meta.json with
#   the frozen init_sr). This script points the hybrid at that exact file so both arms
#   start from the SAME init policy (fairness invariant). The hybrid (p4_top3 +
#   p4.subtask.collect=hybrid) lands in run_<seed>/p4_top3, alongside run_<seed>/diff_dagger.
#
# Run ONLY after the hybrid smoke passes. Usage:
#   ./submit_stack_hybrid.sh           # both seeds
#   ./submit_stack_hybrid.sh 1         # one seed
set -euo pipefail
SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="${SUITE_DIR}/config_stack_hybrid.yaml"
SEEDS=("${@:-1 2}"); SEEDS=(${SEEDS[@]})

for S in "${SEEDS[@]}"; do
  CK="${SUITE_DIR}/results/StackCube-v1/run_${S}/shared_baselines/init_ckpt.pth"
  if [ ! -f "${CK}" ]; then
    echo "ERROR seed ${S}: baseline bootstrap missing: ${CK}" >&2
    echo "  (the diff_dagger baseline for run_${S} must have built its bootstrap first)" >&2
    exit 2
  fi
  echo "[submit] seed ${S}: hybrid reuses ${CK}"
  # Full P4 pipeline incl. VLM ⇒ 3 GPUs (VLM+LLM+orch), matching PushT. h100 ONLY:
  # the VLM renders failures and the render test flagged "h200 hits Vulkan
  # device-lost" (per user: keep these rendering jobs on h100). render fix
  # (headless eval) validated on h100.
  ENV=StackCube-v1 METHODS=p4_top3 SEED="${S}" RUN_ID="${S}" \
    CONFIG="${CFG}" P4_REUSE_INIT_CKPT="${CK}" \
    sbatch --job-name="sc_hyb_s${S}" --array=1 --nodes=1 --gpus-per-node=3 \
           --partition=gpu-large --constraint="gpu-h100" \
           --qos=batch-long --time=7-00:00:00 --export=ALL \
           "${SUITE_DIR}/run_pool_rl_robo.sh"
done
echo "[submit] done. monitor: squeue -u \$USER ; results in results/StackCube-v1/run_<seed>/p4_top3/"
