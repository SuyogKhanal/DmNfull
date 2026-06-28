#!/bin/bash
# submit_plug_hybrid.sh — PlugCharger V3-hybrid (p4_top3 + p4.subtask.collect=hybrid),
# seeds 1,2. Run AFTER the baseline builds the bootstrap AND the hybrid smoke passes.
#
# TEXT-ONLY (config use_vlm:false) ⇒ SKIP_VLM=1 ⇒ 2 GPUs (GPU0 LLM, GPU1 orchestrator),
# NO VLM rendering ⇒ a100/h100/h200 all OK (no Vulkan device-lost risk). Each seed
# reuses its diff_dagger baseline's BYTE-IDENTICAL bootstrap via P4_REUSE_INIT_CKPT so
# both arms start from the SAME init policy (fairness invariant). Lands in
# results/PlugCharger-v1/run_<seed>/p4_top3, alongside .../diff_dagger.
#
# Usage:  ./submit_plug_hybrid.sh        # seeds 1 2
#         ./submit_plug_hybrid.sh 1      # one seed
set -euo pipefail
SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="${SUITE_DIR}/config_plug_hybrid.yaml"
SEEDS=("${@:-1 2}"); SEEDS=(${SEEDS[@]})

for S in "${SEEDS[@]}"; do
  CK="${SUITE_DIR}/results/PlugCharger-v1/run_${S}/shared_baselines/init_ckpt.pth"
  if [ ! -f "${CK}" ]; then
    echo "ERROR seed ${S}: baseline bootstrap missing: ${CK}" >&2
    echo "  (run submit_plug_baseline.sh first and let run_${S} build its bootstrap)" >&2
    exit 2
  fi
  echo "[submit] seed ${S}: FULL P4 VLM hybrid (3 GPU, h100-only) reuses ${CK}"
  # Full P4 pipeline (config use_vlm:true) ⇒ 3 GPUs (GPU0 VLM, GPU1 LLM, GPU2 orch). The VLM
  # RENDERS failure frames. h100-ONLY: the render path works on h100 (StackCube rendered 408/423
  # frames there) but HANGS on h200l (the VLM smoke 102094 deadlocked at env.render() on h200l-m-01).
  ENV=PlugCharger-v1 METHODS=p4_top3 SEED="${S}" RUN_ID="${S}" \
    CONFIG="${CFG}" P4_REUSE_INIT_CKPT="${CK}" \
    sbatch --job-name="pc_hyb_s${S}" --array=1 --nodes=1 --gpus-per-node=3 \
           --partition=gpu-large --constraint="gpu-h100" \
           --qos=batch-long --time=7-00:00:00 --export=ALL \
           "${SUITE_DIR}/run_pool_rl_robo.sh"
done
echo "[submit] hybrid jobs submitted. Results: results/PlugCharger-v1/run_<seed>/p4_top3/"
