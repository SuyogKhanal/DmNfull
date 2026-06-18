#!/bin/bash
#SBATCH --job-name=sc_hyb_nollm
#SBATCH --partition=gpu-large
#SBATCH --qos=batch-short
#SBATCH --constraint=gpu-h100|gpu-h200
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/hyb_nollm_%j.out
#SBATCH --error=slurm_logs/hyb_nollm_%j.err
# Fast 1-GPU, NO-LLM validation of the StackCube V3-hybrid code path. With the LLM
# off, planner.decide(None) takes the SELECT branch → exercises the riskiest NEW
# code (cube-descriptor-from-env replay reading cubeA/cubeB.pose + is_grasping, and
# _correct_onpolicy_from dispatch + telemetry) on a single GPU that schedules fast,
# decoupled from the 3-GPU LLM smoke's Priority wait. BRIDGE + prescriber parsing
# are covered by the full LLM smoke (run_990).
set -eo pipefail
# In an sbatch job BASH_SOURCE points at the read-only spool copy; use the submit dir.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SUITE_DIR="${SLURM_SUBMIT_DIR}"
else
    SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
DMN_ROOT="$(cd "${SUITE_DIR}/../../../.." && pwd)"
PKG="Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo"
mkdir -p "${SUITE_DIR}/slurm_logs" "${SUITE_DIR}/logs"
cd "${DMN_ROOT}"
[ -f "${HOME}/.bashrc" ] && source "${HOME}/.bashrc"
if command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)"; conda activate diffdagger || true; fi
PYTHON_BIN="/home/s226137394/.conda/envs/diffdagger/bin/python"
echo "[nollm-smoke] $(date) gpu=${CUDA_VISIBLE_DEVICES:-?} node=$(hostname)"
"${PYTHON_BIN}" -u -m "${PKG}.orchestrator.run_one" \
    --env StackCube-v1 --methods p4_top3 --run_id 993 --seed 1 \
    --config "${SUITE_DIR}/config_stack_hybrid_smoke.yaml" --no-llm \
    2>&1 | tee "${SUITE_DIR}/logs/hyb_nollm_smoke_${SLURM_JOB_ID:-local}.log"
echo "[nollm-smoke] DONE $(date)"
