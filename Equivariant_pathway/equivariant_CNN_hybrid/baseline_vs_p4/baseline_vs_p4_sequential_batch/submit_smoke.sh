#!/bin/bash
# Submit a 2-job smoke test: run_id 99 and 100, tiny budget, all three
# methods. Exercises the full path end-to-end:
#   - layout setup + per-round correction pool rotation (3 rounds)
#   - rollout + ranking
#   - demo collection (baseline DAgger + P4 sequential + P4 batch)
#   - replay-buffer fine-tune after every demo addition for every method
#   - heldout eval after every round
#
# On the OpenAI cluster:
#   bash submit_smoke.sh
# On the Qwen cluster:
#   SUBMIT_SCRIPT=submit_one_qwen.sh bash submit_smoke.sh
#
# Per-job wallclock is roughly 20-30 min on a single GPU; queue both and
# the production sweep doesn't have to wait long to confirm the wiring
# works.

set -eo pipefail

# Resolve this suite's directory. This script is a launcher (it queues
# sbatch jobs); the intended invocation is ``bash submit_smoke.sh`` from
# the login node. If someone runs ``sbatch submit_smoke.sh`` instead,
# SLURM copies the script into a spool dir before executing it, so
# ${BASH_SOURCE[0]} would resolve there. Prefer $SLURM_SUBMIT_DIR (the
# dir sbatch was invoked from) when present.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${SCRIPT_DIR}"

SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-submit_one.sh}"
CONFIG="${CONFIG:-${SCRIPT_DIR}/config.yaml}"

# Smoke overrides — tiny budget so every method exhausts in a few rounds
# but each method still trains the policy after its demos land.
BUDGET="${BUDGET:-2}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"
CORRECTION_N="${CORRECTION_N:-10}"
BASELINE_ROUND_EPOCHS="${BASELINE_ROUND_EPOCHS:-3}"
ROUND_EPOCHS="${ROUND_EPOCHS:-3}"
INITIAL_EPOCHS="${INITIAL_EPOCHS:-100}"   # only used if upstream bootstrap cache is missing
METHODS="${METHODS:-baseline,p4_sequential,p4_batch}"

mkdir -p slurm_logs logs

EXPORT_BASE="ALL,CONFIG=${CONFIG}"
# sbatch --export uses comma as the variable separator, so a multi-method
# METHODS value (e.g. "p4_sequential,p4_batch") gets truncated to its
# first entry. Encode commas as '+' here; the receiving submit_one*.sh
# decodes them back before passing --methods.
EXPORT_BASE="${EXPORT_BASE},METHODS=${METHODS//,/+}"
EXPORT_BASE="${EXPORT_BASE},BUDGET=${BUDGET}"
EXPORT_BASE="${EXPORT_BASE},MAX_ROUNDS=${MAX_ROUNDS}"
EXPORT_BASE="${EXPORT_BASE},CORRECTION_N=${CORRECTION_N}"
EXPORT_BASE="${EXPORT_BASE},BASELINE_ROUND_EPOCHS=${BASELINE_ROUND_EPOCHS}"
EXPORT_BASE="${EXPORT_BASE},ROUND_EPOCHS=${ROUND_EPOCHS}"
EXPORT_BASE="${EXPORT_BASE},INITIAL_EPOCHS=${INITIAL_EPOCHS}"

SMOKE_RUN_IDS="${SMOKE_RUN_IDS:-99 100}"

echo "[submit_smoke] submit_script=${SUBMIT_SCRIPT}"
echo "[submit_smoke] run_ids=${SMOKE_RUN_IDS}  methods=${METHODS}"
echo "[submit_smoke] budget=${BUDGET}  max_rounds=${MAX_ROUNDS}  correction_n=${CORRECTION_N}"
echo "[submit_smoke] baseline_epochs=${BASELINE_ROUND_EPOCHS}  p4_epochs=${ROUND_EPOCHS}"

JOB_IDS=()
for i in ${SMOKE_RUN_IDS}; do
    JOB_ID=$(sbatch --parsable \
        --job-name="bvp_smoke_${i}" \
        --export="${EXPORT_BASE},RUN_ID=${i}" \
        "${SCRIPT_DIR}/${SUBMIT_SCRIPT}")
    JOB_IDS+=("${JOB_ID}")
    echo "[submit_smoke] run_id=${i}  sbatch_job_id=${JOB_ID}"
done

echo "[submit_smoke] submitted ${#JOB_IDS[@]} smoke jobs: ${JOB_IDS[*]}"
echo ""
echo "After both jobs finish, audit with:"
echo "  for m in baseline p4_sequential p4_batch; do"
echo "    awk -F, 'NR==1{next} END{print \"\$m  training_rounds_total=\"\$3}' \\"
echo "      results/run_99/\$m/results/training_log.csv"
echo "  done"
echo ""
echo "  ls results/run_99/shared/round_*/correction_layouts.yaml"
echo "  python3 -m Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.baseline_vs_p4_sequential_batch.aggregation.aggregate --budget ${BUDGET} --target_sr 0.90"
echo "  cat results/aggregate/contamination_report.json | python3 -m json.tool"
