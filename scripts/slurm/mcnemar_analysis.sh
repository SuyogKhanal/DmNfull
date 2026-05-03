#!/bin/bash
#SBATCH --job-name=dmn_mc_analysis
#SBATCH --partition=cpu
#SBATCH --qos=batch-short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:30:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=s226137394@deakin.edu.au
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# CPU-only follow-on to the mcnemar_eval array job. Runs the McNemar test
# over the three per-profile per_episode_success_policy.json files and the
# prescription-quality paired t-test across active-loop rounds.
#
# Submission pattern (queue this AFTER the eval array job):
#
#   EVAL_JID=$(sbatch --parsable \
#       --dependency=afterok:${JID4}:${JID5}:${JID6} \
#       scripts/slurm/mcnemar_eval.sh)
#   sbatch --dependency=afterok:${EVAL_JID} scripts/slurm/mcnemar_analysis.sh
#
# afterok:<array_job_id> is satisfied only when ALL array tasks exit 0, so
# we never run analysis on a partial set of profiles.
#
# Discovery: with no env override, this script reads the latest run dir
# under results/mcnemar/. Pass MCNEMAR_RUN_DIR=results/mcnemar/run_<id> to
# pin a specific run.
#
# Env vars (all optional):
#   MCNEMAR_RUN_DIR   default: latest results/mcnemar/run_*/
#   P4_LOOP_LOG / P5_LOOP_LOG / P6_LOOP_LOG
#                     default: latest loop_*/ per profile under results/active_loop/
#   NOTIFY_TO         forwarded to scripts/notify.py for END mail
#   PROJECT_ROOT      default /vast/$USER/DmN/DmNfull

set -eo pipefail

module purge
module load Anaconda3
source /home/s226137394/.bashrc
source activate
conda activate maze

PROJECT_ROOT="${PROJECT_ROOT:-/vast/s226137394/DmN/DmNfull}"
cd "$PROJECT_ROOT"

mkdir -p slurm_logs

NOTIFY_TO="${NOTIFY_TO:-}"

P4_PROFILE="p4_vlm_reasoning_kag_cross_plain_llm"
P5_PROFILE="p5_vlm_reasoning_kag_rag_cross_plain_llm"
P6_PROFILE="p6_vlm_reasoning_kag_rag_tkf_cross_plain_llm"

# ---------------------------------------------------------------------------
# Locate the eval results dir written by the array job. Default: latest run.
# ---------------------------------------------------------------------------
if [[ -z "${MCNEMAR_RUN_DIR:-}" ]]; then
    MCNEMAR_RUN_DIR=$(ls -td results/mcnemar/run_* 2>/dev/null | head -n1 || true)
fi
if [[ -z "$MCNEMAR_RUN_DIR" || ! -d "$MCNEMAR_RUN_DIR" ]]; then
    echo "[mcnemar_analysis] ERROR: no eval results directory found under results/mcnemar/."
    echo "                   Set MCNEMAR_RUN_DIR=<path> explicitly, or run mcnemar_eval.sh first."
    exit 2
fi
echo "[mcnemar_analysis] eval results dir: $MCNEMAR_RUN_DIR"

# Sanity-check that all three per-profile success files are present.
for short in p4 p5 p6; do
    case "$short" in
      p4) profile=$P4_PROFILE ;;
      p5) profile=$P5_PROFILE ;;
      p6) profile=$P6_PROFILE ;;
    esac
    fname="$MCNEMAR_RUN_DIR/$profile/per_episode_success_policy.json"
    if [[ ! -f "$fname" ]]; then
        echo "[mcnemar_analysis] ERROR: missing $fname"
        exit 2
    fi
done

# ---------------------------------------------------------------------------
# 1. McNemar paired test over the three success vectors.
# ---------------------------------------------------------------------------
echo "[mcnemar_analysis] running scripts/mcnemar_analysis.py"
python -u scripts/mcnemar_analysis.py --results-dir "$MCNEMAR_RUN_DIR"

# ---------------------------------------------------------------------------
# 2. Prescription-quality paired t-test across active-loop rounds.
#    Auto-detect the latest loop_*/ per profile if env vars aren't set.
# ---------------------------------------------------------------------------
_latest_loop_log() {
    local profile_dir="results/active_loop/$1"
    [[ -d "$profile_dir" ]] || return 1
    local latest=$(ls -td "$profile_dir"/loop_* 2>/dev/null | head -n1)
    [[ -n "$latest" && -f "${latest}/loop_log.json" ]] || return 1
    echo "${latest}/loop_log.json"
}

P4_LOOP_LOG="${P4_LOOP_LOG:-$(_latest_loop_log "$P4_PROFILE" || true)}"
P5_LOOP_LOG="${P5_LOOP_LOG:-$(_latest_loop_log "$P5_PROFILE" || true)}"
P6_LOOP_LOG="${P6_LOOP_LOG:-$(_latest_loop_log "$P6_PROFILE" || true)}"

if [[ -z "$P4_LOOP_LOG" || -z "$P5_LOOP_LOG" || -z "$P6_LOOP_LOG" ]]; then
    echo "[mcnemar_analysis] WARNING: could not auto-detect loop_log.json for all three profiles."
    echo "                   p4=${P4_LOOP_LOG:-<missing>}"
    echo "                   p5=${P5_LOOP_LOG:-<missing>}"
    echo "                   p6=${P6_LOOP_LOG:-<missing>}"
    echo "[mcnemar_analysis] skipping prescription-quality analysis."
else
    echo "[mcnemar_analysis] running scripts/prescription_quality_analysis.py"
    echo "                   p4 loop log: $P4_LOOP_LOG"
    echo "                   p5 loop log: $P5_LOOP_LOG"
    echo "                   p6 loop log: $P6_LOOP_LOG"
    python -u scripts/prescription_quality_analysis.py \
        --p4-loop-log "$P4_LOOP_LOG" \
        --p5-loop-log "$P5_LOOP_LOG" \
        --p6-loop-log "$P6_LOOP_LOG" \
        --out "${MCNEMAR_RUN_DIR}/prescription_quality_results.json"
fi

RC=$?
echo "[mcnemar_analysis] DONE rc=$RC"
echo "[mcnemar_analysis] outputs under: $MCNEMAR_RUN_DIR"

if [[ -n "$NOTIFY_TO" ]]; then
    OUT_FILE="slurm_logs/${SLURM_JOB_NAME:-dmn_mc_analysis}_${SLURM_JOB_ID:-local}.out"
    if [[ -f "$OUT_FILE" ]]; then
        python -u scripts/notify.py \
            --to "$NOTIFY_TO" \
            --subject "[DmN] SLURM_END job=${SLURM_JOB_ID:-?} mc_analysis rc=$RC" \
            --body-file "$OUT_FILE" --tail 400 || true
    fi
fi

exit $RC
