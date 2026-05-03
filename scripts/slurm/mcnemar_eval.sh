#!/bin/bash
#SBATCH --job-name=dmn_mc_eval
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=s226137394@deakin.edu.au
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

# Per-profile per-episode policy evaluation, fanned out as a 3-task slurm
# array: task 0 -> p4, task 1 -> p5, task 2 -> p6. Each task gets its own
# GPU and snapshots only its profile's checkpoint, so the three runs are
# fully independent and execute in parallel when the cluster has capacity.
#
# Submission pattern (run AFTER the three training jobs finish):
#
#   EVAL_JID=$(sbatch --parsable \
#       --dependency=afterok:${JID4}:${JID5}:${JID6} \
#       scripts/slurm/mcnemar_eval.sh)
#   sbatch --dependency=afterok:${EVAL_JID} scripts/slurm/mcnemar_analysis.sh
#
# The analysis script needs ALL three per-profile success vectors, so it
# can't run inside the array. Chain it via afterok:<array_job_id>; slurm
# treats the array job ID as "all tasks succeeded" once every task does.
#
# Snapshot layout (concurrency-safe — each array task owns its subdirectory):
#   checkpoints/snapshots/<array_job_id>/p4/best_model_ema.pth (+ meta + pth)
#                                       /p5/...
#                                       /p6/...
#
# Output layout:
#   results/mcnemar/run_<array_job_id>/p4_.../per_episode_success_policy.json
#                                     /p5_.../...
#                                     /p6_.../...
#
# Env vars (all optional):
#   N_EPISODES   default 500
#   SEED_BASE    default 0
#   NOTIFY_TO    forwarded to scripts/notify.py for the END mail (FAIL also)
#   PROJECT_ROOT default /vast/$USER/DmN/DmNfull

set -eo pipefail

module purge
module load Anaconda3
source /home/s226137394/.bashrc
source activate
conda activate maze

PROJECT_ROOT="${PROJECT_ROOT:-/vast/s226137394/DmN/DmNfull}"
cd "$PROJECT_ROOT"

mkdir -p slurm_logs

N_EPISODES="${N_EPISODES:-500}"
SEED_BASE="${SEED_BASE:-0}"
NOTIFY_TO="${NOTIFY_TO:-}"

# Map array task -> profile short key. SLURM_ARRAY_TASK_ID is set by slurm
# when the script runs as part of an array. For local debugging with a
# single profile, run e.g. SLURM_ARRAY_TASK_ID=0 bash scripts/slurm/mcnemar_eval.sh.
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
case "$TASK_ID" in
  0) SHORT=p4 ;;
  1) SHORT=p5 ;;
  2) SHORT=p6 ;;
  *) echo "[mcnemar_eval] ERROR: unexpected SLURM_ARRAY_TASK_ID=$TASK_ID (expect 0..2)"; exit 2 ;;
esac

# All array tasks share the parent ARRAY_JOB_ID for run-dir / snapshot-dir
# naming, so the analysis follow-up can find everything in one place.
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local_$(date +%s)}}"
SNAPSHOT_ROOT="checkpoints/snapshots/${ARRAY_JOB_ID}"
EVAL_OUT_ROOT="results/mcnemar/run_${ARRAY_JOB_ID}"

echo "[mcnemar_eval] === array task ${TASK_ID} (${SHORT}) ==="
echo "[mcnemar_eval] host         : $(hostname)"
echo "[mcnemar_eval] cwd          : $(pwd)"
echo "[mcnemar_eval] ARRAY_JOB_ID : $ARRAY_JOB_ID"
echo "[mcnemar_eval] N_EPISODES   : $N_EPISODES"
echo "[mcnemar_eval] SEED_BASE    : $SEED_BASE"
echo "[mcnemar_eval] SNAPSHOT_ROOT: $SNAPSHOT_ROOT"
echo "[mcnemar_eval] EVAL_OUT_ROOT: $EVAL_OUT_ROOT"

# ---------------------------------------------------------------------------
# Snapshot only this profile's checkpoint + meta sidecar (concurrency-safe;
# each array task owns its profile subdirectory).
#
# pipeline/rollout._load_meta() looks for best_model_meta.json next to the
# .pth, so the meta sidecar MUST be co-located with the snapshotted weights.
# ---------------------------------------------------------------------------
src_dir="checkpoints/${SHORT}"
dst_dir="${SNAPSHOT_ROOT}/${SHORT}"
if [[ ! -f "${src_dir}/best_model_ema.pth" ]]; then
    echo "[mcnemar_eval] ERROR: missing ${src_dir}/best_model_ema.pth — did the ${SHORT} training job finish?"
    exit 2
fi
mkdir -p "$dst_dir"
cp "${src_dir}/best_model_ema.pth" "${dst_dir}/best_model_ema.pth"
[[ -f "${src_dir}/best_model.pth"      ]] && cp "${src_dir}/best_model.pth"      "${dst_dir}/best_model.pth"
[[ -f "${src_dir}/best_model_meta.json" ]] && cp "${src_dir}/best_model_meta.json" "${dst_dir}/best_model_meta.json"
echo "[mcnemar_eval] snapshot ${src_dir} -> ${dst_dir}"

# ---------------------------------------------------------------------------
# Per-profile per-episode evaluation. --profile <short> tells run_mcnemar_eval
# to skip the cross-profile manifest (which would race with sibling tasks)
# and write only under <out_root>/<profile>/.
# ---------------------------------------------------------------------------
echo "[mcnemar_eval] running scripts/run_mcnemar_eval.py --profile ${SHORT} (${N_EPISODES} episodes)"
python -u scripts/run_mcnemar_eval.py \
    --profile "$SHORT" \
    "--${SHORT}-ckpt" "${dst_dir}/best_model_ema.pth" \
    --n-episodes "$N_EPISODES" \
    --seed-base "$SEED_BASE" \
    --out-root "$EVAL_OUT_ROOT"

RC=$?
echo "[mcnemar_eval] task ${TASK_ID} (${SHORT}) DONE rc=$RC"
echo "[mcnemar_eval] result file : ${EVAL_OUT_ROOT}/$(ls "$EVAL_OUT_ROOT" | grep "^${SHORT}_" | head -n1)/per_episode_success_policy.json"

if [[ -n "$NOTIFY_TO" ]]; then
    OUT_FILE="slurm_logs/${SLURM_JOB_NAME:-dmn_mc_eval}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${TASK_ID}.out"
    if [[ -f "$OUT_FILE" ]]; then
        python -u scripts/notify.py \
            --to "$NOTIFY_TO" \
            --subject "[DmN] mc_eval task=${TASK_ID} profile=${SHORT} rc=$RC" \
            --body-file "$OUT_FILE" --tail 200 || true
    fi
fi

exit $RC
