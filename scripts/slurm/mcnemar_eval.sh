#!/bin/bash
#SBATCH --job-name=dmn_mcnemar
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s226137394@deakin.edu.au
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Run the McNemar + prescription-quality comparison for P4 / P5 / P6 after
# their per-profile active loops have finished training. Designed to be
# chained off the three training jobs:
#
#   JID4=$(PROFILE=p4 ... sbatch --parsable scripts/slurm/round.sh)
#   JID5=$(PROFILE=p5 ... sbatch --parsable scripts/slurm/round.sh)
#   JID6=$(PROFILE=p6 ... sbatch --parsable scripts/slurm/round.sh)
#   sbatch --dependency=afterok:${JID4}:${JID5}:${JID6} \
#          scripts/slurm/mcnemar_eval.sh
#
# The job:
#   1. Snapshots each profile's final EMA checkpoint (and meta sidecar) into
#      checkpoints/snapshots/<job_id>/<profile>/ so the eval reads stable
#      reference weights — even if a later training round in the same
#      checkpoints/<profile>/ dir overwrites the originals.
#   2. Runs run_mcnemar_eval.py for N_EPISODES (default 500) on seeds
#      SEED_BASE..SEED_BASE+N-1, producing per_episode_success_policy.json
#      under results/mcnemar/run_<ts>/<profile>/.
#   3. Runs mcnemar_analysis.py on those vectors -> mcnemar_results.json.
#   4. Auto-detects the latest loop_<id>/ per profile under
#      results/active_loop/ and runs prescription_quality_analysis.py
#      across the rounds, producing prescription_quality_results.json.
#
# Env vars (all optional):
#   N_EPISODES   default 500
#   SEED_BASE    default 0
#   NOTIFY_TO    forwarded to scripts/notify.py for SLURM_START / SLURM_END
#   PROJECT_ROOT default /vast/$USER/DmN/DmNfull
#   P4_LOOP_LOG / P5_LOOP_LOG / P6_LOOP_LOG
#                explicit loop_log.json paths (overrides auto-detection)

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

P4_PROFILE="p4_vlm_reasoning_kag_cross_plain_llm"
P5_PROFILE="p5_vlm_reasoning_kag_rag_cross_plain_llm"
P6_PROFILE="p6_vlm_reasoning_kag_rag_tkf_cross_plain_llm"

JOB_TAG="${SLURM_JOB_ID:-local_$(date +%s)}"
SNAPSHOT_ROOT="checkpoints/snapshots/${JOB_TAG}"

echo "[mcnemar_eval] === Statistical comparison run ==="
echo "[mcnemar_eval] host         : $(hostname)"
echo "[mcnemar_eval] cwd          : $(pwd)"
echo "[mcnemar_eval] N_EPISODES   : $N_EPISODES"
echo "[mcnemar_eval] SEED_BASE    : $SEED_BASE"
echo "[mcnemar_eval] SNAPSHOT_ROOT: $SNAPSHOT_ROOT"
echo "[mcnemar_eval] NOTIFY_TO    : ${NOTIFY_TO:-<unset>}"

if [[ -n "$NOTIFY_TO" ]]; then
    python -u scripts/notify.py --event "SLURM_START" \
        --to "$NOTIFY_TO" \
        --payload-json "{\"job_id\":\"${SLURM_JOB_ID:-?}\",\"kind\":\"mcnemar_eval\",\"n_episodes\":\"$N_EPISODES\",\"seed_base\":\"$SEED_BASE\"}" || true
fi

# ---------------------------------------------------------------------------
# 1. Snapshot per-profile checkpoints + meta sidecars.
#    pipeline/rollout._load_meta() looks for best_model_meta.json next to the
#    checkpoint, so we MUST keep the trained meta sidecar co-located with the
#    snapshotted .pth — otherwise eval falls back to defaults and may build
#    the policy with the wrong horizons / hidden dims.
# ---------------------------------------------------------------------------
for short in p4 p5 p6; do
    src_dir="checkpoints/${short}"
    dst_dir="${SNAPSHOT_ROOT}/${short}"
    if [[ ! -f "${src_dir}/best_model_ema.pth" ]]; then
        echo "[mcnemar_eval] ERROR: missing ${src_dir}/best_model_ema.pth — did the ${short} training job finish?"
        exit 2
    fi
    mkdir -p "$dst_dir"
    cp "${src_dir}/best_model_ema.pth" "${dst_dir}/best_model_ema.pth"
    [[ -f "${src_dir}/best_model.pth"      ]] && cp "${src_dir}/best_model.pth"      "${dst_dir}/best_model.pth"
    [[ -f "${src_dir}/best_model_meta.json" ]] && cp "${src_dir}/best_model_meta.json" "${dst_dir}/best_model_meta.json"
    echo "[mcnemar_eval] snapshot ${src_dir} -> ${dst_dir}"
done

# ---------------------------------------------------------------------------
# 2. Per-episode policy evaluation (writes per_episode_success_policy.json
#    per profile under results/mcnemar/run_<ts>/).
# ---------------------------------------------------------------------------
EVAL_OUT_ROOT="results/mcnemar/run_${JOB_TAG}"

echo "[mcnemar_eval] running scripts/run_mcnemar_eval.py ($N_EPISODES episodes)"
python -u scripts/run_mcnemar_eval.py \
    --p4-ckpt "${SNAPSHOT_ROOT}/p4/best_model_ema.pth" \
    --p5-ckpt "${SNAPSHOT_ROOT}/p5/best_model_ema.pth" \
    --p6-ckpt "${SNAPSHOT_ROOT}/p6/best_model_ema.pth" \
    --n-episodes "$N_EPISODES" \
    --seed-base "$SEED_BASE" \
    --out-root "$EVAL_OUT_ROOT"

# ---------------------------------------------------------------------------
# 3. McNemar analysis on the success vectors.
# ---------------------------------------------------------------------------
echo "[mcnemar_eval] running scripts/mcnemar_analysis.py"
python -u scripts/mcnemar_analysis.py \
    --results-dir "$EVAL_OUT_ROOT"

# ---------------------------------------------------------------------------
# 4. Prescription-quality paired t-test across active-loop rounds.
#    Auto-detect the latest loop_*/ directory per profile if the env var
#    isn't set explicitly.
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
    echo "[mcnemar_eval] WARNING: could not auto-detect loop_log.json for all three profiles."
    echo "                p4=${P4_LOOP_LOG:-<missing>}"
    echo "                p5=${P5_LOOP_LOG:-<missing>}"
    echo "                p6=${P6_LOOP_LOG:-<missing>}"
    echo "[mcnemar_eval] skipping prescription-quality analysis."
else
    echo "[mcnemar_eval] running scripts/prescription_quality_analysis.py"
    echo "                p4 loop log: $P4_LOOP_LOG"
    echo "                p5 loop log: $P5_LOOP_LOG"
    echo "                p6 loop log: $P6_LOOP_LOG"
    python -u scripts/prescription_quality_analysis.py \
        --p4-loop-log "$P4_LOOP_LOG" \
        --p5-loop-log "$P5_LOOP_LOG" \
        --p6-loop-log "$P6_LOOP_LOG" \
        --out "${EVAL_OUT_ROOT}/prescription_quality_results.json"
fi

RC=$?
echo "[mcnemar_eval] DONE rc=$RC"
echo "[mcnemar_eval] outputs under: $EVAL_OUT_ROOT"

if [[ -n "$NOTIFY_TO" ]]; then
    OUT_FILE="slurm_logs/${SLURM_JOB_NAME:-dmn_mcnemar}_${SLURM_JOB_ID:-local}.out"
    if [[ -f "$OUT_FILE" ]]; then
        python -u scripts/notify.py \
            --to "$NOTIFY_TO" \
            --subject "[DmN] SLURM_END job=${SLURM_JOB_ID:-?} mcnemar_eval rc=$RC" \
            --body-file "$OUT_FILE" --tail 400 || true
    else
        python -u scripts/notify.py --event "SLURM_END" \
            --to "$NOTIFY_TO" \
            --payload-json "{\"job_id\":\"${SLURM_JOB_ID:-?}\",\"rc\":$RC,\"out_file\":\"$OUT_FILE\",\"note\":\"out file not found\"}" || true
    fi
fi

exit $RC
