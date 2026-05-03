#!/bin/bash
#SBATCH --job-name=dmn_round
#SBATCH --partition=gpu
#SBATCH --qos=batch-long
#SBATCH --gpus=4
#SBATCH --mem=80G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=68:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s226137394@deakin.edu.au
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Submit one round (train + LLM pipeline) of an active-loop profile.
#
# Usage:
#   PROFILE=p3 sbatch scripts/slurm/round.sh                  # auto-pick latest loop or create one
#   PROFILE=p3 LOOP_ID=loop_20260501_120000 sbatch scripts/slurm/round.sh   # resume that loop
#   PROFILE=p3 NEW_LOOP=1 sbatch scripts/slurm/round.sh       # force a fresh loop
#   PROFILE=p3 N_EPISODES=10 SEED=42 sbatch scripts/slurm/round.sh
#
# After the job finishes (you'll get an email), record demos for the
# recommended layouts the LLM produced — the path is printed at the end of
# the slurm log:
#
#   python scripts/play_maze.py --layouts-from <recommended_layouts.json>
#
# Then resubmit this script with the SAME PROFILE and LOOP_ID to advance to
# the next round. run_round.py auto-detects the next round number under that
# loop directory.

set -eo pipefail

module purge
module load Anaconda3
source /home/s226137394/.bashrc
source activate
conda activate maze

PROJECT_ROOT="${PROJECT_ROOT:-/vast/s226137394/DmN/DmNfull}"
cd "$PROJECT_ROOT"

mkdir -p slurm_logs

# ---------- Parameters from environment (defaults) -------------------------
PROFILE="${PROFILE:-p1}"
LOOP_ID="${LOOP_ID:-}"          # empty -> run_round picks latest or creates one
NEW_LOOP="${NEW_LOOP:-0}"       # 1 -> force a fresh loop directory
ROUND="${ROUND:-}"              # empty -> auto-pick next round number
N_EPISODES="${N_EPISODES:-}"
SEED="${SEED:-}"
TARGET_SR="${TARGET_SR:-0.90}"
BASELINE_CKPT="${BASELINE_CKPT:-checkpoints/baseline.pth}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/${PROFILE}}"
mkdir -p "$CHECKPOINT_DIR"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_BASELINE_RESTORE="${SKIP_BASELINE_RESTORE:-0}"
EXPERT="${EXPERT:-bfs}"         # bfs (default headless) | human | none
AUTO_LOOP="${AUTO_LOOP:-0}"     # 1 -> chain rounds inside this single slurm job
MAX_ROUNDS="${MAX_ROUNDS:-20}"
NOTIFY_TO="${NOTIFY_TO:-}"      # empty -> notifier silent (uses $NOTIFY_TO inside python too)
EXTRA_ARGS="${EXTRA_ARGS:-} --checkpoint-dir $CHECKPOINT_DIR"
export NOTIFY_TO                # exported so notify.py picks it up if --notify-to omitted

echo "[slurm] === DmN active-loop round ==="
echo "[slurm] host         : $(hostname)"
echo "[slurm] cwd          : $(pwd)"
echo "[slurm] python       : $(which python)"
echo "[slurm] PROFILE      : $PROFILE"
echo "[slurm] LOOP_ID      : ${LOOP_ID:-<auto>}"
echo "[slurm] NEW_LOOP     : $NEW_LOOP"
echo "[slurm] ROUND        : ${ROUND:-<auto>}"
echo "[slurm] N_EPISODES   : ${N_EPISODES:-<config default>}"
echo "[slurm] SEED         : ${SEED:-<config default>}"
echo "[slurm] TARGET_SR    : $TARGET_SR"
echo "[slurm] BASELINE_CKPT: $BASELINE_CKPT"
echo "[slurm] SKIP_TRAIN   : $SKIP_TRAIN"
echo "[slurm] SKIP_BASELINE: $SKIP_BASELINE_RESTORE"
echo "[slurm] EXPERT       : $EXPERT"
echo "[slurm] AUTO_LOOP    : $AUTO_LOOP"
echo "[slurm] MAX_ROUNDS   : $MAX_ROUNDS"
echo "[slurm] NOTIFY_TO    : ${NOTIFY_TO:-<unset>}"
echo "[slurm] EXTRA_ARGS   : $EXTRA_ARGS"
echo "[slurm] ============================="

# Tell notify.py where to send mail. Notifier is best-effort: failures inside
# Python will not break the job.
if [[ -n "$NOTIFY_TO" ]]; then
    python -u scripts/notify.py --event "SLURM_START" \
        --to "$NOTIFY_TO" \
        --payload-json "{\"job_id\":\"${SLURM_JOB_ID:-?}\",\"profile\":\"$PROFILE\",\"loop_id\":\"$LOOP_ID\",\"expert\":\"$EXPERT\",\"auto_loop\":\"$AUTO_LOOP\"}" || true
fi

CMD=( python -u scripts/run_round.py
      --profile "$PROFILE"
      --target-sr "$TARGET_SR"
      --baseline-checkpoint "$BASELINE_CKPT"
      --expert "$EXPERT"
      --max-rounds "$MAX_ROUNDS" )

if [[ -n "$LOOP_ID" ]];   then CMD+=( --loop-id "$LOOP_ID" ); fi
if [[ "$NEW_LOOP" == "1" ]]; then CMD+=( --new-loop ); fi
if [[ -n "$ROUND" ]];     then CMD+=( --round "$ROUND" ); fi
if [[ -n "$N_EPISODES" ]];then CMD+=( --n_episodes "$N_EPISODES" ); fi
if [[ -n "$SEED" ]];      then CMD+=( --seed "$SEED" ); fi
if [[ "$SKIP_TRAIN" == "1" ]]; then CMD+=( --skip-train ); fi
if [[ "$SKIP_BASELINE_RESTORE" == "1" ]]; then CMD+=( --skip-baseline-restore ); fi
if [[ "$AUTO_LOOP" == "1" ]]; then CMD+=( --auto-loop ); fi
if [[ -n "$NOTIFY_TO" ]]; then CMD+=( --notify-to "$NOTIFY_TO" ); fi

# EXTRA_ARGS is a free-form string appended verbatim — for example
# 'EXTRA_ARGS="--config configs/some_other.yaml"'.
if [[ -n "$EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARRAY=($EXTRA_ARGS)
    CMD+=( "${EXTRA_ARRAY[@]}" )
fi

echo "[slurm] Running: ${CMD[*]}"
"${CMD[@]}"
RC=$?
echo "[slurm] run_round.py exit code: $RC"

# Mail the tail of the slurm .out so the user can read everything that happened
# without having to log back into the cluster. JOB_NAME / JOB_ID come from
# SBATCH; --output above writes to slurm_logs/<jobname>_<jobid>.out.
if [[ -n "$NOTIFY_TO" ]]; then
    OUT_FILE="slurm_logs/${SLURM_JOB_NAME:-dmn_round}_${SLURM_JOB_ID:-local}.out"
    if [[ -f "$OUT_FILE" ]]; then
        python -u scripts/notify.py \
            --to "$NOTIFY_TO" \
            --subject "[DmN] SLURM_END job=${SLURM_JOB_ID:-?} profile=$PROFILE rc=$RC" \
            --body-file "$OUT_FILE" --tail 400 || true
    else
        python -u scripts/notify.py --event "SLURM_END" \
            --to "$NOTIFY_TO" \
            --payload-json "{\"job_id\":\"${SLURM_JOB_ID:-?}\",\"rc\":$RC,\"out_file\":\"$OUT_FILE\",\"note\":\"out file not found\"}" || true
    fi
fi

exit $RC
