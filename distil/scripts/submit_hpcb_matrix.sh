#!/bin/bash
# submit_hpcb_matrix.sh — HPC-B (cluster: rohan) launcher for its OWNED cells.
# Whole cells only (HANDOFF_HPC2 §3); each (task,modality,arm,seed) is ONE job.
# Ni comes from the per-task config (Wipe=12, Door=4, Lift=8) — NUM_INIT is NOT passed.
# Bootstrap: per-(task,modality) subdir distil/results/shared_bootstrap/<T>_<MODALITY>/.
#
# Parameterized via env:
#   TASKS_OVERRIDE   default "Wipe Door"
#   MODALITY         state | image           (default state)
#   ARMS_OVERRIDE    default = 6 DISTIL arms; for baselines pass the gate family
#   SEEDS            default "1 2 3 4 5"
#   BUDGET           default 20
#
# Examples:
#   MODALITY=image TASKS_OVERRIDE=Door distil/scripts/submit_hpcb_matrix.sh              # Door image DISTIL (30)
#   TASKS_OVERRIDE=Door ARMS_OVERRIDE="diffdagger safe dropout ensemble thrifty stagger" \
#       distil/scripts/submit_hpcb_matrix.sh                                             # Door state baselines (30)
#   distil/scripts/submit_hpcb_matrix.sh --dry-run                                       # preview
#
# SAFETY: DISTIL LLM arms need a real sk-or- key; refuses to submit without one. (Baseline
# arms don't use the LLM, but the key is present anyway, so the guard is a no-op for them.)
set -eo pipefail

ROOT="${DISTIL_ROOT:-$PWD}"
cd "$ROOT"
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1

TASKS=(${TASKS_OVERRIDE:-Wipe Door})
MODALITY="${MODALITY:-state}"
ARMS=(${ARMS_OVERRIDE:-full memory_off allocation_random clustering_off decision_heuristic vlm_off})
SEEDS=(${SEEDS:-1 2 3 4 5})
BUDGET="${BUDGET:-20}"
CONDA_ENV="${CONDA_ENV:-diffdagger}"
PART="${PART:-gpu}"
QOS="${QOS:-batch-long}"
# image runs render every step -> slower; give them 2 days by default.
[ "$MODALITY" = "image" ] && DEFTIME="2-00:00:00" || DEFTIME="1-00:00:00"
TIME="${TIME:-$DEFTIME}"
EXCLUDE="${EXCLUDE:-rtxp6000l-f-01}"   # Blackwell node: no cu121 kernels (PORTABILITY §2)
BSROOT="$ROOT/distil/results/shared_bootstrap"
LEDGER="$ROOT/distil/results/_submitted_hpcb_${MODALITY}.tsv"

# --- key guard ---
KEY="$(grep -E '^OPENROUTER_API_KEY=' "$ROOT/.env" 2>/dev/null | head -1 | cut -d= -f2-)"
if [ "$DRY" = "0" ]; then
  case "$KEY" in sk-or-*) : ;; *) echo "REFUSING: no real sk-or- OPENROUTER_API_KEY in $ROOT/.env"; exit 2;; esac
fi

# --- bootstrap presence guard (per task,modality; Ni baked into the filename) ---
for T in "${TASKS[@]}"; do
  ls "$BSROOT/${T}_${MODALITY}"/${T}_${MODALITY}_ni*.pkl >/dev/null 2>&1 || {
    echo "MISSING bootstrap: $BSROOT/${T}_${MODALITY}/${T}_${MODALITY}_ni*.pkl"
    echo "  gen: python -m distil.run --make-bootstrap --task $T --modality $MODALITY --bootstrap-dir $BSROOT/${T}_${MODALITY}"
    echo "  (image bootstrap renders frames -> run on a GPU node / EGL, not the login node)"
    exit 3; }
done

mkdir -p "$ROOT/distil/slurm_logs"
[ "$DRY" = "0" ] && : > "$LEDGER"
n=0
for T in "${TASKS[@]}"; do
  BOOTSTRAP_DIR="$BSROOT/${T}_${MODALITY}"
  for A in "${ARMS[@]}"; do
    for S in "${SEEDS[@]}"; do
      OUT="$ROOT/distil/results/$T/$MODALITY/$A/seed$S"
      JOBNAME="distil_${T}_${MODALITY}_${A}_s${S}"
      EXPORT="ALL,DISTIL_ROOT=$ROOT,CONDA_ENV=$CONDA_ENV,TASK=$T,MODALITY=$MODALITY,ABLATION=$A,SEED=$S,BUDGET=$BUDGET,OUTPUT_DIR=$OUT,BOOTSTRAP_DIR=$BOOTSTRAP_DIR"
      CMD=(sbatch --parsable --job-name="$JOBNAME" --partition="$PART" --qos="$QOS" --time="$TIME"
           --exclude="$EXCLUDE" --export="$EXPORT" distil/scripts/run_distil.sbatch)
      n=$((n+1))
      if [ "$DRY" = "1" ]; then
        echo "[$n] $JOBNAME  (Ni=config, budget=$BUDGET, bootstrap=$BOOTSTRAP_DIR)"
      else
        JID=$("${CMD[@]}")
        printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$T" "$MODALITY" "$A" "$S" "$JID" "$OUT" >> "$LEDGER"
        echo "[$n] submitted $JOBNAME -> job $JID"
      fi
    done
  done
done
echo "TOTAL: $n jobs ($([ "$DRY" = "1" ] && echo DRY-RUN || echo submitted)). Ledger: $LEDGER"
