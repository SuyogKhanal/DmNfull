#!/bin/bash
# submit_hpcb_matrix.sh — HPC-B (cluster: rohan) launcher for its OWNED state cells:
# Wipe + Door, all 6 arms x 5 seeds = 60 DISTIL jobs (HANDOFF_HPC2 §3; whole cells only).
# Each ablation-branch+seed is ONE job. Ni comes from the per-task config (Wipe=12,
# Door=4 after upstream b5485c09) — NUM_INIT is NOT passed. Per-task bootstrap subdir
# distil/results/shared_bootstrap/<T>_state holds the byte-identical shared bootstrap.
#
# Usage:
#   distil/scripts/submit_hpcb_matrix.sh --dry-run     # preview, submit nothing
#   distil/scripts/submit_hpcb_matrix.sh               # submit for real (needs a real key)
#   SEEDS="1" distil/scripts/submit_hpcb_matrix.sh     # seed subset (staged submission)
#
# SAFETY: refuses to submit unless OPENROUTER_API_KEY is a real sk-or-... key (4/6 arms
# need the live LLM; an empty key silently degrades them to the geometric fallback).
set -eo pipefail

ROOT="${DISTIL_ROOT:-$PWD}"
cd "$ROOT"
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1

TASKS=(${TASKS_OVERRIDE:-Wipe Door})
ARMS=(${ARMS_OVERRIDE:-full memory_off allocation_random clustering_off decision_heuristic vlm_off})
SEEDS=(${SEEDS:-1 2 3 4 5})
BUDGET="${BUDGET:-20}"
CONDA_ENV="${CONDA_ENV:-diffdagger}"   # diffdagger already satisfies every pin on rohan
PART="${PART:-gpu}"
QOS="${QOS:-batch-long}"
TIME="${TIME:-1-00:00:00}"
# PORTABILITY §2: torch 2.4.1+cu121 has NO kernels for the Blackwell node
# (rtxp6000l-f-01 = RTX PRO 6000, sm_100/120) -> "no kernel image". L40S + V100 are fine.
EXCLUDE="${EXCLUDE:-rtxp6000l-f-01}"
BSROOT="$ROOT/distil/results/shared_bootstrap"
LEDGER="$ROOT/distil/results/_submitted_hpcb.tsv"

# --- key guard ---
KEY="$(grep -E '^OPENROUTER_API_KEY=' "$ROOT/.env" 2>/dev/null | head -1 | cut -d= -f2-)"
if [ "$DRY" = "0" ]; then
  case "$KEY" in
    sk-or-*) : ;;
    *) echo "REFUSING: OPENROUTER_API_KEY in $ROOT/.env is not a real sk-or-... key."; exit 2;;
  esac
fi

# --- bootstrap presence guard (per-task subdir; Ni is baked into the filename) ---
for T in "${TASKS[@]}"; do
  ls "$BSROOT/${T}_state"/${T}_state_ni*.pkl >/dev/null 2>&1 || {
    echo "MISSING bootstrap for $T at $BSROOT/${T}_state/. Run:"
    echo "  python -m distil.run --make-bootstrap --task $T --modality state --bootstrap-dir $BSROOT/${T}_state"
    exit 3; }
done

mkdir -p "$ROOT/distil/slurm_logs"
[ "$DRY" = "0" ] && : > "$LEDGER"
n=0
for T in "${TASKS[@]}"; do
  BOOTSTRAP_DIR="$BSROOT/${T}_state"
  for A in "${ARMS[@]}"; do
    for S in "${SEEDS[@]}"; do
      OUT="$ROOT/distil/results/$T/state/$A/seed$S"
      JOBNAME="distil_${T}_${A}_s${S}"
      EXPORT="ALL,DISTIL_ROOT=$ROOT,CONDA_ENV=$CONDA_ENV,TASK=$T,MODALITY=state,ABLATION=$A,SEED=$S,BUDGET=$BUDGET,OUTPUT_DIR=$OUT,BOOTSTRAP_DIR=$BOOTSTRAP_DIR"
      CMD=(sbatch --parsable --job-name="$JOBNAME" --partition="$PART" --qos="$QOS" --time="$TIME"
           --exclude="$EXCLUDE" --export="$EXPORT" distil/scripts/run_distil.sbatch)
      n=$((n+1))
      if [ "$DRY" = "1" ]; then
        echo "[$n] $JOBNAME  (Ni=config, budget=$BUDGET, bootstrap=$BOOTSTRAP_DIR)"
      else
        JID=$("${CMD[@]}")
        printf "%s\tstate\t%s\t%s\t%s\t%s\n" "$T" "$A" "$S" "$JID" "$OUT" >> "$LEDGER"
        echo "[$n] submitted $JOBNAME -> job $JID"
      fi
    done
  done
done
echo "TOTAL: $n jobs ($([ "$DRY" = "1" ] && echo DRY-RUN || echo submitted)). Ledger: $LEDGER"
