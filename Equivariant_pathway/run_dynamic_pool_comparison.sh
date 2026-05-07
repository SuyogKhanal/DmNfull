#!/bin/bash
#SBATCH --job-name=eq_dynamic_pool_comparison
#SBATCH --partition=gpu
#SBATCH --qos=batch-long
#SBATCH --gpus=2
#SBATCH --mem=80G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s226137394@deakin.edu.au
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# Clean head-to-head: p6_dynamic_pool (LLM-guided) vs baseline_dynamic_pool
# (random) starting from the SAME baseline_only/ snapshot.
#
# What this script does:
#   1. Verify baseline_only is in the right state (correction_n=20,
#      initial_best_eq_policy.pth present, >= 20 initial BFS demos).
#      If not, run baseline_only/run.sh to establish that state.
#   2. Print the sha256 of the initial checkpoint so the slurm log
#      records the EXACT bytes both downstream methods will start from.
#   3. Launch p6_dynamic_pool on GPU 0 and baseline_dynamic_pool on
#      GPU 1 IN PARALLEL. Both read baseline_only/ read-only and write
#      only into their own subtrees — no file-level contamination.
#   4. Wait for both to finish.
#   5. Render the final comparison chart that overlays all heldout SR
#      curves plus the dynamic-pool growth panel.
#
# baseline_dynamic_pool is capped to max_pool_expansion_per_round=3 by
# default to match p6's prescription scale. This is the "matched-budget"
# ablation: with the same per-round demo budget, who picks better
# layouts — the LLM or random? To run random uncapped, pass:
#   sbatch ... -- --baseline_dyn_cap 0
# (parsed below; passes through to baseline_dynamic_pool as
#  --max_pool_expansion_per_round 0)

set -eo pipefail

# Capture extras BEFORE any conda sourcing (which would otherwise pick
# up $1 as an env name).
EXTRA_ARGS=("$@")
set --

# Pluck out our own --baseline_dyn_cap and --max_rounds knobs so we can
# split them between the two methods cleanly. Anything else passes
# through to BOTH methods.
P6_EXTRA=()
BO_EXTRA=()
i=0
while [ $i -lt ${#EXTRA_ARGS[@]} ]; do
    arg="${EXTRA_ARGS[$i]}"
    case "$arg" in
        --baseline_dyn_cap)
            i=$((i + 1))
            BO_EXTRA+=(--max_pool_expansion_per_round "${EXTRA_ARGS[$i]}")
            ;;
        --max_rounds)
            P6_EXTRA+=(--max_rounds "${EXTRA_ARGS[$((i+1))]}")
            BO_EXTRA+=(--max_rounds "${EXTRA_ARGS[$((i+1))]}")
            i=$((i + 1))
            ;;
        *)
            P6_EXTRA+=("$arg")
            BO_EXTRA+=("$arg")
            ;;
    esac
    i=$((i + 1))
done
# Default cap = 3 (matches p6's prescription scale).
case " ${BO_EXTRA[*]} " in
    *" --max_pool_expansion_per_round "*) ;;
    *) BO_EXTRA+=(--max_pool_expansion_per_round 3) ;;
esac

module purge
module load Anaconda3
source /home/s226137394/.bashrc
eval "$(conda shell.bash hook)"
conda activate maze

mkdir -p slurm_logs

echo "[$(date)] ============================================================"
echo "[$(date)] DYNAMIC-POOL HEAD-TO-HEAD: p6 (LLM v2 prompt) vs baseline (random)"
echo "[$(date)] ============================================================"
echo "[$(date)]   working directory: $(pwd)"
echo "[$(date)]   p6 extras:                 ${P6_EXTRA[*]}"
echo "[$(date)]   baseline_dynamic extras:   ${BO_EXTRA[*]}"

# ----------------------------------------------------------------------
# STEP 1: ensure baseline_only/ is frozen at correction_n=20
# ----------------------------------------------------------------------
echo "[$(date)] ============================================================"
echo "[$(date)] STEP 1: verify baseline_only/ is at the shared starting point"
echo "[$(date)] ============================================================"
need_run_baseline=false

REQUIRED_BO_FILES=(
    "Equivariant_pathway/baseline_only/heldout_layouts.yaml"
    "Equivariant_pathway/baseline_only/correction_layouts.yaml"
    "Equivariant_pathway/baseline_only/training_layouts.yaml"
    "Equivariant_pathway/baseline_only/checkpoints/initial_best_eq_policy.pth"
    "Equivariant_pathway/baseline_only/checkpoints/initial_last_eq_policy.pth"
)
for f in "${REQUIRED_BO_FILES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "[$(date)]   MISSING: $f"
        need_run_baseline=true
    else
        echo "[$(date)]   found:   $f"
    fi
done

if [ ! -d "Equivariant_pathway/baseline_only/demos" ]; then
    echo "[$(date)]   MISSING dir: Equivariant_pathway/baseline_only/demos"
    need_run_baseline=true
fi

# Confirm correction pool size = 20 (the shared starting state both
# downstream methods must use).
n_pool=""
if [ -f "Equivariant_pathway/baseline_only/correction_layouts.yaml" ]; then
    n_pool=$(python3 -c "
import yaml
with open('Equivariant_pathway/baseline_only/correction_layouts.yaml') as f:
    spec = yaml.safe_load(f) or {}
for k in ('heldout_test_layouts', 'layouts', 'test_layouts', 'training_layouts'):
    v = spec.get(k)
    if isinstance(v, list):
        print(len(v))
        break
") || n_pool=""
    echo "[$(date)]   correction_layouts pool size: ${n_pool:-?} (need 20)"
    if [ "$n_pool" != "20" ]; then
        echo "[$(date)]   pool size mismatch — will re-run baseline_only with correction_n=20"
        need_run_baseline=true
    fi
fi

# Count initial BFS demos in baseline_only (filtered by source field).
if [ -d "Equivariant_pathway/baseline_only/demos" ]; then
    n_init_bfs=$(grep -l '"source": *"equivariant_pathway_bfs"' \
        Equivariant_pathway/baseline_only/demos/*.json 2>/dev/null | wc -l)
    echo "[$(date)]   initial BFS demos in baseline_only/demos: $n_init_bfs (need >= 20)"
    if [ "$n_init_bfs" -lt 20 ]; then
        need_run_baseline=true
    fi
fi

if [ "$need_run_baseline" = true ]; then
    echo "[$(date)] ============================================================"
    echo "[$(date)]   running baseline_only with correction_n=20 to freeze the"
    echo "[$(date)]   shared starting point. This wipes baseline_only/."
    echo "[$(date)] ============================================================"
    bash Equivariant_pathway/baseline_only/run.sh --correction_n 20
    echo "[$(date)] baseline_only complete; re-verifying state..."

    n_pool=$(python3 -c "
import yaml
with open('Equivariant_pathway/baseline_only/correction_layouts.yaml') as f:
    spec = yaml.safe_load(f) or {}
for k in ('heldout_test_layouts', 'layouts', 'test_layouts', 'training_layouts'):
    v = spec.get(k)
    if isinstance(v, list):
        print(len(v))
        break
")
    if [ "$n_pool" != "20" ]; then
        echo "[$(date)] ERROR: baseline_only ran but correction pool size is $n_pool, not 20."
        exit 1
    fi
else
    echo "[$(date)] baseline_only/ is already at the right starting state — skipping rerun."
fi

echo "[$(date)] shared initial checkpoint (both methods will copy this exact file):"
sha256sum Equivariant_pathway/baseline_only/checkpoints/initial_best_eq_policy.pth || true

# ----------------------------------------------------------------------
# STEP 2: launch both methods in parallel on separate GPUs
# ----------------------------------------------------------------------
echo "[$(date)] ============================================================"
echo "[$(date)] STEP 2: launching p6_dynamic_pool (GPU 0) +"
echo "[$(date)]                   baseline_dynamic_pool (GPU 1)  IN PARALLEL"
echo "[$(date)] ============================================================"
echo "[$(date)]   no contamination: p6 writes only to p6_dynamic_pool/,"
echo "[$(date)]   baseline_dynamic_pool writes only to its own folder,"
echo "[$(date)]   baseline_only/ stays read-only for both."

P6_LOG="slurm_logs/parallel_p6_dynamic_pool_${SLURM_JOB_ID:-local}.out"
BO_LOG="slurm_logs/parallel_baseline_dynamic_pool_${SLURM_JOB_ID:-local}.out"

# Detect how many GPUs are visible. Slurm normally sets
# CUDA_VISIBLE_DEVICES to "0,1" (or the actual indices) for our request
# of --gpus=2; we just need indices for sub-process pinning.
all_gpus=()
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra all_gpus <<< "$CUDA_VISIBLE_DEVICES"
fi
n_gpus=${#all_gpus[@]}
if [ "$n_gpus" -lt 2 ]; then
    # Fallback: assume two GPUs at indices 0 and 1. nvidia-smi will tell
    # us if neither is available.
    echo "[$(date)]   WARNING: CUDA_VISIBLE_DEVICES did not expose 2 GPUs (n=$n_gpus)."
    echo "[$(date)]   falling back to CUDA_VISIBLE_DEVICES=0 / 1 for the two child processes."
    all_gpus=(0 1)
fi
GPU_P6="${all_gpus[0]}"
GPU_BO="${all_gpus[1]}"
echo "[$(date)]   GPU mapping: p6_dynamic_pool=$GPU_P6  baseline_dynamic_pool=$GPU_BO"

# p6_dynamic_pool — uses the new prompts (force-floor + diversity).
# Each subprocess sees only one GPU because we override CUDA_VISIBLE_DEVICES
# in the env it starts with.
CUDA_VISIBLE_DEVICES="$GPU_P6" python -u -m Equivariant_pathway.p6_dynamic_pool.pipeline \
    --force_restart "${P6_EXTRA[@]}" \
    > "$P6_LOG" 2>&1 &
P6_PID=$!
echo "[$(date)]   p6_dynamic_pool         pid=$P6_PID  GPU=$GPU_P6  log=$P6_LOG"

CUDA_VISIBLE_DEVICES="$GPU_BO" python -u -m Equivariant_pathway.baseline_dynamic_pool.pipeline \
    --force_restart "${BO_EXTRA[@]}" \
    > "$BO_LOG" 2>&1 &
BO_PID=$!
echo "[$(date)]   baseline_dynamic_pool   pid=$BO_PID  GPU=$GPU_BO  log=$BO_LOG"

# Wait for both. Don't abort the script when one fails; we still want
# to render the chart for whichever method succeeded.
fail=0
echo "[$(date)] waiting for both methods to complete..."
if wait "$P6_PID"; then
    echo "[$(date)] p6_dynamic_pool         (pid=$P6_PID) OK"
else
    rc=$?
    echo "[$(date)] p6_dynamic_pool         (pid=$P6_PID) FAILED rc=$rc — see $P6_LOG"
    fail=1
fi
if wait "$BO_PID"; then
    echo "[$(date)] baseline_dynamic_pool   (pid=$BO_PID) OK"
else
    rc=$?
    echo "[$(date)] baseline_dynamic_pool   (pid=$BO_PID) FAILED rc=$rc — see $BO_LOG"
    fail=1
fi

# ----------------------------------------------------------------------
# STEP 3: render comparison chart
# ----------------------------------------------------------------------
echo "[$(date)] ============================================================"
echo "[$(date)] STEP 3: rendering final comparison charts"
echo "[$(date)] ============================================================"
# baseline_dynamic_pool/chart.py overlays both runs (and any other peer
# learning_curve.json files it finds) — that's the headline figure.
# p6_dynamic_pool/chart.py is a sanity copy with its own annotation.
python -u -m Equivariant_pathway.baseline_dynamic_pool.chart || true
python -u -m Equivariant_pathway.p6_dynamic_pool.chart       || true

echo "[$(date)] charts written:"
ls -la Equivariant_pathway/baseline_dynamic_pool/results/*.png 2>/dev/null || true
ls -la Equivariant_pathway/p6_dynamic_pool/results/*.png 2>/dev/null || true

echo "[$(date)] ============================================================"
echo "[$(date)] HEADLINE CHART:"
echo "[$(date)]   Equivariant_pathway/baseline_dynamic_pool/results/random_vs_llm_dynamic_pool.png"
echo "[$(date)] (overlays p6_dynamic_pool, baseline_dynamic_pool, and any peer"
echo "[$(date)]  learning_curve.json files from baseline_only / p4_only / p6_only)"
echo "[$(date)] ============================================================"

if [ "$fail" -ne 0 ]; then
    echo "[$(date)] one or both methods failed; chart may be partial."
    exit 1
fi
echo "[$(date)] DONE."
