#!/bin/bash
# Launch the P4-LLM training-hyperparameter sweep: generate the grid, submit one
# SLURM job per CHUNK of configs (each chunk runs its (config x run) P4 replays
# in parallel, no LLM), then a ranking job (afterok) that builds the per-demo
# curve-dominance leaderboard vs the fixed production baselines. Results land in
# a SEPARATE dir (results_hpsweep/), never touching the main results/ tree.
#
# Usage:
#   bash submit_sweep_all.sh                         # full grid (36 cfgs) x 2 runs
#   CONFIGS_PER_JOB=2 RUNS="1 6" bash submit_sweep_all.sh
#   SMOKE=1 bash submit_sweep_all.sh                 # 2 cfgs, run 1, epochs=1
#
# Knobs: RUNS, CONFIGS_PER_JOB, MAX_CONCURRENCY, JOB_TIME, OUT_ROOT, SMOKE.

set -eo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then SCRIPT_DIR="${SLURM_SUBMIT_DIR}";
else SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; fi
cd "${SCRIPT_DIR}"

RUNS="${RUNS:-1 6}"
CONFIGS_PER_JOB="${CONFIGS_PER_JOB:-2}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-4}"
JOB_TIME="${JOB_TIME:-08:00:00}"
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/results_hpsweep}"
CONFIG="${CONFIG:-${SCRIPT_DIR}/config_baselines.yaml}"
SMOKE="${SMOKE:-0}"
BUDGET="${BUDGET:-15}"

mkdir -p slurm_logs logs "${OUT_ROOT}"
GRID="${OUT_ROOT}/grid.json"

# ---- generate the grid (medium: 3 x 2 x 3 x 2 = 36 configs) ----
python3 - "$GRID" "$SMOKE" <<'PY'
import json, sys
grid_path, smoke = sys.argv[1], sys.argv[2] == "1"
if smoke:
    EPOCHS=[1]; LR=[5e-5]; RM=[0.5]; DO=[0.0, 0.1]   # 2 tiny configs
else:
    EPOCHS=[90,150,300]; LR=[5e-5,1e-4]; RM=[0.3,0.5,0.7]; DO=[0.0,0.1]
cfgs=[]
for e in EPOCHS:
    for lr in LR:
        for rm in RM:
            for do in DO:
                tag=f"e{e}_lr{lr:g}_rm{rm:g}_do{do:g}"
                cfgs.append({"tag":tag,"epochs":e,"lr":lr,"batch_size":64,
                             "weight_decay":1e-4,"replay_mix":rm,"head_dropout":do})
json.dump(cfgs, open(grid_path,"w"), indent=2)
print(f"[grid] wrote {len(cfgs)} configs -> {grid_path}")
PY

N_CFG=$(python3 -c "import json;print(len(json.load(open('${GRID}'))))")
if [ "${SMOKE}" = "1" ]; then RUNS="${RUNS_SMOKE:-1}"; fi
N_CHUNKS=$(( (N_CFG + CONFIGS_PER_JOB - 1) / CONFIGS_PER_JOB ))
echo "[submit_sweep_all] ${N_CFG} configs, ${CONFIGS_PER_JOB}/job -> ${N_CHUNKS} chunk jobs; runs='${RUNS}'"

# RUNS contains spaces; sbatch --export splits on commas, so encode spaces as
# '+' here and decode them back in submit_sweep_one.sh.
EXPORT_BASE="ALL,CONFIG=${CONFIG},GRID=${GRID},OUT_ROOT=${OUT_ROOT},RUNS=${RUNS// /+},CHUNK_SIZE=${CONFIGS_PER_JOB},MAX_CONCURRENCY=${MAX_CONCURRENCY},BUDGET=${BUDGET}"

JOB_IDS=()
for ((c=0; c<N_CHUNKS; c++)); do
    JID=$(sbatch --parsable --job-name="p4sweep_c${c}" --time="${JOB_TIME}" \
        --export="${EXPORT_BASE},CHUNK_INDEX=${c}" \
        "${SCRIPT_DIR}/submit_sweep_one.sh")
    JOB_IDS+=("${JID}")
    echo "[submit_sweep_all] chunk ${c} -> job ${JID}"
done

DEP="afterok:$(IFS=:; echo "${JOB_IDS[*]}")"
RANK_ID=$(sbatch --parsable --job-name="p4sweep_rank" --time="01:00:00" \
    --dependency="${DEP}" \
    --export="${EXPORT_BASE},RANK=1" \
    "${SCRIPT_DIR}/submit_sweep_one.sh")
echo "[submit_sweep_all] ranking job ${RANK_ID} (afterok:${#JOB_IDS[@]} chunks)"
echo "[submit_sweep_all] leaderboard -> ${OUT_ROOT}/leaderboard.{md,json}, leaderboard_top.png"
