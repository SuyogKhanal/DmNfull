#!/bin/bash
#SBATCH --job-name=d5_gw
#SBATCH --partition=gpu,gpu-large
#SBATCH --qos=batch-short
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=distil/slurm_logs/%x_%j.out
#SBATCH --error=distil/slurm_logs/%x_%j.err
# ===========================================================================
# run_gridworld_d5.sbatch — ONE GridWorld 5x5 run for the D5 compute table.
#
# GridWorld is cheap (a 4-way classifier, no MuJoCo): 1 small GPU, no h100/h200
# constraint, no MUJOCO_GL. The image modality is a plain RGB CNN (RGBCNNPolicy)
# over the 80x80 bird-eye raster; the state modality is the p4m-equivariant CNN.
# The DISTIL reasoning path (VLM frames -> analysis -> decision) is identical in
# both modalities, because the failure descriptor is GEOMETRIC (02_..md #4).
#
# Emits, per round, into $OUTPUT_DIR:
#   result.json               history[].sec  (total s/round)  + history[].tokens
#   telemetry/compute.jsonl   sec_train / sec_eval / sec_screen / sec_llm /
#                             sec_prescribe / sec_reasoning  + the same tokens
#   prompts/round_XXXX/       the exact VLM + analysis + decision prompts
#   kag/GridWorld.kag.txt     the exact KAG block that was injected
#
# Env knobs (pass with --export=ALL,...):
#   MODALITY  image | state         (default: image)
#   ABLATION  full | safe | stagger (default: full)   full = DISTIL, safe = SafeDAgger*
#   SEED      integer               (default: 1)
#   BUDGET    successful demos added on top of Ni (default: 5, == the Door/Wipe D5 runs)
#   NUM_INIT  bootstrap size Ni     (default: config = 20)
#
# The two D5 arms (run both; they share the byte-identical bootstrap + the same
# per-round screening pool, so the only difference is the allocation decision):
#
#   sbatch --job-name=d5_GridWorld_image_full \
#     --export=ALL,MODALITY=image,ABLATION=full,SEED=1,BUDGET=5 \
#     distil/scripts/run_gridworld_d5.sbatch
#
#   sbatch --job-name=d5_GridWorld_image_safe \
#     --export=ALL,MODALITY=image,ABLATION=safe,SEED=1,BUDGET=5 \
#     distil/scripts/run_gridworld_d5.sbatch
# ===========================================================================
set -eo pipefail

ROOT="${DISTIL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ROOT"
mkdir -p distil/slurm_logs

if command -v module >/dev/null 2>&1; then module purge || true; module load Anaconda3 || true; fi
[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"
CONDA_ENV="${CONDA_ENV:-distil}"
if command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)"; conda activate "$CONDA_ENV" || true; fi
PYTHON_BIN="${PYTHON_BIN:-python}"

export DISTIL_ROOT="$ROOT"

MODALITY="${MODALITY:-image}"
ABLATION="${ABLATION:-full}"
SEED="${SEED:-1}"
BUDGET="${BUDGET:-5}"
NUM_INIT="${NUM_INIT:-}"
BOOTSTRAP_DIR="${BOOTSTRAP_DIR:-$ROOT/distil/results/shared_bootstrap}"
# _compute/<task>/<modality>/<arm>/seed<N> — the layout distil/scripts/build_d5_compute.py reads.
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/distil/results/_compute/GridWorld/${MODALITY}/${ABLATION}/seed${SEED}}"

NI_FLAG=""; [ -n "$NUM_INIT" ] && NI_FLAG="--num-init-demos $NUM_INIT"

echo "=== D5 GridWorld: modality=$MODALITY ablation=$ABLATION seed=$SEED budget=$BUDGET Ni=${NUM_INIT:-config} ==="
echo "python: $PYTHON_BIN | out: $OUTPUT_DIR"
nvidia-smi -L || true

"$PYTHON_BIN" -m distil.run \
  --task GridWorld --modality "$MODALITY" --ablation "$ABLATION" --seed "$SEED" \
  --budget "$BUDGET" $NI_FLAG \
  --bootstrap-dir "$BOOTSTRAP_DIR" --output-dir "$OUTPUT_DIR"

echo "=== DONE (exit $?) ==="
