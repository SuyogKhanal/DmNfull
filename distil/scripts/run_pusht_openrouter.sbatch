#!/bin/bash
#SBATCH --job-name=or_pusht
#SBATCH --partition=gpu-large
#SBATCH --qos=batch-long
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
# A Push-T round is expensive: 60-episode failure rollout + from-scratch retrain
# (nd_retrain=1) + 100-episode frozen held-out eval + the VLM/LLM stage. The D5 study
# measured ~1,891 s/round on image, so B=20 is ~10-12 h, plus the Ni=20 bootstrap for
# any seed with no published checkpoint. 3 days leaves real headroom.
#SBATCH --time=3-00:00:00
#SBATCH --output=distil/slurm_logs/%x_%j.out
#SBATCH --error=distil/slurm_logs/%x_%j.err
# ===========================================================================
# run_pusht_openrouter.sbatch — ONE Push-T DISEIL (p4_subtask) run on OPENROUTER.
#
# WHY THIS FILE EXISTS (do not "simplify" it back into run_pool_rl_robo.sh):
# run_pool_rl_robo.sh, whenever a p4_* method is requested, UNCONDITIONALLY
#   (a) starts two local vLLM servers + the qwen proxy, and
#   (b) exports OPENAI_BASE_URL=http://127.0.0.1:<proxy>/v1   (line 167)
# which CLOBBERS any OpenRouter base_url, and demands 3 GPUs. This script drives the
# orchestrator directly instead: no vLLM, no proxy, 1 GPU, OpenRouter only.
#
# The fork's LLM transport (diffdagger/main_analysis/llm_clients.py) speaks the
# OpenAI RESPONSES API (client.responses.create). OpenRouter serves /v1/responses —
# VERIFIED against qwen/qwen3-32b (text) and qwen/qwen3-vl-30b-a3b-instruct (vision,
# input_image) — so NO shim/proxy is needed. The method code is untouched.
#
# CRITICAL: the fork reads OPENAI_API_KEY *only* (no OPENROUTER_API_KEY fallback), so
# we set OPENAI_API_KEY := OPENROUTER_API_KEY here. The repo .env also carries a real
# OpenAI sk-proj- key; sending THAT to OpenRouter is a 401, and sending it to
# api.openai.com with a Qwen model name is a silent 404. DISEIL_STRICT_LLM=1 makes
# p4_subtask/pipeline.py assert the base_url + key shape AND make one live call before
# the loop starts — a misconfigured API aborts instead of degrading to the no-LLM
# fallback (the fork swallows LLM errors and would still write a plausible curve).
#
# Env vars (pass with --export=ALL,...):
#   MODALITY   state | image            (required)
#   SEED       1..5                     (required)
#   RUN_ID     results/<ENV>/run_<ID>   (required; namespaced, never an existing id)
#   BUDGET     default 20 (CoC protocol)
# ===========================================================================
set -eo pipefail

ROOT="${DISTIL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ROOT"
mkdir -p distil/slurm_logs

SUITE_DIR="$ROOT/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_rl_robo"
PKG="Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo"
[ -d "$SUITE_DIR" ] || { echo "[or-pusht] ERROR: suite dir not found: $SUITE_DIR" >&2; exit 2; }

MODALITY="${MODALITY:?set MODALITY=state|image}"
SEED="${SEED:?set SEED}"
RUN_ID="${RUN_ID:?set RUN_ID}"
BUDGET="${BUDGET:-20}"
SMOKE="${SMOKE:-0}"        # 1 => tiny plumbing test (orchestrator forces initial_demos=2)

# ── conda ──────────────────────────────────────────────────────────────────
if command -v module >/dev/null 2>&1; then module purge || true; module load Anaconda3 || true; fi
[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"
CONDA_ENV="${CONDA_ENV:-diffdagger}"
if command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)"; conda activate "$CONDA_ENV" || true; fi
# .bashrc prepends maze/bin to PATH, so a bare `python` can hit the wrong env.
PYTHON_BIN="${PYTHON_BIN:-/home/s226137394/.conda/envs/${CONDA_ENV}/bin/python}"
[ -x "$PYTHON_BIN" ] || { echo "[or-pusht] ERROR: interpreter missing: $PYTHON_BIN" >&2; exit 3; }

# ── .env -> OpenRouter (never echo a key) ──────────────────────────────────
set -a; [ -f "$ROOT/.env" ] && . "$ROOT/.env"; set +a
: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY missing from .env}"
export OPENAI_API_KEY="$OPENROUTER_API_KEY"          # the fork reads OPENAI_API_KEY ONLY
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
export LLM_MODEL_NAME="${LLM_MODEL_NAME:-qwen/qwen3-32b}"
export VLM_MODEL_NAME="${VLM_MODEL_NAME:-qwen/qwen3-vl-30b-a3b-instruct}"

if [ "$OPENAI_BASE_URL" != "https://openrouter.ai/api/v1" ]; then
  echo "[or-pusht] ERROR: OPENAI_BASE_URL is not OpenRouter" >&2; exit 4; fi
case "$OPENAI_API_KEY" in
  sk-or-*) : ;;
  *) echo "[or-pusht] ERROR: OPENAI_API_KEY is not an OpenRouter (sk-or-) key" >&2; exit 4 ;;
esac

export DISEIL_STRICT_LLM=1        # misconfigured/unreachable LLM => ABORT, never fall back
export D5_TELEMETRY=1             # per-call VLM/LLM token records -> telemetry/d5_events.jsonl
export OAI_SDK_TIMEOUT="${OAI_SDK_TIMEOUT:-300}"
export OAI_SDK_MAX_RETRIES="${OAI_SDK_MAX_RETRIES:-2}"
export OAI_MAX_ATTEMPTS="${OAI_MAX_ATTEMPTS:-5}"      # ride out OpenRouter 429s

# ── results namespace: NEVER the published tree ────────────────────────────
export POOL_RESULTS_ROOT="${POOL_RESULTS_ROOT:-$SUITE_DIR/results/rerun_openrouter}"
mkdir -p "$POOL_RESULTS_ROOT"

# ── per-modality config + the shared Ni=20 bootstrap ───────────────────────
if [ "$MODALITY" = "image" ]; then
  CONFIG="$SUITE_DIR/config_pusht_image_or20.yaml"
  export SUITE_HYDRA_CFG="$SUITE_DIR/configs/hydra/pusht_image.yaml"
  CKPT="$SUITE_DIR/results/PushT-v1/run_$((110 + SEED))/shared_baselines/init_ckpt.pth"
else
  CONFIG="$SUITE_DIR/config_pusht_state_or20.yaml"
  CKPT="$SUITE_DIR/results/PushT-v1/run_${SEED}/shared_baselines/init_ckpt.pth"
fi
# Reuse the published byte-identical Ni=20 bootstrap when it exists (fairness
# invariant: same initial policy as the baselines). Read-only; never written.
# SMOKE forces initial_demos=2, so the Ni=20 checkpoint does not apply — never reuse it.
SMOKE_FLAG=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_FLAG="--smoke"
  echo "[or-pusht] SMOKE=1: plumbing test (initial_demos=2, 1 round); bootstrap NOT reused."
elif [ -f "$CKPT" ]; then
  export P4_REUSE_INIT_CKPT="$CKPT"
  echo "[or-pusht] reusing shared bootstrap: $CKPT"
else
  echo "[or-pusht] NOTE: no published bootstrap for seed $SEED ($MODALITY); this run"
  echo "[or-pusht]       BUILDS its own Ni=20 bootstrap (initial_demos are seeds 0..19,"
  echo "[or-pusht]       identical by construction) into $POOL_RESULTS_ROOT."
fi

echo "=== Push-T DISEIL (p4_subtask) | modality=$MODALITY seed=$SEED run_id=$RUN_ID budget=$BUDGET ==="
echo "config  : $CONFIG"
echo "results : $POOL_RESULTS_ROOT/PushT-v1/run_$RUN_ID/p4_subtask/"
echo "llm     : $OPENAI_BASE_URL | llm=$LLM_MODEL_NAME vlm=$VLM_MODEL_NAME (key: OPENROUTER, not printed)"
nvidia-smi -L || true

# NOTE: --methods p4_subtask. Unknown method names are SILENTLY DROPPED by the
# orchestrator (_common.py:349), so `--methods full` would run nothing at all.
"$PYTHON_BIN" -u -m "${PKG}.orchestrator.run_one" \
  --env PushT-v1 --methods p4_subtask \
  --run_id "$RUN_ID" --seed "$SEED" \
  --budget "$BUDGET" --config "$CONFIG"

echo "=== DONE (exit $?) ==="
