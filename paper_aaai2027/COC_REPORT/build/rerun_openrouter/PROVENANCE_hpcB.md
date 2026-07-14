# PROVENANCE — HPC-B (rohan), DISEIL OpenRouter re-run

HPC-B owns 4 cells x 5 seeds = **20 runs**: Lift and Wipe, each under `state` and `image`,
arm `full` (DISEIL) only, no baselines. Push-T and Door are HPC-A's and are not run here.

Every number in `cells/*.json` traces to a run that completed on this cluster. Nothing is
copied from a previous run, and nothing is estimated.

---

## 1. Cluster and environment

| | |
|---|---|
| Cluster | **rohan** (HPC-B), login node `login-f`, SLURM |
| Repo | `/vast/s226137394/DMNFull/DmNfull_diseil` — a **fresh clone** of `coc-report-diseil` |
| GPU partition | `gpu` (V100 / V100L / L40S). **`gpu-large` does not exist on this cluster**, so the sbatch's default `--partition=gpu,gpu-large` is overridden with `--partition=gpu`. |
| Node exclusion | `--exclude=rtxp6000l-f-01` — the RTX PRO 6000 (Blackwell, sm_100/120) node, on which torch cu121 dies with "no kernel image is available" |
| Conda env | **`diffdagger`** (pre-existing on this cluster; satisfies every pin in `distil/environment.yml`) |

Environment was **not rebuilt from scratch**: the existing `diffdagger` env already satisfies
every pin, which was verified rather than assumed:

| package | required | present |
|---|---|---|
| torch | 2.4.1 (+cu121) | **2.4.1+cu121** |
| robosuite | commit **`85abee22`** (NOT pip `1.5.2` — the offscreen-render sync commit is missing from the release) | **`85abee2`**, editable at `/vast/s226137394/robosuite/repo` (on `/vast`, visible from compute nodes; **not** a `/weka` path) |
| mujoco | 3.2.7 | **3.2.7** |
| scikit-learn | **1.6.1** (mandatory — without it `p4/clustering.py` silently falls back to a *different* numpy single-linkage clustering) | **1.6.1** |
| numpy | 1.26.4 | 1.26.4 |
| openai SDK | 2.38.0 | 2.37.0 |

Rendering: `MUJOCO_GL=egl` on GPU nodes (set by the sbatch). Verified that `osmesa` also renders
offscreen on CPU nodes, which is how the smoke run and all four bootstraps were produced without
consuming GPU queue time.

## 2. API — OpenRouter only

    OPENAI_BASE_URL = https://openrouter.ai/api/v1     (the OpenAI SDK's base-url variable, pointed at OpenRouter)
    OPENROUTER_API_KEY = sk-or-v1-...                  (73 chars; never printed or committed)
    VLM_MODEL_NAME = qwen/qwen3-vl-30b-a3b-instruct
    LLM_MODEL_NAME = qwen/qwen3-32b
    DISTIL_REASONING_HIGH = 18432 ; OAI_MAX_OUTPUT_TOKENS = 26624 ; OAI_MAX_ATTEMPTS = 4

No OpenAI API is involved. `.env` is **not** committed (gitignored); it lives at the repo root and
is loaded by `distil/run.py::_load_dotenv` (`run.py:236`) — note the sbatch itself never sources
it, so the credentials reach a compute node only through that call. Verified live on a compute
node: `[env] loaded /vast/.../.env` followed by `[llm-guard] OpenRouter asserted`.

### The silent-fallback hazard, and the guard

`p4/llm.py::make_llm()` returns `None` when `OPENAI_BASE_URL` is unset, and the loop then degrades
to the deterministic geometric planner **and still writes a plausible `result.json`**. The SDK key
also falls back to `OPENAI_API_KEY`, so a missing `OPENROUTER_API_KEY` could have addressed the real
OpenAI API with a Qwen model name. A misconfigured API therefore does not crash — it silently
produces numbers that are not DISEIL.

All 20 runs are submitted with **`DISEIL_STRICT_LLM=1`** (HPC-A's guard, commit `2795c456`), which
refuses to start unless `OPENROUTER_API_KEY` is set, `OPENAI_BASE_URL` is *exactly* the OpenRouter
endpoint, and `make_llm()` returned a live client; and which re-raises instead of swallowing a
mid-run LLM failure (429 / truncated reply) into a geometric round.

> **Exception, stated for honesty:** `Lift/state/seed1` (job 3398450) ran *before* HPC-A's guard
> was pushed, under a functionally equivalent guard of my own (`DISTIL_REQUIRE_LLM=1`, asserting the
> same key + exact base-url + non-None client, plus a live preflight ping) at commit `fed056ca`.
> Its log carries `REQUIRE_LLM preflight OK` instead of `[llm-guard] OpenRouter asserted`. A guard
> can only abort a misconfigured run; it cannot alter the method. The other 19 runs ran at
> `2795c456`. The run is valid and its audit is clean (10/10 demos LLM-prescribed, 0 fallbacks).

## 3. Protocol — confirmed against the code, not the prompt

| quantity | value | where |
|---|---|---|
| Budget **B** | 20 demos acquired **on top of** the bootstrap | `run.py:292` `final_demos = len(init) + budget` |
| **D** | 1 successful demo per round | `p4/loop.py:148-167` (one accepted demo per round) |
| Initial demos **Ni** | **Lift = 8, Wipe = 12** | `config.py` `TASKS` (the sbatch passes `NUM_INIT` empty, so the per-task config value wins) |
| Held-out eval | **fixed 100-episode set**, same seeds every round and across seeds/arms (`eval_seed_base = 5_000_000`, disjoint from collect/dagger/screen/bridge bands) | `config.py:42-43`, `p4/loop.py:74-76` |
| Retraining | **from scratch EVERY round** (1 retrain per acquired demo) | `p4/loop.py:70-72` |
| Bootstrap | shared, byte-identical per (task, modality), `seed_base=0` | `run.py:113-131` |
| Saturation early-stop | stop after 4 consecutive rounds with no usable failures | `p4/loop.py:110-113` (`saturation_patience=4`) |

No hyperparameter was changed.

### Two places where the CoC report's text does not match the code

1. **Retraining cadence.** `05_progress.md:150` states that on the robot tasks "retraining runs once
   every fourth acquired demonstration, so at D=1 the policy is refreshed five times over the
   budget." **The code retrains from scratch every round** for Lift/Wipe/Door — there is no cadence
   parameter anywhere in the `distil` module (`train_every_n=4` exists only in the legacy
   `CNN_pathway/` and `Equivariant_pathway/` trees). The *existing committed results also show this*:
   `Wipe/state/full/seed1` has 21 rounds with `n_demos_at_eval` = 12,13,…,32, i.e. one retrain per
   demo. The author has confirmed that **every-1-demo retraining is the intended protocol**, so the
   code is correct and the report's sentence needs fixing. The Table-8 caveat at `05_progress.md:168`
   ("on the robot tasks it can be up to three demonstrations stale, because retraining runs every
   fourth demonstration there") rests on the same wrong premise and describes staleness that never
   occurs. **Nothing was changed in the code.**

2. **Initial demonstrations.** `05_progress.md:154` states the initial set "holds twenty
   demonstrations for every task". The code — and the runs that produced the current results — use
   **Lift = 8, Wipe = 12** (`config.py`). These runs use the code values.

### Lift does not spend the full budget — by design

Lift reaches ~100% held-out success within a few rounds, screening then finds no usable failures,
and the saturation early-stop ends the run. Lift (state) seeds acquired **10, 7, 7, 7, 4** of B=20.
This is existing designed behaviour, not a truncated budget, and the budget was never reduced to
force a run to finish. It is why Lift is a poor cell for detecting a regression and Wipe is the
informative one.

> Consequence for the shared `cells/` folder: HPC-A's `distil/scripts/collect_rerun_cell.py`
> marks a seed valid only if `demos_acquired == budget` (`:173`). Applied to Lift that voids all
> five seeds and emits an empty cell. The cell files here therefore use
> **valid = zero fallback rounds AND (full budget acquired OR saturation early-stop fired)**, stated
> in each file's `note`.

## 4. How the runs were submitted

One job per (task, modality, seed). `Lift/state/seed1` was run and fully audited **first, alone**,
as a gate; the remaining 19 were released only after it passed.

    sbatch --job-name=or_<Task>_<mod>_s<seed> \
      --partition=gpu --exclude=rtxp6000l-f-01 --qos=batch-long --time=<per-cell> --mem=32G \
      --export=ALL,TASK=<Task>,MODALITY=<mod>,ABLATION=full,SEED=<seed>,BUDGET=20,\
    CONDA_ENV=diffdagger,DISEIL_STRICT_LLM=1,DISTIL_ROOT=<repo>,\
    OUTPUT_DIR=<repo>/distil/results/rerun_openrouter/<Task>/<mod>/full/seed<seed> \
      distil/scripts/run_distil.sbatch

Time limits are sized to **this** cluster: the existing runs' wall times came from `/weka` (HPC-A)
and are ~3.1x faster than rohan's V100s (measured: 12.6 min/round here vs ~4 min/round there). A job
killed at the wall loses everything (there is no resume), so limits are set long:
Lift/state 10 h, Lift/image 24 h, Wipe/state 24 h, Wipe/image 48 h.

Nothing is written outside the new namespace `distil/results/rerun_openrouter/`; no existing result
directory is on the write path.

### SLURM job IDs

| task | modality | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
|---|---|---|---|---|---|---|
| Lift | state | **3398450** (gate) | 3399115 | 3399116 | 3399117 | 3399118 |
| Lift | image | 3399119 | 3399120 | 3399121 | 3399122 | 3399123 |
| Wipe | state | 3399124 | 3399125 | 3399126 | 3399127 | 3399128 |
| Wipe | image | 3399129 | 3399130 | 3399131 | 3399132 | 3399133 |

Supporting jobs: shared bootstraps 3398441-3398444 (CPU partition, `osmesa`); smoke run 3398437
(CPU partition, Lift/state, `--smoke`, budget 1).

## 5. Pre-flight evidence (before any of the 20 runs)

* Live OpenRouter calls to **both** models from this cluster: `qwen/qwen3-32b` replied with real
  token usage; `qwen/qwen3-vl-30b-a3b-instruct` correctly described a synthetic image.
* The guard was tested negatively: with `OPENAI_BASE_URL` pointed at `api.openai.com` it **aborts**;
  with the base-url unset (`make_llm() -> None`, the exact silent-fallback trap) it **aborts**.
* A full smoke run completed end to end with **non-zero VLM (2,128) and LLM (4,841) tokens**, one
  prescription, zero fallback rounds.
* Bootstraps verified to contain exactly Ni demos (Lift 8, Wipe 12) with real rendered frames for
  the image modality.
* Clustering confirmed to use the real sklearn path (`cluster_method: sklearn-silhouette(k*=4)`),
  not the numpy fallback.

## 6. Verification applied to every run

`fallback_audit_hpcB.md` reports, per run, the number of rounds that used the LLM versus the
deterministic fallback, and the provenance of every acquired demonstration. Demo provenance is read
from the `[collect aN] ... choice=` lines of each `run.log` — it does **not** exist in `result.json`,
which records `mode` but never `choice`. Two paths spend real LLM tokens yet let the geometric rule
pick the demo, and both are counted honestly:

* `planner.py:243-248` — the model's output has no parseable `SELECT`/`BRIDGE` tag →
  `geometric_select`/`geometric_bridge`. **Counted as a fallback round.**
* `loop.py:161` — on a retry the LLM's label is discarded → `escalated_select`. This is DISEIL's
  designed in-round escalation after an infeasible prescription (`infeasible_attempts=4`), so it is
  **reported separately and not counted as a fault**.

A run whose `run.log` is missing or truncated is failed, not passed.
