# DISTIL — PORTABILITY / HPC-B notes (read before running on a second cluster)

The DISTIL module is self-contained code with **no hardcoded `/weka` path** and **no
local model weights** (the LLM is an OpenRouter API call). But two host-level
dependencies are NOT carried by a `git clone` and must be reproduced on HPC-B. This
file is the explicit ledger of those (golden rule 8; `08_..md` / `SETUP_HPC2.md`).

## ⚠ 1. `robosuite` is a MODIFIED-PAST-TAG (but PUBLIC) install — pin the commit
On HPC-A the `diffdagger` conda env has `robosuite` installed **editable** from a git
checkout whose `import robosuite` resolves to
`/weka/s226137394/diff-dagger-ur5/repo/robosuite/__init__.py`.

- Reported version is `1.5.2`, **but HEAD is `v1.5.2-8-g85abee22`** — i.e. the v1.5.2
  tag **plus 8 commits**. `pip install robosuite==1.5.2` is therefore **NOT equivalent**
  (it is missing those 8 commits, including `232ce7d4 "synchronize rendering context
  for envs"`, which the **offscreen VLM frame render needs** — without it the multi-GL-
  context render can crash or corrupt frames).
- Good news: those 8 commits are all authored by the ARISE-Initiative team and
  **`origin/HEAD → origin/master → 85abee22` is PUBLIC**. So HPC-B does **not** need a
  private fork — it pins the public commit:

  ```bash
  pip install "git+https://github.com/ARISE-Initiative/robosuite.git@85abee22"
  ```

  (This is exactly what `distil/requirements.txt` and `distil/environment.yml` already
  pin.) Pin the **commit hash `85abee22`**, not `@master` — `master` is a moving branch
  and will drift.
- Verify on HPC-B after install:
  ```bash
  python -c "import robosuite; print(robosuite.__version__)"          # 1.5.2
  python -c "import robosuite, subprocess, os; print(os.path.dirname(robosuite.__file__))"
  # must NOT be a /weka path on HPC-B; must be inside HPC-B's own site-packages/venv.
  ```
- If `85abee22` ever ages off ARISE `master` and the pin fails to fetch: `git clone`
  ARISE robosuite, `git checkout 85abee22`, and either `pip install -e .` it or vendor
  it under `distil/vendor/robosuite/` and add that to `PYTHONPATH`. It is public code, so
  vendoring is allowed; we just prefer the pin to keep the repo light.

## 2. `torch` CUDA build must match HPC-B's driver
HPC-A uses `torch==2.4.1+cu121`. **Do NOT copy HPC-A's conda env** to HPC-B — build/
install torch for HPC-B's local CUDA driver (right `--index-url`), then reinstall the
rest from `requirements.txt`. Confirm `python -c "import torch; print(torch.cuda.is_available())"`.

## 3. Offscreen render needs a GL backend
The VLM sees rendered frames via robosuite offscreen render. On a GPU node set
`MUJOCO_GL=egl` (the `run_distil.sbatch` default; `distil/envs.py` also defaults it). On
a CPU-only login node use `MUJOCO_GL=osmesa`. If frames come back black/empty, this is
the first thing to check.

## 4. `scikit-learn` must be present (reproducibility, not just portability)
`distil/p4/clustering.py` uses sklearn Agglomerative + silhouette (Eq 8) when importable,
**else it silently falls back to a numpy single-linkage path** — a *different* clustering.
For a byte-consistent matrix, install `scikit-learn==1.6.1` on **every** cluster (it is in
`requirements.txt`). The run log line `sklearn-silhouette(k*=…)` vs `numpy-silhouette(…)`
tells you which path ran — it must say `sklearn` everywhere.

## 5. Secrets: `.env` with the OpenRouter key (never committed)
Copy `distil/.env.example` → `.env` at the repo root (or `$DISTIL_ROOT`) and fill
`OPENROUTER_API_KEY`. `distil/run.py` loads it automatically (no hardcoded path). The
`.env` API smoke in `SETUP_HPC2.md §3` proves the key + base_url + slugs before any GPU
time is spent.

## What is NOT a portability risk (carried by the clone)
- The DISTIL code, KAG graphs (`distil/p4/kag/*.json`), and prompts travel with git.
- No `mani_skill` / `r3m` / `vllm` needed for the Phase-1 robot-state tasks (Lift/Wipe/
  Door). PushT-v2 (ManiSkill fork) is Phase 2 and WILL add a vendored custom env — that
  env must be vendored inside `distil/` when PushT is ported (it is NOT on a public pin).
