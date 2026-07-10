# SETUP_HPC2 — bring-up on the SECOND HPC (shared-nothing cluster)

The 2nd HPC is a **different cluster that cannot see `/weka`**. It gets everything through **git**
(code) + **OpenRouter** (the LLM — no local weights) + its **own `.env`**. This is the copy-paste
checklist to stand it up and run its assigned `(task,modality)` cells. Context: `08_...md`.

> Prerequisite: the 1st HPC has already consolidated the code into the `distil/` module, vendored
> the custom envs, and pushed it to GitHub (`00` golden rules 1 & 8). If that isn't done yet, do it
> on HPC-A first — HPC-B only ever *consumes* the repo.

## 1. Clone the code (NOT `/weka`)
```bash
git clone git@github.com:SuyogKhanal/DmNfull.git        # or the dedicated distil repo if split out
cd DmNfull
git checkout <the-branch-with-distil>                    # e.g. stackcube-hybrid-plugcharger-handoff
export DISTIL_ROOT="$PWD"                                 # module uses this, never a hardcoded /weka path
```
If you hit an absolute `/weka/...` path anywhere at runtime, that's a **portability bug** — fix it
to `$DISTIL_ROOT`-relative (golden rule 8), don't symlink `/weka`.

## 2. Python env (pinned)
```bash
conda env create -f environment.yml -n distil    # or: pip install -r requirements.txt
conda activate distil
# heavy deps installed here, not shipped: torch(+CUDA for THIS cluster), robosuite, mani_skill,
# mujoco, diffusers/policy deps. Match CUDA to HPC-B's drivers, not HPC-A's.
python -c "import torch; print(torch.cuda.is_available())"   # must be True
```
Vendored custom envs (PushT-v2, robosuite wrappers) import from inside `distil/` — verify:
```bash
python -c "import distil; from distil.envs import pusht, lift, door, wipe, gridworld; print('envs ok')"
```

## 3. Secrets — the LLM key (never committed)
```bash
cp .env.example .env    # then edit:
#   OPENROUTER_API_KEY=sk-or-...            # HPC-B's own key (or the shared one)
#   VLM_MODEL_NAME=qwen/qwen3-vl-30b-a3b-instruct
#   LLM_MODEL_NAME=qwen/qwen3-32b
#   OPENAI_BASE_URL=https://openrouter.ai/api/v1
python -c "import os,openai; c=openai.OpenAI(base_url=os.environ['OPENAI_BASE_URL'], api_key=os.environ['OPENROUTER_API_KEY']); \
print(c.chat.completions.create(model='qwen/qwen3-32b', messages=[{'role':'user','content':'ping'}], max_tokens=8).choices[0].message.content)"
```
That last line is the **API smoke** — it proves the key + base_url + model slug work from HPC-B
before you burn any GPU time. (For the VLM, smoke a tiny `image_url` request too.)

## 4. Which cells does HPC-B own?
Read `HANDOFF_HPC2.md` (written by the HPC-A session) — it lists the exact remaining
`(task, modality, ablation, seed)` cells assigned here, their priority + est. cost, and the launch
command for each. **Rule: HPC-B runs WHOLE `(task,modality)` cells** (all arms + all 5 seeds of a
cell) so its locally-generated bootstrap stays byte-identical within the cell (`08_...md`). Never
run half a cell here and half on HPC-A.

## 5. Generate this cell's bootstrap locally, then smoke one round
```bash
# byte-identical shared bootstrap for the cell (deterministic; every arm of the cell loads it):
python -m distil.run --make-bootstrap --task <T> --modality <M> --bootstrap-dir results/shared_bootstrap
# 1-round smoke on 1 GPU + 1 cheap OpenRouter call BEFORE the matrix:
python -m distil.run --task <T> --modality <M> --ablation full --seed 1 --budget 1 --smoke \
       --bootstrap-dir results/shared_bootstrap
```
Smoke pass criteria: loads bootstrap, KAG loaded, VLM returns a description from the frames,
cluster→dominant→memory→decision (SELECT/BRIDGE + **confidence**) logged, ≤1 demo collected,
infeasibility loop wired, telemetry + `prompts/` + `kag/` written, no import/path/API errors.

## 6. Submit HPC-B's assigned cells (1 GPU/job, any partition — no h100/h200 constraint)
```bash
sbatch scripts/run_distil.sbatch --task <T> --modality <M> --ablation <A> --seed <s> --budget 20 \
       --bootstrap-dir results/shared_bootstrap
# one ablation branch = one job (golden rule 5); loop seeds inside the job or one job per seed.
```
Register every job in the shared **`RUN_STATE.md`** ledger (`{task,modality,ablation,seed,hpc,jobid,
status,result_path}`) and commit it, so HPC-A + the aggregator see one source of truth.

## 7. Push results back (git, light JSON only)
HPC-B can't write to `/weka`. After jobs finish:
```bash
git add results/**/result.json results/**/config.yaml RUN_STATE.md   # NEVER *.pt / telemetry / frames
git commit -m "HPC-B results: <cells>"
git push origin <results-branch>
```
Aggregation runs on HPC-A: it `git pull`s HPC-B's `result.json` leaves and builds the master table
(`09_...md`). Checkpoints/telemetry stay on HPC-B's local disk (they'd bloat the repo — the 548 GB
incident).

## Gotchas specific to a second cluster
- **CUDA/driver mismatch:** build torch for HPC-B's drivers; don't copy HPC-A's conda env.
- **No `/weka` fallbacks:** if code reaches for `/weka`, it's a bug — patch to `$DISTIL_ROOT`.
- **Separate OpenRouter quota/rate limits:** two clusters hitting the same key doubles RPM; if you
  see 429s, split keys or add backoff (the client must handle 429/5xx → deterministic fallback).
- **Clock/queue differ:** re-estimate per-cell wall-clock here; HPC-B partitions may start faster
  or slower than HPC-A — rebalance the cell assignment if needed and update `HANDOFF_HPC2.md`.
