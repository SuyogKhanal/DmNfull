# D5 — GridWorld 5x5, IMAGE modality: how to run one round of DISEIL and of SafeDAgger

Status: **runnable now** (it was not when this recon started). No jobs submitted in this
phase. Everything below was verified by executing it, except where marked UNMEASURED.

---

## 1. What I found (the two blockers, both confirmed by execution)

The consolidated module `distil/` is the right stack — it is the one the three live
RoboSuite D5 jobs (110355–110360) use, and it already logs everything D5 needs. But
GridWorld/image could not run, for two reasons:

| # | Blocker | Evidence |
|---|---|---|
| 1 | GridWorld image modality not built | `distil/config.py:183` raised `NotImplementedError`; `distil/gridworld/policy.py:55` asserted `modality == "state"`. `gridworld/rgb_policy.py` (the plain RGB CNN) and `gridworld/encoder_rgb.py` existed but **were imported by nothing**, and `encoder_rgb.py` imported `from envs.maze_env import …` (repo-root package), not the in-package `.maze_env`. |
| 2 | GridWorld has **no baseline arm at all** | `distil/run.py` routed *every* GridWorld run to `_run_gridworld` → `run_distil_gridworld` **before** the `BASELINE_ARMS` check. So `--task GridWorld --ablation safe` did not fail — it **silently ran full DISEIL** and would have produced a "SafeDAgger" row that was actually a second DISEIL row. `distil/baselines.py` is diffusion-policy/robot-only (`get_action`, obs windows) and cannot drive the GridWorld classifier. |

The legacy stack (`Equivariant_pathway/…/pool_x_selector`) was rejected as the source
for this row: its P4 pipeline has **no VLM stage, no KAG block and no token accounting**
(`pipeline_p4.py` shells out to a local-Qwen vLLM proxy via `runner.run_analysis`; its
`history` rows carry no `sec` and no token fields), and its policy is the
equivariant/CNN *hybrid*, not the plain CNN. Numbers from it would neither be the method
the paper describes nor comparable with the Door/Wipe rows.

## 2. The wiring (additive, non-breaking; state path byte-identical)

| File | Change |
|---|---|
| `distil/gridworld/encoder_rgb.py` | import `.maze_env` instead of the repo-root `envs.maze_env`. Verified the two modules' `TILE_*` / `AGENT_COLOR` / `TILE_COLORS` constants are **identical**, so the raster is unchanged. The module had no importers, so this cannot regress anything. |
| `distil/gridworld/policy.py` | `modality="image"` → `RGBCNNPolicy(in_channels=3, num_actions=4)` + the RGB encoder, via a new `_new_net()`. Same 4-way logit head ⇒ entropy self-uncertainty, α-quantile threshold, K-patience and the multi-label BCE objective are **unchanged**. The `state` branch builds exactly the same `EquivariantUNetPolicy` as before. |
| `distil/config.py` | dropped the `NotImplementedError` for (GridWorld, image). |
| `distil/gridworld/baselines.py` | **NEW.** SafeDAgger\* + Stagger on GridWorld, gates ported verbatim from `pool_x_selector/selection/iil_baselines.py`. |
| `distil/run.py` | `_run_gridworld` now routes `ablation ∈ BASELINE_ARMS` to the baseline loop (fixes blocker 2). Default `full` path untouched. |
| `distil/compute_log.py` | **NEW.** The per-round compute side-file (§4). Writes `telemetry/compute.jsonl`; **does not touch `result.json`'s schema**. |
| `distil/scripts/run_gridworld_d5.sbatch` | **NEW.** The wrapper (§3). |
| `distil/scripts/build_d5_compute.py` | removed the now-false `BLOCKED[("GridWorld","image")]` entry (the cell reports PENDING until the two leaves land). PushT's entry left untouched. |

**Fairness of the two arms.** Both load the *same* bootstrap demos (GridWorld demos are
layouts + cells, so the pickle is modality-independent and literally shared), screen the
*same* per-round pool (identical seed band `screen_seed_base + rnd*n_screen + i`),
retrain from scratch with the same steps/LR/batch, evaluate on the same frozen 200
held-out layouts, add exactly **one** successful demo per round, and use the **same demo
primitive** (`collect_select_gw`: BFS-optimal path from the takeover cell). The only
difference is *which* failure and *which* takeover cell get the demo — which is exactly
the claim under test.

SafeDAgger\* gate (`baselines.py`, ported): score = fraction of rollout steps whose
action is off the A\* optimal-action set; queryable when score > `tau = 0.10`
(`config_baselines.yaml: baselines.safe.tau`); pick argmax score among queryable
(fallback = top-scoring, so a round with signal is never wasted); expert takes over at
the **first** off-optimal step.

## 3. The two exact commands (one round each)

From the repo root `/weka/s226137394/DmNfull`. `BUDGET=5` matches the live Door/Wipe D5
jobs; the estimator drops round 0, so a budget of ≥2 is the minimum and 5 gives ~5
comparable rounds.

```bash
# (a) DISEIL  — code arm `full` (VLM + KAG + reasoning LLM + Eq-9 allocation)
sbatch --job-name=d5_GridWorld_image_full \
  --export=ALL,MODALITY=image,ABLATION=full,SEED=1,BUDGET=5 \
  distil/scripts/run_gridworld_d5.sbatch

# (b) Baseline — code arm `safe` (SafeDAgger*), matching the three RoboSuite D5 jobs
sbatch --job-name=d5_GridWorld_image_safe \
  --export=ALL,MODALITY=image,ABLATION=safe,SEED=1,BUDGET=5 \
  distil/scripts/run_gridworld_d5.sbatch
```

Outputs land at `distil/results/_compute/GridWorld/image/{full,safe}/seed1/` — the exact
layout `build_d5_compute.py` reads.

Local equivalent (what I actually executed to verify, CPU, no SLURM):

```bash
python -m distil.run --task GridWorld --modality image --ablation full --seed 1 --budget 1 --smoke \
  --bootstrap-dir <dir>/bs --output-dir <dir>/full
python -m distil.run --task GridWorld --modality image --ablation safe --seed 1 --budget 1 --smoke \
  --bootstrap-dir <dir>/bs --output-dir <dir>/safe
```

Both completed; the `state` modality was re-run as a regression and is unaffected.

## 4. Is per-round wall-clock / token usage already recorded?

**Tokens: yes, natively — I instrumented nothing.** `DistilLLM.decide()`
(`distil/p4/llm.py`, class `_Usage`) already records, per round, into
`result.json → history[].tokens`: `prompt / completion / total / calls`, `reasoning`
(the hidden thinking tokens, from `usage.completion_tokens_details.reasoning_tokens`),
a `by_stage` split into **vlm / analysis / decision**, and `kag_calls` + `kag_chars`.
The GridWorld loop already stored that dict per round. Verified live against OpenRouter
on GridWorld/image (smoke round, `analyze_cap=2`):

```json
{"prompt":4674,"completion":1894,"total":6568,"calls":5,"reasoning":1566,
 "by_stage":{"vlm":{"total":1115,"calls":2},"analysis":{"total":3550,"reasoning":1064,"calls":2},
             "decision":{"total":1903,"reasoning":502,"calls":1}},
 "kag_calls":3,"kag_chars":3272}
```

**Total wall-clock per round: yes, natively** — `history[].sec` (already in the loop).

**Reasoning-only wall-clock per round: NO — this is the one thing I added.** A round is
dominated by the from-scratch RETRAIN, which *both* arms pay, so `sec` alone does not
characterise DISEIL's overhead. New side-file `telemetry/compute.jsonl`, one row/round,
written by both arms (`distil/compute_log.py`):

```
sec_total, sec_train, sec_eval, sec_screen, sec_llm, sec_prescribe,
sec_reasoning = sec_screen + sec_llm + sec_prescribe,  tokens{…}
```

Measured on the live smoke round above (CPU, smoke sizes — **not** the production
number): `sec_total=41.1, sec_train=2.7, sec_eval=0.3, sec_screen=1.1, sec_llm=36.9,
sec_prescribe=0.002 → sec_reasoning=38.0`. The baseline arm writes the same row with
`sec_llm=0` and `sec_screen` = its own gate cost, so the overhead is a difference of two
measured quantities, not an estimate.

## 5. KAG token contribution (measured)

`distil/scripts/measure_kag_tokens.py` sends the same prompt with and without the KAG
block at `max_tokens=1` and diffs `usage.prompt_tokens` from the **serving** tokenizer
(`qwen/qwen3-32b`). Executed for GridWorld:

```
[GridWorld] kag_chars=3272  analysis: 301 -> 1169 (+868)  decision: 582 -> 1453 (+871)
```

→ **tokens(KAG block) ≈ 870 per KAG-bearing call.** Per-round contribution =
`kag_calls × 870`, and `kag_calls` (= #analysed failures + 1) is logged per round. In a
production round (`analyze_cap=3`) that is 4 calls ≈ **3.5k tokens/round**, i.e. the KAG
is the single largest prompt-side term. Regenerate the shared file with GridWorld
included before building the table:

```bash
python -m distil.scripts.measure_kag_tokens --tasks Door Wipe GridWorld \
  --out paper_aaai2027/COC_REPORT/build/kag_tokens.json
```

## 6. What is still UNMEASURED, and why

* **Production s/round, tokens/round and Overhead× for this cell.** The jobs have not
  been submitted (explicitly out of scope for this phase). The smoke numbers in §4 are
  CPU + smoke-sized (`initial_train_steps=40`, `screen_episodes=6`, `analyze_cap=2`,
  `eval_episodes=3`) and **must not** be reported as the D5 row. Run §3, then
  `python distil/scripts/build_d5_compute.py`.
* **The other four when-to-query baselines on GridWorld** (dropout / ensemble / thrifty /
  diffdagger). `baselines.py` raises `NotImplementedError` naming them: their gates need
  MC-dropout, an M-member ensemble, or a success-Q head on the classifier. Not needed for
  D5 (baseline arm = SafeDAgger) and not ported.
