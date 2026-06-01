# claude_context.md — `pool_x_selector` suite

You (a future Claude Code session) are picking up an in-flight research
project preparing a **top-A* conference paper**. Read this file end-to-end
before suggesting changes. The suite is fully built, debugged, has run at
production scale, and has a working paper-figure pipeline. This file
contains everything you need to be useful immediately — the design, the
deliberate user-curated knob choices, the bug-fix history, the data-product
catalog, the gotchas already paid for, and the things you must not
re-discover or revert.

The original parent suite this fork came from is
`/weka/s226137394/DmNfull/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/baseline_vs_p4_sequential_batch/` —
its `claude_memory.md` is the canonical reference for the cluster/serving
infrastructure that **still applies here unchanged**. Read its gotcha
catalog when in doubt; §11 below summarises what's load-bearing.

---

## 1. The user

- PhD student. Username `s226137394`. Repo lives at
  `/weka/s226137394/DmNfull/` on the open-source-model (Qwen / vLLM)
  cluster. Conda env: `maze`
  (`/home/s226137394/.conda/envs/maze/bin/python`).
- Action mode. Terse responses; don't restate the diff. Pushes back when
  something feels wrong and **explicitly invites disagreement** — surface
  real downsides rather than silently comply.
- Reviewer-defensible framing for an A* paper: **lead with
  sample-efficiency, own ties honestly, never overclaim**, and
  **no claim duplication across figures/tables** (one claim → one
  artefact; if a table already shows it, don't show it again as a figure).
- Email for SBATCH: `s226137394@deakin.edu.au`.

---

## 2. The experiment in one paragraph

This suite compares **8 demonstration-acquisition methods** on a 2-axis
grid for a 5×5 maze behaviour-cloning task with an
`EquivariantCNNHybridPolicy`. The two axes are **pool** (fixed once per
run vs rotated fresh per round) × **selector** (baseline highest-loss /
baseline random / P4 top-3 failures / P4 all failures). Both P4 variants
prescribe **exactly ONE** layout/round (`mode_max=1`), so the only thing
varying between methods is *how the single demonstration is chosen / how
much failure information the LLM sees*. The held-out evaluation set has
200 layouts (production) or 10 (smoke). The correction pool is size 20
(production) — this is a **load-bearing constraint** for the Qwen 40960
context budget (see §11).

---

## 3. The 8 methods (full pool × selector grid)

Method names follow `<selector>_<pool>`:

| # | Method                       | Kind     | Selector            | Pool    |
|---|------------------------------|----------|---------------------|---------|
| 1 | `baseline_highloss_fixed`    | baseline | `highest_loss`      | fixed   |
| 2 | `baseline_random_fixed`      | baseline | `random`            | fixed   |
| 3 | `baseline_highloss_rotate`   | baseline | `highest_loss`      | rotate  |
| 4 | `baseline_random_rotate`     | baseline | `random`            | rotate  |
| 5 | `p4_top3_fixed`              | p4       | `top_k=3`           | fixed   |
| 6 | `p4_top3_rotate`             | p4       | `top_k=3`           | rotate  |
| 7 | `p4_all_fixed`               | p4       | `top_k=None` (all)  | fixed   |
| 8 | `p4_all_rotate`              | p4       | `top_k=None` (all)  | rotate  |

The dispatch table that ties this together is `METHOD_SPEC` at the top of
[orchestrator/run_one.py](orchestrator/run_one.py). **Five places must
stay in lockstep when adding/removing a method**:
1. `config.yaml::methods` (the list orchestrator iterates).
2. `orchestrator/run_one.py::METHOD_SPEC` (dispatch).
3. `orchestrator/workspace.py::METHOD_DIR_NAMES` (per-run dir creation).
4. `aggregation/aggregate.py::METHOD_DIRS` / `METHOD_COLORS` /
   `METHOD_LABELS` (cross-run aggregation + figures).
5. `nb_plot.py::METHODS` / `LABELS` / `COLORS` (paper-grade plotting).

Comparison groups for the paper: (A) the 4 **fixed**-pool methods share
one pool per run → apples-to-apples within axis. (B) the 4 **rotate**-pool
methods share each round's pool → apples-to-apples within axis. (C)
fixed-vs-rotate per selector. The shared 200-layout held-out set is the
common yardstick across groups.

---

## 4. Repository topology

This suite is a self-contained sibling of `variants_fixed_vs_p4top3/` at
the same depth (depth-1 under `baseline_vs_p4/`). Mark per entry:
`NEW` = added in this session, `EDITED` = modified from the fork,
`KEPT` = verbatim from fork, `DEL` = deleted from fork.

```
pool_x_selector/
├── claude_context.md             NEW   (this file)
├── config.yaml                   EDITED  correction_n=20, methods=8, header
├── nb_plot.py                    EDITED  8 methods + paper LABELS/COLORS (§7)
├── paper_data.py                 NEW    data loaders + toughness + gallery renderer
├── paper_figures.ipynb           NEW    11-cell paper-figure notebook
├── __init__.py                   KEPT   (empty)
│
├── orchestrator/
│   ├── __init__.py               KEPT
│   ├── bootstrap.py              KEPT   upstream-shared init demos + ckpt mirror
│   ├── run_one.py                EDITED  METHOD_SPEC dispatch (table-driven)
│   └── workspace.py              EDITED  METHOD_DIR_NAMES = 8 names
│
├── layouts/
│   ├── __init__.py               KEPT
│   ├── contamination.py          EDITED  KNOWN_METHODS = 8 names
│   └── layout_setup.py           EDITED  SUITE_ROOT path rename
│
├── selection/
│   ├── __init__.py               KEPT
│   ├── rank.py                   KEPT   (-loss, -n_steps, jitter) ranker
│   └── baseline_dagger.py        EDITED  uniform prescribed_loss.json write; package rename
│
├── p4/
│   ├── __init__.py               KEPT
│   ├── demo_collector.py         KEPT   corridor-aware demo recorder
│   ├── pipeline_p4.py            NEW    unified P4 loop (top_k + fixed/rotate, helpers local)
│   ├── prompts.py                EDITED  added top3/all mode_directives w/ OUTPUT REQUIREMENT (§6)
│   ├── runner.py                 KEPT   wraps upstream run_profile_analysis
│   ├── _p4_common.py             DEL    superseded by pipeline_p4.py
│   ├── pipeline_seq.py           DEL    superseded by pipeline_p4.py
│   ├── pipeline_batch.py         DEL    superseded by pipeline_p4.py
│   └── pipeline_top3.py          DEL    superseded by pipeline_p4.py
│
├── corridor/
│   ├── blocker.py                KEPT   "(r,c)->(r,c)" corridor parser + A* mask
│   └── expert_constrained.py     KEPT
│
├── trainer/
│   ├── __init__.py               KEPT
│   └── finetune_replay.py        EDITED  package path rename
│
├── logging_ext/
│   ├── __init__.py               KEPT
│   ├── compression_log.py        EDITED  new column prescribed_loss_mean (§9)
│   ├── prescription_overlap.py   KEPT
│   └── training_log.py           KEPT
│
├── aggregation/
│   ├── __init__.py               KEPT
│   └── aggregate.py              EDITED  8-method METHOD_DIRS + P4_METHODS const + bug fix
│
├── qwen/
│   ├── __init__.py               KEPT
│   └── proxy.py                  EDITED  package path rename only
│
├── submit_one.sh                 EDITED  package path rename
├── submit_one_qwen.sh            EDITED  package rename + production knob shift (§7)
├── submit_all.sh                 EDITED  cosmetic
├── submit_smoke.sh               EDITED  HELDOUT_N=10, METHODS=8, audit hints, --time=8h
├── submit_aggregate.sh           EDITED  package rename
├── run_all.sh                    EDITED  deprecated local launcher; SMOKE METHODS fixed
│
├── results/                      gitignored
│   ├── run_{1..10}/              production runs
│   │   ├── shared/
│   │   │   ├── fixed_pool/correction_layouts.yaml   (fixed methods read)
│   │   │   ├── round_NNN/correction_layouts.yaml    (rotate methods read)
│   │   │   ├── init_demos/                          (20 BFS demos, copied from upstream)
│   │   │   └── init_checkpoints/                    ({best,last}_hybrid_policy.pth)
│   │   ├── <method>/
│   │   │   ├── checkpoints/                         (warm-started + per-round saves)
│   │   │   ├── demos/round_NNN/*.json               (collected corrective demos)
│   │   │   └── results/
│   │   │       ├── learning_curve.json              (history list)
│   │   │       ├── training_log.csv
│   │   │       ├── compression_log.csv              (P4 only; has prescribed_loss_mean)
│   │   │       └── round_NNN/
│   │   │           ├── ranking.json                 (baselines only)
│   │   │           ├── prescribed_loss.json         (NEW: all methods, §9)
│   │   │           ├── failures_layouts.yaml        (P4)
│   │   │           ├── failures_rollout/            (P4 — see §6)
│   │   │           ├── p4_analysis/                 (P4 LLM output)
│   │   │           │   ├── recommended_layouts.json
│   │   │           │   ├── p4_seq_seqbatch_prescription_report.json
│   │   │           │   └── full_output.json
│   │   │           ├── collect_summary.json
│   │   │           └── heldout_eval/
│   │   └── run_summary.json
│   └── aggregate/
│       ├── contamination_report.json
│       ├── summary.json
│       ├── compression_summary.csv
│       ├── figures/{sr_vs_demos_added,sr_vs_training_rounds,compression_dist}.png
│       ├── figures/paper/                            (NEW: paper_figures.ipynb output)
│       │   ├── F1_headline_sr_vs_demos.png
│       │   ├── F3_info_gain_distribution.png
│       │   ├── F4_failures_shown_per_round.png
│       │   ├── F5_infeasibility_rate.png
│       │   ├── F6_layout_distribution.png
│       │   ├── F7_qualitative_gallery.png
│       │   ├── F8_recommendation_evolution.png
│       │   ├── F9_per_run_timeline_run_<N>.png
│       │   ├── T1_final_sr_per_method.{md,tex}
│       │   └── T2_compression_system.{md,tex}
│       └── learning_curves_by_method/                (NEW: user requested)
│           └── <method>/learning_curve_run_{1..10}.json   (80 files total)
├── slurm_logs/                                       (SBATCH stdout/stderr land here)
└── logs/                                             (per-run orchestrator stdout via tee)
```

`UPSTREAM-REUSED` modules (NOT in this dir; imported read-only — never
modify):
- `Equivariant_pathway.layout_sampler` — `sample_layouts`,
  `_load_blocked_signatures`, `_signature`, `write_yaml`.
- `Equivariant_pathway.expert` — `AStarExpert`, `build_grid`,
  `compute_distance_map`, `optimal_action_mask`.
- `Equivariant_pathway.equivariant_CNN_hybrid.{model,dataset,train}` —
  `EquivariantCNNHybridPolicy`, `HybridDemoDataset`,
  `collate_fixed_size`.
- `Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.{baseline_budget,p4_budget}` —
  `_rollout_with_loss`, `_save_corrective_demo`, `_eval_heldout`,
  `_rollout`, `_read_sr`, `_count_demos`, `_load_correction_layouts`,
  `_load_model`, `_find_deviation_step`, `REASONING_ADDENDUM_BASE`,
  `AGGREGATOR_ADDENDUM_BASE`, `_budget_addendum`.
- `Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.nb_helpers` —
  `_layout_rgb` (matplotlib-friendly maze rendering). Used by
  `paper_data.render_gallery()` via deferred import (§8).
- `Equivariant_pathway.collect_demos` — `_build_forced_env`,
  `_record_one`.
- `Equivariant_pathway._analysis_common` — `run_profile_analysis`.
- `pipeline/` package — the upstream LLM analysis pipeline
  (`_oai_retry.py`, `reasoning.py`, `aggregator.py`, `vlm_analyser.py`,
  …). Reads `OPENAI_BASE_URL` from env at import time.

**Hard rule:** never edit any of the above. If something upstream needs
adapting, COPY into `pool_x_selector/` and adapt the copy (see §14).

---

## 5. Key design decisions (the "why")

### 5a. Pool axis — fixed vs rotate
Both primitives live in `layouts/layout_setup.py` and were already in the
fork — no new code:
- `ensure_correction_layouts_for_run(run_id, correction_n, train_yaml,
  heldout_yaml, out_dir)` — fixed: seed `91_000_000 + run_id`, sampled
  once per run, writes to `shared/fixed_pool/correction_layouts.yaml`.
- `ensure_correction_layouts_for_round(run_id, round_idx, correction_n,
  train_yaml, heldout_yaml, run_shared_dir)` — rotate: seed
  `91_000_000 + run_id*1000 + round_idx`, sampled per round, blocked
  against train+heldout+all prior rounds of the same run, writes to
  `shared/round_NNN/correction_layouts.yaml`. Idempotent — methods
  sharing `(run_id, round_idx)` re-read the same yaml.

`orchestrator/run_one.py::main()` samples the fixed pool **only when at
least one `*_fixed` method is selected** (`needs_fixed` guard); rotate
methods sample inside their own round loop.

### 5b. Selector axis
- **Baselines** — `selection/baseline_dagger.py::run(...)` already
  supports BOTH pool modes (`fixed_pool_yaml: Optional[Path]` — set =
  fixed, None = call `ensure_correction_layouts_for_round` per round)
  and BOTH selections (`selection: "highest_loss" | "random"`). No
  logic change in this session — only adds the uniform
  `prescribed_loss.json` write next to `ranking.json` (§9).
- **P4** — one unified module
  [p4/pipeline_p4.py](p4/pipeline_p4.py) parameterized on
  `top_k: Optional[int]` (3 = top-3 failures; None = all failures) and
  `fixed_pool_yaml: Optional[Path]` (set = fixed; None = rotate).
  `mode_max=1` is hard-coded so BOTH variants prescribe exactly 1
  layout/round. Helpers `_finetune` and `_persist_curve` live INSIDE
  `pipeline_p4.py` (they were moved out of the deleted
  `_p4_common.py`). Stop semantics:
  `target_hit` / `budget_exhausted` / `max_rounds` apply to all;
  `pool_solved` (no failures remain) fires **only when
  `fixed_pool_yaml is not None`** — rotate mode continues-on-empty via
  `consecutive_empty`/`no_progress` (default cap
  `max_consecutive_empty=8`).

### 5c. Dispatch — `METHOD_SPEC` table
`orchestrator/run_one.py` near the top defines:
```python
METHOD_SPEC = {
  "baseline_highloss_fixed":  ("baseline", "highest_loss", "fixed"),
  "baseline_random_fixed":    ("baseline", "random",       "fixed"),
  "baseline_highloss_rotate": ("baseline", "highest_loss", "rotate"),
  "baseline_random_rotate":   ("baseline", "random",       "rotate"),
  "p4_top3_fixed":            ("p4", 3,    "fixed"),
  "p4_top3_rotate":           ("p4", 3,    "rotate"),
  "p4_all_fixed":             ("p4", None, "fixed"),
  "p4_all_rotate":            ("p4", None, "rotate"),
}
```
The dispatch loop reads `(kind, sel_or_k, pool)` and routes to
`_run_baseline(..., selection=sel_or_k, fixed_pool_yaml=pool_yaml)` or
`_run_p4(..., method_name=m, top_k=sel_or_k, fixed_pool_yaml=pool_yaml)`.
Adding a 9th method is the 4-line change called out in §3.

### 5d. Replay-buffer fine-tuning
`trainer/finetune_replay.py` (subprocess) — `WeightedRandomSampler` over
`D_old ∪ D_new` with target `replay_mix=0.5`, floor
`replay_mix_floor=0.2`, warm-started weights from
`last_/best_hybrid_policy.pth` (optimizer/scheduler NOT restored —
matches upstream `train.py`). Same fine-tuner for baselines and P4.

### 5e. Corridor blocking
`corridor/blocker.py` parses `"(r,c)->(r,c)->…"` corridor strings and
walls off every FREE cell outside the corridor before A* runs. Infeasible
corridors (step on fire / not 4-adjacent / out of bounds / wrong
endpoints / disconnected) are logged in `compression_log.csv` and fed
back to subsequent rounds' prompts via
`prompts.infeasible_feedback_block`.

---

## 6. The empty-prescription bug + fix — **CRITICAL CONTEXT**

This is the most important thing to internalise from this session.

### 6a. The symptom
For `p4_all_fixed` / `p4_all_rotate` when the LLM was shown many failures
(≥10), the aggregator returned a SHELL prescription:
```json
{
  "failure_clusters": [{"cluster_label": "left_edge_fire_timeout", "episodes_in_cluster": [0,1,2,3,5,7,11], ...}],
  "demonstration_prescriptions": [
    {"cluster": null, "n_demos_needed": null, "recommended_layouts": [], "rationale": ""}
  ],
  "total_demonstrations_needed": 2
}
```
`failure_clusters` was populated and `total_demonstrations_needed >= 1`,
but the inner `recommended_layouts: []` was **empty** — violating the
upstream `AGGREGATOR_ADDENDUM_BASE`'s HARD FLOOR
("the response MUST contain at least one … recommended_layout"). Round
outcome: `prescribed=0 kept=0 collected=0`, budget wasted, the run is
effectively crippled.

### 6b. What this is NOT (the "0/12 success=False" red herring)
A future session inspecting the slurm `.out` will see blocks like:
```
[hybrid-rollout] ep 0 (corr_r101_007): steps=60 success=False
…
[hybrid-rollout] ep 11 (corr_r101_009): steps=60 success=False
[hybrid-rollout] DONE successes=0/12
```
**This is NOT the bug.** Look at the immediately preceding log lines —
this `[hybrid-rollout]` block runs `rollout_test --layouts
failures_layouts.yaml`, i.e. the policy re-rolls out the 12 KNOWN
failures so the LLM analysis pipeline has rendered frames + trajectories
to ingest. They were just identified as failures (line printed by the
loss-rollout in `pipeline_p4`: `pool=16 successes=4 failures=12`). Of
course rolling them out again with the same policy + same seed produces
the same 12 failures. **The expert/BFS is fine** — see
`compression_log.csv` for the actual demo-collection result
(`n_demos_collected`, `n_corridor_infeasible`). Do not chase this.

### 6c. The fix (load-bearing — do NOT weaken)
[p4/prompts.py](p4/prompts.py) `mode_directive("top3")` and
`mode_directive("all")` now each include an explicit OUTPUT REQUIREMENT
block. The "all" version reads (truncated):

> ALL-FAILURES MODE — you are shown ALL failures from this round's
> correction-pool rollout. Prescribe EXACTLY ONE layout / demonstration…
>
> OUTPUT REQUIREMENT (strict, non-negotiable):
> - The single recommended_layout MUST be CONCRETE and FULLY specified:
>   provide start_pos: [r, c], goal_pos: [r, c], fire_positions:
>   [[r, c], ...] (all integers in 0..4), AND a non-empty `steps`
>   corridor string of the form "(r,c)->(r,c)->..." from start to goal.
> - DO NOT emit an empty recommended_layouts list, a null cluster, or
>   null n_demos_needed. An incomplete prescription is treated as ZERO
>   prescriptions and this round will collect ZERO demos.
> - When in doubt with many failures shown, pick ONE failure episode
>   from the LARGEST cluster verbatim (copy its start/goal/fires) and
>   design a safe corridor that avoids every fire cell. Better to
>   prescribe a known-failing layout than to emit nothing.

After the fix: smoke runs 103/104 — every one of 32 method-rounds
(`8 methods × 2 runs × 2 rounds`) reported
`prescribed=1 kept=1 collected=1 infeasible=0`. Production then ran 10×
clean.

**Hard rule:** never weaken this block. If you change anything in
`p4/prompts.py`, re-run the smoke and verify `p4_all_fixed`/
`p4_all_rotate` still produce non-zero prescriptions at large failure
counts (≥10). If they don't, the prompt regressed.

---

## 7. Recent user-curated changes (intentional — do NOT revert)

The user edited these files after the smoke validation. Treat them as
authoritative; if you must change them, ASK first.

### 7a. `submit_one_qwen.sh` — production knob shift
At the top of the env-default block:
- `BUDGET="${BUDGET:-9999}"` — was 15 (config default). 9999 effectively
  disables budget-based stopping; methods now stop only on
  `target_hit` / `max_rounds` (or `pool_solved` for fixed) /
  `no_progress`. This lets the sweep explore the saturation regime.
- `TARGET_SR="${TARGET_SR:-1.0}"` — was 0.90 (config default). Combined
  with `BUDGET=9999` this means methods run until **every** held-out
  layout is solved or `max_rounds` is hit.
- `ROUND_EPOCHS="${ROUND_EPOCHS:-600}"` — P4 fine-tune epochs/round
  (mapped to `--p4_finetune_epochs`). Was unset (would inherit
  `trainer.finetune_epochs=20`). 600 is a heavy fine-tune.
- `BASELINE_ROUND_EPOCHS="${BASELINE_ROUND_EPOCHS:-90}"` — baseline
  fine-tune epochs/round (mapped to `--baseline_finetune_epochs`).
  Heavier than the previous 20.
- Walltime stays `#SBATCH --time=120:00:00`.

### 7b. `nb_plot.py` — paper-grade LABELS and COLORS
`METHODS` is unchanged (still 8 names). `LABELS` and `COLORS` have been
curated for paper presentation:

```python
LABELS = {
    "baseline_highloss_fixed":  "baseline highest-loss (fixed)",
    "baseline_random_fixed":    "baseline random (fixed)",
    "baseline_highloss_rotate": "DAgger-Discrete highest-loss",
    "baseline_random_rotate":   "DAgger-Discrete first pick",
    "p4_top3_fixed":            "P4 top-3 (fixed)",
    "p4_top3_rotate":           "P4 top-3 compress",
    "p4_all_fixed":             "P4 all (fixed)",
    "p4_all_rotate":            "P4 all compress",
}
COLORS = {
    "baseline_highloss_fixed":  "tab:orange",
    "baseline_random_fixed":    "tab:green",
    "baseline_highloss_rotate": "tab:blue",    # intentional overlap
    "baseline_random_rotate":   "tab:olive",
    "p4_top3_fixed":            "tab:blue",    # intentional overlap
    "p4_top3_rotate":           "tab:orange",  # intentional overlap
    "p4_all_fixed":             "tab:purple",
    "p4_all_rotate":            "tab:red",
}
```

The repeated colors (`tab:blue` for both `baseline_highloss_rotate` and
`p4_top3_fixed`; `tab:orange` for `baseline_highloss_fixed` and
`p4_top3_rotate`) are deliberate — the user composes paper subfigures
that show only ONE method per shared color. If you build a NEW figure
that overlays both methods sharing a color, override the palette inside
that cell instead of editing `nb_plot.COLORS`.

`aggregation/aggregate.py::METHOD_COLORS`/`METHOD_LABELS` were set
earlier with non-overlapping defaults; the user has not touched them.
The notebook's `paper_figures.ipynb` calls `nb_plot.compare()` for F1
(headline) which uses the user-curated palette above.

---

## 8. The paper-figure pipeline

### 8a. `paper_data.py` — NEW data-loader module
Self-contained module the notebook imports. **Heavy upstream imports are
DEFERRED into the functions that use them** so module reload is cheap
(this is load-bearing — moving them back to the top level was a
10-minute slowdown on Weka; see §11/L4). Specifically:
- `from Equivariant_pathway.expert import build_grid, compute_distance_map`
  is inside `compute_toughness()`.
- `from Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.nb_helpers
  import _layout_rgb` is inside `render_gallery()`.

Public surface (in declaration order):
- `available_runs() -> List[int]`
- `iter_prescribed_losses(method, runs=None) -> Iterator[dict]` — yields
  `{run, round, method, layout, policy_loss, policy_loss_sum, n_steps,
  ep_seed, success}` per picked/prescribed demo. Data:
  `results/run_*/<method>/results/round_*/prescribed_loss.json`.
- `iter_p4_prescriptions(method, runs=None) -> Iterator[dict]` — yields
  `{run, round, method, layout}` from
  `p4_analysis/recommended_layouts.json`.
- `iter_correction_pool_layouts(runs=None) -> Iterator[dict]` —
  method-agnostic; yields
  `{run, pool_kind: 'fixed'|'rotate', round, layouts: List}` from
  `shared/fixed_pool/` and `shared/round_*/`.
- `compute_toughness(layout) -> dict` — `{n_fires, bfs_opt_len,
  start_quadrant, goal_quadrant, manhattan, start_pos, goal_pos}` using
  upstream `expert.build_grid + compute_distance_map`. `bfs_opt_len=-1`
  if unreachable.
- `parse_corridor_steps(steps_str) -> List[(r,c)]` — parses
  `"(r,c)->(r,c)->…"`.
- `render_gallery(items, ncols=4, suptitle=None, panel_size=2.4)` —
  matplotlib grid of maze panels with orange corridor polyline.
- `final_sr_dataframe(runs=None) -> pd.DataFrame` — drives T1.
- `compression_dataframe(method=None, runs=None) -> pd.DataFrame` —
  drives T2 / F4 / F5; includes the new `prescribed_loss_mean` column.
- `save_figure(fig, name, dpi=150) -> Path`
- `save_table(df, name, caption="") -> (Path, Path)` — writes BOTH
  `.md` and `.tex` to `results/aggregate/figures/paper/`.
- `_df_to_markdown(df)` — tabulate-free markdown formatter
  (`pandas.DataFrame.to_markdown()` needs `tabulate` which is not in the
  maze env; this is the reason for the manual formatter).
- Constants: `METHODS`, `P4_METHODS`, `BASELINE_METHODS`,
  `RESULTS_ROOT`, `PAPER_FIG_ROOT`.

### 8b. `paper_figures.ipynb` — NEW 11-cell notebook
Cell 0 (markdown): title + perf note.
Cell 1 (code, setup): adds REPO_ROOT to sys.path; imports
`paper_data`, `nb_plot`, matplotlib/pandas/numpy/json; has a
`RELOAD = False` flag. **Downstream cells do NOT reload modules.** Set
`RELOAD = True` ONLY when iterating on `paper_data.py`/`nb_plot.py` and
re-run Cell 1. Per-cell `importlib.reload(...)` was the prior 10-minute
slowdown — don't put it back.

Cells 2–11 (one figure or table each, each starting with
`# CLAIM / DATA / USES`):

| Cell | Artefact | Claim | Data |
|---|---|---|---|
| 2  | F1 | sample efficiency, all 8 methods, mean ± std | `learning_curve.json` |
| 3  | T1 | per-method final SR, extras-to-target, stopped-reason counts | `learning_curve.json` + `run_summary.json` |
| 4  | F3 | info-gain proxy distribution per method | `prescribed_loss.json` |
| 5  | F4 | failures shown per round, per P4 method | `compression_log.csv` |
| 6  | F5 | corridor infeasibility rate per P4 method | `compression_log.csv` |
| 7  | F6 | layout-distribution histograms (BFS-opt-len + #fires) | pool YAMLs + `prescribed_loss.json` |
| 8  | F7 | qualitative gallery: prescribed mazes w/ corridor | `recommended_layouts.json` |
| 9  | F8 | recommendation toughness vs round, per P4 method | `recommended_layouts.json` |
| 10 | F9 | per-run timeline (latest run by default) | `learning_curve.json` |
| 11 | T2 | per-P4-method compression + system aggregate | `compression_log.csv` |

Total notebook runtime: **~24 seconds** end-to-end on the 10-run
production data (with deferred imports, no per-cell reload).
Output dir: `results/aggregate/figures/paper/`. Tables emit BOTH `.md`
and `.tex`.

---

## 9. Info-gain logging (NEW data product)

The user wants a paper claim around **"P4 selects more informative
demos than DAgger"**. The proxy is the **pre-finetune policy's per-step
BCE loss on the chosen/prescribed layout** — high = the demo carries
learning signal, low = the policy already handles this layout.

### 9a. The artefact
Per round per method, written uniformly across all 8 methods at
`results/run_<id>/<method>/results/round_<NNN>/prescribed_loss.json`.
Schema:
```json
{"round": 7, "method": "<method_name>",
 "picks": [{"layout": {...}, "policy_loss": 0.42,
            "policy_loss_sum": 8.4, "n_steps": 20,
            "ep_seed": 7099, "success": false}]}
```

### 9b. Where it gets written
- **Baselines** ([selection/baseline_dagger.py](selection/baseline_dagger.py))
  — after `ranking.json` is persisted, the picked failure's
  `(layout, policy_loss, policy_loss_sum, n_steps, ep_seed, success)`
  is extracted from `ranked[i][3]` and written. No new rollout needed
  (the loss was already computed in the round's loss-rollout).
- **P4** ([p4/pipeline_p4.py](p4/pipeline_p4.py)) — after
  `cap_layouts(rec_path, mode_max=1)` and BEFORE
  `demo_collector.collect(...)`, the CURRENT (pre-finetune) policy is
  rolled out via `_bb._rollout_with_loss(model, kept_layout, ep_seed,
  max_steps, device)` on each kept layout. The result populates the
  `picks` list; `prescribed_loss_mean` is the mean of `policy_loss`
  across kept layouts (always 1 for the current mode_max).

### 9c. The CSV column
`logging_ext/compression_log.py::CompressionLog.write_row` accepts a new
`prescribed_loss_mean: Optional[float] = None` kwarg and emits a
`prescribed_loss_mean` column. Empty cell when the round produced no
kept prescriptions. Read by `paper_data.compression_dataframe` →
drives T2's `prescribed_loss_mean` aggregate.

### 9d. Why uniform across methods
For baselines the picked layout IS one of the round's loss-rolled-out
failure layouts; the loss is already known. For P4 the prescribed
layout is NEW (generated by the LLM), so we must roll it out fresh.
Persisting the same `prescribed_loss.json` schema across both lets the
notebook read **one filename** for every method (see
`paper_data.iter_prescribed_losses`).

---

## 10. Data product catalog

Quick reference for "where is X on disk." All paths relative to the
suite root unless noted.

| Artefact | Path | Producing module | Notes |
|---|---|---|---|
| Learning curve | `results/run_<R>/<method>/results/learning_curve.json` | `baseline_dagger._persist_curve`, `pipeline_p4._persist_curve` | top-level `method`, `history[]` |
| Training log | `results/run_<R>/<method>/results/training_log.csv` | `logging_ext/training_log.py` | round, demos_added_total, training_rounds_total, heldout_sr, finetune_epochs, lr, replay_mix |
| Compression log | `results/run_<R>/<P4>/results/compression_log.csv` | `logging_ext/compression_log.py` | + new `prescribed_loss_mean` column |
| Ranking (baselines) | `results/run_<R>/<bl>/results/round_NNN/ranking.json` | `baseline_dagger.run` | list of {rank, ep_idx, policy_loss, policy_loss_sum, n_steps, jitter, layout_name, picked} |
| **Prescribed loss (NEW)** | `results/run_<R>/<method>/results/round_NNN/prescribed_loss.json` | both pipelines | uniform schema across baselines + P4 |
| Failures yaml (P4) | `results/run_<R>/<P4>/results/round_NNN/failures_layouts.yaml` | `pipeline_p4.run` (via `write_yaml`) | top-3 or all failures depending on `top_k` |
| Failures rollout (P4) | `results/run_<R>/<P4>/results/round_NNN/failures_rollout/` | `_pp._rollout` | feeds LLM analysis pipeline |
| P4 analysis | `results/run_<R>/<P4>/results/round_NNN/p4_analysis/` | upstream `run_profile_analysis` | `recommended_layouts.json`, `p4_seq_seqbatch_prescription_report.json`, `full_output.json`, per-episode dirs |
| Collect summary (P4) | `results/run_<R>/<P4>/results/round_NNN/collect_summary.json` | `demo_collector.write_summary` | `n_saved`, `n_infeasible`, `saved[]`, `infeasible[]` |
| Heldout eval | `results/run_<R>/<method>/results/round_NNN/heldout_eval/` | upstream rollout_test | `success_rate.json` |
| Demo files | `results/run_<R>/<method>/demos/round_NNN/*.json` | upstream `_record_one` | maze_name, start/goal/fires, trajectory, observations, images, actions, rewards, success |
| Fixed pool yaml | `results/run_<R>/shared/fixed_pool/correction_layouts.yaml` | `ensure_correction_layouts_for_run` | one per run |
| Rotate pool yaml | `results/run_<R>/shared/round_NNN/correction_layouts.yaml` | `ensure_correction_layouts_for_round` | one per (run, round) |
| Pool sampling report | `results/run_<R>/shared/round_NNN/layout_setup_report.json` | `layout_setup._save_report` | seed, n_correction, overlap_train/heldout |
| Run summary | `results/run_<R>/run_summary.json` | `orchestrator/run_one.py::main` | `{run_id, methods, results: {<method>: {stopped_reason, n_history}}}` |
| Aggregate summary | `results/aggregate/summary.json` | `aggregation/aggregate.py::main` | per-method final-SR mean±std |
| Contamination report | `results/aggregate/contamination_report.json` | `layouts/contamination.py::cross_run_check` | `correction_pool_intra_run_overlap` MUST be `[]` |
| Per-method curve index | `results/aggregate/learning_curves_by_method/<method>/learning_curve_run_<N>.json` | bash one-liner the user requested | 80 files (8 methods × 10 runs) |
| Paper figures | `results/aggregate/figures/paper/F*.png`, `T*.{md,tex}` | `paper_figures.ipynb` | 8 figs + 2 tables (× 2 formats) |

Method dir names — listed in `workspace.METHOD_DIR_NAMES` and the dispatch
table:
`baseline_highloss_fixed`, `baseline_random_fixed`,
`baseline_highloss_rotate`, `baseline_random_rotate`,
`p4_top3_fixed`, `p4_top3_rotate`,
`p4_all_fixed`, `p4_all_rotate`.

---

## 11. Gotchas (inherited + new)

### Inherited from the parent suite (`baseline_vs_p4_sequential_batch`) — still apply unchanged
The full catalog lives in
`baseline_vs_p4_sequential_batch/claude_memory.md` §"Gotchas". The most
load-bearing items, condensed:

- **C1 `$SLURM_SUBMIT_DIR`.** SLURM copies the script into a spool dir;
  `${BASH_SOURCE[0]}` mis-resolves. The submit scripts already use
  `${SLURM_SUBMIT_DIR}` when set. Don't break this.
- **C2 launchers are bash, not sbatch.** `submit_all.sh`,
  `submit_smoke.sh`, `submit_aggregate.sh` (chained) — invoke with
  `bash`, never `sbatch`. They call `sbatch` internally for the real
  job script.
- **C3 `--gpus=N` can split across nodes.** Use
  `--nodes=1 --gpus-per-node=2`. The Qwen sbatch already does this.
- **C4 unique ports per job.** Derived from `SLURM_JOB_ID`
  (`20000 + (JOB_ID % 12000)*3`); the Qwen sbatch does this. Do not
  revert to hardcoded 8000/8001/8002.
- **C5 `sbatch --export` truncates METHODS at commas.** Encoded as `+`
  in the launchers, decoded back in `submit_one*.sh`. Don't break
  either side.
- **D1 `maze` env has no vllm.** The vLLM servers launch with
  `VLLM_PYTHON` (default
  `/home/s226137394/.conda/envs/vllm_embed/bin/python`).
- **D2 `bitsandbytes`** must be installed in `VLLM_PYTHON`'s env.
- **S4 Responses API.** The pipeline uses
  `client.responses.create(...)` → `/v1/responses`. The
  [qwen/proxy.py](qwen/proxy.py) MUST route this path. If proxy log
  shows 404 on `/v1/responses`, the proxy regressed.
- **S5 `input_image.detail` schema drift.** Proxy injects
  `detail: "auto"` on every `input_image` content item that lacks one.
  Don't remove this patch.
- **S6 Qwen3 `<think>` blocks.** Proxy strips
  `<think>...</think>` from `/v1/responses` JSON responses
  (`_strip_think_recursive`). If you see
  `phase_c.parsed_prescription.raw_output` starting with `<think>`,
  the strip regressed.
- **S7 retry death-spiral.** `OAI_SDK_TIMEOUT=900`,
  `OAI_MAX_IN_FLIGHT=2`, `OAI_SDK_MAX_RETRIES=2`, plus
  `PROXY_MAX_OUTPUT_TOKENS=8192` cap. Exported in
  `submit_one_qwen.sh`.
- **S8 `OPENAI_BASE_URL`** must be set BEFORE importing pipeline
  modules (the upstream `_CLIENT` is built at module-level import
  time). The Qwen sbatch sets it before `python -m run_one`.
- **B1–B4 context-length budget.** Model ceiling `40960`; output capped
  to 8192 at proxy; input bounded by `correction_n=20`. ALL THREE are
  load-bearing. Do NOT raise `correction_n` above 20.
- **L1 don't stop on 0-demo round.** Implemented in both
  `baseline_dagger` (via `saved=0` → `no_new_demos` only ends in fixed
  mode? — actually baselines DO end on `no_new_demos`; check before
  changing) and `pipeline_p4` (`consecutive_empty`/`no_progress`).
- **L2 infeasible-corridor feedback.** Rejected corridors are persisted
  in `infeasible_memo` and rendered into next-round prompts via
  `prompts.infeasible_feedback_block`. Don't disable.
- **L3 layout-yaml keys.** Probe
  `("layouts", "test_layouts", "training_layouts", "heldout_test_layouts")`
  when reading. `paper_data._read_layouts_yaml` already does this.

### New gotchas specific to this suite

- **N1 `parents[5]` depth coupling.** Every module computes
  `REPO_ROOT = Path(__file__).resolve().parents[5]`. This suite MUST sit
  at depth-1 under `baseline_vs_p4/` (same nesting as
  `variants_fixed_vs_p4top3/`). Do NOT nest the suite folder deeper or
  imports break with no obvious error.
- **N2 METHOD lists must stay in lockstep.** Five places: `config.methods`,
  `METHOD_SPEC`, `METHOD_DIR_NAMES`, `aggregate.METHOD_DIRS`,
  `nb_plot.METHODS`. A 9th method is a 5-line change (not "one line" —
  see §3).
- **N3 The `0/12 success=False` log line after `failures_layouts.yaml`
  is NOT a bug.** See §6b. Don't chase it.
- **N4 The OUTPUT REQUIREMENT prompt block is load-bearing.** Weakening
  it brings the empty-prescription bug back. See §6c.
- **N5 `paper_data.py` heavy imports are DEFERRED inside functions** —
  `expert` is imported inside `compute_toughness()`, `_layout_rgb`
  inside `render_gallery()`. Do not move them to the top level — on
  Weka filesystem latency, top-level placement makes module reload take
  many seconds; per-cell `importlib.reload(...)` made the prior
  notebook take **10+ minutes**. The current notebook runs in ~24 s.
- **N6 No per-cell `importlib.reload(...)` in `paper_figures.ipynb`.**
  Cell 0 reloads only when `RELOAD = True`. Downstream cells must not
  reload. See §8b.
- **N7 `aggregation/aggregate.py` had a latent `KeyError`** (the
  compression section iterated hardcoded `("p4_sequential","p4_batch")`).
  Now uses `P4_METHODS` constant. If you add/remove P4 methods, edit
  `P4_METHODS` at the top of `aggregate.py` (not hardcode again).
- **N8 `contamination.py::KNOWN_METHODS`** was wrong in the fork; now
  set to the 8 method dir names. The intra-run pool-overlap check
  globs `shared/round_*` only, so the fixed pool at
  `shared/fixed_pool/` is correctly excluded — that's NOT contamination
  even though the two seeds are independent.
- **N9 Demo cross-run overlap of `n_overlap=20` per method is BENIGN.**
  Those are the 20 shared initial bootstrap demos copied into every
  run's per-method `demos/`. The contamination report may say
  `status: violation` for this reason; read the
  `correction_pool_intra_run_overlap` field, not just `status`.
- **N10 `aggregate.py` aggregate dir/aggregate command path** — invoke
  from REPO_ROOT:
  `python3 -m Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_x_selector.aggregation.aggregate --budget 15 --target_sr 0.90`.

---

## 12. How to run

### Smoke (validate wiring)
From the suite dir (`pool_x_selector/`). Current smoke defaults set in
[submit_smoke.sh](submit_smoke.sh):
```
BUDGET=2  MAX_ROUNDS=3  CORRECTION_N=16  HELDOUT_N=10
BASELINE_ROUND_EPOCHS=3  ROUND_EPOCHS=3  INITIAL_EPOCHS=100
METHODS=<all 8>   SMOKE_RUN_IDS="99 100"
--time=08:00:00 (overrides production 120h for backfill speed)
```
Launch:
```
SUBMIT_SCRIPT=submit_one_qwen.sh bash submit_smoke.sh
# optional overrides:
SUBMIT_SCRIPT=submit_one_qwen.sh SMOKE_RUN_IDS="103 104" bash submit_smoke.sh
```
Smoke completes in ~2–3 hours when GPU resources are available.

### Production
From the suite dir. Defaults in [submit_one_qwen.sh](submit_one_qwen.sh)
reflect the user's recent shift (§7a):
```
BUDGET=9999  TARGET_SR=1.0  ROUND_EPOCHS=600  BASELINE_ROUND_EPOCHS=90
walltime 120h
```
Launch (10 runs × 8 methods, chained aggregation):
```
AGGREGATE_AFTER=1 SUBMIT_SCRIPT=submit_one_qwen.sh bash submit_all.sh
```

### Aggregation by hand (CPU-only)
```
python3 -u -m \
  Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_x_selector.aggregation.aggregate \
  --budget 15 --target_sr 0.90
```
Or via SLURM: `sbatch submit_aggregate.sh`.

### Paper notebook
Open [paper_figures.ipynb](paper_figures.ipynb) in Jupyter (kernel:
maze) → run cells top-down. Outputs land under
`results/aggregate/figures/paper/`. Total runtime ~24 s on the 10-run
production set.

If you edit `paper_data.py` or `nb_plot.py`: set `RELOAD = True` in
Cell 0, re-run Cell 0 only, then re-run whichever downstream cells you
need. Don't re-add per-cell reload.

### Per-method curve archive
Already materialised at
`results/aggregate/learning_curves_by_method/<method>/learning_curve_run_<N>.json`
(80 files). Reproduction recipe:
```
OUT=results/aggregate/learning_curves_by_method
mkdir -p "$OUT"
for m in baseline_highloss_fixed baseline_random_fixed \
         baseline_highloss_rotate baseline_random_rotate \
         p4_top3_fixed p4_top3_rotate p4_all_fixed p4_all_rotate; do
  mkdir -p "$OUT/$m"
  for rid in $(seq 1 10); do
    src=results/run_${rid}/$m/results/learning_curve.json
    [ -f "$src" ] && cp "$src" "$OUT/$m/learning_curve_run_${rid}.json"
  done
done
```

---

## 13. Verification (green-flags for the next session)

Sanity-check on a completed production sweep — every item should pass.

1. **`squeue`/`sacct`** — all 10 jobs `COMPLETED 0:0`, single-node
   allocations.
2. **Proxy log** —
   `grep "POST /v1/responses" slurm_logs/proxy_<jobid>.log | grep -c status=200`
   non-zero for BOTH `model='qwen3-32b'` and `model='qwen3-vl-32b'`.
3. **vLLM logs** — both `vllm_llm_<jobid>.log` and `vllm_vlm_<jobid>.log`
   reached `Application startup complete`; no `Address already in use`,
   `max_model_len > 40960`, or `ModuleNotFoundError`.
4. **Per-run dirs** — all 10 `results/run_*/` exist; each has
   `shared/fixed_pool/correction_layouts.yaml` AND
   `shared/round_NNN/correction_layouts.yaml`.
5. **Curves complete** — `results/run_*/<method>/results/learning_curve.json`
   exists for every (run, method) cell. `history` non-empty, last
   `heldout_sr` populated.
6. **Info-gain logging present** —
   `find results/run_*/*/results/round_*/prescribed_loss.json | wc -l`
   should equal Σ rounds across all methods.
7. **Compression log column** — first line of
   `results/run_*/<P4>/results/compression_log.csv` ends with
   `,prescribed_loss_mean`.
8. **P4 rounds collected** — for each P4 method, grep the run log:
   `grep -hE "<P4>.*collected=" logs/run_*.log` should show no
   `prescribed=0` rounds (the §6 bug regression test).
9. **Contamination** — `results/aggregate/contamination_report.json`
   has `correction_pool_intra_run_overlap == []` (the load-bearing
   field). The `status` may say `violation` due to the benign
   `demo_cross_run_overlap == 20` (shared init demos, §N9).
10. **Aggregate figures** — `results/aggregate/figures/*.png` and
    `results/aggregate/summary.json` exist with per-method final-SR
    mean±std.
11. **Paper notebook** — runs in ~24 s end-to-end; produces 8 F*.png
    + T1/T2 in both `.md` and `.tex`.

---

## 14. Things to NEVER do

- Never edit upstream code in `baseline_vs_p4/` or in the sibling suites
  (`baseline_vs_p4_sequential_batch/`, `variants_fixed_vs_p4top3/`).
  COPY into `pool_x_selector/` and adapt.
- Never weaken the OUTPUT REQUIREMENT block in `p4/prompts.py` (§6c).
- Never raise `correction_n` above 20 — Qwen 40960-ctx ceiling +
  proxy 8192 output cap leaves room for ≤20 failure summaries.
- Never re-introduce per-cell `importlib.reload(...)` in
  `paper_figures.ipynb` — that was the 10-minute slowdown (§N5/N6).
- Never move `paper_data.py`'s heavy upstream imports back to the top
  level — same reason.
- Never restart the `[hybrid-rollout] DONE successes=0/N` confusion
  (§6b).
- Never run production from a Claude session without explicit user
  approval. Smoke tests on small budgets are fine when the user asks.
- Never edit the upstream package path `…baseline_vs_p4.pool_x_selector.…`
  in the submit scripts / subprocess `python -m` calls — the suite's
  imports + finetune subprocess depend on it.
- Never revert the user-curated `nb_plot.py` LABELS/COLORS (§7b) or
  `submit_one_qwen.sh` knob defaults (§7a) without explicit
  confirmation.
- Never `sbatch` the launchers — they are bash launchers that call
  `sbatch` internally.

---

## 15. Open / next-iteration items

- **Production sweep with the new knobs** (§7a:
  `BUDGET=9999 / TARGET_SR=1.0 / ROUND_EPOCHS=600 /
  BASELINE_ROUND_EPOCHS=90`) — this is what the user most recently
  configured. The 10-run results currently on disk are the artefact
  set the notebook reads.
- **System-cost figures** are NOT in the notebook yet. If the user
  wants them: add `time.perf_counter()` round-timing in
  `pipeline_p4.py` and `selection/baseline_dagger.py` (persist to a new
  `round_timing.json`), and add per-call token logging in
  `qwen/proxy.py` (append to `slurm_logs/tokens_<jobid>.csv`). Plot in
  a new F10/T3 cell. This was Option C ("Maximum") in the original
  scope question, which the user did NOT pick — only add on request.
- **The 80 per-method curve files** at
  `results/aggregate/learning_curves_by_method/` are ready to feed to
  any external plotting tool / paper script the user is composing.
- **Paper-grade figure curation continues** — the user is iterating on
  `nb_plot.LABELS`/`COLORS` for camera-ready presentation. Treat those
  edits as authoritative.

---

## 16. Pointers — the files you'll touch most

- [p4/pipeline_p4.py](p4/pipeline_p4.py) — unified P4 loop
  (top_k + fixed/rotate + info-gain rollout + helpers).
- [p4/prompts.py](p4/prompts.py) — strengthened `mode_directive("top3")`
  and `mode_directive("all")` (do NOT weaken).
- [selection/baseline_dagger.py](selection/baseline_dagger.py) —
  baseline loop; writes uniform `prescribed_loss.json`.
- [orchestrator/run_one.py](orchestrator/run_one.py) — `METHOD_SPEC`
  dispatch + `needs_fixed` guard for fixed-pool sampling.
- [orchestrator/workspace.py](orchestrator/workspace.py) —
  `METHOD_DIR_NAMES` tuple of 8.
- [orchestrator/bootstrap.py](orchestrator/bootstrap.py) — idempotent
  init demos + checkpoint mirror.
- [layouts/layout_setup.py](layouts/layout_setup.py) — fixed + rotated
  pool primitives.
- [layouts/contamination.py](layouts/contamination.py) — cross-run
  contamination check.
- [logging_ext/compression_log.py](logging_ext/compression_log.py) —
  new `prescribed_loss_mean` column.
- [aggregation/aggregate.py](aggregation/aggregate.py) — 8-method
  aggregation + `P4_METHODS` constant.
- [paper_data.py](paper_data.py) — paper-figure data loaders.
- [paper_figures.ipynb](paper_figures.ipynb) — the 11-cell notebook.
- [nb_plot.py](nb_plot.py) — paper-grade LABELS/COLORS (user-curated).
- [config.yaml](config.yaml) — single source of truth for knobs.
- [submit_one_qwen.sh](submit_one_qwen.sh) — production SBATCH
  (`BUDGET=9999`, `TARGET_SR=1.0`, `ROUND_EPOCHS=600`,
  `BASELINE_ROUND_EPOCHS=90`, 120h).
- [submit_smoke.sh](submit_smoke.sh) — smoke launcher
  (`HELDOUT_N=10`, 8h cap).
- [submit_all.sh](submit_all.sh) — production launcher.

---

## 17. The 30-second mental model for a fresh session

You are working on `pool_x_selector/` — an 8-method
**pool × selector** comparison for a top-A* paper. The suite is built,
production has run 10× cleanly, and the paper notebook is at
[paper_figures.ipynb](paper_figures.ipynb).

- If the user asks for a NEW paper figure → add a cell to the notebook;
  if you need a new data loader → add a function to
  [paper_data.py](paper_data.py); keep heavy imports DEFERRED.
- If the user asks about results → start with
  `results/aggregate/summary.json` + the figures in
  `results/aggregate/figures/paper/`.
- If the user asks about a method behaving oddly → check
  `results/run_*/<method>/results/learning_curve.json` and the
  `compression_log.csv`; for P4, also the
  `p4_analysis/recommended_layouts.json` to see what the LLM
  prescribed.
- If you see "0/N success=False" in the slurm log → it's the failures
  re-rollout, NOT a bug (§6b).
- If `p4_all_*` ever produces `prescribed=0` rounds → check whether
  someone weakened the OUTPUT REQUIREMENT in
  [p4/prompts.py](p4/prompts.py) (§6c).
- If imports / notebook are slow → check for per-cell reloads
  (§N5/N6).
- If you need to add a 9th method → 5-line change across the lockstep
  files (§3 / §N2).

Read this whole file before suggesting changes. Then you're ready.

---

## 18. IIL-baseline comparison suite (NEW — P4-LLM vs 5 baselines)

A later session added a **rotated-only** comparison of the already-run
`p4_top3_rotate` (relabeled **"P4-LLM"**) against the 5 interactive-IL
baselines from `baseline_implementations_guide.md`: **SafeDAgger\***,
**DropoutDAgger**, **EnsembleDAgger**, **ThriftyDAgger**, **Stagger**. P4 is
NOT re-run — the baselines reuse P4's seeds, per-round pools, fine-tuner, and
held-out yardstick; only the *which-layout-to-query* rule differs.

### 18a. New files (all in this suite)
- `selection/iil_baselines.py` — 5 decision rules + shared run loop (clone of
  `baseline_dagger.run`'s skeleton). Per-layout query scores; picks ONE
  layout/round; reuses `_rollout_with_loss`/`_save_corrective_demo`/
  `_eval_heldout`/`_find_deviation_step`/`_finetune`. **Stop semantics follow
  P4's ROTATE pipeline** (`target_hit`/`budget_exhausted`/`no_progress` after
  `max_consecutive_empty` empty rounds/`max_rounds`) — NOT `no_new_demos` — so
  baselines may run more rounds than P4 to reach budget=15.
- `selection/uncertainty.py` — ensembles (`finetune_ensemble` calls the replay
  fine-tuner M× into `checkpoints/member_{m}/`; `ensemble_rollout`,
  `ensemble_eval_heldout` mean-action eval), MC-dropout (`load_dropout_model`
  instantiates the policy with `head_dropout=d>0`, keeps Dropout active),
  shared `action_discrepancy`.
- `selection/success_q.py` — `SuccessQNet` + MC success-to-go trainer for
  ThriftyDAgger risk (`risk = 1 - Q`), persisted at
  `<method>/success_q/q_net.pth`.
- `layouts/layout_setup.py::ensure_correction_layouts_for_round_filled` — REUSE
  P4's `shared/round_NNN` pool when present; else **seed-bump fill** (bump the
  sampler seed +1 until a full size-20 pool, since `sample_layouts` raises on
  short pools). Records `seed_bump`/`reused_existing` in the layout report.
- `orchestrator/run_baselines.py` — baseline-only driver; `METHOD_SPEC` of 5;
  no fixed-pool path; writes **`run_summary_baselines.json`** (separate file so
  P4's `run_summary.json` is never clobbered).
- `config_baselines.yaml` — budget=15, target_sr=0.90 (matches P4 on disk),
  `baselines:` block with per-kind knobs (tau, M, dropout p/N/d, alpha_h, …).
- `submit_baseline_one.sh` — single-GPU SBATCH, **NO vLLM/proxy** (baselines
  don't use the LLM). `submit_baselines_all.sh` — 10 parallel jobs + chained
  aggregate; same comma→`+` METHODS encoding as `submit_all.sh`.

Run: `AGGREGATE_AFTER=1 bash submit_baselines_all.sh`.
Smoke: `N_RUNS=2 SMOKE_FLAG=--smoke SMOKE_RUN_IDS="99 100" METHODS=safe_dagger+stagger bash submit_baselines_all.sh`.

### 18b. Authorized edits to the analysis pipeline (do NOT treat as reversions)
The user explicitly chose "relabel only, P4 vs 5 baselines". These edits are
intentional:
- `orchestrator/workspace.py::METHOD_DIR_NAMES` — extended with the 5 baseline
  dir names (`safe_dagger`, `dropout_dagger`, `ensemble_dagger`,
  `thrifty_dagger`, `stagger`); the 8 old names are kept.
- `aggregation/aggregate.py`, `paper_data.py`, `nb_plot.py` — `METHODS`/
  `METHOD_DIRS`/`LABELS`/`COLORS` now target the **6-method comparison set**
  (5 baselines + `p4_top3_rotate`); `P4_METHODS = ("p4_top3_rotate",)` (only
  P4-LLM has `compression_log.csv`). `nb_plot.LABELS["p4_top3_rotate"]` is now
  **"P4-LLM"** (authorized relabel — supersedes the §7b "P4 top-3 compress").
  The old fixed/vague-rotate keys remain in the dicts so legacy wrappers work.
- `paper_data._stopped_reason` now also reads `run_summary_baselines.json`.
- `paper_figures.ipynb` cells F1 (Cell 2) and F3 (Cell 4) now use the 6-method
  set; T1/F6 already key off `paper_data` constants and auto-adapt; F4/F5/F7/
  F8/T2 stay **P4-only** (they read `compression_log.csv`/`recommended_layouts.json`
  which the baselines do NOT produce — baselines DO write `prescribed_loss.json`,
  `learning_curve.json`, `training_log.csv`).

### 18c. Faithfulness caveats (report in the paper; do NOT silently "fix")
- **MC-dropout is fusion-head-only** — the shared checkpoint has no backbone
  dropout (§model.py); DropoutDAgger's uncertainty comes from the one head
  Dropout. Retraining with backbone dropout would break the shared warm-start.
- **Per-timestep→per-layout reduction** (SafeDAgger, Thrifty) — switch rules
  applied at layout granularity (1 demo/round); Thrifty quantiles are over
  per-layout novelty/risk.
- **EnsembleDAgger round-1 degeneracy** — members start identical (doubt==0);
  `_pick` deliberately picks a *queryable* layout even at score 0 (by highest
  loss) so a demo is collected and members can diverge. Do NOT re-add a
  `score>0`-only filter to `_pick` (it deadlocks the ensemble).
- **Stagger ≈ baseline_random_rotate** — intentional low-information control.

---

## 19. Fairness analyses + HP sweep — KEY NEGATIVE RESULT (do NOT re-discover)

After the 6-method comparison, the user pushed on **whether P4-LLM actually beats
the baselines fairly**. Two no-LLM analyses were run (both reuse P4's already-
prescribed demos — no new LLM calls). **Verdict: on this 5×5 maze, P4 does NOT
beat the baselines; the task saturates ~0.90–0.91 and demonstration *selection*
does not separate the methods.** Don't re-run these expecting a different answer
— the lever is the environment, not more tuning.

### 19a. What was tried
1. **Matched-budget retrain (`p4_top3_rotate_e90`).** P4 originally trained at
   **500 epochs/round**; the baselines at **90**. Retraining P4's exact demos at
   90 epochs (`orchestrator/replay_p4.py`, 10 runs) → final SR **0.886** vs
   P4@500 **0.896** vs baselines **0.891–0.901**. So part of P4's @500 edge was
   the training budget; at matched 90 epochs P4 ties/slightly trails.
2. **Training-hyperparameter sweep (`results_hpsweep/`).** 36 configs
   (epochs {90,150,300} × lr {5e-5,1e-4} × replay_mix {0.3,0.5,0.7} ×
   head_dropout {0,0.1}) × runs {1,6}, retraining P4's demos, comparing per-demo
   against the **max-over-5-baselines envelope**. **0/36 configs (and 0/72
   config×seed cells) put P4 above all baselines at every demo count.** Best
   config `e150_lr1e-4_rm0.5_do0`, worst-demo margin only **−0.055**. Trend:
   lr=1e-4 > 5e-5; epochs 90–150 ≥ 300; head_dropout/replay_mix minor.
   Leaderboard: `results_hpsweep/leaderboard.{md,json}` + `leaderboard_top.png`.

### 19b. New machinery (no-LLM; all reuse the replay path)
- `orchestrator/replay_p4.py` — retrains ANY method's fixed demos
  (`--source_method`) at arbitrary epochs/lr/batch/wd/replay_mix/`--head_dropout`,
  warm-starting from the run's init checkpoint; `--out_root` writes to a separate
  tree while READING demos+ckpt from the main `results/`. Per-round 200-layout
  eval dominates cost (~1.5 min/round) — replays are eval-bound, not train-bound.
- `orchestrator/sweep_hp.py` (per-job parallel `(config×run)` driver) +
  `orchestrator/sweep_rank.py` (per-demo dominance leaderboard vs the fixed
  baseline envelope; `min_margin`, `margin@5/@10/@15`, `dominates` flag).
- `submit_p4_replay_one.sh`, `submit_sweep_one.sh`, `submit_sweep_all.sh`
  (partition `gpu`, **no vLLM**; `submit_sweep_all.sh SMOKE=1` for a 2-config check).
- `trainer/finetune_replay.py` gained `--head_dropout` (parameter-free →
  warm-start-safe; default 0.0 = unchanged for every other caller).

### 19c. Methodology guardrail (the user asked, was talked out of it)
**Do NOT cherry-pick the seeds where P4 wins and average only those** — selection
bias / garden-of-forking-paths; won't replicate; instant A*-paper reject. The
honest forms: win-rate over ALL seeds, paired per-seed test, or a seed
train/test split (pick config on dev seeds, report on held-out seeds). Here even
the per-seed check was 0/72, so there was nothing to pick anyway.

### 19d. Recommended next step (where a real win can come from)
Build a **harder / non-saturating environment** (larger grid, more fires, longer
horizon, lower BC ceiling) so demonstration *quality* actually gates success —
the regime where P4's LLM selection has room to **reproducibly** beat the
baselines. Tuning training hyperparameters or hunting seeds will not flip the
verdict on the current 5×5 maze. (Architecture/cold-start search was deliberately
excluded — it breaks warm-start and, given saturation, is unlikely to help.)
