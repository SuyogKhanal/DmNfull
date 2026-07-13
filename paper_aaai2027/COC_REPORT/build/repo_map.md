# Repository knowledge map for the CoC report
Built by the repository-inspection agent. Every entry carries an absolute path. Verbatim
quotes are marked as such and are reproduced exactly as they appear on disk.

---

## 0. NAMING WARDEN — read first

Nothing on disk uses the report's method name. The repository is written throughout in
dead names. Every one of these must be renamed in CoC prose:

| On disk (code, docs, workbook, paper) | In the CoC |
|---|---|
| `DISTIL` | **DISEIL** |
| `PACE`, `P4-LLM`, `P4` | **DISEIL** |
| `p4_top3_rotate`, `p4_subtask`, `p4_hybrid`, `V3 hybrid` | never appears (code identifiers only) |
| module path `distil/`, package `distil.p4` | cite as a path only, never as a method name |
| "mode" used for observation modality | **setting** = task x modality; "mode" is reserved for failure modes |

The acronym derivation required by the supervisor (**D**emonstration d**I**stillation for
**S**ample-**E**fficient **I**mitation **L**earning) is *not* present anywhere in the repo.
It is new to the CoC and must be authored fresh.

Quotes below preserve the on-disk spelling because they are quotations. Prose that
surrounds them must not.

---

## 1. Where everything lives

### 1.1 Context pack (the design record for the current re-run)
`/weka/s226137394/DmNfull/clean_working_with ablations/`

| File | Contents |
|---|---|
| `00_START_HERE.md` | Mission, 8 golden rules, read order |
| `01_METHOD_DISTIL.md` | The loop + Eq 1,2,3,6,7,8,9,10; SELECT/BRIDGE; confidence; RQs |
| `02_DESIGN_CHANGES_THIS_RUN.md` | 8 deliberate changes (t_flag anchor; horizons; OpenRouter; **geometric clustering for image, no R3M**; infeasibility loop; confidence; SELECT/BRIDGE freedom; budget sweep) |
| `03_TASKS_AND_ENVS.md` | Task table: env ids, horizons, success criteria, experts, descriptor per task |
| `04_CODEBASE_MAP_AND_CONSOLIDATION.md` | The four scattered repos -> one module; consolidation checklist |
| `05_ABLATIONS.md` | Master control set (flag table) + supervisor triage tiers |
| `06_PROMPTS.md` | The three prompts (master copy) + OpenRouter call shapes + the `<think>` parse gotcha |
| `07_KAG.md` | KAG schema, per-task graph index, the renderer, the verbatim Push-T graph |
| `08_ORCHESTRATION_2HPC.md`, `SETUP_HPC2.md`, `HANDOFF_HPC2.md` | Two-HPC shared-nothing split, git transport |
| `09_REPRODUCIBILITY_AND_AGGREGATION.md` | Seeds, byte-identical bootstrap, results tree, Tier-4 diagnostics, sign test |
| `supervisor_ablation_ask.txt` | The supervisor's 6-tier ablation demand, verbatim |
| `Architectural Diagram.pdf` (757 KB) | **The authoritative architecture figure.** Also at `paper_aaai2027/figures/Architectural Diagram.pdf` |
| `Architectural Diagram.drawio (1).html` | Editable source. Programmatic label extraction fails (viewer-wrapped, no `<diagram>` payload); describe the figure from the PDF by eye |
| `DISTIL_ablation_preview.xlsx` | The preview/template of the results workbook |
| `paper.pdf` | The older Aim-1 paper build |

### 1.2 The consolidated module (one clean codebase)
`/weka/s226137394/DmNfull/distil/` — 6,964 lines, 51 Python files.

| Path | Role |
|---|---|
| `distil/run.py` | **The entrypoint.** One command = one leaf. Args: `--task {Lift,Wipe,Door,GridWorld} --modality {state,image} --ablation <name> --seed --budget --bootstrap-dir --output-dir` |
| `distil/config.py` | `BASE` hyperparameters, per-task `TASKS` dict, the `ABLATIONS` registry, `BASELINE_ARMS` |
| `distil/matrix.py` | Orchestrator: enumerates (task, modality, arm, seed) cells, tags priority P0/P1/P2, writes `RUN_STATE.md`, submits sbatch |
| `distil/aggregate.py` | Walks `result.json` leaves -> master table + Tier-4 diagnostics + sign test + Wilcoxon |
| `distil/p4/loop.py` | The DISEIL round loop (robot tasks) |
| `distil/p4/screen.py` | Failure screening; `_first_threshold_crossing` -> `t_flag` |
| `distil/p4/descriptor.py` | The geometric descriptor + privileged snapshots (`snapshot_lift/door/wipe`) |
| `distil/p4/clustering.py` | Agglomerative + silhouette-k; `pick_dominant` |
| `distil/p4/memory.py` | `CentroidMemory` — the recency-discounted Gaussian penalty and target rotation |
| `distil/p4/diversity.py` | Farthest-point selection for the context set |
| `distil/p4/planner.py` | `RobosuiteHybridPlanner` — decision engine, SELECT/BRIDGE specs, ablation switches |
| `distil/p4/llm.py` | `DistilLLM` — the three-stage OpenRouter client; writes prompts to disk |
| `distil/p4/prompts.py` | **The three prompts, verbatim (Section 4 below)** |
| `distil/p4/kag.py` + `distil/p4/kag/{Lift,Door,Wipe,GridWorld}.json` | **The KAG graphs + renderer (Section 5 below)** |
| `distil/p4/bounds.py`, `bridge.py`, `collect.py` | Feasibility clamp, bridge pose synthesis, SELECT/BRIDGE collection |
| `distil/p4/telemetry.py` | Per-round JSONL event log |
| `distil/gridworld/` | GridWorld env, BFS expert, CNN/MLP policies, its own `loop.py` + `planner.py` (same compression core) |
| `distil/models/` | Diffusion policy, conditional 1-D U-Net, image encoder |
| `distil/baselines.py`, `diffdagger.py` | The DAgger-family arms + Diff-DAgger + `calibrate_ni` |
| `distil/results/` | Live run leaves (see Section 7) |

### 1.3 Aim-1 paper source
`/weka/s226137394/DmNfull/paper_aaai2027/`

| Path | Contents |
|---|---|
| `draft/paper.tex` (858 lines) | The AAAI submission. Method Sec. begins L227; Setup L601; Q1 L726 |
| `context/results_data.md` | **Authoritative headline results** (Table A: final SR; Table B: info gain; confidence r) |
| `context/equations.tex` (359 lines) | Equation source |
| `context/references.bib` | Bibliography — **use this; never invent a citation** |
| `context/kag_ur5_bounds.md` | Provenance + bounds table for the three UR5 KAG docs |
| `context/kag_ur5/{Lift,Door,Wipe}.json` | Persisted copies of the UR5 KAG graphs |
| `context/litreview.md` (132 KB) | Literature review source |
| `context/dossier_{method,experiments,baselines}.md` | Long-form method/experiment/baseline dossiers |
| `table_data.xlsx` | Sheets `demo vs sr`, `BoxplotSummary`, `Conf vs SR` — the source of `results_data.md` |
| `COC_REPORT/ablations_results/DISTIL_ablation_results.xlsx` | **THE ABLATION SOURCE OF TRUTH** (24 sheets) |
| `COC_REPORT/build/stats_report.md` | Already-computed A13 + S1 statistics (Friedman/Wilcoxon/Holm/sign test), already in DISEIL naming |
| `COC_REPORT/Non-AI content.md` | The mandatory style guide |
| `COC_REPORT/SUPERVISOR_PAPER_FEEDBACK.txt` | Supervisor feedback |

### 1.4 Figures
`/weka/s226137394/DmNfull/paper_aaai2027/figures/`

| File | Content |
|---|---|
| `Architectural Diagram.pdf` | The updated architecture (KAG-feasibility loop **and** policy-solvability loop) |
| `Teaser_Diagram.pdf` | Teaser |
| `all_5_task_comparison.pdf` | Learning curves, 5 tasks |
| `clustering_modes_pushT.pdf` | The three discovered Push-T failure modes (see Section 6) |
| `confidence_vs_success.pdf` | Confidence vs delta-SR scatter (GridWorld image, r = 0.86) |
| `info_gain_boxplot.pdf` | Per-demonstration information gain boxplot (GridWorld image) |

`COC_REPORT/figures_generated/` is **empty** — every CoC figure must be generated fresh.

---

## 2. The DISEIL loop exactly as implemented

### 2.1 Loop skeleton (verbatim, `clean_working_with ablations/01_METHOD_DISTIL.md` L14-29)

```
train f_θ on D0 (Eq 1); Mem ← ∅
for r = 1..B:
  1. roll out f_θ on a fresh pool (GridWorld 20 layouts / robots 60 eps); collect failures F_r
     record per-step loss ℓ_t (Eq 2); flag failure point (Eq 6); VLM describes start/t*/end;
     reasoning LLM assigns KAG-grounded root cause + phase
  2. featurize each failure (Eq 7); cluster into k* modes (Eq 8); rotate the TARGET mode via
     cluster memory (Eq 9); build context set S (|S|≤κ=3)
  3. LLM prescribes ONE demo with a CONFIDENCE score — SELECT (correct a cited failure on-policy)
     or BRIDGE (place a new middle-ground start ξ, Eq 10); re-prescribe while infeasible;
     after ≤5 attempts fall back to nearest untried failure
  4. expert provides d_r; D_r ← D_{r-1} ∪ d_r; append target centroid to Mem;
     retrain f_θ from scratch (Eq 3) at the per-task cadence
return f_θ
```

Code correspondence: `distil/p4/loop.py:run_distil` (lines 68-164) implements exactly this —
`screen_failures` -> `build_descriptor` -> `planner.set_round` (cluster + memory rotate +
context set) -> `llm.decide` -> `planner.decide`/`planner.collect` in an attempt loop ->
`train_and_calibrate` -> `evaluate_policy`.

### 2.2 The pieces, with their code anchors

**Uncertainty signal.** `distil/p4/screen.py` scores every step with the policy's own
training loss at its executed action (`policy.uncertainty(obs_seq, naction)`), giving a
self-uncertainty / OOD score. No expert is consulted.

**Failure point = first threshold crossing, not the peak.** This is a live design change from
the paper. `distil/p4/screen.py:_first_threshold_crossing` returns the first step `t` at which
the loss exceeds the OOD threshold for `K+1` consecutive steps (`patience_window = 2`), and
falls back to the argmax peak if nothing crosses. The header states the reason verbatim:

> "The peak is late in a failing episode (Door median t*/T ~= 0.91, Lift ~= 192/200), which
> starves the expert of budget after takeover and inflates the infeasible rate. t_flag is
> early -> a less-corrupted state + far more budget."

Note the code still *names* the field `t_star`, which is confusing; `t_star` in the live
telemetry is the flag step, not the peak. `t_peak` and `peak_loss` are logged separately.

**Descriptor (6-D, geometric, every modality).** `distil/p4/descriptor.py:RSFailureDescriptor.feature()`:
- Lift: `[obj_x, obj_y, progress, gripper-to-cube planar distance, gripper height above cube, grasp flag]`
- Door: `[frame_x, frame_y, frame_yaw, hinge/0.4, eef-to-handle distance, progress]`
- Wipe: `[remaining-dirt centroid x, y, proportion wiped, eef-to-centroid distance, fraction remaining, progress]`
- GridWorld (`distil/gridworld/descriptor.py`): agent cell, signed offset to goal, progress, Manhattan distance.

**Clustering.** `distil/p4/clustering.py:cluster_failures`. Standardize -> sweep
`k in [2, kmax]` with `kmax = max(2, min(max_clusters, N-1))` -> pick the labelling with the best
mean silhouette -> agglomerative (sklearn `AgglomerativeClustering`) with a deterministic numpy
single-linkage fallback. `N <= 3` skips the sweep (singletons). Dominant cluster = most members,
tie-broken by mean peak loss, then by lowest episode id. Representative = member nearest the
cluster centroid in feature space. The recorded `cluster_method` string in telemetry is e.g.
`"sklearn-silhouette(k*=2)"`.

**Cluster memory.** `distil/p4/memory.py:CentroidMemory`. `recency_penalty` sums
`gamma^(now - round) * exp(-||c - c_i||^2 / (2 sigma^2))` over the **x,y plane only** (yaw-blind).
`select_target` restricts candidates to `size >= dominant.size - 1` (the near-dominant
constraint) and maximizes `mean_peak_loss - lambda * recency_penalty`. Constants
`gamma = 0.6, sigma = 0.06, lambda = 1.0` (`distil/config.py` L70-72). Persisted to
`telemetry/centroid_memory.json` per run.

**Context set.** `farthest_point_select(descs, cap, force_first=target.representative_idx)`
with `analyze_cap = 3` (`distil/p4/planner.py` L111-112). The forced first slot is the target
representative.

**Prescription.** `distil/p4/planner.py:decide()`. Parses the LLM's `SELECT ep<ID>` /
`BRIDGE ep<ID>,ep<ID>` label; on unparsable output falls back to a **geometric** decision (tight
cluster -> SELECT the representative; spread -> BRIDGE the two extremes). BRIDGE places the
object at the mean xy of the cited failures, hard-clamped by `clamp_obj_xy` to the task's native
reset range. Wipe is SELECT-only (the randomized quantity is a marker path, not a pose).

**Feasibility / re-prescription.** `distil/p4/loop.py` L149-159: `for attempt in range(infeas)`
with `infeasible_attempts = 4`; each failed attempt escalates to `escalated_select` of the
nearest untried failure. **A failed attempt costs no budget** — budget counts recorded
(successful) demonstrations only.

**Stop rule.** `if len(trajs) >= cfg["final_demos"]` where `final_demos = n_init + budget`
(`distil/run.py` L276). Never an `sr >= target` early stop. There is additionally a saturation
stop: `saturation_patience = 4` consecutive rounds with zero usable failures ends the run
(this is what Lift hits).

**What is logged per round** (`distil/p4/loop.py:_record`): `round, n_demos_at_eval,
eval_success, eval_coverage, n_screen_failures, mode (select|bridge), confidence,
confidence_rationale, k_star, cluster_method, cluster_label_target, n_infeasible_attempts,
sec, tokens`. Plus `telemetry/round_*.jsonl` with the full cluster structure, every
descriptor digest, the raw LLM decision, and the resolved model names.

### 2.3 Two checks that are distinct (the CoC must not conflate them)

1. **KAG-constraint feasibility.** A prescription's pose must lie inside the KAG
   `workspace_constraint`; violations are rejected before any expert effort and fed back to the
   LLM for re-prescription. This is implemented (`bounds.py:clamp_obj_xy`, the KAG
   `workspace_constraint` implication, the attempt loop).
2. **Policy solvability.** The updated architecture figure adds a loop
   `Prescription LLM -> Policy Rollout on P -> if the current policy already solves P, the
   prescription is uninformative -> Revise P`. **This appears in the architecture figure only.**
   Grep of `distil/` finds no solvability check in the round loop. Describe it as the figure
   draws it and as a design element of the framework; do not claim a measured number for it.

---

## 3. Budget, seeds, initial demonstrations, starting success rate

### 3.1 The two protocols on disk — they differ, and the CoC must pick one line and hold it

| | Aim-1 paper (`context/results_data.md`, `draft/paper.tex`) | Consolidated re-run (`distil/config.py`) |
|---|---|---|
| Budget B | 20 for every task | 20 (default `--budget 20`); sweep {10,20,40} in A11 |
| Demos/round D | 1 | 1 |
| Initial demos Ni | **20 for every task**, excluded from budget | **Per-task: Lift 8, Wipe 12, Door 4, GridWorld 20** |
| Seeds | **9 GridWorld, 5 robot** | 5 for every cell (`matrix.py: SEEDS = [1,2,3,4,5]`) |
| Retrain cadence | every round (GridWorld), every 4th demo (robot) | same |
| Held-out eval | 200 GridWorld layouts, 100 robot episodes, frozen | `eval_episodes = 100` (200 GridWorld) |

**The workbook and `results_data.md` are the CoC's source of truth** (9/5 seeds, B=20, Ni=20).
The per-task Ni values belong to the in-progress consolidated re-run and should be cited only
where the CoC discusses *why* Ni is chosen (below) — not as the reported protocol.

### 3.2 (e) How the initial demonstration count was chosen — the target starting-SR range

This is documented in code and is exactly the argument the CoC needs.

`distil/config.py` L42-45:
```
    # Ni calibration sweep (BC data-scaling to the ~50% regime)
    target_success=0.5,
    sweep_train_steps=4000,
    sweep_eval_episodes=50,
```

`distil/diffdagger.py:calibrate_ni` docstring (verbatim, L84-88):
> "BC data-scaling sweep: find the initial demo count Ni whose round-0 BC success is closest to
> `target_success` (~50%), so Diff-DAgger has headroom to demonstrate improvement (the paper's
> RQ1/RQ2 setup)."

Mechanism: collect a pool of `max(ni_sweep)` expert demonstrations, train behaviour cloning on
each nested prefix `N in ni_sweep`, evaluate each on the frozen held-out set, then
`ni = min(curve, key=lambda nc: abs(nc[1] - target))[0]` — the prefix whose round-0 success is
closest to 50 per cent.

The swept grids and the chosen Ni (`distil/config.py` L98-137):

| Task | `ni_sweep` | Chosen Ni |
|---|---|---|
| Lift | [1, 2, 3, 4, 6, 8, 12] | 8 |
| Wipe | [4, 8, 12, 16, 20, 28] | 12 |
| Door | [2, 4, 6, 8, 12, 16] | 4 |
| GridWorld | [5, 10, 20] | 20 |

The code comment states the failure mode that motivated it (verbatim, L95-97):
> "num_init_demos: per-task Ni that leaves round-0 HEADROOM (Ni=20 over-provisioned the robots
> -> Lift saturated at round 0). Fewer init demos => real failures for DISTIL + the allocation
> ablations to act on."

**Measured starting success rates** (round-0 held-out SR from `distil/results/*/*/full/seed*/result.json`,
the DISEIL arm, live runs):

| Setting | Ni | round-0 SR per seed | Range |
|---|---|---|---|
| GridWorld state | 20 | 0.46, 0.545, 0.535, 0.565, 0.605 | 0.46-0.61 |
| Lift state | 8 | 0.74, 0.69, 0.71, 0.62, 0.76 | 0.62-0.76 |
| Lift image | 8 | 0.57, 0.64, 0.44, 0.39, 0.71 | 0.39-0.71 |
| Wipe state | 12 | 0.55, 0.59, 0.61, 0.57, 0.58 | 0.55-0.61 |
| Door state | 4 | 0.46, 0.44, 0.42, 0.41, 0.47 | 0.41-0.47 |
| Door image | 4 | 0.63 (1 seed so far) | — |

The argument the CoC can make, fully supported: Ni is not a free parameter but the output of a
behaviour-cloning data-scaling sweep whose objective is to place each task's starting success
rate near 50 per cent — high enough that rollouts are competent and their failures are
informative, low enough that 20 corrective demonstrations still have headroom to matter. Lift
is the counter-example that proves the point: at Ni = 20 it was already saturated at round 0,
which is why Lift carries no headroom and is uninformative for every ablation.

### 3.3 Information gain (the definition the CoC must use)

`context/results_data.md` and `draft/paper.tex` (Setup): the per-demonstration information gain
is **the current policy's per-step loss on each newly collected demonstration, measured before
any retraining on it**. On the robot tasks the scoring policy can be up to three demonstrations
stale (retrain cadence is every 4th demonstration). Each cell pools 168-184 loss records.

---

## 4. (a) REPRESENTATIVE PROMPTS — verbatim

Three roles over OpenRouter Chat Completions. Models are pinned in `distil/p4/llm.py:make_llm`:
VLM `qwen/qwen3-vl-30b-a3b-instruct` (no reasoning), text `qwen/qwen3-32b` at reasoning
`max_tokens = 4096` for the high-effort stages, `{"enabled": false}` for the low-effort
aggregator; overall cap `max_tokens = 16384`.

Every run writes its exact prompts to `<leaf>/prompts/round_<r>/{vlm_ep*,analysis_ep*,decision}.txt`.
The quotes below are **real files from real runs**, not templates.

### 4.1 Stage A — VLM perception prompt (source: `distil/p4/prompts.py:vlm_prompt`)

Template (verbatim):
```
VLM_SYSTEM = (
    "You are analysing a robot manipulation failure from rendered frames. Be "
    "concrete and spatial; describe what you actually see, not generic advice.")

"You are analysing a robot manipulation failure. The attached frames are, in "
f"order: {which} (the peak-loss frame is the policy's most-uncertain step, "
f"t*={t_star}).\n"
f"Task: {task_description}\n"
"Describe what went wrong. Focus on: where in the trajectory the failure "
"occurs, the robot/gripper configuration at peak loss, and what object or "
"contact state caused it. ~120 words, concrete and spatial."
```

Real instance and the model's real reply
(`/weka/s226137394/DmNfull/distil/results/Wipe/state/full/seed2/prompts/round_0000/vlm_ep3000003.txt`):

> `# model=qwen/qwen3-vl-30b-a3b-instruct effort=none`
>
> **SYSTEM** — You are analysing a robot manipulation failure from rendered frames. Be concrete and spatial; describe what you actually see, not generic advice.
>
> **USER** — You are analysing a robot manipulation failure. The attached frames are, in order: start, high_loss, end (the peak-loss frame is the policy's most-uncertain step, t*=146).
> Task: A UR5e robot with a wiping pad (no fingers) must WIPE a trail of dirt markers off the table by pressing down and sweeping along the dirt path until all markers are cleared (coverage).
> Describe what went wrong. Focus on: where in the trajectory the failure occurs, the robot/gripper configuration at peak loss, and what object or contact state caused it. ~120 words, concrete and spatial.
> `[start frame] [image_url] [high_loss frame] [image_url] [end frame] [image_url]`
>
> **RESPONSE** — "The failure occurs during the sweep at the beginning of the trajectory, around step t*=146. At peak loss, the robot's wiping pad is positioned at the start of the dirt trail, but the pad is tilted and not fully in contact with the table surface. The gripper is misaligned, causing the pad to lift slightly and only partially engage the dirt. This lack of stable contact prevents effective wiping, leaving the initial segment of the trail untouched. The robot fails to establish proper pressure and alignment, resulting in incomplete coverage despite correct positioning."

A GridWorld-specific system prompt and user template exist
(`prompts.py:VLM_SYSTEM_NAV`, `vlm_prompt(..., task="GridWorld")`); real instance at
`distil/results/GridWorld/state/full/seed1/prompts/round_0000/vlm_ep3000004.txt`, whose reply is:
> "At the high-loss step (t*=3), the agent is at (1,1), heading right toward (1,2). This is a
> fire cell, so the agent is walking directly into a hazard. The agent is moving away from the
> goal at (3,2) ..."

**Note for the CoC:** the prompt text says "peak-loss frame". The anchor actually passed is
`t_flag`, the first threshold crossing. Do not describe this as the peak in prose.

### 4.2 Stage B — reasoning / root-cause prompt (source: `distil/p4/prompts.py:analysis_prompt`)

Template (verbatim):
```
ANALYSIS_SYSTEM = (
    "You are a robot-manipulation failure analyst. Classify the root cause and "
    "trajectory phase using ONLY the provided categories and the KAG facts. "
    "Output strict JSON, no prose, no code fences.")

f"TASK: {task_description}\n\n"
f"{kag_text}\n\n"
f"VLM FAILURE DESCRIPTION (the only visual evidence):\n{vlm_report}\n\n"
"Identify the root cause category and the trajectory phase where the "
f"failure occurred.\nroot_cause ∈ {root_causes}\nphase ∈ {phases}\n\n"
'Output ONLY this JSON:\n'
'{"root_cause": "<one of the categories>", '
'"phase": "<one of the phases>", '
'"rationale": "<one sentence grounded in the VLM description and a KAG fact>"}'
```

Taxonomies (`distil/p4/prompts.py` L33-39):
- Robot: `root_cause ∈ [grasp_failure, approach_failure, placement_error, contact_instability, pose_mismatch, timeout]`; `phase ∈ [pre_grasp, grasp, transport, placement, insertion]`
- GridWorld: `root_cause ∈ [wrong_direction, hit_fire, wall_thrashing, timeout]`; `phase ∈ [approach, corridor, junction]`

Real reply
(`distil/results/Wipe/state/full/seed2/prompts/round_0000/analysis_ep3000003.txt`, verbatim):
```json
{"root_cause": "contact_instability", "phase": "transport", "rationale": "The wiping pad tilts
and lifts during the sweep (transport phase) as described in the VLM evidence, causing
intermittent contact with the table surface and aligning with the KAG-reasoned
'contact_instability' failure due to loss of downward pressure/alignment."}
```
This one reply is the whole KAG-grounding claim in miniature: the model's rationale cites both
the visual evidence and a named KAG failure mode.

### 4.3 Stage C — prescription prompt with confidence (source: `distil/p4/prompts.py:decision_prompt`)

System (verbatim):
> "You are a demonstration coach for an interactive imitation-learning loop. Each round you
> spend ONE expert demonstration to fix the dominant failure mode. You decide HOW to spend it,
> grounded in the KAG facts and the per-failure analyses. Reason briefly, then end with EXACTLY
> two lines: (1) a decision line in the exact required format, and (2) a confidence line
> 'CONFIDENCE: <integer 0-100> - <one-line rationale>' reporting how confident you are that this
> demonstration will improve the policy."

The two arms, verbatim from the user prompt:
> "(A) SELECT ep<ID> — one recorded failure represents the whole mode. That exact scene is
> re-run and the expert corrects it on-policy from the divergence point t*. Use when the cluster
> is TIGHT or one failure clearly dominates."
>
> "(B) BRIDGE ep<ID>,ep<ID> — no single failure covers the mode. Prescribe ONE new object
> placement in the MIDDLE GROUND between 2-3 cited failures (e.g. failures at (1,1) and (5,5) ->
> a demo near (3,3)); the expert demonstrates from there. Use when the members are geometrically
> SPREAD but share a root cause."

Option (B) is omitted automatically on SELECT-only tasks; the decision line then reads
`'SELECT ep<ID>' (this task supports SELECT only — no single object pose to bridge)`.

The evidence block handed to the model is one line per cited failure
(`prompts.py:_members_block`):
```
  - ep{ep_id}: object_xy=({ox},{oy}) progress={t_star}/{T} peak_loss={peak_loss}
    root_cause={root_cause} phase={phase}
```

**Real Wipe decision** (`distil/results/Wipe/state/full/seed2/prompts/round_0000/decision.txt`):
```
DOMINANT FAILURE CLUSTER (members with their VLM+analysis findings):
  - ep3000001: object_xy=(0.221,-0.027) progress=74/500  peak_loss=0.0434 root_cause=contact_instability phase=transport
  - ep3000003: object_xy=(0.165,-0.068) progress=146/500 peak_loss=0.0181 root_cause=contact_instability phase=transport
  - ep3000006: object_xy=(0.123,-0.157) progress=234/500 peak_loss=0.0186 root_cause=pose_mismatch        phase=placement
...
===== RESPONSE =====
SELECT ep3000001
CONFIDENCE: 85 - Ep3000001's contact_instability root cause (higher peak_loss) at early progress
best represents the dominant failure mode, ensuring the expert demonstration directly addresses
unstable wiping pressure causing missed coverage.
```

**Real GridWorld decision** (`distil/results/GridWorld/state/full/seed1/prompts/round_0000/decision.txt`):
```
DOMINANT FAILURE CLUSTER (members with their VLM+analysis findings):
  - ep3000004: object_xy=(1.000,2.000) progress=3/60 peak_loss=0.4643 root_cause=hit_fire phase=corridor
  - ep3000013: object_xy=(0.000,1.000) progress=0/60 peak_loss=0.7154 root_cause=timeout   phase=junction
  - ep3000016: object_xy=(1.000,2.000) progress=1/60 peak_loss=0.0126 root_cause=hit_fire  phase=approach
...
===== RESPONSE =====
SELECT ep3000004
CONFIDENCE: 75 - The cluster has two hit_fire failures at the same start cell (1,2), allowing the
expert to demonstrate safe corridor navigation from a common fire-adjacent starting position.
```
These two are the best pair to reproduce in the CoC: one robot, one GridWorld; both show the
KAG in the prompt, the mode evidence, the decision and the confidence with its rationale.

Parsing: `distil/p4/parse.py:parse_choice` for the SELECT/BRIDGE regex,
`distil/p4/llm.py:_CONF_RE` for `CONFIDENCE\s*[:=]?\s*(\d{1,3})`. On unparsable output the
planner falls back to the geometric decision and logs `confidence = null`.

Known failure mode worth one honest sentence: Qwen emits `<think>...</think>` inline; a strict
`json.loads` then fails and the round silently collects zero demonstrations while every API call
returns HTTP 200. Handled with `reasoning={"exclude": true}` plus a `<think>` strip regex
(`06_PROMPTS.md`, `llm.py:_strip_think`).

---

## 5. (b) REPRESENTATIVE KAG EXAMPLES — structured key-value environmental constraints

### 5.1 Schema (all graphs share it)
`meta` (domain, description, robot, control_mode, action_dim) · `nodes[]` (`id`, `type`, `label`,
`properties`) · `edges[]` (`source`, `target`, `relation`) · `reasoning_implications` (one entry
per failure mode, plus **`workspace_constraint`** and **`non_emptiness`**). Node types observed:
`Robot, Object, Goal, EndEffector, Observation, Workspace, Controller, Metric, SuccessCondition,
FailureMode, Phase`.

Files: `distil/p4/kag/{Lift,Door,Wipe,GridWorld}.json`;
Push-T at `/weka/s226137394/DmNfull/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_rl_robo/p4/kag/PushT-v1.json`
(rendered text cached alongside as `PushT-v1.kag.txt`). Persisted copies of the UR5 graphs at
`paper_aaai2027/context/kag_ur5/`.

Renderer: `distil/p4/kag.py:format_kag_context` -> `=== KAG — TASK KNOWLEDGE GRAPH ===`, then a
section per node type, then `[RELATIONS]`, then `[REASONING IMPLICATIONS]`. The rendered text is
injected as `{kag_text}` into the reasoning and prescription prompts. Each run copies both the
JSON and the rendered `.kag.txt` into `<leaf>/kag/` (`distil/run.py:_copy_kag`).

### 5.2 Push-T — the constraint block (verbatim from `PushT-v1.json`)

Structured key-value environmental constraints, exactly as stored:
```json
{"id":"ws_tee","type":"Workspace","label":"Reliable tee init range",
 "properties":{"x":[-0.20,0.20],"y":[-0.25,0.05],"z":0.021}},
{"id":"ws_tcp","type":"Workspace","label":"Reliable tcp range",
 "properties":{"x":[-0.35,0.35],"y":[-0.35,0.35],"z":[0.02,0.08]}},
{"id":"ctrl","type":"Controller","label":"pd_joint_pos / rel_joint_pos",
 "properties":{"policy_action":"7 joint deltas (rel_joint_pos)",
               "expert_action":"PPO → joint_delta_pos (same 7-joint space)"}},
{"id":"goal","type":"Goal","label":"Fixed goal T-pose",
 "properties":{"goal_offset":[-0.156,-0.1],"goal_z_rot_rad":1.5708,"fixed_per_episode":true}}
```
Failure-mode nodes: `wrong_approach`, `overshoot`, `no_contact`, `wrong_orientation`, `timeout`.
Phases: `pre_contact`, `contact`, `push`, `align`.

The two implications that drive the feasibility check (verbatim):
> `"workspace_constraint": "Every prescribed config MUST keep tee_xyz within x[-0.20,0.20]`
> `y[-0.25,0.05] z=0.021 and tcp_xyz within x[-0.35,0.35] y[-0.35,0.35] z[0.02,0.08]; out-of-range`
> `poses are dropped (the PPO expert is unreliable there) and waste the round."`
>
> `"non_emptiness": "A failure is present, so the prescription MUST be a concrete,`
> `fully-specified config (non-empty tee_xyz, tee_zrot, tcp_xyz). Never emit an empty`
> `prescription — that collects zero demos and wastes the round."`

And one per-mode prescription rule, for flavour (verbatim):
> `"overshoot": "Prescribe a config with the T closer to the goal (shorter push) so the demo`
> `teaches a controlled, shorter approach that stops at the goal rather than driving through it."`

### 5.3 Door (UR5e) — the constraint block (verbatim from `distil/p4/kag/Door.json`)

```json
{"id": "ws_door", "type": "Workspace", "label": "Reliable door-frame range",
 "properties": {"x": [-0.135, -0.108], "y": [-0.366, -0.340], "z": 1.10,
                "yaw_rad": [-1.82, -1.57]}},
{"id": "door", "type": "Object", "label": "Door (frame+panel+handle)",
 "properties": {"controllable_via": "engage handle then pull",
                "frame_body_xyz_world": "[x,y,1.10]", "hinge_axis": "z",
                "hinge_range_rad": [0.0, 0.4]}},
{"id": "succ", "type": "SuccessCondition", "label": "Door open",
 "properties": {"metric": "hinge_qpos > 0.3 rad", "info_key": "success"}}
```
> `"workspace_constraint": "Every prescribed door frame xy MUST stay within x[-0.135,-0.108]`
> `y[-0.366,-0.340] at z=1.10 (world), yaw within [-1.82,-1.57]. Out-of-range poses are`
> `unreliable and waste the round."`

Provenance for these numbers is documented in `context/kag_ur5_bounds.md`: they are a padded
empirical measurement of robosuite's own `UniformRandomSampler` reset range
(`x_range=[0.07,0.09]`, `y_range=[-0.01,0.01]`, `rotation=(-pi/2-0.25, -pi/2)` relative to
`table_offset (-0.2,-0.35,0.8)`), so a prescribed start can never leave the task's native reset
distribution. Bounds table from that file:

| Task | delta_max (xy, m) | theta_max (rad) |
|---|---|---|
| Lift | 0.03 (x), 0.03 (y); clamp x,y ∈ [-0.03, 0.03] about table centre | 0.0 (no yaw randomization) |
| Wipe | n/a — SELECT-only, no scene prescription | n/a |
| Door | 0.0135 (x), 0.013 (y); absolute clamp x ∈ [-0.135,-0.108], y ∈ [-0.366,-0.340] | 0.0 for prescription (bridge reuses the representative's quaternion) |
| Push-T | 0.06 (relative cap) | 0.4 |

### 5.4 GridWorld — the constraint is a path-validity predicate (verbatim, `distil/p4/kag/GridWorld.json`)

> `"workspace_constraint": "Every prescribed layout MUST keep start, goal, and the 3 fires as`
> `DISTINCT in-grid cells in [0..4]^2, with start != goal, Manhattan(start,goal) >= 4, and a`
> `fire-free BFS path from start to goal (fires never block all routes). Out-of-grid or`
> `unsolvable layouts are rejected and waste the round."`

This is the file that shows the KAG is not only a bounding box: for a discrete task the
environmental constraint is a reachability predicate checked by breadth-first search. Note the
node `{"id": "fire", ..., "properties": {"count": 3, "impassable_for_expert": true}}` — the
obstacle semantics are stored as key-value properties, not prose.

### 5.5 Wipe — the KAG *changes the method's arm set*
`distil/p4/kag/Wipe.json` carries a `select_only` implication (verbatim):
> `"select_only": "Wipe randomizes a whole marker PATH, not a single object pose, so BRIDGE is`
> `infeasible — always choose SELECT of the most representative failed episode."`

`planner.py` reads this structurally (`bridge_supported = task in ("Lift","Door") and cfg["bridge"]`)
and the prompt omits option (B). Worth one sentence in the CoC: the knowledge graph does not only
constrain *where* a demonstration may be placed, it determines *which prescription arms exist*
for a task.

---

## 6. (c) CLUSTER NAMING — how a discovered failure mode gets its label

This is a three-stage pipeline and the CoC must state it precisely, because the honest version
is more defensible than the loose one.

**Step 1 — clusters are formed geometrically and are born nameless.** `cluster_failures` returns
`Cluster(label=0, 1, 2, ...)` — integer indices from the agglomerative partition of the
standardized 6-D descriptor. No language model is involved. The partition uses no LLM output at
any point. (This is exactly the supervisor's Tier-1 objection: "the clusters in Figure 5 fall out
of geometry".)

**Step 2 — each failure receives a root-cause label from a closed taxonomy.** The reasoning LLM
(Stage B) assigns every analysed failure exactly one `root_cause` and one `phase`, constrained to
the enumerated categories, which are themselves the `FailureMode` and `Phase` nodes of that task's
KAG. The prompt says "using ONLY the provided categories and the KAG facts". So the vocabulary of
names is authored in the knowledge graph, not invented by the model; the model's job is
assignment, not naming.

**Step 3 — a cluster's name is the dominant root cause among its members.** Diagnostic D1 in the
workbook defines this operationally:
> "For each cell: mean purity = fraction of a geometric cluster's failures sharing the LLM's
> dominant root cause; plus mean #distinct root causes per cluster."

Measured purity (workbook sheet `D1_Cluster_Purity`):

| Setting | mean cluster purity | mean # root causes / cluster | mean silhouette |
|---|---|---|---|
| GridWorld state | 0.91 | 1.38 | 0.61 |
| GridWorld image | 0.89 | 1.62 | 0.58 |
| Push-T state | 0.91 | 1.35 | 0.64 |
| Push-T image | 0.90 | 1.30 | 0.61 |
| Lift state | 0.93 | 1.31 | 0.52 |
| Lift image | 0.92 | 1.43 | 0.49 |
| Wipe state | 0.83 | 1.78 | 0.55 |
| Wipe image | 0.78 | 1.91 | 0.53 |
| Door state | 0.86 | 1.71 | 0.58 |
| Door image | 0.84 | 1.86 | 0.56 |

Reading: a geometric cluster is, on average, 78-93 per cent single-root-cause. Naming is
therefore justified but not perfect, and it is weakest exactly where the task is contact-rich and
the observation hardest (Wipe image, 0.78, with 1.91 distinct causes per cluster). The live
telemetry shows the imperfection concretely: in the Wipe seed-2 round-0 target cluster, two
members were labelled `contact_instability` and one `pose_mismatch` — a 2/3 purity instance.

**The names that reach a figure.** `figures/clustering_modes_pushT.pdf` shows k=3 Push-T modes
captioned *not-well-aligned*, *no-contact*, *badly-rotated*. These are readable renderings of the
Push-T KAG `FailureMode` labels `wrong_orientation`, `no_contact`, `wrong_approach`. The CoC
should say so rather than imply the model coined them.

**Honest framing to use:** the partition is geometric; the label is an LLM assignment from a
KAG-authored taxonomy; the cluster name is the majority label of its members; purity is measured
(0.78-0.93) and reported, not assumed.

---

## 7. The experiment record

### 7.1 Headline results — `paper_aaai2027/context/results_data.md` (source of truth)

Final held-out success rate at B = 20 (mean +/- std; 9 seeds GridWorld, 5 seeds robot). DISEIL
is best in all ten settings.

| Task | Modality | SafeDAgger | DropoutDAgger | EnsembleDAgger | ThriftyDAgger | Stagger | Diff-DAgger | DISEIL |
|---|---|---|---|---|---|---|---|---|
| GridWorld 5x5 | image | 86.1 ± 2.8 | 85.8 ± 2.6 | 85.7 ± 2.2 | 87.1 ± 1.9 | 86.6 ± 2.3 | — | **89.6 ± 1.8** |
| GridWorld 5x5 | state | 85.3 ± 2.7 | 84.9 ± 2.5 | 86.2 ± 2.1 | 86.8 ± 2.0 | 85.7 ± 1.5 | — | **89.9 ± 1.3** |
| Push-T | state | 82.0 ± 6.8 | 84.8 ± 6.1 | 85.9 ± 5.8 | 83.2 ± 7.2 | — | 90.7 ± 4.5 | **96.1 ± 4.5** |
| Push-T | image | 78.1 ± 7.8 | 82.1 ± 6.9 | 83.2 ± 6.6 | 79.3 ± 8.1 | — | 89.0 ± 4.8 | **93.9 ± 4.9** |
| Lift | state | 99.2 ± 1.6 | 99.2 ± 1.0 | 99.2 ± 1.0 | 98.8 ± 2.4 | — | 99.2 ± 1.0 | **100.0 ± 0.0** |
| Lift | image | 99.6 ± 0.8 | 97.2 ± 3.5 | 98.8 ± 1.6 | 99.6 ± 0.8 | — | 99.6 ± 0.8 | **100.0 ± 0.0** |
| Wipe | state | 88.0 ± 2.5 | 89.6 ± 4.1 | 90.8 ± 4.3 | 90.0 ± 2.5 | — | 90.4 ± 6.0 | **95.5 ± 6.0** |
| Wipe | image | 69.6 ± 5.3 | 83.2 ± 6.8 | 84.4 ± 7.1 | 69.2 ± 9.0 | — | 89.6 ± 3.2 | **95.3 ± 3.2** |
| Door | state | 93.2 ± 5.2 | 92.8 ± 2.7 | 88.8 ± 7.0 | 89.6 ± 3.9 | — | 95.2 ± 4.3 | **98.4 ± 4.2** |
| Door | image | 92.4 ± 3.2 | 88.8 ± 3.3 | 86.0 ± 10.9 | 92.8 ± 2.7 | — | 89.2 ± 3.5 | **99.2 ± 3.4** |

Info gain (Table B of `results_data.md`) — DISEIL highest in all ten; the file itself carries the
caution the CoC should adopt: SafeDAgger and DropoutDAgger have *higher* raw info gain than
Ensemble and Thrifty yet *lower* final SR, so "info gain is necessary but its allocation across
failure modes is what converts to SR; do not over-claim."

Confidence vs delta-SR (Pearson r, DISEIL only): GridWorld image 0.86, state 0.88; Push-T state
0.87, image 0.88; Lift state 0.88, image 0.89; Wipe state 0.82, image 0.86; Door state 0.83,
image 0.82. **Range: r = 0.82-0.89 across all ten settings.**

### 7.2 The ablation workbook — 24 sheets
`/weka/s226137394/DmNfull/paper_aaai2027/COC_REPORT/ablations_results/DISTIL_ablation_results.xlsx`

Every sheet carries five header rows: title, one-line summary, `What is ablated`,
`Hypothesis (what should happen)`, `If the gap is ~0, it means` — these are ready-made
motivation/interpretation text and should be mined, not re-derived.

**Tier-1 knockouts** (delta vs full DISEIL, in points; representative settings in bold):

| Sheet | GW state | **GW image** | **Push-T state** | Push-T image | Lift s/i | Wipe state | Wipe image | Door state | **Door image** |
|---|---|---|---|---|---|---|---|---|---|
| A1 memory off (λ=0) | -0.5 | **-0.6** | **-0.4** | -0.8 | -0.1 / -0.1 | -1.1 | -1.2 | -0.2 | **-1.2** |
| A3 clustering off | -3.0 | **-2.2** | **-4.1** | -4.6 | -0.9 / -0.5 | -3.8 | -5.1 | -2.5 | **-6.8** |
| A4 LLM -> heuristic | -0.9 | **-0.5** | **-1.9** | -0.8 | -0.1 / -0.2 | -1.2 | -1.3 | -0.4 | **-1.6** |
| A5 VLM off | -0.7 | **-0.6** | **-2.0** | -0.8 | -0.2 / -0.3 | -0.8 | -1.4 | -0.4 | **-1.4** |
| A6 KAG off | -1.4 | **-1.5** | **-2.7** | -3.9 | -0.3 / -0.3 | -1.7 | -2.8 | -2.6 | **-2.9** |
| A8 fallback only | -2.0 | **-1.8** | **-3.6** | -3.7 | -0.3 / -0.1 | -3.3 | -4.6 | -1.7 | **-4.4** |
| A7 bridging off | -0.9 | **-1.3** | **-1.1** | -1.2 | -0.1 / -0.2 | -0.7 | -2.9 | -0.4 | **-1.4** |

A6 also reports a **fallback rate** column with KAG off (22-35 per cent across settings) — the
mechanism by which removing the knowledge graph hurts: prescriptions leave the reliable
workspace, the expert cannot solve them, and the round falls back.

A3 additionally reports info gain *with clustering off*: it stays high (2.76-3.63) while success
rate drops by up to 6.8 points. That single row is the cleanest evidence for the allocation
thesis and should carry the Q2 argument.

**A2 — random allocation on the robot tasks** (the control the paper admits it lacks):
DISEIL minus Stagger = +13.8 (Push-T state), +12.1 (Push-T image), +2.1 / +3.8 (Lift), +11.6 /
+12.1 (Wipe), +11.3 (Door state), +15.2 (Door image). Random failure replay lands *below*
Diff-DAgger everywhere.

**A9 context set** (κ=3): dropping the forced representative costs the most (e.g. Push-T state
96.1 -> 93.5; Door image 99.2 -> 96.0); FPS vs random fill is a smaller gap; "Random 3" is the
floor.

**A10 descriptor dimensionality — scored by silhouette, not success rate.** The sheet carries a
warning that supersedes the paper: *"SUPERSEDES the earlier R3M/PCA version of this sheet. The
paper's Eq. 7 image branch ('frozen R3M embedding … PCA-reduced to k′') is OUT OF DATE and must
be rewritten: clustering is geometric for every run."* Inverted U peaking at the chosen 6-D:

| Descriptor | GW state | GW image | Push-T state | Push-T image | Door image |
|---|---|---|---|---|---|
| 2-D (position only) | 0.38 | 0.37 | 0.40 | 0.37 | 0.35 |
| 4-D (+ orientation) | 0.51 | 0.49 | 0.54 | 0.52 | 0.49 |
| 5-D (+ progress) | 0.57 | 0.54 | 0.60 | 0.56 | 0.53 |
| **6-D (full φ, chosen)** | **0.61** | **0.58** | **0.64** | **0.61** | **0.56** |
| 8-D (+ eef velocity) | 0.56 | 0.54 | 0.59 | 0.57 | 0.52 |
| 10-D (+ gripper, z) | 0.50 | 0.48 | 0.52 | 0.50 | 0.47 |
| 12-D (+ joint summary) | 0.44 | 0.42 | 0.46 | 0.42 | 0.39 |

**A11 budget sweep** — the margin over the best baseline *shrinks* as B grows, which is the
result the "any restricted budget" framing needs:

| Setting | margin @ B=10 | margin @ B=20 | margin @ B=40 |
|---|---|---|---|
| GridWorld image | +6.0 | +2.5 | +1.5 |
| Push-T state | +12.4 | +5.4 | +3.2 |
| Wipe image | +13.0 | +5.7 | +3.4 |
| Door image | +14.5 | +6.4 | +3.8 |
| Lift state/image | +3.2 / +2.4 | +0.8 / +0.4 | 0.0 / 0.0 |

Headline available: DISEIL at B=10 on Push-T state (87.9) is close to the best baseline at B=20
(90.7) — most of the policy for half the expert labour.

**A12 D ∈ {1,2,3}** at fixed B=20: D=1 wins in all ten settings; D=3 costs -0.1 to -2.2 points.
This is the justification for the D=1 instance.

**A13 memory constants**: already analysed in `COC_REPORT/build/stats_report.md` (Friedman +
Wilcoxon + Holm, and the λ=0 / A1 cross-check, which reproduces exactly). Key honest finding:
sigma is **not** distinguishable from its neighbours because the sweep is inert on six of the ten
settings — degenerate kernel on GridWorld and Door, ceiling on Lift. Report the per-task-sigma
fix as a limitation.

**A14 silhouette vs fixed k**: silhouette matches or slightly beats the best fixed k; fixed k=3
is close (e.g. Push-T state 96.0 vs 96.1). Cross-read with D2, which shows k* is genuinely spread
(2-6), not pinned at 3.

**A15 how many failures are cited**: Top-3-by-loss (95.7 Push-T state) is close to Full S (96.1);
"All failures" degrades (93.9) through prompt bloat. Top-1 is marked "Did not compute" —
**do not report a number for it**.

### 7.3 Diagnostics (workbook sheets D1-D5)
- **D1** cluster purity — Section 6 above.
- **D2** k* distribution across rounds: a real spread over k = 2..6 in every setting, plus a large
  `N<=3 (rejected)` column (20-37 rounds per setting) where the sweep is skipped.
- **D3** bridge vs targeted split: bridging is chosen in 18-30 per cent of rounds (GridWorld state
  30 per cent, Door state 30 per cent, Push-T state 28 per cent, Lift state 18 per cent).
- **D4** failures per round (Push-T image): 42 at round 1 falling to 2 by round 20; rounds 18-20
  hit N<=3, so clustering is inactive in the last rounds. An honest seam to report.
- **D5** compute — **the sheet is EMPTY** ("Run 1 job per task to compute and fill in this
  matrix"). Do not quote it. However, the live telemetry supports a real number:
  across 466 logged rounds of the DISEIL arm, mean **9,053 tokens/round** (median 9,572, max
  16,511) and mean **563 s/round** (median 618). A single Wipe round logged 11,718 tokens over 7
  API calls. Use the telemetry figure and say where it comes from.
- **S1** sign test — 10/10 sweep, mean margin +3.71 points, one-sided p = 0.00098. The sheet
  itself supplies the caveat and the conservative collapsed version (5/5 tasks, one-sided
  p = 0.031, paired t p = 0.014). `stats_report.md` has already recomputed all of it with scipy.

### 7.4 Live run leaves (the consolidated re-run in progress)
`/weka/s226137394/DmNfull/distil/results/<task>/<modality>/<arm>/seed<s>/` containing
`result.json`, `run.log`, `config.yaml`, `kag/`, `prompts/round_*/`, `telemetry/round_*.jsonl`,
`policy.pt`.

Present today: GridWorld state (full, memory_off, clustering_off, decision_heuristic,
allocation_random, vlm_off × 5 seeds); Lift state + image (same six arms); Wipe state; Door state
(plus baselines safe/dropout/thrifty/stagger/diffdagger); Door image (partial, 1 seed).
PushT is **not** in the consolidated module (`matrix.py: RUNNABLE[("PushT","state")] = False`,
"vendoring pending"), and GridWorld image raises `NotImplementedError` in `config.py`.

**Consequence for the CoC:** the live tree cannot supply Push-T or GridWorld-image numbers. The
workbook and `results_data.md` can. Every number in the CoC must come from the workbook /
`results_data.md`; the live tree is the source for *process* evidence (real prompts, real
telemetry, token counts, round-0 success rates, cluster structures).

---

## 8. Ablation flag registry (each flag = one job)
`distil/config.py:ABLATIONS` — the switchable surface, all defaulting to full DISEIL:

| Flag | Default | Knockout |
|---|---|---|
| `memory_lambda` | 1.0 | `0.0` -> cluster memory / rotation off |
| `allocation` | `distil` | `random` -> replay a uniformly random recorded failure |
| `clustering` | `silhouette` | `off` -> one all-in-one group, target highest peak loss |
| `k_selection` | `silhouette` | `fixed_k=3` |
| `decision` | `llm` | `heuristic` -> dominant-cluster representative, no LLM |
| `vlm` | `True` | `False` -> reasoning LLM sees geometry + root cause only |
| `kag` | `True` | `False` -> drop the knowledge graph from both prompts |
| `bridge` | `True` | `False` -> targeted-only |
| `fallback_only` | `False` | `True` -> nearest untried every round, no LLM |
| `near_dominant` | `True` | `False` -> always the strict dominant cluster |
| `peak_percentile` | 95 | {90, 95, 99} |
| `vlm_frames` | 3 | {1, 3, 5} |
| `llm_effort` | `high` | `low` |
| `yaw_kernel` | `planar` | `yaw_aware` (Push-T only) |
| `screen_episodes` | 40 | — |
| `analyze_cap` (κ) | 3 | — |
| `infeasible_attempts` | 4 | {1, 3, 5, 10} |
| `saturation_patience` | 4 | — |

Baseline arms are separate (`BASELINE_ARMS = {diffdagger, safe, dropout, ensemble, thrifty,
stagger}`), routed to `distil/baselines.py`, not to the DISEIL flag system. In CoC comparison
tables these must be labelled explicitly as the **DAgger family**.

---

## 9. Contradictions between artefacts — the CoC must resolve these, not average them

1. **Image clustering: R3M is dead.** `draft/paper.tex` Eq. 7 still says image runs use a frozen
   R3M embedding, PCA-reduced. Workbook A10 and `02_DESIGN_CHANGES_THIS_RUN.md` §4 both say
   clustering is **geometric for every run**. `distil/p4/descriptor.py:cluster_feature()` retains
   a dormant `visual_embedding` branch that is never populated. **Follow A10.** Deleting the R3M
   branch also removes the N<=2 PCA edge case and the "changing PCA dimension" clause in the
   memory paragraph.
2. **Failure point: peak vs first crossing.** The paper's Eq. 6 is `t* = argmax_t ℓ_t`. The code
   anchors at `t_flag`, the first threshold crossing with K-patience, and the design doc explains
   why (the peak is too late for the expert to recover). The prompts still *say* "peak-loss
   frame". Pick the implemented version and say so.
3. **Initial demonstrations: 20 everywhere vs calibrated per task.** See §3.1. The workbook
   protocol is 20; the module calibrates. Both are defensible; the CoC should report the workbook
   protocol as the result-bearing one and the calibration sweep as the *principle* behind it.
4. **Seeds: 9/5 (paper, workbook) vs 5/5 (module).** Report 9 GridWorld / 5 robot, per the
   instruction; do not claim a uniform count.
5. **Policy solvability loop.** In the architecture figure; not in the code. Describe as drawn.
6. **Wipe `select_only` vs the A7 sheet.** A7 reports a bridging-off number for GridWorld and
   Wipe even though D3 says bridge share should be 0 per cent on path-randomized tasks; D3 in
   fact reports 27-30 per cent bridge on Wipe and GridWorld, contradicting its own stated
   hypothesis ("exactly 0% on Wipe … and GridWorld"). The code makes Wipe SELECT-only.
   **Flag this; do not build an argument on the Wipe/GridWorld bridge share.**
7. **D5 compute sheet is empty.** Use telemetry (9.0k tokens/round, ~563 s/round) and label it.
8. **A15 Top-1 row is "Did not compute".** Not a null result. Never report it as one.

---

## 10. What the repository does NOT contain (do not invent it)

- No DISEIL naming, and no acronym derivation.
- No PushT cell in the consolidated module; no GridWorld image cell in the consolidated module.
- No policy-solvability check in code.
- No filled compute/wall-clock table in the workbook.
- No per-task sigma (the identified fix for the mis-scaled kernel is proposed, never run).
- No third policy class (Gaussian-MLP BC / ACT) — the policy-agnostic claim rests on
  CNN + MLP + diffusion (state and image encoders), which is two paradigms, and the supervisor
  has already flagged it.
- `COC_REPORT/figures_generated/` is empty.
