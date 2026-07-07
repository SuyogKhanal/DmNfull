# PACE Method Dossier (math-ready)

Verified against the implementation, not hearsay. Every parameter below is quoted
with its **exact code name** and **default**. Sources are absolute paths; a symbol
table for the equations agent is in §8.

PACE = **P**erceive → **P**artition → **P**rioritize → **P**rescribe. The four
stages map to code as:

| Stage | Role | Implementation |
|---|---|---|
| Perceive  | VLM localises the failure from 3 frames + KAG | `p4/vlm.py`, `p4/runner.py`, `p4/kag.py` |
| Partition | code-level clustering of the round's failures | `p4_subtask/clustering.py`, `p4_subtask/embedding.py` |
| Prioritize| farthest-point diversity + forced dominant rep + worst-loss seed | `p4_subtask/diversity.py`, `p4_subtask/memory.py` |
| Prescribe | LLM SceneCommand → ResetSpec / sub-task entry with escalation | `p4_subtask/planner.py`, `p4_subtask/subtask_entry.py`, `p4_subtask/collect.py` |

The **robot task** (T2 Push = ManiSkill PushT) is the fully-instantiated path.
The **toy grid** (T1) uses the same VLM→reason→prescribe→cap-1 engine but a
corridor prescription instead of a pose (`pool_x_selector/p4/`). T3–T5 (Lift/
Wipe/Door, RoboSuite) reuse the generic 3-component `p4/` modules
(`vlm/analysis/prescription/config_generator`) and are not yet wired to a fork
pipeline (the p4 package explicitly says the fresh modules "are the
implementation path for the non-PushT tasks").

---

## 1. Failure DESCRIPTOR feature vector (Partition input, per failure)

Source: `p4_subtask/descriptor.py`.

### 1.1 Which frame is analysed (the "failure point" t\*)
Each failed episode writes a `meta.json` via `HighLossImageSaver`
(`diffdagger/util/high_loss_image_saver.py`). During the failure-discovery
rollout the policy is queried in DAgger mode at every step and returns a scalar
**diffusion loss** `ℓ_t` (`action_dict["diffusion_loss"]`); `log_step` buffers
`(timestep, loss, cdf_value, query, frame, state)`. A frame is written to
`high_loss/` iff `ℓ_t > loss_threshold` (default `loss_threshold = 0.05`,
overridden by the policy's adaptive p95 threshold: `_compute_adaptive_threshold`,
`percentile = 95` = "top 5% are high loss"). The saver **always** keeps the
single episode-max frame `_peak_frame`; if nothing crossed the threshold it
back-fills `high_loss_frames = [_peak_frame]` so t\* is never empty.

`descriptor._peak_frame(meta)` selects the analysed frame:
1. **peak-loss frame**: over `high_loss_frames` with a non-null `obj_pose_xyz`,
   take `argmax` of `diffusion_loss` → this is **t\*** (peak diffusion-loss
   timestep = "the moment the policy was most uncertain ≈ the failure point").
2. **fallback**: `end_frame_state`, then `start_frame_state` (first with a valid
   `obj_pose_xyz`). Returns `None` (→ planner inert, geometric fallback) if none.

`peak_loss` per failure = `meta["peak_loss"]` if present, else the frame's
`diffusion_loss`; `mean_loss = meta["mean_loss"]`. `_augment_meta_losses`
(`pipeline.py`) derives `peak_loss = max` and `mean_loss = mean` over all
frame losses (high-loss frames + start/end states).

### 1.2 Raw state at t\* (from `_extract_env_state`, PushT / Panda-stick)
`obj_pose_xyz` (tee, 3), `obj_pose_quat_wxyz` (tee, 4), `arm_qpos_rad` (7 arm
joints), `gripper_finger_pos_m` (2 finger entries — PushT panda_stick has no
gripper so these are the last 2 of the 9-d `agent_qpos`), `tcp_pose_xyz` (stick
TCP, 3). Stored on the `FailureDescriptor`:
`tee_xyz` (3), `tee_theta` (scalar yaw), `arm_qpos` (7), `full_qpos = arm+grip`
(9, exactly what `robot.set_qpos` wants), `tcp_xyz` (3). Yaw is recovered from
the quaternion (`yaw_from_quat_wxyz`, z-only rotation):
`θ = atan2(2(wz+xy), 1−2(y²+z²))`.

### 1.3 The 6-D clustering feature vector `feature()`
The **geometric descriptor** is **6-dimensional** (quaternion-free — orientation
as sin/cos, so neither the LLM nor the clusterer reasons about raw quaternions):

```
φ = [ tee_x,
      tee_y,
      sin(tee_θ),
      cos(tee_θ),
      progress = t*/max(1,T),
      tcp_to_tee = ||tcp_xyz − tee_xyz||₂ ]     # 3-D Euclidean contact distance
```

Components: (1–2) tee planar position; (3–4) tee orientation sin/cos; (5) task
progress `t*/T` (T = `total_steps`/`end_timestep`); (6) TCP-to-tee 3-D distance
= a **contact discriminator** (near-contact vs no-contact failures).

`cluster_feature()` returns `visual_embedding` (§2) for image-based runs when one
was attached this round, **else** the 6-D `feature()`. Prescription/reset never
use `cluster_feature()` — they use the raw tee/qpos geometry — so the visual↔
geometric swap is confined to *mode discovery*.

---

## 2. Visual embedding path (image-based runs only)

Source: `p4_subtask/embedding.py`; wired in `planner._attach_visual_embeddings`.
Active only when `cfg.cluster_features == "visual"` (default `"geometric"`).

### 2.1 Key-frame selection per failure
`descriptor._frame_paths` returns absolute paths for **3 key-frames**:
`start` = `start_frame.png`; `peak` = the highest-`diffusion_loss` buffered frame
in `high_loss/` (**= t\***; falls back to `start` if none); `end` =
`end_frame.png`. Missing files are dropped.

### 2.2 Encoder
`FrameEmbedder(model_name="r3m", pool="concat", image_size=224)`. Each frame →
RGB, resized to 224×224, **fed to R3M in [0,255]** (no ImageNet normalisation,
per R3M convention) → **512-d** per frame. `model_name`:
- `"r3m"` (default) = `r3m.load_r3m("resnet18")`, weights cached offline at
  `~/.r3m/r3m_18/model.pt`.
- `"resnet18"`/`"imagenet"` = torchvision ResNet-18 (`IMAGENET1K_V1`) with
  `fc=Identity` → 512-d.
Encoder is frozen (`requires_grad_(False)`, `eval()`), lazily loaded once,
per-path cached. Any import/frame error → `embed_failure` returns `None`.

### 2.3 Pooling to a per-failure vector
- `pool="concat"` (default): keep start→t\*→end ordering; pad/truncate to
  exactly 3 slots → **1536-d** (`3×512`).
- `pool="mean"`: mean over frames → **512-d**.

### 2.4 Round-level PCA reduction
Across the round's `N` failures, stack raw vectors `M ∈ ℝ^{N×d}`
(`d`=1536 or 512), then `pca_reduce(M, k = embed_pca_dim = 16)`:
standardize columns (z-score), centered **economy SVD** `Xs = UΣVᵀ`, project
onto the top `k' = min(k, N−1, d)` right-singular vectors → **N×k'** scores
(pure numpy, no sklearn). `N ≤ 2` returns the standardized features unchanged.
Every failure must embed (`require EVERY failure embedded`) or the whole round
**falls back to geometric** (`cluster_features` stays effectively geometric that
round). Reduced vector is written to `desc.visual_embedding`; clustering +
diversity then run in this 16-D visual space, but **cluster geometry
(centroid_xyz/θ) and all prescription geometry stay on the raw tee pose**.

---

## 3. Partition — CLUSTERING

Source: `p4_subtask/clustering.py`, `cluster_failures(descs, max_k=6)`.

- **Feature space**: `cluster_feature()` per failure = visual embedding (§2) or
  6-D geometric (§1.3). Assembled into `feats ∈ ℝ^{N×D}`.
- **Standardization** (`_standardize`): per-column z-score
  `X̃ = (X − μ)/σ`, with `σ` floored: `σ[σ<1e-8]=1`. Applied to the feature
  matrix before clustering AND (separately) for representative selection.
- **Distance metric**: **Euclidean** in standardized space (silhouette uses
  `np.linalg.norm` pairwise L2; sklearn `AgglomerativeClustering` default
  affinity = euclidean, default linkage = ward).
- **Small-N guards**: `N==0` → empty; `N ≤ 3` → each failure is its own cluster
  (`method="singletons"`).
- **k selection** (`_best_k_clustering`): sweep `k ∈ [2, kmax]`,
  `kmax = _choose_k(N, max_k) = max(2, min(max_k, N−1))`, pick the labelling
  with the **highest mean silhouette**.
- **Primary path**: `sklearn.cluster.AgglomerativeClustering(n_clusters=k)` +
  `sklearn.metrics.silhouette_score`. Method string
  `"sklearn-silhouette(k*=…)"`.
- **Fallback path** (sklearn absent): deterministic **numpy single-linkage**
  agglomerative (`_numpy_single_linkage`: iteratively merge the two clusters
  with the minimum inter-cluster min-distance until `k` remain) +
  `_numpy_silhouette`. Method string `"numpy-silhouette(k*=…)"`. The chosen
  path is recorded in telemetry for reproducibility.
- **Per-cluster geometry** (always from RAW tee pose, feature-space-independent):
  centroid `(cx,cy)` = mean of member `tee_xyz`; circular-mean yaw
  `θ̄ = atan2(mean sin θ, mean cos θ)`; `centroid_xyz = KB.clamp_tee([cx,cy,TEE_Z])`;
  `mean_peak_loss` = mean member `peak_loss`.
- **Representative** = the member **nearest the cluster centroid in standardized
  feature space** (`argmin_i ||X̃_i − mean_cluster(X̃)||`).

### 3.1 Dominant cluster (`pick_dominant`)
Lexicographic key **maximised**: `(size, mean_peak_loss, −min_episode_id)`
— most members; tie-break highest mean peak-loss; final tie-break lowest min
episode id (determinism).

---

## 4. Prioritize — DIVERSITY + memory rotation

### 4.1 Farthest-point (k-center greedy, max-min) selection
Source: `p4_subtask/diversity.py`, `farthest_point_select(descs, k, force_first)`.
Same standardized feature space as clustering. Build selection set `S` (returns
≤ `k` global indices, `k` = `cap` = `top_k`, default 3):
1. **force-include** `force_first` = the **target cluster's representative**
   (`target.representative_idx`) — guarantees the failure we anchor the demo on
   is analysed.
2. **worst-loss seed**: add `argmax_i peak_loss_i` (the hardest failure) if not
   already in `S`.
3. **greedy max-min** while `|S|<k`:
   `next = argmax_{i∉S} min_{j∈S} ||X̃_i − X̃_j||₂` (k-center farthest-point).

So `S` = { forced dominant/target representative } ∪ { worst-peak-loss } ∪
{ farthest-point cover }. When `diversity=False` (ablation), selection is
peak-loss-ordered top-cap with the rep forced first.

### 4.2 Cross-round centroid MEMORY (target-cluster rotation)
Source: `p4_subtask/memory.py` (`CentroidMemory`), used by
`memory.select_target`. Because we retrain + re-rollout after **every** demo, the
failure distribution shifts each round; a persistent mode could be prescribed
repeatedly. Memory rotates coverage.
- Persisted JSON `telemetry/centroid_memory.json`; each success appends
  `(round, centroid_xyz, centroid_theta)`.
- **Recency penalty** (Gaussian, recency-discounted) at query point `c`:
  `P(c) = Σ_i γ^{max(0, now−r_i)} · exp( −‖c_xy − c_i,xy‖² / (2σ²) )`
  with `memory_gamma (γ) = 0.6`, `memory_sigma (σ) = 0.06`. `P≈1` on a very
  recently covered point, `→0` far away.
- **Target selection** among clusters that tie the dominant within one member
  (`size ≥ dominant.size − 1`): pick
  `argmax_c ( mean_peak_loss(c) − λ · P(centroid_c) )`, `lam (λ) = 1.0`.
  Chosen cluster = the round's **target** (≠ dominant when rotation kicks in).
The **anchor** = `build_anchor(descs[target.representative_idx])`.

---

## 5. Prescribe — LLM SceneCommand → ResetSpec / sub-task entry

Source: `p4_subtask/planner.py` (`reset_spec_for`, `_hybrid_spec`),
`p4_subtask/subtask_entry.py`, `p4_subtask/collect.py`. Default
`collect = "hybrid"`.

### 5.1 The LLM output (SceneCommand `cmd`)
Fields read: `cmd.tee_xyz` (absolute in-bounds tee, 3), `cmd.tee_zrot` (scalar
yaw), `cmd.tcp_xyz` (3), `cmd.label` (carries the decision tag). Guidance is
injected via always-on per-round prompt addenda (fork hook B); prompt bounds are
the hard KAG bounds (§7).

### 5.2 Collection modes (`collect`)
- **`"hybrid"` (default, v3)** — LLM decides per round:
  - **SELECT ep<ID>** → on-policy correction of one recorded failure.
  - **BRIDGE ep<ID>,ep<ID>[,ep<ID>]** → one new middle-ground pose covering
    2–3 cited failures. A BRIDGE pose must lie within `bridge_max_xy = 0.10` m /
    `bridge_max_theta = 0.6` rad of the nearest cited member. Without an explicit
    tag: pose within `snap_eps = 0.02` of a member ⇒ SELECT, else BRIDGE.
- **`"onpolicy"` (v2)** — always snap the LLM pose to the nearest **untried**
  real failure and correct it on-policy (pure select-by-pointing).
- **`"teleport"` (v1, ablation)** — always anchored-perturbation reset.

### 5.3 The four ResetSpec constructors (`subtask_entry.py`)
`ResetSpec(tee_xyz, tee_zrot, tcp_xyz, agent_qpos, mode, seed, provenance)`.
- **ONPOLICY** (`onpolicy_correction_spec`) — `mode="onpolicy_correction"`,
  `seed = member.seed`, `agent_qpos=None`; the demo = re-roll the failure's scene
  seed with the *current* policy, then the expert corrects on-policy from the
  policy's own divergence state t\* (Diff-DAgger-faithful; the tee fields are
  informational). This is the winning mechanism (`p4_select` beat Diff-DAgger
  5/5 on PushT). Implementation `collect.collect_onpolicy_correction`:
  re-roll → if re-roll now *succeeds*, budget-free **skip**
  (`skipped="solved_on_reroll"`); else replay prefix to argmax-loss t\*, expert
  takes over, only the expert segment is the demo.
- **PERTURB** (`apply_coverage_perturbation`) — `mode="perturbed"`,
  `agent_qpos = anchor.full_qpos`. Shift the anchor tee toward the LLM coverage
  point, **L2-capped** at `max_xy`: if `‖Δxy‖ > max_xy` scale to `max_xy`; yaw
  shift `Δθ = clamp(wrap_π(zrot_llm − θ_anchor), −max_theta, +max_theta)`. TCP is
  re-derived in the tee frame (`_rederive_tcp`: express anchor TCP-offset in the
  anchor tee frame, re-apply at the new tee/orientation) so the stick stays on
  the same push face. Everything clamped to KAG bounds.
- **EXACT** (`exact_anchor_resetspec`) — `mode="exact_anchor"`; zero-perturbation
  replay of the recorded t\* state (`agent_qpos = anchor.full_qpos`). Safe,
  penetration-free fallback.
- **FRESH** (`fresh_tee_resetspec`) — `mode="fresh_tee"`, `agent_qpos=None`; the
  LLM's absolute pose with a canonical robot (≡ p4_top3 behaviour). Last resort.

### 5.4 ESCALATION (deterministic, keyed on within-round re-prescribes)
`_escalation` = # of re-prescribes this round (0 on the first prescription;
bumped on each infeasible/empty retry). Caps: `p4_represcribe_attempts = 5`,
`p4_infeasible_attempts = 5`.
- **hybrid**: escalation 0 honours the LLM SELECT/BRIDGE; escalation ≥ 1 forces
  **SELECT of the nearest untried real failure** (safety floor — a bad bridge
  can never waste a round). Member pool exhausted → other clusters by
  (size desc, peak-loss desc), then repeat overall worst failure.
- **teleport (v1)** escalation ladder: `esc≤0` → PERTURB; `esc==1` (with
  `exact_anchor_fallback=True`) → EXACT; `esc≥2` → FRESH.

### 5.5 Coverage perturbation ranges (defaults, `planner.__init__` cfg)
`perturb_max_xy = 0.06` m, `perturb_max_theta = 0.4` rad (teleport PERTURB);
`bridge_max_xy = 0.10` m, `bridge_max_theta = 0.6` rad (hybrid BRIDGE);
`snap_eps = 0.02` (SELECT snap threshold); `max_clusters (max_k) = 6`;
`diversity = True`; `exact_anchor_fallback = True`;
`confirm_target_with_heldout = True` (a rollout-SR target hit must be confirmed
by the frozen held-out eval before stopping).

### 5.6 Snap metric (which real failure a pose maps to)
`_pose_dist(d, x, y, θ) = hypot(d.tee_x−x, d.tee_y−y) + 0.1·|wrap_π(d.tee_θ−θ)|`
(planar distance + 0.1 m/rad orientation term).

---

## 6. Perceive — VLM + KAG context injection

### 6.1 VLM three-frame localisation
Source: `p4/vlm.py` `analyze_failure`. Frames `start_frame`, `high_loss_frame`
(**t\* = peak diffusion-loss timestep**), `end_frame` are each captioned by the
VLM (`client.analyze_frame`, system prompt `VLM_SYSTEM`, per-role user prompt
`vlm_prompt(task_description, role, t_star)`, ~120 words each). Failure modes are
**not hard-coded** — inferred from frames + task text. Output = one combined
`=== VLM FAILURE REPORT ===` with a section per frame.

Frame selection for the VLM (`frame_selector.select_frames`, fork engine) picks
exactly **5 frames**: `start` + **top-3** highest-`diffusion_loss` frames + `end`
(`top_k = analyzer.frames.top_k_high_loss`). In P4-top3/subtask the config sets
`top_k_high_loss = 1`, so the VLM sees start + single t\* + end (3 frames).

Transport (`p4/runner.py` `P4Client`): fork Responses-API clients
(`VLMClient`/`ReasoningClient`/`PlainClient`). VLM model
`VLM_MODEL_NAME = "qwen3-vl-32b"` (effort `"low"`); reasoning/config
`LLM_MODEL_NAME = "qwen3-32b"` (reasoning effort `"high"`, plain `"low"`);
`max_output_tokens = 16384`. VLM is the only **cached** stage
(`_vlm_cached`, keyed `(iteration, episode_id)`, on disk under
`work_dir/vlm_cache`). `make_client` returns `None` if `OPENAI_BASE_URL` unset
(graceful text-only fallback).

The full downstream chain (robot tasks, fresh modules): VLM report →
**Analysis LLM** (`analysis_prompt`, classifies `root_cause ∈
{grasp_failure, approach_failure, placement_error, contact_instability,
pose_mismatch, timeout}`, `phase ∈ {pre_grasp, grasp, transport, placement,
insertion}`) → **Prescription LLM** (`prescription_prompt`, or cube variant) →
**Config LLM** (`config_prompt` → simulator reset config). Each prescription is
hard-capped to **one** corrective demo, with a **non-empty HARD-FLOOR**
(`OUTPUT_REQUIREMENT`: an empty/refused prescription raises
`PrescriptionEmptyError` and is retried — an empty prescription = zero demos =
the worst outcome).

### 6.2 KAG (Knowledge-Augmented Generation)
Source: `p4/kag.py`, graph `p4/kag/PushT-v1.json`. Per-task knowledge **graph**
(`meta`, `nodes`, `edges`, `reasoning_implications`) rendered to prompt text by
`format_kag_context` (nodes-by-type, relations, reasoning implications) and
written to `p4/kag/{env}.kag.txt`, injected via `analyzer.kag_path`. The PushT
graph encodes: robot (Panda panda_stick, 7 arm joints, no gripper), tee, fixed
goal (`goal_offset=[-0.156,-0.1]`, `goal_z_rot_rad`), TCP, 21-d state obs,
controller (`pd_joint_pos`, action_dim 7 = rel_joint_pos deltas), success
(T∩goal), a **failure taxonomy** (`wrong_approach`, `overshoot`, `no_contact`,
`wrong_orientation`, `timeout`), 4 phases, and the **workspace bounds**
(`ws_tee`, `ws_tcp`) that ground every prescription. `reasoning_implications`
give per-failure-mode prescription rules + the `workspace_constraint` /
`non_emptiness` hard rules.

### 6.3 Grounded semantic directions (prompt side)
Source: `p4_subtask/semantic_map.py`. Goal T at `GOAL_XY=(-0.156,-0.1)`,
`GOAL_ZROT=(5/3)π`. `semantic_lines` give the anchor→goal unit vector + distance
+ bearing and the anchor→centroid offset, so the LLM's absolute-pose output is
geometrically anchored (Fix-3 spirit; PushT is quaternion-free with scalar
`tee_zrot`).

---

## 7. KAG / hard workspace bounds (`p4_subtask/kag_bounds.py`, PushT)

Single source of truth for clamping (mirrors `PushT-v1.json` `ws_tee`/`ws_tcp`):
- tee (movable T): `TEE_X = (−0.20, 0.20)`, `TEE_Y = (−0.25, 0.05)`,
  `TEE_Z = 0.021`.
- tcp (stick EE): `TCP_X = (−0.35, 0.35)`, `TCP_Y = (−0.35, 0.35)`,
  `TCP_Z = (0.02, 0.08)`.
`clamp_tee`/`clamp_tcp` project poses into these boxes; out-of-box prescriptions
are dropped/clamped (the PPO expert is unreliable outside its training region).

---

## 8. Toy grid (T1) prescription path (`pool_x_selector/p4/`)

Same VLM→reason→prescribe engine (sequential profile, prescription **hard-capped
to 1**, `mode_directive`), but the failure signal and prescription are grid-shaped.
Failures are ranked by per-step loss (`rank_failures`); the LLM is shown the
**top-3** highest-loss failures (`top_k=3`, p4_top3) or **all** (`top_k=None`,
p4_all). It prescribes ONE `layout` = `{start_pos:[r,c], goal_pos:[r,c],
fire_positions:[[r,c]…], steps:"(r,c)->…"}` (5×5 grid, ints 0..4). The prescribed
corridor is materialised by an A\* expert forced down the corridor
(`demo_collector.collect`, corridor blocking); an A\*/BFS feasibility check
rejects corridors that step on fire / leave the grid / skip cells / don't connect
start→goal, and rejections are fed back into later prompts
(`infeasible_feedback_block`). An **info-gain proxy** rolls the current
pre-finetune policy on each kept layout (high loss = high info gain). Maze env
(`envs/maze_env.py`): obs is a `Dict{state: Box(14,), image: Box(80,80,3)}` →
STATE policy = equivariant MLP; IMAGE policy = plain CNN. Stop reasons:
`target_hit / budget_exhausted / pool_solved / no_progress
(max_consecutive_empty = 8) / max_rounds / llm_error`.

---

## 9. NOTATION (define every symbol for the equations agent)

**Indices / sets**
- `r` — DAgger round index (a.k.a. `rnd`, `now` for memory). One prescribed demo
  per round (sequential).
- `i, j` — failure indices within a round; `N` — number of usable failures
  (descriptors) this round.
- `𝔉_r = {f_1,…,f_N}` — the round's failure set.
- `S` — the diversity-selected failure subset, `|S| ≤ k`.
- `𝒞 = {C_1,…,C_m}` — clusters of `𝔉_r`; `k` (a.k.a. `k*`, `kmax`) — number of
  clusters (silhouette-chosen). `|C|` = cluster size.

**Per-failure quantities**
- `ℓ_t ∈ ℝ_{≥0}` — diffusion loss at rollout step t (`diffusion_loss`).
- `t*_i = argmax_t ℓ_t` — peak-loss timestep of failure i (the analysed frame).
- `T_i` — total steps of failure i (`total_steps`).
- `peak_i = max_t ℓ_t` (`peak_loss`); `mean_i = mean_t ℓ_t` (`mean_loss`).
- `p_i^tee = (tee_x, tee_y, tee_z) ∈ ℝ³` (`tee_xyz`); `θ_i` = tee yaw
  (`tee_theta`).
- `p_i^tcp ∈ ℝ³` (`tcp_xyz`); `q_i^arm ∈ ℝ⁷` (`arm_qpos`);
  `q_i^full ∈ ℝ⁹` (`full_qpos` = arm ⊕ 2 finger).
- Progress `ρ_i = t*_i / max(1, T_i)`; contact distance
  `δ_i = ‖p_i^tcp − p_i^tee‖₂`.

**Feature vectors**
- `φ_i = [tee_x, tee_y, sin θ_i, cos θ_i, ρ_i, δ_i] ∈ ℝ⁶` — geometric descriptor
  (`feature()`).
- `v_i ∈ ℝ^{d}` — raw R3M key-frame embedding (`d=1536` concat / `512` mean);
  `ṽ_i ∈ ℝ^{k'}`, `k' = min(16, N−1, d)` — PCA-reduced visual feature
  (`visual_embedding`).
- `ψ_i = cluster_feature(i) = ṽ_i` (visual mode) `else φ_i`.
- `X = [ψ_1;…;ψ_N] ∈ ℝ^{N×D}`; standardized `X̃ = (X − μ)/σ`
  (`_standardize`, `σ` floored at 1e-8).

**Clustering / dominance**
- `sil(k)` — mean silhouette at k clusters; `k* = argmax_k sil(k)`,
  `k ∈ [2, kmax]`, `kmax = max(2, min(max_k, N−1))`, `max_k = 6`.
- `c_C^{xy}` — cluster centroid (mean member tee xy, clamped to `TEE_Z`);
  `θ̄_C = atan2(Σ sin θ, Σ cos θ)` — circular-mean yaw;
  `L̄_C = mean_{i∈C} peak_i` (`mean_peak_loss`).
- `rep(C) = argmin_{i∈C} ‖X̃_i − mean_{C}(X̃)‖₂` — cluster representative.
- dominant cluster `C* = argmax_C (|C|, L̄_C, −min_i∈C episode_id)` (lexicographic).

**Prioritize / memory**
- `k = cap = top_k` (default 3) — # failures analysed / selected.
- k-center greedy: `S ← {rep(target)} ∪ {argmax_i peak_i}`, then
  `S ← S ∪ {argmax_{i∉S} min_{j∈S} ‖X̃_i − X̃_j‖₂}` until `|S|=k`.
- Recency penalty `P(c) = Σ_{i} γ^{max(0, r−r_i)} exp(−‖c − c_i‖²/(2σ²))`;
  `γ = 0.6` (`memory_gamma`), `σ = 0.06` (`memory_sigma`); memory entries
  `(r_i, c_i, θ_i)`.
- Target cluster `C_tgt = argmax_{C: |C|≥|C*|−1} (L̄_C − λ·P(c_C^{xy}))`,
  `λ = 1.0` (`lam`).

**Prescribe / reset**
- Anchor `A = (p_A^tee, θ_A, q_A^full, p_A^tcp, t*_A, T_A)` from `rep(C_tgt)`.
- LLM command `cmd = (p^{tee}_{llm}, θ_{llm}, p^{tcp}_{llm}, label)`.
- Perturb: `Δxy = p^{tee}_{llm,xy} − p_A^{tee,xy}`, capped
  `Δxy ← Δxy · min(1, max_xy/‖Δxy‖)`;
  `Δθ = clamp(wrap_π(θ_{llm} − θ_A), −max_theta, +max_theta)`; new tee
  `p^tee = clamp_tee(p_A^tee + [Δxy,0])`, `θ = θ_A + Δθ`; TCP re-derived by
  rotating the anchor TCP-offset from `−θ_A` to `+θ` and re-adding.
  `(max_xy, max_theta)` = `(0.06, 0.4)` perturb / `(0.10, 0.6)` bridge.
- Snap distance `d_snap(f, p, θ) = ‖p_f^{tee,xy} − p_{xy}‖₂ + 0.1·|wrap_π(θ_f − θ)|`.
- `wrap_π(a) = ((a+π) mod 2π) − π`.
- Escalation `e ∈ {0,1,2,…}` (`_escalation`); ladder (teleport): `e=0`→perturb,
  `e=1`→exact, `e≥2`→fresh; (hybrid): `e=0`→LLM SELECT/BRIDGE, `e≥1`→forced SELECT.

**Bounds** — `TEE_X=(−.20,.20)`, `TEE_Y=(−.25,.05)`, `TEE_Z=.021`;
`TCP_X=TCP_Y=(−.35,.35)`, `TCP_Z=(.02,.08)`.

---

## 10. Config knobs & defaults (one place)

`SubtaskPlanner` cfg (`cfg["p4_subtask"]`): `max_clusters=6`, `diversity=True`,
`perturb_max_xy=0.06`, `perturb_max_theta=0.4`, `exact_anchor_fallback=True`,
`collect="hybrid"`, `signal="diffusion"`, `cluster_features="geometric"`,
`embed_model="r3m"`, `embed_pool="concat"`, `embed_pca_dim=16`, `snap_eps=0.02`,
`bridge_max_xy=0.10`, `bridge_max_theta=0.6`, `confirm_target_with_heldout=True`,
`memory_gamma=0.6`, `memory_sigma=0.06`.
Engine (`k[...]`): `top_k` (VLM/cluster cap, =3), `frames.top_k_high_loss=1`,
`p4_represcribe_attempts=5`, `p4_infeasible_attempts=5`, `p4_high_loss_percentile`
(threshold percentile, p95), `mode="sequential"` (1 demo/round),
`heldout_seed_base=7777`. VLM `qwen3-vl-32b` (low), LLM `qwen3-32b` (high),
`max_output_tokens=16384`, threshold `loss_threshold=0.05` / adaptive p95.

Source paths (all absolute):
- `/weka/s226137394/DmNfull/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_rl_robo/p4_subtask/{descriptor,clustering,diversity,embedding,subtask_entry,memory,planner,collect,kag_bounds,semantic_map}.py`
- `/weka/s226137394/DmNfull/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_rl_robo/p4/{vlm,kag,pipeline,prompts,runner}.py`, `p4/kag/PushT-v1.json`
- `/weka/s226137394/DmNfull/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_x_selector/p4/{pipeline_p4,prompts,demo_collector}.py`
- `/weka/s226137394/diff-dagger/diffdagger/util/high_loss_image_saver.py`, `diffdagger/main_pipeline/{sim_bridge,pipeline}.py`, `diffdagger/main_analysis/frame_selector.py`
- `/weka/s226137394/DmNfull/envs/maze_env.py`
