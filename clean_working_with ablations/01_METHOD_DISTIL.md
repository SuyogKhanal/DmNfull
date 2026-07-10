# 01 — DISTIL, the method (from `paper.pdf` + the new architecture)

**DISTIL = "Demonstration Distillation for Sample-Efficient Imitation Learning."** It reframes
budget-constrained interactive IL: instead of deciding *when* to query the expert (the DAgger
question), decide *which* failure modes to correct and *how* to place each corrective demo,
spending a fixed budget only on the most informative corrections. **Policy-agnostic** — needs
only a learner `f_θ` whose per-step training loss `ℓ_t` can be read along rollouts (cross-entropy
for GridWorld CNN/MLP; v-prediction/denoising loss for the diffusion policies).

> The `Architectural Diagram.pdf/html` in this folder is the authoritative picture. This file is
> the text version. Where the paper's default differs from THIS run, the delta points to
> `02_DESIGN_CHANGES_THIS_RUN.md`.

## Algorithm 1 — the DISTIL loop (one demonstration per round, budget B=20)
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
Cadence: retrain **every round** (GridWorld), **every 4th demo** (robots). Budget counts only
**recorded (successful) demonstrations**; a no-failure pool or an infeasible prescription costs
nothing.

## Equations (define every symbol; quote-faithful)
- **Eq 1 — train:** `θ* = argmin_θ E_{(s,a)~D}[ ℓ(f_θ;s,a) ]`. `ℓ` = cross-entropy (GridWorld) / v-prediction denoising loss (robots).
- **Eq 2 — per-step uncertainty:** `ℓ_t = ℓ(f_θ; s_t, a_t^nov)` — the training loss at the learner's OWN executed action (novice), a self-uncertainty/OOD signal along the rollout.
- **Eq 3 — aggregate+retrain:** `D_{r+1}=D_r ∪ d_r`, `θ_{r+1}=argmin_θ E_{D_{r+1}}[ℓ]` (from scratch).
- **Eq 6 — failure point:** `t*_i = argmax_t ℓ_t^(i)`, `peak_i = max_t ℓ_t^(i)`, over steps kept by an **adaptive threshold = 95th percentile** of the episode's loss.  ⟵ **DELTA (`02_...md` #1): this run uses `t_flag` = the FIRST threshold crossing, not the argmax peak, for the descriptor anchor AND the SELECT takeover.**
- **Eq 7 — descriptor:** `φ_i = [x, y, sinθ, cosθ, ρ=t*/T, δ=eef-contact-dist]` (6-D, quaternion-free), anchored at the failure point. Paper: image runs use a frozen **R3M** embedding (PCA→≤16-D). ⟵ **DELTA (`02_...md` #4): this run uses the GEOMETRIC φ for image runs too — no R3M.**
- **Eq 8 — k-selection:** `k* = argmax_{k∈[2,k_max]} sil(k)`, `k_max=max(2,min(6,N−1))`; `N≤3` → singletons; **dominant `C*` = largest cluster, tie → mean peak loss `L̄_C`**; `rep(C)=argmin_{i∈C}‖X̃_i−X̃_C‖`.
- **Eq 9 — cluster memory + target rotation (the allocation mechanism, the crown-jewel claim):**
  `P_mem(c)=Σ_{(r_i,c_i)∈Mem} γ^{r−r_i}·exp(−‖c_xy−c_{i,xy}‖²/(2σ²))`;
  `C_tgt = argmax_{C:|C|≥|C*|−1} ( L̄_C − λ·P_mem(c_C) )`; **γ=0.6, σ=0.06, λ=1**.
  Recency-discounted spatial coverage penalizes recently-corrected regions → **rotation** to new
  modes. **Near-dominant constraint `|C|≥|C*|−1`** keeps rounds on near-dominant modes (rare modes
  reached only after big ones are retrained + penalized). Kernel is **planar-xy only** (yaw-blind).
- **Context set S:** `|S|≤κ=3` — forced **worst-peak-loss** representative + **farthest-point
  selection** fill in descriptor space.
- **Eq 10 — BRIDGE (Push-T):** `Δ_xy = cmd.p_xy − p_{A,xy}` capped to `‖Δ_xy‖≤Δ_max`; `Δ_θ =
  clamp(wrap_π(cmd.θ−θ_A), ±θ_max)`; `ξ = clamp_W(p_A+[Δ_xy,0], θ_A+Δ_θ, q_A)`. Caps **(0.06 m,
  0.4 rad)** on Push-T; **Lift/Door use an absolute clamp** to the reset range (Lift ±0.03 xy,
  θ_max=0; Door small xy, reuse rep orientation); **Wipe: BRIDGE infeasible → SELECT-only**;
  GridWorld: prescribe a corridor layout. `q_A` = representative's carried-over height+orientation.

## The two arms (SELECT ⟷ BRIDGE — LLM chooses freely every round)
- **SELECT (targeted correction):** re-instantiate a cited failure's scene, replay the policy to
  its divergence point (`t_flag` this run), expert takes over. `d_r = {(s_t, π*(s_t)) : t ≥ t_flag}`.
- **BRIDGE (bridging placement):** place ONE new start `ξ` (Eq 10) between 2–3 cited failures of a
  mode; expert demonstrates the whole episode from `ξ`; one demo covers several failures.

## Step 5b — feasibility + re-prescription (the infeasibility loop)
Every prescription is validated (robot: respect KAG bounds `W`, cite real failures; GridWorld:
A*/BFS corridor validity) BEFORE any expert effort. Infeasible/empty → rejected + re-prescribed
with the reason fed back; after **≤5 attempts** → deterministic fallback to the nearest untried
failure. **No budget on an infeasible prescription.** (This run also invokes it when the expert
*attempts and fails* — esp. PushT's clockwise-only PPO capability gap — see `02_...md` #5.)

## Confidence (Q3) — REQUIRED every round
The prescription carries a **self-reported confidence**. `02_...md` #6: the decision prompt must
explicitly ask for it; log per round. The paper reports confidence↔ΔSR **Pearson r=0.82–0.89**
(with the pseudoreplication caveat — see `05_...md` Tier 5 / Q3).

## Research questions + contributions (what you're proving)
- **Q1** sample efficiency: at B=20, higher final SR than when-to-query baselines.
- **Q2** informativeness→allocation: more info gain per demo, tied to allocation across modes.
- **Q3** confidence predicts improvement + modes are semantically meaningful.
- Contributions: (1) reframe as demonstration *distillation*; (2) the DISTIL loop; (3) 5 tasks × 2
  modalities × ≤6 baselines, best mean in all 10 at B=20; (4) online↔offline bridge (each
  prescription is a standing artifact — expert can record it later).

## The claims a reviewer attacks (know these; they drive `05_ABLATIONS.md`)
Overlapping 1-σ error bars (defend by *ranking consistency* + sign test); non-standard pooled
denominator; **fairness/privileged info** (authored KAG + object poses even on image runs);
**missing random-allocation control on robots** (Stagger was GridWorld-only); oracle experts;
info-gain is a novelty *proxy* not a formal measure; correlation pseudoreplication (~25 pts);
per-task KAG + hand-designed φ + memory constants are *designed not learned*; bridging needs a
**resettable simulator**. The one clean causal story is **allocation (clustering+memory) → coverage
→ success** — spend compute proving THAT.

## Protocol (baseline; adjust per `00`/`02`)
Same BC init per seed (**20 init demos, EXCLUDED from budget**); **B=20**, 1 demo/round; retrain
from scratch (every round GridWorld / every 4th robots); held-out eval on a frozen set (200
GridWorld layouts / 100 robot episodes) at every retrain. **Seeds: this run = 5 for every cell**
(the user's instruction; the paper used 9 GridWorld / 5 robot — note the change). Baselines:
Diff-DAgger + the gate family (Safe, Dropout, Ensemble, Thrifty, Stagger). **GridWorld expert is
a human in the paper; the code uses an AStar/BFS oracle — decide which for the clean run and state
it** (BFS oracle is what makes it automatable).
