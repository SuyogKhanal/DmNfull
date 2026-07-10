# 05 — Ablation matrix (from `supervisor_ablation_ask.txt` → config flags)

Read `supervisor_ablation_ask.txt` in full for the reasoning; this file is the actionable
distillation. **Design principle (his words): you ablate what you claim or design; you don't
ablate what you inherit.** Each ablation is **one config flag = one job**. Every flag defaults
to the full-DISTIL value; a run sets exactly one (or, for the matrix, is the full method).

## The master control set (implement as config flags)
| flag | full-DISTIL default | knockout meaning |
|---|---|---|
| `memory.lambda` | 1.0 | `0` → cluster memory / rotation OFF (Eq 9 inert) |
| `allocation` | `distil` | `random` → replay a random recorded failure each round (Stagger control) |
| `clustering` | `silhouette` | `off` → prescribe from raw failures (target highest-peak-loss or random), no modes |
| `decision` | `llm` | `heuristic` → always target dominant-cluster rep / always bridge centroid (no LLM) |
| `vlm` | `on` | `off` → reasoning LLM sees only geometric descriptor + root-cause, no frame description |
| `kag` | `on` | `off` → drop the per-task knowledge graph from the prompt |
| `bridge` | `on` | `off` → targeted(SELECT)-only |
| `fallback_only` | `false` | `true` → deterministic nearest-untried every round, no LLM at all |
| `context_set` (κ=3) | `rep+worstpeak+fps` | knock out each: forced-target-rep, worst-peak seed, FPS fill (vs random fill) |
| `near_dominant` (|C|≥|C\*|−1) | `on` | `off` → always target the strict dominant cluster |
| `descriptor_feature` | all 6 (Eq 7) | drop one at a time: contact-dist δ, orientation sinθ/cosθ, progress ρ |
| `k_selection` | `silhouette` (Eq 8) | `fixed_k=3` |
| `budget` | 20 | sweep {10, 20, 40} (+5 if cheap) |
| `demos_per_round` (D) | 1 | {2, 3} |
| `memory_const` (γ,σ,λ) | 0.6, 0.06, 1 | small grid around them (one task) |
| `peak_percentile` | (current) | {90, 95, 99} |
| `vlm_frames` | 3 | {1, 3, 5} |
| `attempts_before_fallback` | (current) | {1, 3, 5, 10} |
| `llm_effort` | high/16k | {low, small token cap} |
| `yaw_kernel` | planar | yaw-aware (PushT only) |

## TRIAGE — do exactly this (his explicit plan)

### RUN BEFORE SUBMISSION (the allocation thesis lives or dies here) — highest priority, HPC time
1. **`memory.lambda=0`** (memory off) — *dangerous, top priority*. If SR barely moves, Eq 9 is decorative.
2. **`allocation=random`** on the **robot tasks** (Push-T, Wipe especially) — *dangerous*. The null
   hypothesis "random failure replay already wins" is unrebutted until you produce this number.
3. **`clustering=off`** — *dangerous*. With 1–2 this triangulates the whole allocation claim.
4. **`decision=heuristic`** (LLM vs fixed heuristic) — *dangerous, the sharp one*. Descriptor (Eq 7)
   is geometric / frozen-embedding and does NOT use the LLM; the FM stack's only job is the
   prescription decision. If a heuristic matches, the 16k-token reasoning LLM is unjustified.
5. **`vlm=off`** — *dangerous*. If root-causing + prescription hold up without the frame, the "V"
   is dead weight (distinct from #4: this tests visual perception, that tested the decision).
6. **Sign test** (free, Tier 5): 10/10 clean sweep under coin-flip null → p≈0.001. Compute it and
   put it in the text; it converts "overlapping error bars" into a real aggregate significance
   statement. Stronger: Wilcoxon signed-rank over the 10 paired cell means.
7. **Four cheap diagnostics** (Tier 4, appendix — see `09_...md`): cluster-label agreement,
   k\* distribution, targeted-vs-bridge usage split, failure-count-per-round.

### COMPUTE + KEEP IN DRAWER (rebuttal-tier)
`kag=off`; `bridge=off` on Push-T/Lift/Door; `fallback_only=true`; budget sweep {10,20,40};
`demos_per_round`∈{2,3}; memory-const grid; `descriptor_feature` drops; per-checkpoint Q3
correlation (the honest ~25-point version); the compute/wall-clock table.

### DEFEND WITH WORDS / NARROW THE CLAIM (already in Limitations)
policy-agnostic scope (add one genuinely different policy — Gaussian-MLP BC or ACT — on one
task, or narrow the claim); privileged-info exchange (image runs use object poses for clustering
+ memory centroids — be ready to run "image descriptors without privileged poses" or argue the
exchange is favorable); ΔSR-measured-on-targeted-set (have the clean held-out ΔSR ready);
reroll-fairness (does DISTIL see more env interaction? report it).

## Statistical power (Tier 5 — the single biggest rebuttal risk)
- 5 seeds (robot) with overlapping 1-σ bars: the Q1 claim rests on *ranking consistency*. Run the
  **sign test now**; be **ready to add seeds to 10** on Push-T + Wipe (the most likely single
  rebuttal demand).
- **Q3 pseudoreplication**: robot correlations rest on ~25 points (5 seeds × 5 retrainings);
  compute the honest per-checkpoint correlation (or mixed-effects), don't let a reviewer derive
  the smaller effective-n.

## What "full DISTIL" is (the reference every ablation compares to)
All flags at default, all tasks (GridWorld/Lift/Wipe/Door/PushT) × {state, image} × 5 seeds ×
budget 20, vs the Diff-DAgger baseline family. This is the headline table; aggregate it and every
ablation into the `DISTIL_ablation_preview.xlsx` shape (`09_...md`).

> Reality check from the supervisor: several knockouts *may come back negative* (VLM-off or
> LLM-heuristic matching full DISTIL). **That is the point** — discover it in your own logs and
> reframe the contribution honestly, rather than have a reviewer discover it. Concentrate compute
> on proving the allocation thesis (clustering + memory → coverage → success); be honest about the
> rest.
