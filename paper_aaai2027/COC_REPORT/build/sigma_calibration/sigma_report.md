# Calibrating the Gaussian memory-kernel width σ from existing cluster geometry

**Scope.** Calibrates σ from centroids that *already exist* in completed runs. No DISEIL loop was
run, no rollout executed, and **no code default changed** — this is measurement + recommendation only.

**SLURM job `110500`** (`sigma_calib`, partition `gpu`, `--gres=none`, pure CPU). Script
`distil/scripts/analyze_sigma.py`; report emitted from the CSV by `distil/scripts/write_sigma_report.py`.

> **Revision note.** A first version of this study was produced and then adversarially audited.
> The audit found real defects — the penalty columns had been computed over a *pooled* distance set
> while σ was calibrated on a different one (producing a 55-order-of-magnitude self-contradiction on
> GridWorld), the GridWorld length scale used grid *pitch* rather than spawn *extent*, `_smoke` runs
> were silently folded in, and the decisive decision-level statistic was computed and dropped.
> The analysis script was fixed and re-run; every number below comes from that corrected run.

---

## 0. Ground truth — re-verified, plus three corrections

The kernel is exactly as the brief states. Confirmed against source:

| Claim | Verdict |
|---|---|
| `recency_penalty` uses **x,y only** (z, θ ignored); RAW planar Euclidean; `pen += γ^(now−r)·exp(−d²/2σ²)` | **CONFIRMED** — `distil/p4/memory.py:56-70` |
| `score = mean_peak_loss − λ·pen` | **CONFIRMED** — `distil/p4/memory.py:86-87` |
| the centroid is the mean of members' `centroid_pos()`, RAW/unstandardized; the standardized 6-D space is used only for clustering + representative | **CONFIRMED** — `distil/p4/clustering.py:158-166` |
| `memory_sigma = 0.06` shared by robots **and** GridWorld (γ=0.6, λ=1.0) | **CONFIRMED** — `distil/config.py:71` |
| robots: `centroid_pos()` = `obj_xyz` in **metres** | **CONFIRMED** — `distil/p4/descriptor.py:121` |
| GridWorld: `centroid_pos()` = `[agent_row, agent_col, 0]` in **grid-cell indices** | **CONFIRMED** — `distil/gridworld/descriptor.py:54` |

It is a **units** mismatch, not merely a scale mismatch. Confirmed.

**Correction 1 — the centroid key differs by codebase, and the brief was right about the fork.**
In `distil`, `round_setup` carries per-cluster **`centroid_xy`** (robots) / **`centroid_rc`**
(GridWorld), both 2-D; `centroid_xyz` (3-D) appears only in `telemetry/centroid_memory.json`, for the
one chosen target per round. In the `pool_rl_robo` **fork**, every PushT `round_setup` carries
per-cluster **`centroid_xyz`** (3-D) plus `mean_peak_loss` directly. The harvester reads all three
keys. A harvester written against the `distil` schema alone would silently drop **all of PushT**
(1,096 clusters — and one of the only two cells where σ is currently live).

**Correction 2 — `mean_peak_loss` is never persisted by `distil`** (computed at `memory.py:87`, then
discarded). It is reconstructed offline by joining `clusters[].members` → `descriptors[].peak_loss`.
The fork persists it directly.

**Correction 3 — the constraint that bounds this entire study.** σ can only change the decision when
the candidate set has ≥2 members: `select_target` scores only `cands = [c for c in clusters if c.size
>= dominant.size - 1]` (`memory.py:80`). With a singleton candidate set the dominant cluster is
returned **whatever σ is** — yet the penalty is still computed and logged, so the run *looks* like
memory is active while σ is provably inert. Quantified in §3.3 and §3.5.

---

## 1. Harvest

Sources: `distil/results/**/telemetry/round_*.jsonl` (`round_setup`),
`distil/results/**/telemetry/centroid_memory.json`, and the fork under
`Equivariant_pathway/.../pool_rl_robo/**`.

**115 run directories, 1,751 cluster-carrying rounds, 4,157 clusters, 115 `centroid_memory.json` files.**

`_smoke` runs are **excluded** (plumbing tests, not experiments). `pool_x_selector` was **not**
walked: ~835k `episode_data.json` files and **zero** centroid artifacts (verified: 0
`centroid_memory.json`, 0 `telemetry/` dirs). It prescribes discrete grid layouts, not centroids —
a structural zero, not a search failure.

| task × modality | runs | rounds | clusters | NN pairs | unit | data verdict |
|---|---:|---:|---:|---:|---|---|
| GridWorld / state | 30 | 600 | 1518 | 1404 | grid cells | **OK** |
| Lift / image | 28 | 558 | 1224 | 1118 | m | **OK** |
| PushT / state | 12 | 413 | 1096 | 1096 | m | **OK** |
| Lift / state | 30 | 161 | 267 | 175 | m | **OK** |
| Door / image | 1 | 5 | 17 | 17 | m | INSUFFICIENT DATA |
| Door / state | 1 | 5 | 12 | 12 | m | INSUFFICIENT DATA |
| Wipe / image | 1 | 5 | 12 | 12 | m | INSUFFICIENT DATA |
| Wipe / state | 3 | 3 | 9 | 9 | m | INSUFFICIENT DATA |
| GridWorld / image | 1 | 1 | 2 | 2 | grid cells | INSUFFICIENT DATA |
| PlugCharger / state | 3 | 0 | 0 | 0 | m | INSUFFICIENT DATA |
| StackCube / state | 5 | 0 | 0 | 0 | m | INSUFFICIENT DATA |

**Why Door and Wipe are so thin — and it is *not* that the clustering never ran.** Door executed
**1,278 production rounds** across its arms (726 of them record a `k_star`, proving the clustering
did execute) and retained **zero** centroid geometry: all **54** production leaves under
`distil/results/Door` hold only `config.yaml` + `result.json`, with **0 `telemetry/` directories**.
Wipe ran **631** production rounds (600 with `k_star`) and retained **3** — **0.48 %**. The
coordinates were computed and then never kept (or were swept). The only Door/Wipe geometry that
survives anywhere is in the `_compute` D5 runs plus 3 Wipe/state production rounds.
**Recovering Door/Wipe geometry properly requires a re-run**, which is out of scope here.

StackCube and PlugCharger use the fork's older `p4_top3` serializer, whose `round_setup` emits no
`clusters[]` array at all — per-cluster coordinates are unrecoverable *by schema*. (They do retain
chosen-target memory, which is why they appear with 0 clusters rather than not at all.)

---

## 2. The distances the kernel actually sees

Two distributions, two different questions — **they are reported separately and never pooled**:

- **Nearest-neighbour (NN) inter-centroid distance**, within a round: sets whether the penalty can
  *discriminate between candidates*. This is the quantity σ must resolve, and **the set σ is
  calibrated on**.
- **Chosen-target ↔ stored-memory-entry distance**: sets the *absolute magnitude* of the applied penalty.

### 2.1 Nearest-neighbour inter-centroid distance

| task × modality | unit | min | p25 | **median** | p75 | max | n (NN) |
|---|---|---:|---:|---:|---:|---:|---:|
| GridWorld / image | grid cells | 2.32 | 2.32 | **2.324** | 2.32 | 2.32 | 2 |
| GridWorld / state | grid cells | 0 | 1.41 | **1.947** | 2.51 | 5 | 1404 |
| Wipe / state | m | 0.0632 | 0.123 | **0.1686** | 0.259 | 0.261 | 9 |
| PushT / state | m | 0 | 0.0611 | **0.118** | 0.193 | 0.431 | 1096 |
| Wipe / image | m | 0.0155 | 0.068 | **0.08782** | 0.126 | 0.165 | 12 |
| Lift / image | m | 0.000412 | 0.0152 | **0.02233** | 0.0316 | 0.108 | 1118 |
| Lift / state | m | 0.00104 | 0.0132 | **0.01983** | 0.0423 | 0.0698 | 175 |
| Door / image | m | 0.00197 | 0.00537 | **0.00805** | 0.00944 | 0.0132 | 17 |
| Door / state | m | 0.00407 | 0.00485 | **0.005465** | 0.00564 | 0.0122 | 12 |

### 2.2 Chosen-target ↔ memory-entry distance (corroboration)

| task × modality | n | median |
|---|---:|---:|
| GridWorld / state | 5700 | 1.6 grid cells |
| StackCube / state | 3552 | 0.189 m |
| PlugCharger / state | 1243 | 0.116 m |
| PushT / state | 12180 | 0.104 m |
| Wipe / image | 10 | 0.0639 m |
| Lift / state | 410 | 0.0313 m |
| Lift / image | 5283 | 0.0162 m |
| Door / image | 10 | 0.00843 m |
| Door / state | 10 | 0.00565 m |

The two distributions agree to within ~1.6× on every task, so the length scale is robust to which
one you calibrate against. **The measured length scale spans 425× (2.63 orders of magnitude)** across tasks — from Door (~0.005465 m) to
GridWorld (~2.32 grid cells) — and even *within the metric tasks alone* it spans
**31×** (0.005465 m → 0.1686 m). One σ cannot serve that.

Histograms and kernel curves: `figures/<Task>_<modality>.{pdf,png}`.

---

## 3. Diagnosis at the current σ = 0.06

`exp(−d²/2σ²)` evaluated over the **NN set** (the set σ is calibrated on):

| task × modality | median penalty @ σ=0.06 | IQR @ σ=0.06 | regime |
|---|---:|---:|---|
| GridWorld / image | 0 | 0 | **COLLAPSED** |
| GridWorld / state | 1.74e-229 | 2.3e-121 | **COLLAPSED** |
| Wipe / state | 0.0193 | 0.122 | partial |
| PushT / state | 0.145 | 0.59 | **LIVE** |
| Wipe / image | 0.358 | 0.415 | **LIVE** |
| Lift / image | 0.933 | 0.0982 | saturated |
| Lift / state | 0.947 | 0.196 | saturated |
| Door / image | 0.991 | 0.00829 | **SATURATED** |
| Door / state | 0.996 | 0.00115 | **SATURATED** |

### 3.1 GridWorld — collapsed to an identical-centroid indicator

Measured median NN separation: **1.947 grid cells**. In cell units:

```
d = 1.0   (adjacent cells)     ->  exp(-1.0^2 /(2*0.06^2)) = 4.8e-61
d = 1.947 (measured median) ->  exp(-1.947^2/(2*0.06^2)) = 1.7e-229
```
The median penalty over the NN set is **1.74e-229**. The kernel is a numerical
delta function: ~0 for any two *distinct* cluster centroids, firing only when a centroid lands within
~0.15 cells of a previously prescribed one. It is not a strict no-op — it acts as a degenerate
**identical-centroid suppressor** (see §3.5, where it still flips 10 decisions) — but it is blind to
all real geometry.

**This matters for a claim already in the workbook.** The recorded "GridWorld allocation null result"
(`memory_off` ≡ full DISEIL) is **not** evidence that the allocation mechanism does nothing on
GridWorld — it is evidence that at σ=0.06 the two arms run very nearly the *same* effective algorithm.
GridWorld at the paper σ cannot argue for or against the memory mechanism. (It is *not*, however, the
sole explanation of that null: `clustering_off`, `allocation_random` and `decision_heuristic` also
match full DISEIL there, and σ explains none of those.)

### 3.2 Door and Lift — saturated, so the penalty cannot move the argmax

Door's reset clamp is ±0.0135 m (x) / ±0.013 m (y), and the measured NN separations are
correspondingly tiny (median **0.005465 m** state, 0.00805 m image). At σ=0.06 every
candidate scores a penalty of **~0.996**, with an interquartile spread of only
**0.0011**. Subtracting a near-identical constant from every candidate leaves the argmax of
`mean_peak_loss − λ·pen` unchanged. The memory term is present in the arithmetic and absent from the
decision. Lift is the same failure mode one notch weaker (median penalty
0.947–0.933).

### 3.3 σ is inert in most rounds *regardless of its value*

Fraction of rounds whose candidate set had ≥2 members — i.e. rounds where σ could matter **at all**:

| task × modality | rounds with \|cands\| ≥ 2 | fraction | data |
|---|---|---:|---|
| GridWorld / image | 1 / 1 | 100.0 % | thin |
| Door / image | 3 / 5 | 60.0 % | thin |
| GridWorld / state | 262 / 600 | 43.7 % | OK |
| Lift / state | 57 / 161 | 35.4 % | OK |
| Door / state | 1 / 5 | 20.0 % | thin |
| Wipe / image | 1 / 5 | 20.0 % | thin |
| PushT / state | 77 / 413 | 18.6 % | OK |
| Lift / image | 91 / 558 | 16.3 % | OK |
| Wipe / state | 0 / 3 | 0.0 % | thin |

On the four adequately-powered cells the candidate set is a singleton in
**56–84 % of rounds**, so the dominant cluster is returned
regardless of σ. Re-scaling σ has leverage on at most the complement. The thin cells are worse, not
better (Wipe/state: 0 of 3 rounds had a multi-candidate set). **This bounds the realistic upside of
the entire recommendation** and is stated here, not buried in the limits.

### 3.4 Lift — the ceiling, and what it forbids

DISEIL is at **100.0 ± 0.0** on Lift in `GT_SR` (both modalities). No headroom, no variance. A flat
σ-sweep on Lift is **flat by ceiling, not by kernel geometry** — Lift can neither confirm nor refute
anything about σ. I read no null result from it, and neither should the paper.

### 3.5 The decision-level test — does the penalty actually change the choice?

Everything above is geometry. This is the only measurement that tests the *decision*. For every round
with a multi-candidate set and a non-empty memory, I replay `select_target` exactly as deployed —
the **true γ-weighted sum** over the actual stored entries, `mean_peak_loss` reconstructed from the
descriptors — and ask whether `argmax(mpl − λ·pen)` differs from `argmax(mpl)`:

| task × modality | decidable rounds | argmax flips @ σ=0.06 | argmax flips @ σ recommended |
|---|---:|---:|---:|
| GridWorld / state | 257 | 10 (4 %) | 101 (39 %) |
| Lift / image | 87 | 44 (51 %) | 47 (54 %) |
| PushT / state | 77 | 41 (53 %) | 38 (49 %) |
| Lift / state | 34 | 4 (12 %) | 7 (21 %) |
| Door / image | 3 | 1 (33 %) | 3 (100 %) |
| Door / state | 1 | 0 (0 %) | 1 (100 %) |
| Wipe / image | 1 | 0 (0 %) | 0 (0 %) |

This is the strongest evidence in the study, and it cuts both ways:

- **GridWorld/state: 10 → 101 flips of 257** (3.9 % → 39 %). At the paper σ the memory term touches
  almost nothing; at the recommended σ it becomes a genuinely active allocation mechanism. This is
  the fix working.
- **PushT/state: 41 → 38 of 77** (53 % → 49 %). Essentially unchanged — σ=0.06 was **already**
  right for PushT, and the recommendation correctly leaves it alone. That the rule *declines to
  change* the one task that already works is the best available evidence it is not overfitting.
- **Lift: 44 → 47 of 87** (image). A small change, and on Lift it is unobservable anyway (§3.4).

---

## 4. Recommendation: σ_task = α · L_task

### 4.1 Choosing L_task

**(a) Spawn/reset extent** — the randomisation extent of each task, from `distil/p4/bounds.py`
`TASK_BOUNDS` + `paper_aaai2027/context/kag_ur5_bounds.md`; for GridWorld the spawn extent of the 5×5
lattice (index span 0..4 = **4 cell-widths**, *not* the 1-cell grid pitch):

| task × modality | spawn extent | d_median / extent |
|---|---:|---:|
| Door / state | 0.027 m | 0.202 |
| PushT / state | 0.4 m | 0.295 |
| Door / image | 0.027 m | 0.298 |
| Lift / state | 0.06 m | 0.331 |
| Lift / image | 0.06 m | 0.372 |
| GridWorld / state | 4.0 grid cells | 0.487 |
| GridWorld / image | 4.0 grid cells | 0.581 |
| Wipe / image | **DOES NOT EXIST** | — |
| Wipe / state | **DOES NOT EXIST** | — |

Once the *correct* GridWorld denominator is used, this ratio is a fairly well-behaved
**0.20–0.58** (2.9× spread) across every task that *has* bounds —
GridWorld is **not** wildly out of family. (An earlier draft of this report claimed GridWorld was
"5–10× off" and used that to reject the spawn-extent rule; that claim was an artifact of dividing by
the grid *pitch* instead of the spawn *extent*, and it is **withdrawn**.) The spawn extent is a
legitimate fallback length scale, and §4.4 notes it is roughly collinear with the measured geometry.

It nevertheless fails as *the* rule, for one decisive reason: **Wipe has no reset bounds at all.**
Wipe is SELECT-only, absent from `TASK_BOUNDS` entirely (`clamp_obj_xy("Wipe")` would raise
`KeyError`); the randomised quantity is a ~100-marker dirt path, not an object pose. **The rule has no
denominator on Wipe — one of the only two cells where σ is currently live.** A σ rule that cannot be
evaluated on the task it most needs to serve is not the rule to ship.

**(b) Measured median NN inter-centroid distance.** Defined on every task (including Wipe and
GridWorld), in that task's own units, and it is *exactly* the spacing the kernel must resolve to
discriminate between candidates. **This is the recommended L_task.** Its cost is that it requires
prior runs — for a brand-new task with no telemetry, fall back to (a) with α·(d/extent) ≈ 0.85 × 0.3 ≈ **0.25 × spawn extent**.

### 4.2 Choosing α — and an honest word about what it is

**Criterion:** the penalty must span a useful dynamic range over the observed distances — neither
collapsing to 0 nor saturating at 1. The natural operating point puts a *typical* pair of centroids at
the kernel's half-maximum, so closer-than-typical is penalised more and farther-than-typical less:

```
exp(-d_median^2 / (2*sigma^2)) = 0.5   <=>   sigma = d_median / sqrt(2 ln 2)
```
With `L_task := d_median` this yields one dimensionless constant shared by every task:

```
alpha  = 1 / sqrt(2 ln 2) = 0.8493
sigma_task = alpha * d_median(task)
```
**Be clear about the epistemic status of this.** Given `L_task := d_median`, a shared α is true *by
construction* — it is an analytic consequence of the half-maximum criterion, **not** an empirical
regularity discovered across tasks. α is a *choice of operating point*; all the genuine per-task
adaptation lives in the measured L_task. The empirical content of this section is therefore **not**
"one α fits all" (a tautology) but: (i) the measured L_task varies by 356× across tasks, so a single
σ *cannot* be right everywhere; and (ii) at the resulting per-task σ the kernel demonstrably starts
changing decisions where it previously did not (§3.5). Any α in ~0.6–1.2 keeps the penalty usable;
0.849 centres it.

### 4.3 Resulting σ, and the penalty distribution before vs after

| task × modality | unit | L_task (d_med) | σ current | **σ recommended** | median pen @0.06 | IQR @0.06 | median pen @ new | **IQR @ new** | data |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| GridWorld / image | grid cells | 2.324 | 0.06 | **1.973** | 0 | 0 | 0.5 | **0** | thin |
| GridWorld / state | grid cells | 1.947 | 0.06 | **1.654** | 1.74e-229 | 2.3e-121 | 0.5 | **0.378** | OK |
| Wipe / state | m | 0.1686 | 0.06 | **0.1432** | 0.0193 | 0.122 | 0.5 | **0.495** | thin |
| PushT / state | m | 0.118 | 0.06 | **0.1002** | 0.145 | 0.59 | 0.5 | **0.674** | OK |
| Wipe / image | m | 0.08782 | 0.06 | **0.07459** | 0.358 | 0.415 | 0.505 | **0.419** | thin |
| Lift / image | m | 0.02233 | 0.06 | **0.01896** | 0.933 | 0.0982 | 0.5 | **0.477** | OK |
| Lift / state | m | 0.01983 | 0.06 | **0.01684** | 0.947 | 0.196 | 0.5 | **0.694** | OK |
| Door / image | m | 0.00805 | 0.06 | **0.006837** | 0.991 | 0.00829 | 0.5 | **0.349** | thin |
| Door / state | m | 0.005465 | 0.06 | **0.004642** | 0.996 | 0.00115 | 0.5 | **0.101** | thin |
| PlugCharger / state | m | — | 0.06 | **UNMEASURED** | — | — | — | — | INSUFFICIENT (schema) |
| StackCube / state | m | — | 0.06 | **UNMEASURED** | — | — | — | — | INSUFFICIENT (schema) |

The median penalty at the new σ is **0.5 on every row by construction** — that is the criterion, not a
finding. The column that carries information is the **IQR**, the kernel's usable dynamic range:

- Under σ=0.06 it is ~0 on GridWorld (10⁻¹²¹) and 0.001–0.008 on Door — the penalty carries
  essentially **no information**.
- Under the recommended σ it rises to **0.38–0.69** on the four
  well-powered cells. It does **not** reach a uniform band on every cell: Door/state only reaches
  0.1 (its NN distances are tightly bunched, n=12), and
  GridWorld/image is 0 because it has a single NN pair. A single α does **not** buy uniform dynamic
  range — it buys a correctly *centred* one.

Door moves by **12.9×** and GridWorld by
**27.6×**, while PushT and Wipe barely move — their σ was already
close to right. **The rule leaves the working tasks alone and fixes the broken ones**, and §3.5 shows
this at the decision level, not merely the geometric one.

### 4.4 Cross-check against the A13 sheet — it *predicts* A13

A13 asserts σ is LIVE only on Push-T and Wipe; INERT on GridWorld/Door (degenerate kernel) and on
Lift (ceiling). Each falls out of the measurement:

- **"LIVE only on Push-T and Wipe."** These are exactly the two tasks whose measured d_median
  (0.118 m, 0.0878–0.169 m) is *comparable to* σ=0.06, so 0.06 already
  sits in the discriminating band. Everywhere else the geometry is far from 0.06 and the kernel pins
  at one end. ✔ **Explained.**
- **"Lift's kernel comes alive at σ=0.02."** A13 infers this analytically. My *measured* Lift
  d_median is 0.01983 m / 0.02233 m, and the half-maximum rule returns
  **σ = 0.0168 / 0.019 ≈ 0.02**. ✔ **Reproduced from data.**
- **"Door … you would need σ≈0.005."** Measured Door d_median 0.005465 m → rule returns
  **σ = 0.00464**. ✔ **Reproduced from data.**
- **GridWorld "identical-centroid check at EVERY swept σ."** Confirmed and *understated*: A13 quotes
  ≈4e−6 (that is for σ=0.20); at the paper σ=0.06 an adjacent-cell pair gives **4.8e−61**. ✔

**Caveat on how much independent confirmation this really is.** A13's estimates were themselves
derived from the reset ranges, and §4.1 shows d_median ≈ 0.20–0.58 × spawn extent. The two routes are
therefore **partly collinear**, not fully independent: agreement is reassuring and shows the
arithmetic is consistent, but it is weaker evidence than it first appears. The genuinely independent
content is §3.5 (decision flips), which A13 does not measure.

Two defects in A13 itself, found while cross-checking: (i) its Spread column is stored as *formulas
with no cached values*, so `openpyxl(data_only=True)` returns `None` — any programmatic consumer must
recompute it; (ii) the verdict criterion it declares ("spread vs paired-seed noise") has **no noise
column anywhere on the sheet**, so as written the test cannot be executed from A13 alone.

---

## 5. Limits — stated plainly

- **This calibrates σ from existing geometry. It does NOT prove the new σ improves success rate.**
  §3.5 shows the recommended σ changes *which cluster gets prescribed* far more often; it does **not**
  show those are *better* choices. Establishing that requires re-running the DISEIL loop at the
  per-task σ and comparing held-out SR. **That re-run has not been done and is out of scope here.**
- **Even a correctly scaled σ is inert in most rounds** (§3.3): the candidate set is a singleton in
  56–84 % of rounds on the well-powered cells, and σ cannot act there at any value.
- **Door and Wipe recommendations rest on thin data** (9–17 clusters, 1–3 runs), because their
  production telemetry was not retained. Both are marked INSUFFICIENT DATA in the CSV. Door's σ is
  corroborated independently by A13's analytic estimate; **Wipe's is corroborated by nothing**, and
  Wipe is one of the two cells where σ is currently live. Treat Wipe's σ as **provisional**.
- **Wipe is called LIVE on an arithmetic standard it cannot meet on a decision standard.** Its penalty
  IQR at σ=0.06 is wide, but it has **0 of 3** multi-candidate rounds on record — so there is no
  observed round in which σ could have changed a Wipe decision at all. The LIVE verdict for Wipe rests
  on A13's SR sweep, not on this geometry.
- **GridWorld/image, StackCube and PlugCharger** have too few (or structurally zero) per-cluster
  centroids to support a recommendation. Marked INSUFFICIENT DATA, not guessed.
- The penalty is bounded by Σγ^Δ ≤ 1/(1−γ) = **2.5**, so with λ=1 the memory term can never flip a
  decision whose `mean_peak_loss` gap exceeds 2.5, at any σ.
- **σ is not the only explanation of the GridWorld null.** `clustering_off`, `allocation_random` and
  `decision_heuristic` also match full DISEIL there; a mis-scaled σ explains the `memory_off` null
  specifically, not the whole set.
- Single-source: all robot centroids come from one descriptor implementation; no cross-validation
  against an independent geometric pipeline was possible.
