# Paper 2 — Reverse VLA · Concept & Proposal

> **Thesis:** *Leveraging Large Language Models for Sample-Efficient Imitation Learning*
> **Paper 1:** *Demonstration Distillation for Sample-Efficient Imitation Learning* (done)
> **Paper 2 (this doc):** proposed — the **Reverse-VLA** direction
>
> Companion figure: `paper2_reverse_vla_architecture.drawio` (Slate-Academic style, matches your Paper-1 diagram).

---

## 1. Proposed title

**Recommended:**
> **Know What You've Taught: A Reverse-VLA Coverage Memory for Sample-Efficient Imitation Learning**

The hook names the exact gap Paper 1 leaves (the selector has no model of what has already been taught) and pairs it with the two load-bearing terms — *Reverse-VLA* and *coverage memory*.

**Strong alternatives** (in the family "… for Sample-Efficient Imitation Learning"):
1. *Reverse VLA: Captioning Trajectories, Actions, and Failures for Sample-Efficient Imitation Learning* — cleanest "brand + method"; the three caption types double as a one-line abstract.
2. *Naming the Gap: Coverage-Aware Demonstration Selection for …* — foregrounds the outcome (which demo, and why) without leaning on the "Reverse VLA" coinage, if a reviewer finds the term unfamiliar.
3. *Inverting the VLA: From Vision and Action to Language for …*
4. *Teaching the Teacher: Dataset-Self-Aware Demonstration Selection for …*

---

## 2. The gap Paper 1 leaves (motivation)

Paper 1's loop works, but its LLM components are **stateless and dataset-blind**:

```
seed demos → train policy → rollout (held-out) → flag uncertainty @ t*
   → Vision LLM (3 frames) → Reasoning LLM (root cause)
   → Cluster Engine + Cluster Memory + KAG → Prescription LLM (config P)
   → expert demo → feasibility → add ONE demo → retrain → repeat
```

The **Vision / Reasoning / Prescription** LLMs are three separate calls that pass lossy text between them, and **none of them holds a model of what trajectories/skills are already in the training set.** They can say "this rollout failed because X," but they cannot say "…and we already have six demonstrations of X, so that is *not* the gap." Selection is therefore driven by *local* failure/uncertainty, never by *global* coverage of the taught skill space.

**Consequence:** demonstrations can be spent re-teaching what the policy already knows — the opposite of sample efficiency.

---

## 3. Core idea — Reverse VLA (Vision + Action → Language)

A standard **VLA** maps **(Vision, Language) → Action** in one model: instruction in, motor commands out.

**Reverse VLA inverts the mapping: (Vision, Action) → Language.** Given a demonstration or rollout trajectory — the *visual observations together with the executed action sequence* — it emits natural-language **captions** at three granularities:

| Head | Output | Example (PushT) |
|---|---|---|
| **Trajectory caption** | holistic intent / skill | "Align the T's long edge, then rotate it clockwise into the goal pose." |
| **Action caption** | sub-skill spans (dense) | "contact bottom face" → "translate left" → "rotate CW about centroid" |
| **Failure caption** (anchored at `t*`) | root cause | "Contacted the wrong face at `t*`; block over-rotated past goal; contact lost." |

These captions are aggregated into a **persistent, language-indexed coverage memory** of *what the policy has already been taught*. A **single** grounded LLM then reasons over `(current failure caption + coverage memory)` to prescribe the **minimal complementary** demonstration — the one that fills a **coverage gap**, not merely a locally uncertain state — and projects its reasoning into language so a human operator understands **why** so few demonstrations are needed.

> **The one-line contribution:** it is *not* the V+A→L captioner (that primitive exists). It is **closing the loop** — captioning → dataset self-awareness → coverage-gap-driven demonstration selection — for sample-efficient active imitation learning, with a language rationale for *why* few demos suffice.

---

## 4. Method

### 4.1 The captioner `C_φ`

- **Input** a trajectory `τ = {(oₜ, sₜ, aₜ)}` — RGB obs `oₜ`, (privileged sim) state `sₜ`, executed action `aₜ`.
- **Encoding.** Sample `N≈8` keyframes (forced to include `0`, `t*`, `H−1`), encode with a frozen ViT (SigLIP/DINOv2); feed the **full action sequence** as quantized tokens (the actions are the discriminative "reverse" signal — they carry skill semantics vision alone misses); interleave and project into a small LLM backbone (LLaVA-style adapter over Qwen/Llama).
- **Output** the three heads above, selected by a query token; the **failure head is Paper 1's Reasoning-LLM root cause, expressed in language** and anchored at `t*`.

### 4.2 Training & label sources (no manual captioning)

1. **Sim ground-truth → templated "silver" captions.** Privileged state gives programmatic predicates (contact/no-contact, displacement, rotation, contacted face, task phase, distance-to-goal); a grammar renders correct-but-stiff captions.
2. **Distillation from a large VLM teacher**, *constrained to the template facts* (fact-locking curbs teacher hallucination); student is SFT/seq-KD'd on the paraphrases.
3. **Self-supervised action segmentation** via change-point detection on action/velocity/contact signals → sub-skill spans, named by template+teacher.
4. **Contrastive failure labels:** for a failed `τ`, retrieve the nearest *successful* demo of the same intent and describe the divergence at `t*`. **Deliberately over-sample failed/OOD rollouts** (see Risk 3).

**Objective** `L = L_LM + λ_seg·L_seg + λ_align·L_align + λ_fact·L_fact`, where `L_align` (InfoNCE between caption and trajectory embeddings) builds the shared language↔trajectory space the memory reuses, and `L_fact` re-parses generated captions and penalizes mismatch vs the sim predicates (anti-hallucination).

### 4.3 Coverage / skill memory

Store every demo's captions + embeddings: `M = {(eᵢ, ℓᵢ, metaᵢ)}`, a vector store over trajectory- and primitive-level embeddings with per-skill support counts.

- **Coverage** of a query `ρ(e_q) = (1/|M|) Σ k(e_q, eᵢ)`; **Novelty** `= 1 − maxᵢ sim(e_q, eᵢ)`.
- **Competence weighting (key):** weight each stored skill by the policy's *measured success rate* — a skill that is present but not yet mastered still reads as a gap ("taught **and** learned").
- The **coverage gap** is the low-`ρ` region near the current failure embedding.

### 4.4 The single unified selector — collapsing 3 → 1

`P, R = LLM_θ( ℓ_fail , g , Retrieve(M, e_fail) )` — retrieval-augmented generation over the memory. Because all captions live in **one** grounded language space from **one** captioner, the selector has genuine **dataset self-awareness**: *"you have 6 demos of top-face push + CW rotate, 0 demos recovering from bottom-left over-rotation → prescribe that."* `P` = prescribed demo config; `R` = the operator rationale.

| Paper 1 (3 stateless, dataset-blind LLMs) | Paper 2 |
|---|---|
| Vision LLM (reads 3 frames) | subsumed by `C_φ` — perception→language, precomputed & shared |
| Reasoning LLM (root cause) | `C_φ`'s **failure head** (`ℓ_fail`) |
| Prescription LLM (`P`) | the single `LLM_θ`, now grounded in `M` |
| Cluster Engine + Cluster Memory + KAG | absorbed into `M` (embedding clusters + geometric `g`) |

Net: **one grounded captioner front-end + one memory-conditioned selector**, replacing three blind calls plus the clustering stack. Only one LLM call happens at decision time.

### 4.5 One round of the loop

1. Roll out `π_r` on held-out episodes.
2. Paper-1 detection: flag uncertainty at `t*`, extract geometric descriptor `g`.
3. `C_φ` → `ℓ_fail` for each flagged failure.
4. Embed & cluster failures (replaces the standalone Cluster Engine); pick the dominant/most-novel mode.
5. `Retrieve(M, e_fail)` → nearest skills + coverage histogram.
6. `LLM_θ ⇒ (P, R)` — the minimal complementary demo for the low-coverage region.
7. Feasibility check → if not solvable, LLM **revises `P`** (inner loop).
8. Expert gives **one** demo `d`.
9. `C_φ` captions `d`; **memory grows** `M_{r+1} ← M_r ∪ captions(d)` — *the statefulness Paper 1 lacked.*
10. `D_{r+1} = D_r ∪ {d}`; retrain → `π_{r+1}`.
11. Stop at target success, else next round.

---

## 5. Novelty & positioning

**Genuinely novel (defensible):**
1. **Captioning-as-self-awareness loop** — V+A→L captions build a *persistent, language-indexed coverage memory*, and selection is conditioned on a **coverage gap** rather than state-uncertainty (DAgger) or embedding-diversity (curation). No prior work closes this loop.
2. **Unification over Paper 1** — three stateless LLMs → one grounded, stateful, coverage-aware component.
3. **Interpretability *for* sample efficiency** — a language rationale for *why* few demos suffice.
4. **Tri-granularity inversion anchored at `t*`.**

**Honest about what's borrowed:** the V+A→L captioner itself, uncertainty-triggered querying, and `t*` localization are adopted building blocks — frame them as such, not as the contribution.

**Nearby work and how we differ** (fuller list + refs in the appendix):

| Area | Representative | Difference |
|---|---|---|
| Forward VLA | RT-2, OpenVLA, Octo | they map (V,L)→A; we invert to language *about* the dataset |
| Language-as-intermediate | RT-H, ECoT | they emit language *en route to an action*; we describe *already-executed* trajectories to choose what to collect |
| Robot trajectory captioning | proprioception-enhanced VLM captioning (2025/26), Dense Motion Captioning | those are open-loop description scored on caption quality; we use captions as the *substrate for a selection decision* |
| Inverse dynamics / latent action | CLAM, AMPLIFY | they invert to *actions/latents*; we invert to *language* |
| Active IL (DAgger family) | DAgger, Ensemble/Lazy/Safe/Diff-DAgger | they query where the policy is *locally uncertain*, blind to dataset content; we add a dataset-global *coverage* criterion |
| Demo/dataset selection | STRAP, Data-Quality-in-IL | they curate an *existing pool* by embedding distance; we reason in *language* and *prescribe new* demos |
| LLM failure explanation | REFLECT | per-episode recovery; ours is cross-episode dataset design |

**"Isn't this just ___?" — the three you'll hear, and the rebuttals:**
- *"…Joint Action-Language Modelling / the action-understanding half of a bidirectional VLA?"* → that reconstructs the instruction to make **one policy** transparent; it builds **no memory and does no data selection**. Their captioner could be a drop-in front-end to our loop — which shows the captioning is not our claim.
- *"…robot trajectory captioning?"* → those are description benchmarks scored on caption fidelity; the closest (proprioception-enhanced captioning) *explicitly leaves IL integration to future work*. We evaluate on **sample efficiency**, not BLEU. Description → **decision** is the delta.
- *"…REFLECT / ECoT with extra steps?"* → REFLECT recovers a *single episode*; ECoT reasons *to emit an action*. Neither keeps a cross-episode coverage model or closes an *active-collection* loop that minimizes expert demos.

---

## 6. The hard questions your panel will ask (and how to answer)

Ordered hardest-first. **#1–#3 are load-bearing** — they all probe the same nerve: *does language do causal work, or is it decoration on a retrieval loop you could build without it?*

**1. Is the language actually causal, or an epiphenomenal readout?**
The existential question. **Settle it with a matched-information ablation:** freeze the selection loop and vary *only* the representation the selector consumes, at equal capacity — (i) generated captions, (ii) Paper-1 geometric KAG signatures, (iii) a learned trajectory/action embedding at the same budget, (iv) **content-scrambled captions** (placebo). Add an **oracle-caption ceiling** (human captions). The claim holds *only if* generated language beats the strong learned-embedding baseline (iii) and the placebo (iv) and trends toward the oracle. **Design the paper so this is the first results figure.** Pre-commit: if (iii) matches (i), reframe the contribution as *interpretability*, not efficiency.

**2. Circularity — captions come from the policy you're improving.**
Decouple architecturally: ground `C_φ` on a **frozen, independently-pretrained VLM** (not fine-tuned on this policy's rollouts) and anchor coverage in the **policy-independent geometric state**. Then test: inject **held-out failure modes the policy has never produced** and show the captioner still describes them and the selector requests the right demo — proving the memory generalizes beyond the policy's current failure distribution.

**3. Does language add anything over Paper 1's geometric cluster signatures?**
Geometry can't do three things: (a) **task-agnostic composability** (a PushT contact descriptor is meaningless for pouring; language is one interface a shared selector reasons over across tasks); (b) **priors** (a pretrained LLM judges the novelty of a mode it never numerically clustered); (c) the **interpretability deliverable**. **Experiment:** cross-task transfer — build memory on task A, show the language selector generalizes to task B while the geometric-signature selector must be re-engineered. *Include ≥1 task where Paper-1 descriptors do not transfer, or this objection stands.*

**4. How do you train/evaluate the captioner and prove it isn't hallucinating?**
Privileged sim state → template captions + small human-validation set + **contrastive grounding** (caption must let a retrieval model recover *which* trajectory it describes). Report **grounding-consistency F1** (not fluency). Falsification: inject captions at rising corruption rates and show monotonic downstream degradation — proving the system *uses* caption content.

**5. "Isn't this just inverse dynamics + captioning?"** Inverse dynamics recovers *actions*; we output a **multi-granularity semantic index** that drives closed-loop selection. Baseline that isolates it: an off-the-shelf video captioner (no action-conditioning, no memory, no `t*`) — the gap is the contribution, quantified.

**6. Why is one unified LLM better than three?** Don't claim "one model"; claim **grounded + stateful + coverage-aware joint reasoning**, which *happens* to be one component. Ablate: (i) three LLMs; (ii) three LLMs **+ memory bolted on**; (iii) single unified model. If (ii) closes most of the gap, honestly report the win is the *memory*; if (iii) > (ii), joint reasoning is the mechanism. Both are publishable.

**7. Generalize beyond PushT?** Commit to ≥3 environments (PushT + RoboMimic + Meta-World/LIBERO). The language interface is *why* it should transfer (ties to #3). State the honest degradation plan for the no-privileged-state / real-robot regime.

**8. Compute/latency of an LLM in the loop?** The LLM runs **only at demo-selection time**, never in the control loop — off the robot's critical path, amortized over a retrain. The scarce resource is *human demonstration time*; trading GPU inference for **fewer human demos** is favorable. Provide a budget table (LLM tokens vs human-minutes saved); note the single model is cheaper than Paper 1's three.

---

## 7. Evaluation plan

**Benchmarks (tiered):**
- **PushT** — continuity/sanity vs Paper 1 (apples-to-apples numbers).
- **RoboMimic** (Lift/Can/Square/Transport/Tool-Hang; PH/MH) — **primary** sample-efficiency testbed; graded difficulty makes demos-to-threshold curves discriminative.
- **LIBERO** (Spatial/Object/Goal/Long) — ships **language instructions** → free ground truth for caption scoring and a defined skill taxonomy for coverage metrics.
- **Meta-World** (ML/MT) — a large **named skill inventory** to prove the memory prescribes *complementary* (novel) skills, not redundant ones.

**Metrics:**
- *Primary (sample efficiency):* **Demonstrations-to-threshold `D@τ`** (τ ∈ {50, 70, 90}%) and **success-vs-#demos AUC**; report `D@τ` ratio vs Paper 1 ("N% fewer demos to 80%").
- *Secondary:* final/asymptotic success (no ceiling loss); **caption quality** vs GT language / human (incl. failure root-cause accuracy at `t*`, and caption→trajectory retrieval); **coverage** (fraction of skill set represented) and **novelty / redundant-demo rate** per added demo; **interpretability** (can a human, reading only the rationale, predict the requested demo above chance? is the rationale *faithful* — ablate the cited gap, selection should change?); **cost/latency** (expected to *drop* vs 3-LLM Paper 1).

**Baselines** (identical policy + retrain loop, vary only selection): BC (passive) · random selection · vanilla DAgger · **uncertainty-only** (Paper-1 trigger, no LLM) · **Paper-1 full method** (the key head-to-head) · Reverse-VLA (ours). Optional oracle skyline (ground-truth coverage).

**Ablations that isolate each contribution:**
- **A — coverage memory:** −memory (prescribe from failure caption alone, blind) vs +memory; plus language-indexed vs raw-embedding memory (shows the *language* index matters).
- **B — 1 vs 3 LLMs:** memory ON, swap reasoning core; report success *and* cost/latency.
- **C — caption granularity:** toggle {trajectory, action, failure} independently.
- **D — grounding:** with/without `t*`/geometric anchoring on the failure caption.

**What a positive result looks like:** Reverse-VLA's success-vs-#demos curve **stochastically dominates** all baselines; **`D@τ` lower than Paper 1** with non-overlapping CIs over ≥5 seeds; **no ceiling loss**; ablations confirm the mechanism (removing memory raises the redundant-demo rate; unified LLM matches/beats three at lower cost); captions are grounded; and humans reading only the rationale predict the requested demo above chance. → *strictly better sample efficiency than Paper 1 with a single grounded component, plus human-legible justification.*

---

## 8. Riskiest design decisions (be upfront with supervisors)

1. **Caption faithfulness (top risk).** Everything downstream reasons over `ℓ`; a hallucinated cause corrupts the memory. Mitigations must be load-bearing: fact-locked decoding, `L_fact`, sim-state anchoring; validate captions against held-out predicates as a *first-class* metric.
2. **Is language the right coverage index?** If kinematically-different skills collapse to identical captions, novelty is mis-measured. Likely fix: **fuse the caption embedding with a geometric/action embedding** rather than indexing on pure language.
3. **Train/deploy shift.** `C_φ` trains mostly on smooth expert demos but must caption the OOD *failed* rollouts. Over-sample failed/perturbed trajectories; report captioner accuracy **specifically on the failure slice**.

---

## 9. Contributions (for the intro bullets)

1. **Reverse VLA** — a (V+A)→L captioner producing trajectory/action/failure captions, with the failure head anchored at the policy's own peak-loss step `t*`.
2. A **language-indexed, competence-weighted coverage memory** giving the demonstration selector *dataset self-awareness*.
3. A **single grounded selector** that prescribes the **minimal complementary** demonstration by coverage gap — collapsing Paper 1's three stateless LLMs + clustering stack into one component.
4. **Interpretability for sample efficiency:** a language rationale for *why* so few demonstrations are needed.
5. Empirical **sample-efficiency gains over Paper 1** and active-IL baselines, with ablations isolating each contribution.

---

## 10. Relation to the thesis

- **Thesis** — *Leveraging LLMs for Sample-Efficient Imitation Learning.*
- **Paper 1** distills *which failures* to fix (demonstration distillation). Its selector is powerful but **dataset-blind**.
- **Paper 2** gives the selector **memory of what's already been taught**, in language, via Reverse VLA — so it spends each demonstration on a genuine *gap* and can *explain* why. Same north star (fewer expert demos), next lever (dataset self-awareness + unification + interpretability).

---

## Appendix — references (verify recent ones before circulating)

- Brohan et al. **RT-2.** CoRL 2023. arXiv:2307.15818
- Kim et al. **OpenVLA.** CoRL 2024. arXiv:2406.09246
- Octo Model Team. **Octo.** RSS 2024.
- Belkhale et al. **RT-H: Action Hierarchies Using Language.** 2024. arXiv:2403.01823
- Zawalski et al. **Embodied Chain-of-Thought (ECoT).** CoRL 2024. arXiv:2407.08693
- Wulff et al. **Joint Action Language Modelling for Transparent Policy Execution.** 2025. arXiv:2504.10055 *(closest inverse/bidirectional threat)*
- **Proprioception Enhances VLM in Generating Captions and Subtask Segmentations for Robot Task.** 2025/26. arXiv:2512.20876 *(closest captioning mechanism; explicitly NOT IL-integrated)*
- **Dense Motion Captioning (DEMO / CompMo).** 2025. arXiv:2511.05369
- Krishna et al. **Dense-Captioning Events in Videos.** CVPR 2017. arXiv:1705.00754
- Ross, Gordon, Bagnell. **DAgger.** AISTATS 2011.
- Menda et al. **EnsembleDAgger**; Hoque et al. **LazyDAgger/ThriftyDAgger**; Zhang & Cho **SafeDAgger.**
- Liu, Bahety, Song. **REFLECT.** CoRL 2023. arXiv:2306.15724
- **STRAP: Sub-Trajectory Retrieval for Augmented Policy Learning.** 2024. arXiv:2412.15182
- Belkhale, Cui, Sadigh. **Data Quality in Imitation Learning.** NeurIPS 2023.
- **How to Leverage Diverse Demonstrations in Offline IL.** 2024. arXiv:2405.17476
- Cazenavette et al. **Dataset Distillation by Matching Training Trajectories (MTT).** CVPR 2022. arXiv:2203.11932
- Benchmarks: **RoboMimic** (robomimic.github.io/study), **LIBERO**, **Meta-World**.

> ⚠️ A few 2025/2026 arXiv items (e.g. 2512.20876, 2511.05369, 2504.10055) were surfaced by automated search; confirm titles/venues manually before this goes to your supervisors.
