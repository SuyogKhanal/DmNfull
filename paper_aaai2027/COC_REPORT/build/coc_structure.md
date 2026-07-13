# Confirmation of Candidature — detailed section plan

Student: Suyog Khanal (s226137394), A2I2, Deakin University.
Supervisors: A/Prof Santu Rana; Dr Arun Kumar Anjanapura Venkatesh.
Thesis: *Leveraging Large Language Models for Sample-Efficient Imitation Learning*.
Candidature start 13 Nov 2025 · CoC 13 Aug 2026 · thesis submission Nov 2028.

**Target total: ~30,800 words of prose** (no page limit; depth is wanted), plus figures, tables and appendices.

---

## Global drafting constraints (apply to every section)

These are binding and must be checked section by section before any prose is accepted.

1. **Naming.** The method is **DISEIL**. The strings *DISTIL*, *PACE*, *P4*, *p4_top3_rotate*, *p4_subtask* must appear nowhere in the report. Every repository asset that carries an old label (figure PDFs, workbook sheet titles, `sections/*.md`) must be relabelled at import; the file `build/stats_report.md` already uses DISEIL and is the correct precedent.
2. **Acronym derivation.** Aim-1 paper title: *Demonstration Distillation for Sample-Efficient Imitation Learning*. In the abstract, at first mention, bold exactly six letters to show the derivation: **D**emonstration d**I**stillation for **S**ample-**E**fficient **I**mitation **L**earning → D-I-S-E-I-L. Do not bold the letters in the title, in headings, or anywhere else. Do not take "IS" as a block.
3. **Setting vs modality vs mode.** A *setting* = one task under one observation modality. 5 tasks × 2 modalities = 10 settings. "Modality" is always state or image. "Mode" is reserved for *failure modes* (clusters). Sweep the whole document for collisions.
4. **Framework vs instance.** The framework works under any fixed budget B and any demonstrations-per-round D. B = 20 and D = 1 are the validated instance. Symbols in the method and algorithm; concrete values only in the setup subsections.
5. **Method contains only novelty.** The behaviour-cloning objective, dataset aggregation, the generic query-gate template, the silhouette criterion, A*/BFS, k-centre/farthest-point selection and the Diff-DAgger rule all live in the background chapter, cited, and are flagged "we follow standard practice" wherever they reappear.
6. **Policy-agnostic.** A policy is any function f_θ with a per-step loss: MLP (GridWorld state), CNN (GridWorld image), state and image diffusion policies (robot tasks). Never write "a framework for diffusion policies".
7. **Clustering is geometric for every run** (A10). The 6-D descriptor is: robot [p_x, p_y, sin θ, cos θ, ρ, δ]; GridWorld [agent cell (2), signed offset (2), progress, Manhattan distance]. The frozen-R3M-embedding + PCA image branch is out of date and must not appear anywhere, including in the method equations imported from `sections/method.md` (its Eq. 7 visual branch is superseded).
8. **Lift is uninformative for every ablation** (100.0 ± 0.0: no headroom, no variance). Say this explicitly the first time Lift appears in an ablation, and never read a null result on Lift as evidence about a mechanism.
9. **Style.** `Non-AI content.md`, strictly. No AI vocabulary, minimal em dashes, no negative parallelism, no rule-of-three padding, no copula avoidance, no elegant variation, no formulaic conclusions, no semicolons in headings, no paragraph opening on a bare "This is". Every non-obvious claim cited or cut. Never fabricate a citation, DOI or statistic.
10. **Source of truth for numbers:** `ablations_results/DISEIL_ablation_results.xlsx` (currently named `DISTIL_ablation_results.xlsx`) and `build/stats_report.md`. No number enters the report from memory.

---

## Front matter

### F.1 Cover page — 60 words
**Purpose.** Identify the document to the panel.
**Assets.** `A2I2_Logo_Stacked_2025_Keyline.png` at the top of the page.
**Content.** Deakin University; Deakin Applied Artificial Intelligence Initiative (A2I2); "Confirmation of Candidature Report"; thesis title; Suyog Khanal; s226137394; both supervisors with titles; candidature start 13 November 2025; Confirmation of Candidature date 13 August 2026.
**Convention.** Both reference reports use a clean, centred, unnumbered title page with the logo block and a supervisory-panel block ruled off from the title. Follow that layout. No page number.

### F.2 Table of contents, list of figures, list of tables — auto-generated
Both reference reports carry all three. Figures and tables are numbered continuously across the report, each with a caption that states what the reader should conclude, not merely what is plotted.

### F.3 Executive summary (abstract) — 650 words
**Purpose.** A standalone statement of the problem, the idea, what has been done, and what remains. Read on its own by a panel member who reads nothing else.
**What must be argued.**
- Expert demonstrations are the binding cost in imitation learning, and interactive imitation learning spends its budget on a single decision: when to ask.
- The programme's claim: a large language model, given a structured description of *how the policy is failing* and *what the environment permits*, can decide **which** failure to correct and **where** the corrective demonstration should start, so that each demonstration in a restricted budget carries more information.
- First mention of DISEIL with the six bolded letters (rule 2 above).
- Aim-1 evidence in three sentences, each a single claim: best mean success rate in all ten settings; mean margin of 3.7 points over the strongest baseline in each setting; the conservative aggregate test (collapsed to five task means, 5/5, one-sided sign test p = 0.031, paired t-test t(4) = 4.15, p = 0.014) from `build/stats_report.md` §5.
- One sentence each on Aim 2 and Aim 3 as the continuation.
**Constraints.** Do not enumerate every framework component. Break the closing sentence into separate claims. Define "setting" before "(five baselines per setting)" is used.

---

## 1. Introduction and research vision — 3,000 words

### 1.1 The demonstration is the scarce resource — 550 words
**Purpose.** Establish the economic premise of the whole thesis.
**Argue.** Behaviour cloning scales with demonstrations; demonstrations are collected by a human or a scripted expert at a cost that does not fall with compute. In deployment settings the realistic constraint is not "collect more data" but "you have B demonstrations, spend them well". Name the constraint once, symbolically, and keep it: a fixed, restricted budget B.
**Assets.** `figures/Teaser_Diagram.pdf` (Figure 1). Caption must not open by foregrounding the LLM (supervisor note); open on the decision the framework makes.

### 1.2 What interactive imitation learning answers, and what it leaves open — 600 words
**Argue.** The DAgger family corrects covariate shift by labelling learner-visited states, and its members differ only in the scalar gate that decides *when* to hand over. Three consequences, stated plainly and without a rule-of-three flourish: the gates are per-state, so within a batch of failed rollouts they cannot tell which failures are redundant; they are memoryless across rounds, so one persistent failure mode can absorb the budget; and they inherit whatever start state tripped the gate, so they cannot place a demonstration at a more instructive point. This is the opening the programme works in.
**Note.** All six baselines are named here only in passing; their mechanics belong to §2.3.

### 1.3 Central idea and thesis statement — 450 words
**Argue.** A language model is not being used as a controller. It is used as the component that reasons over a *structured summary of the policy's own failures* plus *explicit environmental constraints*, and returns a demonstration request. The thesis statement: language models can raise the information content of each demonstration under a restricted budget, and doing so is what makes imitation learning sample-efficient.

### 1.4 Research questions — 400 words
Three questions, one per aim, each written to be answerable and each mapped to its aim.
- **RQ1.** Under a fixed budget B, does choosing which failure mode to correct and where the corrective demonstration begins yield a policy with a higher success rate than choosing only when to intervene?
- **RQ2.** Can a policy's demonstration selector be made aware of what it has already been taught, by inverting the vision-language-action mapping to produce language descriptions of executed trajectories, and does coverage-gap selection beat failure-local selection?
- **RQ3.** (stated in §6) Can demonstration demand be expressed, priced and satisfied across tasks and embodiments, so that a generalist policy asks a non-expert human for the specific demonstrations it lacks?

### 1.5 Contributions to date — 350 words
Enumerated, restricted to Aim 1, each contribution one sentence and each traceable to a section.

### 1.6 Scope and constraints — 350 words
Simulation only. Five tasks, two observation modalities. Nine seeds on GridWorld, five on the robot tasks (state the counts; do not fabricate a uniform count and do not hide the asymmetry). Expert oracles are scripted or planner-based, not human, and the A*/BFS routine is the feasibility and path-validity checker on GridWorld, never the expert. No human-subject data at any stage of the programme.

### 1.7 Report outline — 300 words
One paragraph per chapter.

---

## 2. Background and literature review — 5,500 words

Organised as the reference reports organise it: from the general formulation, through the machinery this work uses but does not own, to the specific literature the aims sit inside, closing on the gap. Every abbreviation is re-expanded at first use in the body.

### 2.1 Imitation learning and behaviour cloning — 500 words
The behaviour-cloning objective (imported from `sections/method.md` Eq. 3, relabelled), covariate shift, why offline cloning degrades on learner-visited states. Cited, flagged as standard, and never re-derived later.

### 2.2 Dataset aggregation and the interactive loop — 500 words
DAgger and its descendants; the aggregate-and-retrain skeleton (Eq. 4 of the current method draft) restated here once, "for completeness", so that the Aim-1 method never has to.

### 2.3 The DAgger-family query gates — 900 words
SafeDAgger, DropoutDAgger, EnsembleDAgger, ThriftyDAgger, Stagger (GridWorld only) and Diff-DAgger (robot only), presented as instances of one generic gate template. **Qualitative descriptions only; no reproduction of their hyperparameters** (supervisor instruction). This is where the material currently sitting in `sections/method.md` §"A Unified Query Framework" is moved to. State explicitly that Diff-DAgger's per-step diffusion loss as an uncertainty signal is *its* idea, used here as a signal and as a baseline.

### 2.4 Policy classes and why the framework is agnostic to them — 450 words
A policy is any f_θ with a per-step loss. Multilayer perceptron, convolutional network, and diffusion policies. Argue that the framework only requires the loss to be evaluable per step, which is what makes the same loop run on a discrete grid policy and a continuous manipulation policy.

### 2.5 Standard machinery this work uses and does not claim — 500 words
Agglomerative clustering as one instantiation of a generic clustering step (K-means and others would serve); the silhouette criterion for choosing the number of clusters; farthest-point / k-centre selection; A* and breadth-first search as the corridor-validity checker. Each cited. One sentence per item making the ownership explicit.

### 2.6 Language and vision-language models as reasoners in robotics — 900 words
Language models as planners, critics and reward designers; vision-language models as failure describers. What the literature shows they are reliable at (naming a cause given structured evidence) and what they are not reliable at (metric geometry unaided). This is the argument that motivates giving the model a *geometric descriptor and a constraint store* rather than raw pixels alone.

### 2.7 Structured environmental knowledge and constraint grounding — 500 words
Knowledge stores as explicit key-value constraint sets: workspace bounds, reachability, object and spawn ranges, controller limits. Position the knowledge-augmented graph (KAG) against retrieval-augmented generation: the KAG is not a document store, it is a constraint store queried during verification.

### 2.8 Demonstration selection, curation and active learning — 700 words
Coreset and diversity selection; dataset curation by embedding distance; active learning acquisition functions. Argue what all of these share: they select from an existing pool. The Aim-1 framework *prescribes a demonstration that does not exist yet*, which is a different problem.

### 2.9 Vision-language-action models — 550 words
The forward mapping (vision, language) → action, and language-as-intermediate work. This subsection exists to set up Aim 2's inversion, and says so in its last sentence.

### 2.10 Open problems and the gap addressed by this programme — 400 words
Three gaps, one per aim, each traceable to a subsection above. No formulaic close.

---

## 3. The research programme — 1,200 words

### 3.1 One question in three stages — 700 words
**Purpose.** Make the panel see one programme, not three papers.
**Argue.** Aim 1 decides *which failure to fix and where to start*, using the policy's own failures. Its selector is dataset-blind: it can say "the policy failed here" but not "and we already hold six demonstrations of that". Aim 2 gives the selector a memory of what it has been taught, in language. Aim 3 turns the memory outward, across tasks, embodiments and teachers. Each aim is the correction to the limitation the previous aim's evaluation exposed, and that is the sentence the transition must earn each time.
**Asset.** New figure: three-panel programme diagram (Figure 2), to be produced; it reuses the visual grammar of `figures/Architectural Diagram.pdf`.

### 3.2 Methodology and validation strategy — 500 words
Quantitative, deductive, matched-comparison design: one loop skeleton, one policy backbone per setting, one expert, one evaluation protocol, and the demonstration-acquisition rule as the only free variable. Ablation as the mechanism test, comparison as the performance test. Define ΔSR here or at first use in §4.6, whichever comes first: **ΔSR is the change in the policy's success rate on the round-level rollout evaluation**.

---

## 4. Aim 1 — DISEIL: Demonstration Distillation for Sample-Efficient Imitation Learning — 9,000 words

The core chapter. Written so that a reader who skips the rest of the report can still evaluate it.

### 4.1 Motivation and problem statement — 500 words
The budget is fixed; the only lever is the information content of each demonstration. State the problem formally with symbols: budget B, demonstrations per round D, policy f_θ, per-step loss ℓ_t.

### 4.2 The gap in the DAgger family — 450 words
Sharpen §1.2 into a stated deficiency: *when* is one of three decisions, and the other two are unclaimed. Name them: which failure mode, and where the demonstration starts.

### 4.3 Problem formulation — 550 words
The interactive loop as a skeleton shared by every method compared (cited to §2.2, not re-derived). Define the round, the rollout evaluation, the failure set, and the stopping rule symbolically. B and D remain symbols throughout.

### 4.4 The DISEIL framework — 2,400 words
Only the novelty. Four stages, each with its own short subsection.

**4.4.1 Perceive — 500 words.** Failure localisation at the peak-loss step t*; the 6-D geometric descriptor φ, given for both domain families (robot and GridWorld); the vision-language model reading three key frames around t*. State once, unambiguously: **clustering is geometric for every run, state and image alike**; there is no visual-embedding branch.
**4.4.2 Partition — 400 words.** A generic clustering step C over the standardised descriptors, instantiated here as agglomerative clustering; the number of clusters chosen by the silhouette criterion (standard, cited in §2.5). The output is a set of failure modes with a representative and a mean peak loss each.
**4.4.3 Prioritise — 550 words.** The cross-round cluster memory: a recency-discounted Gaussian coverage penalty over already-corrected centroids, and the target-mode rotation it induces. The within-round diversity selection of which failures enter the context set S. This is the genuinely novel pair and the chapter should say so.
**4.4.4 Prescribe — 700 words.** The prescription LLM emits a scene command; the command becomes a corrective reset or sub-task-entry specification. **Two distinct checks, presented separately and never conflated:**
  - *Feasibility verification against the KAG.* The LLM proposes a prescription; constraints are retrieved from the knowledge-augmented graph (workspace bounds, reachability, object and spawn ranges, controller limits); the prescription is checked against them; if a constraint is violated the violation is returned to the LLM as feedback and a revised prescription is requested, until a feasible one is produced. This is the mechanism of Equation 10.
  - *Policy solvability.* The prescribed scenario P is rolled out under the current policy. If the current policy already solves P, the prescription carries no information and P is revised ("Solvable ⇒ Revise P").
**4.4.5 Algorithm — 250 words + algorithm float.** Atomic steps, readable standalone, an imitation-learning reader who skips the prose gets the whole method from it. Loop header `for r = 1 to B`. Separate steps for: roll out; localise failures; describe frames; featurise; cluster; select target mode under memory; select context set; prescribe; verify against KAG; check solvability; collect D demonstrations; aggregate; retrain.

### 4.5 Architecture — 700 words
**Asset.** The **updated** architecture figure (Figure 3), replacing `figures/Architectural Diagram.pdf`. It must be described exactly as drawn, including both loops: the KAG-constraint feasibility loop and the policy-solvability loop (Prescription LLM → Policy Rollout on P → Solvable ⇒ Revise P).
Walk each block: rollout and failure detection; VLM frame description; descriptor extraction; clustering; cluster memory; prescription LLM; KAG; feasibility verification; policy-solvability check; expert; aggregation; retrain. One paragraph per block, each stating what the block consumes, what it emits, and what breaks without it (forward-referencing the ablation that proves the last claim).

### 4.6 Implementation and experimental setup — 1,100 words
All concrete values live here and nowhere else.
- **Tasks.** GridWorld 5×5 (discrete, 3 obstacles, start → goal, human expert; A*/BFS is the feasibility and path-validity checker only), Push-T (ManiSkill), Lift / Wipe / Door (RoboSuite, UR5/UR5e).
- **Modalities and settings.** 5 tasks × 2 modalities = 10 settings. Define "setting" before it is used.
- **Policies.** GridWorld image = convolutional network; GridWorld state = multilayer perceptron; robot tasks = state and image diffusion policies.
- **Budget and protocol.** B = 20 as the validated instance; D = 1 demonstration per round, with A12 cited as the justification.
- **Initial demonstrations.** State the count per task and the reasoning: the count was chosen to place each task's starting success rate inside a target initial range, high enough that rollout failures are meaningful and low enough that the budget has headroom to matter. This is the first half of the information-gain argument in §4.8.
- **Seeds.** Nine on GridWorld, five on the robot tasks.
- **Baselines.** Named and described qualitatively, each one line, and **explicitly labelled as the DAgger family** in the table header and in the text: SafeDAgger, DropoutDAgger, EnsembleDAgger, ThriftyDAgger, Stagger (GridWorld only), Diff-DAgger (robot only). No baseline hyperparameters.
- **Metrics.** Final held-out success rate; per-demonstration information gain; ΔSR, defined at first use as the change in success rate on the round-level rollout evaluation.

### 4.7 Results — 1,300 words
**Assets and presentation.**
- Table 1: final success rate, all 10 settings × 7 methods, from workbook sheet `GT_SR`. Baseline columns grouped under a "DAgger family" header rule.
- Figure 4: success rate versus demonstrations for the representative settings, from `figures/all_5_task_comparison.pdf` (relabelled).
- Figure 5: the aggregate result as a paired-difference plot (DISEIL minus best baseline, one point per setting), which communicates 10/10 better than a table does. Source: `S1_SignTest` and `build/stats_report.md` §5.
**Argue.**
- Best mean in all ten settings; mean margin +3.71 points (sd 2.05).
- Report the aggregate test honestly and in the conservative form: the ten settings are not ten independent experiments, because the state and image variants of a task share the expert, the reset distribution and the reward structure. Collapsed to five task means the result is 5/5 (one-sided sign test p = 0.031; paired t-test t(4) = 4.15, p = 0.014); the two-sided sign test at n = 5 does not reach significance (p = 0.063). Take the collapsed result as the claim of record.
- Per-task margins: GridWorld +2.80, Push-T +5.15, Lift +0.60, Wipe +5.20, Door +4.80.
- Flag the Wipe (image) non-plateau: both DISEIL and the strongest baseline are still rising at demonstration 20, so the claim rests on the final gap (95.3 versus 89.6) and not on a demonstrated plateau.
- State the Lift ceiling here, before the ablations, so it never has to be re-explained.

### 4.8 Information gain, starting performance, and why the gain is real — 900 words
**Purpose.** This is a claim with an argument, not a hypothesis, and the section must read that way.
**Assets.** Workbook sheet `GT_InfoGain`; `figures/info_gain_boxplot.pdf`; `figures/confidence_vs_success.pdf`; `table_data.xlsx` sheets `BoxplotSummary`, `Conf vs SR`, `demo vs sr`.
**Argue, in this order.**
1. **Definition.** Information gain is the policy's per-step loss on a newly acquired demonstration, measured *before* retraining on it.
2. **Starting performance.** The number of initial demonstrations per task was chosen to place the starting success rate inside a target range: enough competence that rollout failures are informative rather than uniform, low enough that the budget has room to change the outcome. Without this, a high pre-retrain loss would be uninterpretable.
3. **The disjunction.** A high pre-retrain loss means either (a) the demonstration covers an underrepresented region of the state space, or (b) the demonstration is suboptimal or invalid.
4. **Ruling out (b) by construction.** Prescriptions pass the KAG feasibility check, and the demonstrations come from the expert. Neither an infeasible scenario nor a bad action survives to the dataset.
5. **Therefore.** High pre-retrain loss identifies genuinely novel, underrepresented data, and DISEIL's higher information gain in every setting (sheet `GT_InfoGain`) is a statement about coverage, not about noise.
6. **Prescription confidence as a predictor of realised improvement**, with the Pearson correlation from `Conf vs SR` reported as a single claim in its own sentence.

### 4.9 Ablation studies — 2,700 words
**Scope (author instruction).** Representative studies only, on three primary settings: **GridWorld (image), Push-T (state), Door (image)**. Every other cell is retained for supplementary and rebuttal and is signposted as such, not tabulated in full. Every study gets: motivation, setup, findings, why it works or fails, implications, limitations, and its effect on the final framework. Choose the clearest presentation for each; do not force everything into tables.

Grouped into three families:

**4.9.1 Knockouts — what is load-bearing (900 words).**
Grouped bar chart (margin retained, % of the DISEIL-over-best-baseline margin) across the three primary settings, one bar per knockout. Sheets: `A1_Memory_Off`, `A3_Clustering_Off`, `A4_LLM_vs_Heuristic`, `A5_VLM_Off`, `A6_KAG_Off`, `A7_Bridging_Off_`, `A8_Fallback_Only`, `A2_RandomAlloc_Robots`.
Must argue: clustering (A3) and the full reasoning stack (A8) carry most of the margin; the memory (A1) is smaller than the sweep of §4.9.3 suggests it should be, and the reason is the σ mis-scaling identified in A13, not the memory being unimportant; the LLM prescription (A4) and the VLM (A5) have small marginal value *because clustering has already done most of the localisation work*, which is a finding about where the mechanism lives and should be stated as such rather than defended away; the KAG (A6) shows its effect as a rise in the fallback rate (27–34%) more than as a success-rate drop, so its function is to bound the damage of a bad prescription rather than to raise the ceiling; A2 (uniform-random allocation, i.e. Stagger's rule applied to the robot tasks) lands *below* the gate baselines, which is the control that shows structured allocation is doing work.

**4.9.2 Design choices — is the instantiation the right one (900 words).**
- **A10, descriptor dimensionality.** Line plot of mean silhouette against descriptor dimension (2, 4, 5, 6, 8), showing the inverted U peaking at 6-D. This is also the section that states, once and for the record, that clustering is geometric in every run and that the frozen-embedding image branch was retired.
- **A14, cluster count.** Silhouette-adaptive k against fixed k ∈ {2,3,4,5}; adaptive matches the best fixed k without knowing it in advance. Pair with `D2` (the distribution of chosen k*) to show the adaptivity is real and not a disguised constant.
- **A9 and A15, the context set.** What goes into S (forced representative, worst-loss seed, farthest-point fill) and how many failures it needs (plateau at 2–3). Note the Top-1 row of A15 is confounded (n = 1 makes bridging impossible) and must not be read as a data point.
- **A11, budget sweep** (B ∈ {10, 20, 40}). The margin is largest at the smallest budget and narrows as B grows. This is the empirical backing for the "any fixed budget, B = 20 is the instance" framing, and it belongs here, not in the method.
- **A12, demonstrations per round** (D ∈ {1,2,3}). D = 1 wins at fixed budget in all settings; this is the evidence that justifies the D = 1 instance and it is cited from §4.6.

**4.9.3 Sensitivity and diagnostics — where the instantiation is wrong (900 words).**
- **A13, the memory constants**, with the statistics already computed in `build/stats_report.md`. Report all three families:
  - γ = 0.6 is significantly best (Friedman χ²(4) = 33.13, p = 1.1e-06; Holm-corrected p = 0.008 against every alternative), with a symmetric inverted U on either side.
  - λ = 1.0 is significantly best (Holm-corrected p = 0.008 throughout), including against λ = 0. **Cross-check to state explicitly: the λ = 0 column reproduces `A1_Memory_Off` in all ten settings, as it must.**
  - **σ is mis-scaled and this is reported as a limitation, not a virtue.** σ = 0.06 is directionally best but not distinguishable from its neighbours (corrected p = 0.125). The sweep is inert on six of ten settings: on GridWorld and Door the centroid separations produced by the reset distributions lie outside the range where a kernel of any swept width discriminates (GridWorld centroids are in grid-cell units; the Door reset range is on the order of a centimetre), and on Lift the 100.0 ± 0.0 ceiling makes nothing observable. A per-task σ, expressed as a fraction of each task's reset range, would make the memory term active everywhere. State this as the identified fix.
- **Diagnostics.** `D1` (root-cause label purity per geometric cluster: 0.78–0.93, lowest on Wipe where failures are least separable), `D3` (bridging chosen in 19–30% of accepted prescriptions, so it is used and is not a decorative equation), `D4` (failures per round falling 42 → 14 over the budget, which is why the context set can be small).
- **Deferred to supplementary.** `D5` (compute and token cost) is incomplete in the workbook; either report it as work in progress or omit it. Do not invent the numbers.

### 4.10 What the ablations changed in the framework — 350 words
A short, honest section: which design choices survived contact with the evidence, which were retired (the visual-embedding clustering branch), and which are flagged as wrong but not yet fixed (the global σ).

### 4.11 Limitations — 550 words
Simulation only. Scripted and planner experts, not humans. The global memory kernel width is mis-scaled for narrow-reset tasks. Lift has no headroom and therefore contributes nothing to any mechanism claim. The Wipe (image) curve has not been shown to plateau within the budget. The selector is dataset-blind, which is the limitation Aim 2 exists to remove, and that sentence is the transition into §5.

### 4.12 Status and publication — 150 words
Submitted to AAAI 2027 main track, July 2026.

---

## 5. Aim 2 — Reverse-VLA: a coverage memory of what has already been taught — 4,000 words

Source: `paper2_reverse_vla_concept.md`. Architecture figure extracted from `paper2_preview.pdf`, captioned exactly **"Proposed architecture for Aim 2"**.

### 5.1 The limitation Aim 1 leaves — 500 words
The Aim-1 selector is stateless and dataset-blind. It can identify the failure and its cause; it cannot say "and the training set already contains six demonstrations of that cause, so it is not the gap". Demonstrations can therefore be spent re-teaching what the policy already knows, which is the opposite of the programme's objective.

### 5.2 Core idea — inverting the vision-language-action mapping — 600 words
A vision-language-action model maps (vision, language) → action. The inversion maps (vision, action) → language: given a trajectory's observations and its executed action sequence, emit captions at three granularities (trajectory intent, sub-skill spans, failure root cause anchored at t*). Argue why the *action* sequence is the discriminative signal that vision alone lacks.

### 5.3 Method — 1,000 words
The captioner and its heads; label sources that require no manual captioning (privileged simulator state → templated captions, constrained distillation from a larger teacher, self-supervised action segmentation, contrastive failure labels); the training objective and its anti-hallucination term. Then the coverage memory: a language-indexed store of every demonstration's captions and embeddings, with per-skill support counts weighted by the policy's measured success rate on that skill, so that a skill that is present but not yet learned still reads as a gap. Then the single memory-conditioned selector that replaces Aim 1's three separate calls.

### 5.4 Architecture — 500 words
**Asset.** Figure extracted from `paper2_preview.pdf`, captioned "Proposed architecture for Aim 2". Walk each component and, for each, state which Aim-1 component it subsumes: the captioner subsumes the frame-describing VLM; its failure head subsumes the root-cause reasoning; the memory subsumes the cluster memory and the KAG; the single selector subsumes the prescription LLM.

### 5.5 Novelty and positioning — 550 words
What is genuinely new: the captioning-as-self-awareness loop; selection driven by a *coverage gap* rather than local uncertainty; a language rationale for why few demonstrations suffice. What is borrowed and must be framed as borrowed: the captioner primitive, uncertainty-triggered querying, t* localisation. Differentiate from forward VLA models, language-as-intermediate work, robot trajectory captioning, inverse dynamics, and the DAgger family, one clause each.

### 5.6 Evaluation strategy — 550 words
Primary metric: demonstrations-to-threshold and the area under the success-versus-demonstrations curve, against Aim 1 as the key head-to-head, on Push-T (continuity with Aim 1) plus a broader manipulation suite. The load-bearing experiment, stated as such: a matched-information ablation that varies only the representation the selector consumes (generated captions; Aim-1 geometric descriptors; a learned trajectory embedding at equal budget; content-scrambled captions as a placebo) with an oracle-caption ceiling. Pre-commit to the interpretation: if the learned embedding matches the captions, the contribution is interpretability, not sample efficiency.

### 5.7 Risks — 300 words
Caption faithfulness is the top risk, because everything downstream reasons over the caption. Circularity, since the captioner is trained on rollouts of the policy it is improving. Train-deploy shift, since the captioner trains mostly on clean expert demonstrations and must caption out-of-distribution failures.

### 5.8 Relationship to Aim 1 and target venue — 200 words
CoRL 2027, submission late May 2027.

---

## 6. Aim 3 — Demonstration demand across tasks, embodiments and teachers — 3,000 words

Original, ambitious, extends Aim 2, completes the story. The panel should read it as the natural third step, not as a bolt-on.

### 6.1 The limitation Aim 2 leaves — 400 words
Aim 2 gives one policy, on one task, a memory of what it has been taught. That memory is task-local and its supplier is a scripted expert who is always available and always correct. Neither holds in the setting the programme is ultimately about: a generalist policy, many tasks, one embodiment or several, and a human teacher whose time is the actual budget.

### 6.2 The idea — a demonstration demand model — 700 words
Aim 3 proposes to make demonstration *demand* an explicit, transferable object. The coverage memory of Aim 2 becomes a cross-task, cross-embodiment skill inventory; the selector becomes a demand model that answers three questions in one language-expressible request: which skill is missing, on which task and embodiment it should be demonstrated, and what the demonstration is worth relative to its cost in human time. Because the request is in language, it can be issued to a **non-expert human** rather than to a scripted oracle, and because the skill inventory is shared across tasks, a demonstration collected for one task can be credited against the demand of another. The research question: can a generalist policy learn to ask for exactly the demonstrations it lacks, from a teacher who does not know the policy's internals?

### 6.3 Proposed method — 900 words
Three components, each extending an Aim-2 component rather than replacing it.
- **A cross-task skill inventory.** Captions from Aim 2 are aggregated into a shared, embodiment-annotated skill space, so coverage is measured over skills, not over trajectories of one task.
- **A demand model with a price.** Each candidate demonstration request carries an expected information gain (the Aim-1 measure, now predicted rather than measured after the fact) and an expected human cost. Selection maximises information gain per unit of teacher time, which turns the budget from a demonstration count into a time budget and makes the framework's central quantity, the value of one demonstration, explicit.
- **A non-expert teaching interface.** The request is rendered as a natural-language instruction plus a scene specification that the KAG-style feasibility check has already verified, so an untrained human can satisfy it. Demonstrations that fail the check are never requested.
- **Transfer credit.** When a demonstration is collected, it is captioned and credited against every task in the inventory whose demand it partially satisfies, which is what makes the demonstration economy cross-task rather than per-task.

### 6.4 Evaluation strategy — 600 words
Multi-task manipulation suites with a defined skill taxonomy, so that coverage is measurable against a ground-truth inventory. Metrics: demonstrations-to-threshold across a task family; teacher-time-to-threshold as the primary economic metric; transfer credit (how many tasks a single demonstration advances); and a human study in which non-expert teachers satisfy the generated requests, measured on request-satisfaction rate and on downstream policy gain. Controls: per-task demand (no transfer credit), uniform demand, and Aim-2 single-task selection.

### 6.5 Risks and what completes the story — 400 words
The human study is the first point in the programme where humans enter, and the ethics of that are addressed in §9. The risk that language is too coarse a coverage index across embodiments; the mitigation, fusing the caption embedding with a geometric one, is already flagged in Aim 2's risk list. Close on the arc: Aim 1 decides which failure to fix, Aim 2 knows what it has already been taught, Aim 3 knows what it is worth to be taught next and can ask a person for it. Target venue CoRL 2028, submission late May 2028.

---

## 7. Coherence of the research programme — 1,200 words

### 7.1 One thesis, three levers — 500 words
Each aim raises the information content of a demonstration at a different level: within a round (Aim 1), across the dataset (Aim 2), across tasks and teachers (Aim 3). State the through-line in one sentence and then show it holds by tracing a single quantity, the value of one demonstration, through all three.

### 7.2 What each aim contributes to the thesis chapters — 400 words
Map aims to thesis chapters and to the three papers.

### 7.3 Contingencies — 300 words
If Aim 2's matched-information ablation shows language adds no efficiency over a learned embedding, the contribution is reframed as interpretability and Aim 3 proceeds on the geometric inventory. If the Aim-3 human study is not approvable in time, the demand model is validated against scripted teachers with simulated cost models. Neither contingency breaks the arc.

---

## 8. Project plan and Gantt chart — 1,500 words

### 8.1 Completed work — 400 words
Literature review; Aim-1 framework, implementation across 10 settings and 6 baselines; the ablation programme (15 studies plus 5 diagnostics); the statistical analysis; the Aim-1 manuscript submitted to AAAI 2027 main track in July 2026.

### 8.2 Publication plan — 300 words
Aim 1 → AAAI 2027 main track, submitted July 2026. Aim 2 → CoRL 2027 (abstract and full paper late May 2027; conference October–November 2027). Aim 3 → CoRL 2028 (abstract and full paper late May 2028; conference early November 2028).

### 8.3 Milestone table — 300 words + Table
A milestone table in the style of the reference report's Table 18: three year-blocks (Year 1 Nov 2025 – Aug 2026, completed; Year 2 Aug 2026 – Oct 2027, Aim 2; Year 3 Nov 2027 – Nov 2028, Aim 3 and thesis), each row a milestone with a completion or target date. Anchors: candidature start Nov 2025; CoC Aug 2026; AAAI-27 author response and camera-ready; CoRL-27 submission May 2027; mid-candidature review; CoRL-28 submission May 2028; thesis chapters compiled; pre-submission review; **thesis submission November 2028**.

### 8.4 Gantt chart — 500 words of surrounding text + full-width figure
**Asset.** New figure, to be generated. Professional HDR-style horizontal Gantt covering Nov 2025 – Nov 2028 at month resolution, with milestone diamonds. Bars: literature review; problem formulation; methodology development; implementation; experimentation; evaluation; paper writing; conference submissions; revisions; Aim 1 completion; Aim 2 completion; Aim 3 completion; thesis writing; thesis submission; examination preparation. The chart must be consistent with §8.3 to the month; any disagreement between the two is a defect.

---

## 9. Ethical considerations — 500 words
All work to date is in simulation, with no human participants and no personal data. The only point at which humans enter the programme is the Aim-3 teaching study, and that study will be submitted for Deakin human-research ethics approval before any participant is recruited; participants demonstrate robot tasks in simulation, no personal data is collected beyond what the demonstration itself contains, and no demonstration request is issued that has not already passed the feasibility check. Language-model use is confined to the demonstration-selection loop and never to the robot's control loop.

---

## 10. Higher-degree research training and other research activities — 600 words
**Assets.** `SSC900 Academic Writing Result.pdf`, `Research_Integrity_Deakin_Safety_and_Research_Integrity_Training_KHANAL.pdf`, `Certificate_of_Completion_-_Respect_at_Deakin_HDR_...KHANAL.pdf`, `Compulsory Training Status.png`.
Prose summary of the completed compulsory modules with completion dates as they appear on the certificates (transcribe; do not guess a date), plus institute seminars and reading groups. Certificates reproduced in Appendix A, the compulsory-training status screenshot included as the summary table.

---

## 11. Conclusion — 700 words
What has been established (Aim 1, with its evidence and its two identified defects), what is under way, and what the programme will have shown by November 2028. End on a concrete next step, not on a formulaic summary of challenges.

---

## 12. References
Author-year, in the style of the more recent reference report (parenthetical citations with hyperlinked years). Sources: `draft/references.bib`, extended for the Aim-2 and Aim-3 literature. Every reference verified before inclusion; no fabricated DOIs, venues or years. The 2025/2026 arXiv items listed at the end of `paper2_reverse_vla_concept.md` are flagged there as unverified and must be checked or dropped.

---

## Appendices

- **A. HDR training certificates** — the three certificate PDFs, one per page, plus the compulsory-training status image.
- **B. Full ablation tables** — all ten settings for every study in the workbook, for the studies presented on the three primary settings in §4.9.
- **C. Supplementary results** — the cells and studies not tabulated in the main body, signposted from §4.9 as retained for supplementary and rebuttal.
- **D. Statistical appendix** — the Friedman, Wilcoxon, Holm-Bonferroni, sign-test and paired-t procedures and their outputs, reproduced from `build/stats_report.md`, with the analysis script referenced.

---

## Asset checklist

| Asset | Status | Used in |
|---|---|---|
| `A2I2_Logo_Stacked_2025_Keyline.png` | present | Cover page |
| `figures/Teaser_Diagram.pdf` | present, relabel | Fig. 1, §1.1 |
| Programme diagram (3 aims) | **to produce** | Fig. 2, §3.1 |
| Updated architecture figure (both loops) | **to produce**; supersedes `figures/Architectural Diagram.pdf` | Fig. 3, §4.5 |
| `figures/all_5_task_comparison.pdf` | present, relabel | Fig. 4, §4.7 |
| Paired-difference plot (DISEIL − best baseline) | **to produce** from `S1_SignTest` | Fig. 5, §4.7 |
| `figures/info_gain_boxplot.pdf` | present, relabel | §4.8 |
| `figures/confidence_vs_success.pdf` | present, relabel | §4.8 |
| `figures/clustering_modes_pushT.pdf` | present, relabel | §4.4.2 or §4.9.2 |
| Knockout bar chart | **to produce** from A1–A8 | §4.9.1 |
| Descriptor-dimension line plot (inverted U) | **to produce** from `A10` | §4.9.2 |
| Budget-sweep plot | **to produce** from `A11` | §4.9.2 |
| Memory-constant sweep panels | **to produce** from `A13` | §4.9.3 |
| Aim-2 architecture (from `paper2_preview.pdf`) | present, extract | §5.4, captioned "Proposed architecture for Aim 2" |
| Gantt chart | **to produce** | §8.4 |
| Training certificates (3 PDFs + status PNG) | present | §10, Appendix A |
| `ablations_results/DISTIL_ablation_results.xlsx` | present; **rename to DISEIL** | §4.7–4.9, Appendix B/C |
| `build/stats_report.md`, `build/stats_results.csv` | present, already uses DISEIL | §4.7, §4.9.3, Appendix D |

---

## Word-count summary

| Section | Words |
|---|---|
| F.1 Cover page | 60 |
| F.3 Executive summary | 650 |
| 1. Introduction and research vision | 3,000 |
| 2. Background and literature review | 5,500 |
| 3. The research programme | 1,200 |
| 4. Aim 1 — DISEIL | 9,000 |
| 5. Aim 2 — Reverse-VLA | 4,000 |
| 6. Aim 3 — Demonstration demand | 3,000 |
| 7. Coherence of the research programme | 1,200 |
| 8. Project plan and Gantt | 1,500 |
| 9. Ethical considerations | 500 |
| 10. HDR training | 600 |
| 11. Conclusion | 700 |
| **Total prose** | **~30,900** |
