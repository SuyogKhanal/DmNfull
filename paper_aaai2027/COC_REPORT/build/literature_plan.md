# Literature plan and citation map — Confirmation of Candidature

Student: Suyog Khanal (s226137394), A2I2, Deakin University.
Thesis: *Leveraging Large Language Models for Sample-Efficient Imitation Learning*.
Bibliography for the report: `build/references_coc.bib` (101 entries: 52 carried over, 49 newly verified).

This document maps every claim the report needs to make onto the citekey that supports it. It follows the section numbering of `build/coc_structure.md`. A claim with no citekey beside it is either a statement about this project's own results (source of truth: the ablation workbook) or a definition, and must be written as such.

## Rules that bind the writing agents

1. The method is **DISEIL**. The strings *DISTIL*, *PACE* and *P4* must not appear in the report. The code identifiers `p4_top3_rotate` and `p4_subtask` are code identifiers and must not appear in prose. Note that `context/references.bib` and `context/litreview.md` still say "PACE" in their comment headers and per-paper "Relation to" fields; those are internal notes, not report text, and the header comment has already been rewritten in `references_coc.bib`.
2. Every non-obvious claim carries a citekey from `references_coc.bib` or it is cut. No citation is to be invented, and no citekey outside that file is to be used.
3. Standard machinery is cited where it is used and explicitly flagged as standard (supervisor instruction). The behaviour-cloning objective, the aggregate-and-retrain skeleton, the generic query-gate template, the silhouette criterion, A*, breadth-first search and the Diff-DAgger rule all belong to Background, not to the Method.
4. Baselines are described qualitatively. Their hyperparameters are not reproduced (supervisor instruction), even though `context/litreview.md` records them.
5. **Stagger has no citation and must never be given one.** `context/dossier_baselines.md` §5 states plainly that Stagger is a uniform-random control implemented in this repository, not a published method. In the comparison tables the five published gates (SafeDAgger, DropoutDAgger, EnsembleDAgger, ThriftyDAgger, Diff-DAgger) are grouped and labelled as the DAgger family; Stagger is placed in its own row, labelled as a uniform-random control, and the caption says so. Labelling it a DAgger-family method would be a false attribution.
6. Terminology: a **setting** is one task under one observation modality. **Mode** is reserved for failure modes (clusters). The word "mode" is never used for a modality anywhere in the report.

---

## Part A — Shared background (Chapter 2)

### 2.1 Imitation learning and behaviour cloning — 500 words

| Claim | Citekey |
|---|---|
| A policy can be learned by supervised regression onto expert actions; the earliest working system did this for road following. | `pomerleau1988alvinn`, `pomerleau1991efficient` |
| Behaviour cloning as a named framework: fit a policy to logged expert state-action pairs. | `bain2000cloning` |
| The imitation-learning problem, its variants and its formal treatment. | `argall2009survey`, `osa2018algorithmic` |
| Training and deployment distributions differ once the learner's own actions determine the states it visits; the reweighting literature calls this covariate shift. | `shimodaira2000covariate` |
| A behaviour-cloned policy's error compounds over the horizon, and the naive supervised reduction admits a quadratic-in-horizon regret; the interactive reduction removes it. | `ross2010reductions`, `ross2011dagger` |
| The recovery problem was visible in the very first systems: Pomerleau's lateral image shifts synthesise the off-centre corrections a centre-driving expert never demonstrates. | `pomerleau1991efficient` |
| Noise injection into the expert's control stream is the off-policy answer to the same problem. | `laskey2017dart` |

Write the objective once, flag it as standard, and never re-derive it in Chapter 4.

### 2.2 Dataset aggregation and the interactive loop — 500 words

| Claim | Citekey |
|---|---|
| The aggregate-and-retrain skeleton: roll out, query the expert on the states the learner visits, aggregate, retrain. | `ross2011dagger` |
| Cost-to-go rather than action agreement as the aggregation signal. | `ross2014aggrevate`, `sun2017aggrevated` |
| Interactive imitation learning as a field, its taxonomy and its feedback types. | `celemin2022iil` |
| Human-gated variants, where the human rather than the robot decides when to take over. | `kelly2019hgdagger` |
| Intervention data is worth more than on-policy data and is reweighted accordingly. | `mandlekar2020iwr`, `liu2023sirius` |

### 2.3 The DAgger-family query gates — 900 words

Present all published gates as instances of one template: a scalar signal, a threshold, a handover. Qualitative only.

| Gate | Signal | Citekey |
|---|---|---|
| SafeDAgger | learned safety classifier predicting large policy-expert deviation | `zhang2017safedagger` |
| DropoutDAgger | Monte-Carlo dropout spread of the novice's action distribution | `menda2017dropoutdagger` |
| EnsembleDAgger | ensemble variance as "doubt", combined with action discrepancy | `menda2019ensembledagger` |
| ThriftyDAgger | novelty plus a learned risk estimate, under a target switching rate | `hoque2021thriftydagger` |
| LazyDAgger | asymmetric switching thresholds to reduce context switches (context, not a baseline here) | `hoque2021lazydagger` |
| Diff-DAgger | the diffusion policy's own per-step training loss as the uncertainty signal | `lee2025diffdagger` |
| Stagger | uniform-random query time — **no citation, internal control** | — |

Two supporting claims. The dropout and ensemble signals are the two canonical deep uncertainty estimators, imported from the uncertainty literature rather than invented by the DAgger papers (`gal2016dropout`, `lakshminarayanan2017ensembles`). Diff-DAgger's use of the per-step diffusion loss as an uncertainty signal is *its* idea; DISEIL uses that signal and also compares against it as a baseline (`lee2025diffdagger`). Say this once, plainly.

The section closes on the deficiency that Chapter 4 sharpens: every gate answers *when* to hand over. None of them chooses *which* of a batch of failures to correct, and none of them chooses *where* the corrective demonstration begins.

### 2.4 Policy classes and why the framework is agnostic to them — 450 words

| Claim | Citekey |
|---|---|
| Explicit regression policies are a poor fit for multimodal action distributions; energy-based and generative formulations fit them better. | `florence2021implicitbc` |
| Denoising diffusion as a generative model. | `ho2020ddpm` |
| Diffusion applied to trajectories for planning. | `janner2022diffuser` |
| Diffusion applied to visuomotor policies, which is the robot-task learner used here. | `chi2023diffusionpolicy` |
| Offline human demonstration data and what matters when learning from it (the source of the RoboSuite-family policy conventions used here). | `mandlekar2021robomimic` |

The argument to make: the framework requires only that the policy expose a per-step loss. A multilayer perceptron on GridWorld state, a convolutional network on GridWorld images and a diffusion policy on the robot tasks all satisfy that. Never write that the framework is "for diffusion policies".

### 2.5 Standard machinery this work uses and does not claim — 500 words

| Item | Citekey | Sentence to write |
|---|---|---|
| Clustering step C, instantiated as agglomerative clustering | `ward1963hierarchical` | Agglomerative clustering is one instantiation of a generic partition step; k-means or another partition method would serve. |
| k-means, as the named alternative | `lloyd1982kmeans` | — |
| Choosing the number of clusters | `rousseeuw1987silhouette` | The silhouette criterion is standard and is used unmodified. |
| Diversity selection of the context set | `eldar1997fps` | Farthest-point / k-centre selection is standard. |
| A* as the feasibility and path-validity checker on GridWorld | `hart1968astar` | A* and breadth-first search check that a prescribed grid configuration admits a valid path. They are **never** the expert; the GridWorld expert is a human. |
| Breadth-first search | `cormen2022algorithms` | — |
| Implementation of the clustering, silhouette and standardisation steps | `pedregosa2011sklearn` | — |
| Visual representation used for the image-modality policies | `nair2022r3m` | R3M supplies the visual encoder for the image policies. **It does not supply the clustering features.** Per correction A10, clustering is geometric for every run, state and image alike; the frozen-embedding-plus-PCA branch is out of date and must not be described anywhere. |

`radosavovic2022mvp` and `ma2023vip` are cited once, in a single sentence, as the alternative pre-trained visual representations that were available.

### 2.6 Language and vision-language models as reasoners in robotics — 900 words

| Claim | Citekey |
|---|---|
| Language models can decompose an instruction into a plan over a fixed skill repertoire, provided the plan is grounded in what the robot can actually do. | `ahn2022saycan` |
| Language models can write executable policy code against a perception and control API. | `liang2023codeaspolicies`, `singh2023progprompt` |
| Multimodal models can take sensor input directly into the language model's embedding space. | `driess2023palme` |
| Language models can design rewards and cost maps. | `ma2024eureka`, `yu2023language2rewards`, `huang2023voxposer` |
| Closed-loop textual feedback improves an embodied planner. | `huang2022innermonologue` |
| Verbal self-critique and iterative self-refinement are established language-model patterns. | `shinn2023reflexion`, `madaan2023selfrefine` |
| A vision-language model can summarise a robot's experience and name the cause of a failure. | `liu2023reflect` |
| A vision-language model can be trained specifically to detect and reason over manipulation failures. | `duan2025aha` |
| Language models can be made to recognise when they do not know, and to ask a human. | `ren2023knowno` |
| Vision-language models are unreliable at metric and spatial reasoning from pixels alone. | `chen2024spatialvlm`, `fu2024blink` |
| The specific backbones used here. | `bai2025qwen3vl` (perception), `yang2025qwen3` (prescription) |

The argument this subsection exists to make: the literature shows these models are good at naming a cause given structured evidence and poor at metric geometry unaided (`chen2024spatialvlm`, `fu2024blink`). That is precisely why DISEIL hands the model a low-dimensional geometric descriptor and a constraint store instead of raw pixels alone. This sentence is the bridge to §2.7 and to the Method.

### 2.7 Structured environmental knowledge and constraint grounding — 500 words

| Claim | Citekey |
|---|---|
| Retrieval-augmented generation retrieves passages from a document store to condition generation. | `lewis2020rag` |
| Graph-structured retrieval organises the store rather than treating it as flat text. | `edge2024graphrag` |
| Robot knowledge bases have long stored explicit, queryable environmental and action knowledge. | `tenorth2013knowrob` |
| A language model's proposal can be checked by an external solver, and the solver's verdict returned as feedback. | `liu2023llmp` |
| The check can be iterated: propose, verify, return the violation, revise. | `chen2024autotamp` |

Position the knowledge-augmented graph (KAG) against retrieval-augmented generation in one sentence: the KAG is not a document store, it is a store of explicit key-value environmental constraints (workspace bounds, reachability, object and spawn ranges, controller limits) queried during verification. Then state the workflow that Equation 10 now denotes, in this order and no other: the prescription language model proposes a prescription; constraints are retrieved from the KAG; the prescription is checked against them; if a constraint is violated the violation is returned to the model as feedback and a revised prescription is requested; the loop repeats until a feasible prescription is produced. `liu2023llmp` and `chen2024autotamp` are the closest precedents for the propose-verify-revise pattern, and the report should say so, because it makes the KAG loop legible to a panel that has seen it before.

The policy-solvability check is a **separate** mechanism with no precedent in that list, and Chapter 4 must present the two checks separately (see Part B).

### 2.8 Demonstration selection, curation and active learning — 700 words

| Claim | Citekey |
|---|---|
| Active learning: the acquisition function decides which unlabelled point to label. | `settles2009active` |
| Coreset selection: cover the representation space rather than chase uncertainty. | `sener2018coreset` |
| Batch active learning that combines uncertainty with diversity. | `ash2020badge` |
| Expected information gain as a Bayesian acquisition criterion. | `houlsby2011bald` |
| Demonstration data is curated by retrieving sub-trajectories from an existing corpus. | `memmel2025strap` |
| Not all demonstrations are equal; data quality in imitation learning can be characterised. | `belkhale2023dataquality` |
| Diverse (including suboptimal) demonstrations can be exploited in offline imitation learning. | `yue2024diversedemos` |
| Data mixtures for large-scale imitation learning can be optimised. | `hejna2024remix` |
| Imitation-learning performance follows a scaling law in demonstration count and diversity. | `lin2024datascaling` |
| A dataset can be distilled into a smaller synthetic set that trains as well. | `cazenavette2022mtt` |

The argument, stated once and not repeated: every method above **selects from an existing pool** (or synthesises training data offline). DISEIL **prescribes a demonstration that does not exist yet**, and then has an expert produce it. That is a different problem, and it is why coreset and curation methods are background rather than baselines. `cazenavette2022mtt` is the closest name-level neighbour (dataset distillation) and the report should distinguish it explicitly, since the Aim-1 paper title uses the word *distillation*: dataset distillation compresses data that has already been collected; demonstration distillation decides which demonstration to collect next.

### 2.9 Vision-language-action models — 550 words

| Claim | Citekey |
|---|---|
| The forward mapping (vision, language) to action, at scale. | `brohan2023rt1`, `brohan2023rt2` |
| Open generalist policies trained on pooled cross-embodiment data. | `kim2024openvla`, `octo2024`, `oneill2023openx` |
| Flow-matching and other action-decoder variants of the same mapping. | `black2025pi0` |
| Generalist multi-embodiment agents. | `reed2022gato`, `bousmalis2023robocat` |
| Language as an intermediate representation *en route to* an action. | `belkhale2024rth`, `zawalski2024ecot` |

Last sentence of the subsection: every one of these maps vision and language to action. Aim 2 inverts the mapping. Say it once, here, and do not restate the inversion until §5.2.

### 2.10 Open problems and the gap addressed by this programme — 400 words

Three gaps, one per aim, each traceable upward.

- Gap 1 (from §2.3, §2.8). Query gates choose *when*. Selection methods choose *from a pool*. Nothing chooses which failure mode to correct and where the corrective demonstration should begin. → Aim 1.
- Gap 2 (from §2.8, §2.9). A selector that reasons only over the current failure is dataset-blind: it cannot tell a novel failure from one the training set already covers six times. → Aim 2.
- Gap 3 (from §2.6, §2.9). Demonstration demand is never made explicit, priced against a human teacher's time, or shared across tasks and embodiments. → Aim 3.

No formulaic close.

---

## Part B — Aim 1 (DISEIL), Chapter 4

The acronym derivation is shown once, in the executive summary, at first mention, by bolding exactly six letters of the Aim-1 paper title: **D**emonstration d**I**stillation for **S**ample-**E**fficient **I**mitation **L**earning. Not in the title. Not in headings. No other derivation.

### 4.2 The gap in the DAgger family

Re-use §2.3's citekeys; add nothing new. The three unclaimed decisions are stated against `zhang2017safedagger`, `menda2017dropoutdagger`, `menda2019ensembledagger`, `hoque2021thriftydagger`, `lee2025diffdagger` collectively, not individually.

### 4.3 Problem formulation

| Claim | Citekey |
|---|---|
| The interactive loop skeleton is shared by every method compared and is not re-derived. | `ross2011dagger` (cross-reference to §2.2) |
| The framework works under any fixed, restricted budget B; B = 20 is the validated instance and the value appears only in §4.6. | — (project fact; supervisor principle 1) |
| D = 1 demonstration per round is the tested instance, justified by ablation A12. | — (workbook) |

### 4.4 The DISEIL framework — only the novelty

**4.4.1 Perceive.** Failure localisation at the peak-loss step t*. The per-step loss as the localisation signal is used the way Diff-DAgger uses it (`lee2025diffdagger`) and that is acknowledged. The vision-language model reads three key frames around t* (`bai2025qwen3vl`; capability precedent `liu2023reflect`, `duan2025aha`). The descriptor is 6-D and geometric: robot [p_x, p_y, sin θ, cos θ, ρ, δ]; GridWorld [agent cell (2), signed offset (2), progress, Manhattan distance]. State once, unambiguously, that **clustering is geometric for every run, state and image alike** — there is no visual-embedding branch (correction A10). Ablation A10 shows an inverted-U in silhouette peaking at 6-D, which is the empirical justification for the descriptor's width.

**4.4.2 Partition.** A generic clustering step C over standardised descriptors, instantiated as agglomerative clustering (`ward1963hierarchical`), with the number of clusters chosen by the silhouette criterion (`rousseeuw1987silhouette`, standard, cited in §2.5). k-means (`lloyd1982kmeans`) is named as an equally admissible instantiation. Implementation: `pedregosa2011sklearn`.

**4.4.3 Prioritise.** The genuinely novel pair, and the chapter says so.
- Cross-round cluster memory: a recency-discounted Gaussian coverage penalty over already-corrected centroids, and the target-mode rotation it induces. Nearest ideas in the literature, cited as *related but not the same*: coreset coverage (`sener2018coreset`), diversity-plus-uncertainty batch acquisition (`ash2020badge`), intervention reweighting (`mandlekar2020iwr`, `liu2023sirius`). None of them is a cross-round memory over failure modes.
- Within-round diversity selection of the context set S: farthest-point selection (`eldar1997fps`), standard, cited as such.

**4.4.4 Prescribe.** The prescription language model (`yang2025qwen3`) emits a scene command, which becomes a corrective reset or a sub-task-entry specification. Two distinct checks, presented separately and never conflated.
- *Feasibility verification against the KAG* (Equation 10). Precedent for propose-verify-revise: `liu2023llmp`, `chen2024autotamp`. Precedent for an explicit robot knowledge store: `tenorth2013knowrob`. Contrast with document retrieval: `lewis2020rag`.
- *Policy solvability.* The prescribed scenario P is rolled out under the current policy; if the current policy already solves P, the prescription carries no information and P is revised ("Solvable ⇒ Revise P"). The nearest relatives are curriculum and reset-state work, which choose start states by what the learner can and cannot yet do: `florensa2017reversecurriculum`, `eysenbach2018leavenotrace`. Cite them as the intellectual neighbours; do not claim they do this check.

Both loops appear in the updated architecture figure and the figure must be described exactly as drawn.

**4.4.5 Algorithm.** Atomic steps, readable standalone, loop header `for r = 1 to B`. No citations inside the float.

### 4.6 Implementation and experimental setup

| Item | Citekey |
|---|---|
| Push-T environment | `mu2021maniskill`, `gu2023maniskill2`, `tao2024maniskill3` |
| Lift, Wipe, Door (UR5/UR5e) | `zhu2020robosuite` |
| Policy conventions for offline human-style demonstration data | `mandlekar2021robomimic` |
| Robot-task learner | `chi2023diffusionpolicy` (built on `ho2020ddpm`) |
| Image-modality visual encoder | `nair2022r3m` |
| GridWorld path-validity checker | `hart1968astar`, `cormen2022algorithms` |
| Language backbones | `yang2025qwen3`, `bai2025qwen3vl` |

Concrete values live here and nowhere else: B = 20, D = 1, 9 seeds on GridWorld, 5 seeds on the robot tasks. State the seed asymmetry; do not fabricate a uniform count. Define ΔSR at first use: **the change in the policy's success rate on the round-level rollout evaluation**.

### 4.7 Results and 4.8 Information gain

Results are project facts from the workbook and carry no citations. Two places need support:

| Claim | Citekey |
|---|---|
| Information gain is measured as the policy's per-step loss on a newly acquired demonstration, evaluated **before** retraining on it; the per-step loss of a diffusion policy is a usable signal of how out-of-distribution a datum is. | `lee2025diffdagger` |
| High loss on an unseen datum is the standard expected-model-change / information-gain intuition in active learning. | `settles2009active`, `houlsby2011bald` |
| Demonstration count and coverage govern imitation-learning performance, which is why the initial demonstration count was chosen to place each task's starting success rate inside a target range. | `lin2024datascaling` |

The argument to write out in §4.8, as a claim with an argument and not as a hypothesis: high pre-retrain loss means either (a) the demonstration covers an underrepresented region, or (b) the demonstration is suboptimal or invalid. Alternative (b) is ruled out by construction, because every prescription passes the KAG feasibility check and every demonstration comes from the expert. Therefore high pre-retrain loss identifies genuinely novel, underrepresented data. The initial demonstration count is set so that each task's starting success rate falls inside a target range: high enough that rollout failures are informative rather than uniform, low enough that the budget has headroom to matter.

### 4.9 Ablation studies

Representative studies only, on three primary settings: GridWorld (image), Push-T (state), Door (image). All other cells are retained for supplementary and rebuttal material and should be described as such, not tabulated in full. The Excel workbook is the source of truth for every number. Choose the clearest presentation per ablation (line plot, grouped or stacked bar, scatter, heatmap, radar; a table only where exact numbers matter). Every ablation discusses motivation, setup, findings, why some combinations work and others fail, implications, limitations, and its influence on the final DISEIL framework.

Three corrections are binding and must be written honestly:

- **Lift is uninformative for every ablation** (100.0 ± 0.0: no headroom, no variance). A null result on Lift is never evidence about any mechanism. Say so explicitly, once, and refer back to it wherever a Lift column is flat.
- **A13: σ is mis-scaled for narrow-reset tasks** (Door, GridWorld). The memory kernel is degenerate there, and on Lift the degeneracy is masked by the ceiling. A per-task σ, set as a fraction of each task's reset range, would make the memory function everywhere. This is reported as a limitation, not dressed up as a design choice.
- **A13 cross-check:** the λ = 0 column must reproduce A1_Memory_Off. If it does not, the ablation is not reportable.

Literature support in this section is thin by design; the only citekeys that recur are `rousseeuw1987silhouette` (for the descriptor-dimension study, A10) and `lee2025diffdagger` (wherever the diffusion loss is the measured quantity).

### 4.11 Limitations

| Claim | Citekey |
|---|---|
| Simulation only; the sim-to-real gap for demonstration-driven manipulation is a known open problem. | `khazatsky2024droid` (as the counterexample: real, in-the-wild demonstration collection at scale) |
| The expert is scripted or planner-based, not human, so the human cost the thesis is ultimately about is modelled, not measured. | `mandlekar2018roboturk`, `liu2023sirius` |
| σ mis-scaling (A13). | — (workbook) |
| Lift's ceiling (no headroom). | — (workbook) |

---

## Part C — Aim 2 (Reverse VLA), Chapter 5

### 5.1 The limitation Aim 1 leaves

No new citations; this is an argument about Aim 1's own selector.

### 5.2 Core idea — inverting the vision-language-action mapping

| Claim | Citekey |
|---|---|
| A vision-language-action model maps (vision, language) to action. | `brohan2023rt2`, `kim2024openvla`, `octo2024`, `black2025pi0` |
| Language has been used as an intermediate *en route to* an action, which is not the inversion. | `belkhale2024rth`, `zawalski2024ecot` |
| Describing an already-executed trajectory in language is an existing primitive, and must be framed as borrowed. | `krishna2017densecap`, `xu2026densemotion`, `suzuki2025proprioception`, `wulff2025jalm` |
| Proprioceptive and action signals carry skill semantics that frames alone miss — the reason the *action* sequence is the discriminative input. | `suzuki2025proprioception` |

The closest threat to novelty is `wulff2025jalm` (Joint Action Language Modelling for Transparent Policy Execution): it reconstructs the instruction to make one policy transparent. It builds no memory and performs no data selection. `suzuki2025proprioception` is the closest captioning mechanism and, by its own account, leaves imitation-learning integration to future work. Both facts must appear in §5.5, because a panel will find these papers.

### 5.3 Method

| Claim | Citekey |
|---|---|
| Frozen pre-trained visual encoders for the keyframe encoding. | `nair2022r3m`, `radosavovic2022mvp`, `ma2023vip` |
| Retrieval over the memory to condition the selector. | `lewis2020rag`, `edge2024graphrag` |
| Coverage and novelty as selection criteria over an embedding space. | `sener2018coreset`, `ash2020badge` |
| Sub-trajectory retrieval as the nearest existing use of a trajectory index. | `memmel2025strap` |
| Competence weighting: a skill that is present in the dataset but not yet learned still reads as a gap — the underlying observation that data presence is not data quality. | `belkhale2023dataquality` |
| Self-critique and revision as an established language-model loop for the selector's inner revision step. | `shinn2023reflexion`, `madaan2023selfrefine` |

### 5.4 Architecture

Figure extracted from `paper2_preview.pdf`, captioned exactly **"Proposed architecture for Aim 2"**. No citations in the caption.

### 5.5 Novelty and positioning — one clause each

| Neighbour | Citekey | The difference to state |
|---|---|---|
| Forward vision-language-action models | `brohan2023rt2`, `kim2024openvla`, `octo2024` | They map vision and language to action; Aim 2 inverts to language *about the dataset*. |
| Language-as-intermediate | `belkhale2024rth`, `zawalski2024ecot` | They emit language on the way to an action; Aim 2 describes already-executed trajectories in order to choose what to collect. |
| Robot trajectory captioning | `suzuki2025proprioception`, `xu2026densemotion`, `krishna2017densecap` | Open-loop description scored on caption quality; Aim 2 uses captions as the substrate for a selection decision and is scored on sample efficiency. |
| Joint action-language modelling | `wulff2025jalm` | Transparency for one policy; no memory, no selection. |
| Inverse dynamics and latent action | `liang2025clam`, `collins2025amplify` | They invert to actions or latents; Aim 2 inverts to language. |
| The DAgger family | `ross2011dagger`, `menda2019ensembledagger`, `hoque2021thriftydagger`, `lee2025diffdagger` | They query where the policy is locally uncertain and are blind to what the dataset already contains. |
| Demonstration and dataset selection | `memmel2025strap`, `belkhale2023dataquality`, `yue2024diversedemos` | They curate an existing pool by embedding distance; Aim 2 reasons in language and prescribes new demonstrations. |
| Language-model failure explanation | `liu2023reflect`, `duan2025aha` | Per-episode explanation and recovery; Aim 2 is cross-episode dataset design. |

### 5.6 Evaluation strategy

| Claim | Citekey |
|---|---|
| Continuity benchmark with Aim 1. | `mu2021maniskill`, `gu2023maniskill2` |
| Graded-difficulty manipulation suite for demonstrations-to-threshold curves. | `mandlekar2021robomimic`, `zhu2020robosuite` |
| A suite that ships language instructions, giving free ground truth for caption scoring and a defined skill taxonomy. | `liu2023libero` |
| A large named skill inventory, to test whether the memory prescribes complementary rather than redundant skills. | `yu2019metaworld` |
| Language-conditioned long-horizon manipulation, if the suite is broadened. | `mees2022calvin`, `james2019rlbench` |
| The scaling relationship the sample-efficiency claim is measured against. | `lin2024datascaling` |

The load-bearing experiment is the matched-information ablation, and the report must say so: freeze the selection loop and vary only the representation the selector consumes — generated captions, Aim-1 geometric descriptors, a learned trajectory embedding at equal budget, and content-scrambled captions as a placebo — with an oracle-caption ceiling. Pre-commit to the interpretation: if the learned embedding matches the captions, the contribution is interpretability, not sample efficiency.

### 5.7 Risks

| Risk | Citekey |
|---|---|
| Caption faithfulness; vision-language models hallucinate spatial and metric detail. | `chen2024spatialvlm`, `fu2024blink` |
| Circularity: the captioner is trained on rollouts of the policy it improves. Mitigation is a frozen, independently pre-trained backbone. | `nair2022r3m`, `bai2025qwen3vl` |
| Train-deploy shift: the captioner trains on clean expert demonstrations and must caption out-of-distribution failures. | `duan2025aha` |

---

## Part D — Aim 3 (demonstration demand across tasks, embodiments and teachers), Chapter 6

Aim 3 is a proposal, so its literature has one job: show the panel that each of its four components rests on something real, and that the combination does not yet exist.

### 6.1 The limitation Aim 2 leaves

| Claim | Citekey |
|---|---|
| Aim 2's memory is task-local. Generalist policies are trained across many tasks and embodiments, so a task-local coverage model does not compose. | `oneill2023openx`, `octo2024`, `kim2024openvla`, `reed2022gato`, `bousmalis2023robocat` |
| The supplier of demonstrations in Aims 1 and 2 is a scripted expert who is always available and always correct. Real demonstration collection is a human-time cost, at scale. | `mandlekar2018roboturk`, `khazatsky2024droid` |

### 6.2 The idea — a demonstration demand model

| Claim | Citekey |
|---|---|
| The value of a demonstration is not constant: performance follows a scaling law in demonstration count and diversity, so demonstrations have a measurable marginal return. | `lin2024datascaling` |
| Data mixtures can be optimised, which is the dataset-level analogue of pricing demand. | `hejna2024remix` |
| Expected information gain is the standard way to price a candidate query before it is answered. | `houlsby2011bald`, `settles2009active` |
| Demonstration quality varies and can be characterised, so demand must be expressed over quality, not only over count. | `belkhale2023dataquality` |

The research question: can a generalist policy learn to ask for exactly the demonstrations it lacks, from a teacher who does not know the policy's internals?

### 6.3 Proposed method — four components, each extending an Aim-2 component

| Component | Supporting literature | What it borrows and what is new |
|---|---|---|
| **Cross-task skill inventory.** Captions from Aim 2 are aggregated into a shared, embodiment-annotated skill space; coverage is measured over skills, not over trajectories of one task. | `oneill2023openx`, `octo2024`, `yu2019metaworld`, `liu2023libero`, `wang2023voyager` | Cross-embodiment pooling and open-ended skill libraries exist. Measuring *demonstration demand* over such an inventory does not. |
| **A demand model with a price.** Each request carries an expected information gain (the Aim-1 measure, now predicted rather than measured after the fact) and an expected human cost; selection maximises information gain per unit of teacher time. | `houlsby2011bald`, `settles2009active`, `lin2024datascaling`, `hejna2024remix` | Cost-sensitive acquisition is standard in active learning. Pricing a *robot demonstration* against a human's time, and turning the budget from a demonstration count into a time budget, is the new object. |
| **A non-expert teaching interface.** The request is rendered as a natural-language instruction plus a scene specification that the KAG-style feasibility check has already verified, so an untrained human can satisfy it; infeasible requests are never issued. | `ren2023knowno`, `mandlekar2018roboturk`, `khazatsky2024droid`, `christiano2017preferences`, `brown2019trex` | `ren2023knowno` is the direct precedent for a language model that knows when to ask a human. Preference and suboptimal-demonstration work (`christiano2017preferences`, `brown2019trex`) is the evidence that non-expert human input is usable signal. |
| **Transfer credit.** A collected demonstration is captioned and credited against every task in the inventory whose demand it partially satisfies. | `memmel2025strap`, `yue2024diversedemos`, `oneill2023openx` | Sub-trajectory retrieval already shows one trajectory can serve several tasks. Crediting it against an explicit demand ledger is new. |

### 6.4 Evaluation strategy

| Claim | Citekey |
|---|---|
| Multi-task suites with a defined skill taxonomy, so coverage is measurable against a ground-truth inventory. | `yu2019metaworld`, `liu2023libero`, `mees2022calvin`, `james2019rlbench` |
| Cross-embodiment evaluation. | `oneill2023openx` |
| The human study: non-expert teachers satisfy the generated requests. Precedents for crowdsourced and in-the-wild demonstration collection, and for the interfaces that make it tractable. | `mandlekar2018roboturk`, `khazatsky2024droid`, `liu2023sirius` |

Primary economic metric: teacher-time-to-threshold. Secondary: demonstrations-to-threshold across a task family, transfer credit (how many tasks one demonstration advances), request-satisfaction rate, downstream policy gain. Controls: per-task demand (no transfer credit), uniform demand, and Aim-2 single-task selection.

### 6.5 Risks

| Risk | Citekey |
|---|---|
| Language may be too coarse a coverage index across embodiments; kinematically different skills can collapse to the same caption. Mitigation, already flagged in Aim 2: fuse the caption embedding with a geometric one. | `chen2024spatialvlm`, `fu2024blink` |
| Non-expert demonstrations are suboptimal by construction, and the demand model must be robust to that. | `brown2019trex`, `yue2024diversedemos`, `belkhale2023dataquality` |
| The human study is the first point in the programme where humans enter; ethics is handled in Chapter 9 and no human-subject data exists at Aims 1 or 2. | — |

---

## Part E — Verification record

### Reused from the verified bibliography — 52

All 52 entries of `context/references.bib` are carried into `references_coc.bib` verbatim. Every one of them is used at least once in the map above.

### Newly verified and added — 49

Each was checked against the arXiv API, Crossref, dblp or PMLR during this task. The venue recorded in the bib entry is the venue the retrieved metadata confirmed. Where the metadata did not confirm a published venue, the entry is `@misc` against its arXiv identifier rather than an invented conference line.

| Citekey | Identifier | Verified against |
|---|---|---|
| `bain2000cloning` | OUP *Machine Intelligence 15*, pp. 103–129 | publisher record (see caveat below) |
| `argall2009survey` | doi 10.1016/j.robot.2008.10.024 | Crossref |
| `osa2018algorithmic` | arXiv 1811.06711 | arXiv (journal_ref: Foundations and Trends in Robotics) |
| `shimodaira2000covariate` | doi 10.1016/S0378-3758(00)00115-4 | Crossref |
| `ross2010reductions` | PMLR v9, pp. 661–668 | PMLR |
| `hart1968astar` | doi 10.1109/TSSC.1968.300136 | Crossref |
| `cormen2022algorithms` | MIT Press, 4th ed. | publisher record |
| `rousseeuw1987silhouette` | doi 10.1016/0377-0427(87)90125-7 | Crossref |
| `ward1963hierarchical` | doi 10.1080/01621459.1963.10500845 | Crossref |
| `pedregosa2011sklearn` | arXiv 1201.0490 | arXiv (journal_ref: JMLR 2011) |
| `houlsby2011bald` | arXiv 1112.5745 | arXiv |
| `lin2024datascaling` | arXiv 2410.18647 | arXiv |
| `hejna2024remix` | arXiv 2408.14037 | arXiv |
| `memmel2025strap` | arXiv 2412.15182 | arXiv + ICLR 2025 confirmed |
| `belkhale2023dataquality` | arXiv 2306.02437 | arXiv |
| `yue2024diversedemos` | arXiv 2405.17476 | arXiv (journal_ref: ICML) |
| `cazenavette2022mtt` | arXiv 2203.11932 | arXiv (journal_ref: CVPR 2022) |
| `lewis2020rag` | arXiv 2005.11401 | arXiv (comment: NeurIPS 2020) |
| `edge2024graphrag` | arXiv 2404.16130 | arXiv |
| `tenorth2013knowrob` | doi 10.1177/0278364913481635 | Crossref |
| `liu2023llmp` | arXiv 2304.11477 | arXiv |
| `chen2024autotamp` | arXiv 2306.06531 | arXiv (comment: ICRA 2024) |
| `ren2023knowno` | arXiv 2307.01928 | arXiv (comment: CoRL 2023) |
| `chen2024spatialvlm` | arXiv 2401.12168 | arXiv |
| `fu2024blink` | arXiv 2404.12390 | arXiv (comment: ECCV 2024) |
| `brohan2023rt1` | arXiv 2212.06817 | arXiv |
| `kim2024openvla` | arXiv 2406.09246 | arXiv + dblp (CoRL 2024) |
| `octo2024` | arXiv 2405.12213 | arXiv + dblp (RSS 2024) |
| `belkhale2024rth` | arXiv 2403.01823 | arXiv + dblp (RSS 2024) |
| `zawalski2024ecot` | arXiv 2407.08693 | arXiv + dblp (CoRL 2024) |
| `black2025pi0` | arXiv 2410.24164 | arXiv (journal_ref: RSS 2025) |
| `oneill2023openx` | arXiv 2310.08864 | arXiv |
| `bousmalis2023robocat` | arXiv 2306.11706 | arXiv (journal_ref: TMLR 12/2023) |
| `reed2022gato` | arXiv 2205.06175 | arXiv (journal_ref: TMLR 11/2022) |
| `wang2023voyager` | arXiv 2305.16291 | arXiv |
| `krishna2017densecap` | arXiv 1705.00754 | arXiv |
| `xu2026densemotion` | arXiv 2511.05369 | arXiv (comment: accepted to 3DV 2026) |
| `suzuki2025proprioception` | arXiv 2512.20876 | arXiv |
| `wulff2025jalm` | arXiv 2504.10055 | arXiv |
| `liang2025clam` | arXiv 2505.04999 | arXiv (journal_ref: IROS 2026) |
| `collins2025amplify` | arXiv 2506.14198 | arXiv |
| `liu2023libero` | arXiv 2306.03310 | arXiv |
| `yu2019metaworld` | arXiv 1910.10897 | arXiv (journal_ref: CoRL 2019) |
| `mees2022calvin` | arXiv 2112.03227 | arXiv (comment: IEEE RA-L) |
| `james2019rlbench` | arXiv 1909.12271 | arXiv |
| `mandlekar2018roboturk` | arXiv 1811.02790 | arXiv (comment: CoRL 2018) |
| `khazatsky2024droid` | arXiv 2403.12945 | arXiv |
| `christiano2017preferences` | arXiv 1706.03741 | arXiv |
| `brown2019trex` | arXiv 1904.06387 | arXiv (comment: ICML 2019) |

### Rejected as unverifiable — 0

Every reference listed in the appendix of `COC_REPORT/paper2_reverse_vla_concept.md`, including the three items that document itself flagged as surfaced by automated search, was located and verified. Nothing needs to be dropped.

- arXiv 2504.10055 — **verified.** Wulff, Maharjan, Chi, Cangelosi, "Joint Action Language Modelling for Transparent Policy Execution", 14 Apr 2025. → `wulff2025jalm`
- arXiv 2512.20876 — **verified.** Suzuki, Shimizu, Ogata, "Proprioception Enhances Vision Language Model in Generating Captions and Subtask Segmentations for Robot Task", 24 Dec 2025. Note the exact title differs from the concept doc's paraphrase ("Proprioception Enhances VLM in Generating Captions and Subtask Segmentations for Robot Task"); the bib carries the exact title. → `suzuki2025proprioception`
- arXiv 2511.05369 — **verified.** Xu, Liberatori, Varol, Rota, "Dense Motion Captioning", 7 Nov 2025, accepted to 3DV 2026. The concept doc's parenthetical "(DEMO / CompMo)" refers to the method and dataset names inside the paper, not to the title; do not put them in the citation. → `xu2026densemotion`

### Corrections to the concept document's appendix

These are errors in `paper2_reverse_vla_concept.md`, not in the bibliography, and the report must not propagate them.

1. "Krishna et al. **Dense-Captioning Events in Videos.** CVPR 2017" — the arXiv record does not confirm CVPR, and this paper is an ICCV 2017 publication. The bib cites it as an arXiv preprint (`krishna2017densecap`), which is correct under any reading. Do not write "CVPR 2017".
2. "Octo Model Team. **Octo.** RSS 2024" — correct; confirmed by dblp.
3. "Kim et al. **OpenVLA.** CoRL 2024" — correct; confirmed by dblp.
4. Meta-World is cited in the concept doc under "Benchmarks" with no reference; it is `yu2019metaworld` (CoRL 2019), not the 2021 journal version.
5. "CLAM, AMPLIFY" appear in the positioning table with no references; they are `liang2025clam` (arXiv 2505.04999) and `collins2025amplify` (arXiv 2506.14198).

### Caveat on one entry

`bain2000cloning` is the only entry whose **year** could not be pinned to a single authoritative record. The chapter "A Framework for Behavioural Cloning" by Bain and Sammut appears in *Machine Intelligence 15: Intelligent Agents* (Oxford University Press), pages 103–129; different indexes record the volume year as 1995, 1999 or 2000. The bib entry uses 2000 and carries a `note` field stating the ambiguity. Confirm against the Deakin library record before final submission, or cite behaviour cloning to `pomerleau1988alvinn` and `argall2009survey` alone and drop the entry.
