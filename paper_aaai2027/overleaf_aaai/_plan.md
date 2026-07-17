# DISEIL AAAI-2027 — the plan (Agent 1, principal research scientist)

Binding: `PAPER_SPEC.md`. Truth: `COC_REPORT/build/v2/*.md`. Method name **DISEIL**, always.
7.0 content pages, references separate, supplementary a separate PDF. The only legal way to fit is to
write less. No layout manipulation of any kind.

---

## 1. THE STORY IN EIGHT BEATS

This is the spine. Every section serves it. Any sentence that does not advance a beat is cut.

1. **Problem.** A demonstration is the one input whose cost does not fall when compute is bought or a
   simulator is downloaded, so under a fixed allowance the only remaining lever on policy performance
   is what each demonstration in that allowance contains.
2. **Gap.** The query-efficient members of the interactive imitation learning family all reduce to one
   template, compute a scalar at a visited state and hand over at the first threshold crossing, which
   answers *when* to call the expert and leaves *which* of a round's failures the expert's time is
   spent on and *where* the corrective demonstration begins fixed by default.
3. **Insight.** A round's failures are not independent events but a small set with structure, and that
   structure is recoverable from geometry: a low-dimensional descriptor of the configuration at the
   step where the policy first becomes unreliable partitions the failures into behaviourally distinct
   modes, which a scalar attached to one state cannot represent.
4. **Method.** DISEIL partitions each round's failures into modes over a six-dimensional geometric
   descriptor, targets the near-dominant mode of highest mean peak loss, and prescribes the
   configuration of a demonstration that does not exist yet, verified against an explicit model of
   what the environment permits before any expert time is spent.
5. **Experiments.** Five tasks under two observation modalities at a budget of twenty demonstrations,
   against the five published query gates and a uniform-random allocation control, with the round-zero
   success rate held near the band in which a fixed budget can be spent well or badly.
6. **Evidence.** DISEIL holds the best mean in all ten settings (collapsed task-level paired
   $t(4) = 4.10$, $p = 0.015$ two-sided), acquires higher-information demonstrations than Diff-DAgger
   in all eight settings where Diff-DAgger runs even though Diff-DAgger gates on the very loss the
   metric is computed from, and removing the partition costs 4.37 points while per-demonstration
   information gain does not fall, which locates the advantage in allocation and not in
   per-demonstration informativeness.
7. **Limitations.** The selector reasons about one round and holds no representation of what the
   training set already contains, the descriptor is designed by hand and recovers cause only where
   configuration determines cause, and every expert here is a simulator oracle that answers instantly,
   correctly and at the same price for any request.
8. **Conclusion.** Deciding which failure mode a demonstration is spent on and where it begins is
   worth more than deciding when to ask, and it is worth most exactly when the budget is smallest.

**The hand-off chain** (each section closes into the next): budget is the only lever -> the gates do
not pull it -> failures have recoverable structure -> here is the machinery that uses it -> here is
how we tested it -> it wins, and here is the honest size of the win -> here is what buys the win ->
here is what it does not yet do.

---

## 2. THE CONTRIBUTION, AS A REVIEWER WOULD WANT IT

**What is genuinely new.** The pairing of a partition of a round's failures into modes with a
prescription that is verified for feasibility against an explicit model of the environment *before an
expert is asked to satisfy it*. The unit of novelty is the **object being verified**: a request for a
training demonstration that has not been collected, rather than a plan to be executed. This is what
makes DISEIL the first framework to compute all three parts of the acquisition rule
$A = (t^\star, C_{\mathrm{tgt}}, \xi)$ instead of the first part only.

**Why the obvious neighbours are not this.** Selection literatures (active learning, core-set,
sub-trajectory retrieval) rank items in a pool that already exists; DISEIL prescribes an item that
does not exist, so those methods do not answer the question and are not baselines. Query gates decide
handover timing from a per-state scalar. Propose-verify-revise with an external checker is a standard
pattern; what is new is not the checker.

**What we do NOT claim** (write these down so no drafting agent drifts into them):
- Not "we used an LLM". The ablations say the opposite and the paper says so in the main text: the
  prescription model is worth 1.33 points and the vision-language model 1.33 points, each comparable
  to the seed standard error of the corresponding full run. The advantage is in the allocation step.
- Not the cluster memory. It is a configurable, task-specific component (A1, 0.73 points, the least
  damaging of seven). Named once as configurable, never sold, and absent from the architecture figure.
- Not the feasibility check as a mechanism.
- Not the per-step diffusion loss. That is Diff-DAgger's idea, used here for localisation and also
  compared against as a baseline. Both facts stated plainly wherever the signal appears.
- Not ten independent experiments. Two modalities of a task share expert, reward and reset
  distribution, so the effective sample size is nearer five than ten. Lead with the collapsed test.
- Not "DISEIL at $B{=}10$ matches the best baseline at $B{=}20$". Retracted in the CoC. It must not
  reappear.
- Not Eq. 11 (solvability). A design element, not exercised in any reported run. Flagged as such.

**The three contribution bullets that end the intro** (AAAI norm, ~0.25 pp):
1. The three-part acquisition rule, and the observation that the DAgger family fixes two of its three
   parts by default at no cost under an unbounded budget and at full cost under a budget of twenty.
2. DISEIL: a geometric partition of a round's failures paired with a feasibility-verified prescription
   of a demonstration that does not yet exist.
3. Evidence that **allocation, not per-demonstration informativeness, is the mechanism**, from a
   dissociation no other work supplies: remove the partition and information gain is unchanged
   (mean $+0.02$) while success falls 4.37 points.

Bullet 3 is the paper. It is what converts Table 8 from a decoration into evidence.

---

## 3. SECTION PLAN (hard budget, sums to 7.00)

| # | section | pp | purpose / what it argues | carries |
|---|---|---|---|---|
| 1 | Title, abstract, introduction | **1.15** | Beats 1-3 plus the headline. Abstract: problem, gap, method, one headline result, **no ablations**, "novel" at most once. Intro: the budget is the only lever; the gates answer one of three questions; failures have structure; 3 bullets. | no figure (see §4) |
| 2 | Related work | **0.55** | Three run-in blocks: interactive IL and the DAgger family; demonstration selection and coverage; LLM/VLM guidance in robot learning. Each states what prior work does, the **condition** it assumes, one contrast clause. Carries the *dataset distillation* name clash. From `02_background.md`. | — |
| 3 | Problem formulation | **0.50** | Budgeted interactive IL, the BC objective, the aggregation loop, and Eq. 4, the framing equation. Argues that the allocation decision exists and is currently taken by default. | Eqs. 1-4 |
| 4 | **Method: DISEIL** | **1.85** | **The largest block; reviewers pay for the idea.** Perceive / Partition / Prioritise / Prescribe as run-in headings. Argues why each stage is necessary, not merely present: first crossing not peak (the peak inherits a ruined state); sine/cosine yaw; the size constraint in Eq. 8; feasibility before expert time. Cluster memory named once as configurable and task-specific. | **Fig. 1** architecture (`figure*`, 0.35); Eqs. 5-11; **Algorithm 1** (0.25) |
| 5 | Experimental setup | **0.50** | 5 tasks x 2 modalities, $B{=}20$, $D{=}1$, seeds (9 GridWorld / 5 robot), experts, policies, baselines. States **once**: SE $=$ std$/\sqrt{n}$; the $N_i$ band rationale; the ten settings are not ten independent experiments; Stagger is ours and not a DAgger method. | — |
| 6 | Main results | **1.35** | Beat 6, part one. Prose under **finding-stating** bold run-ins. Collapsed task-level test is the claim of record. Lift at ceiling flagged **here**, in the body, not a footnote. | **Table 1** (= CoC Table 7, `table*`, 0.40); **Fig. 2** learning curves (`figure*`, 0.15) |
| 7 | Analysis and ablations | **0.75** | Beat 6, part two: **the dissociation**. Information gain, then the knockouts. A4/A5 small and A3-below-baseline stated in the same plain register as the wins. Pointer to supplementary for A1..A18. | **Table 2** (= CoC Table 8, 0.18); **Fig. 3** knockout summary (0.25) |
| 8 | Limitations and conclusion | **0.35** | Beats 7-8. Limitations in body voice, not a defensive list. Conclusion: one paragraph, **no numbers**, at most one forward-looking clause. No Aim 2/Aim 3. | — |
| | **total** | **7.00** | | |
| | references | (2.00) | separate allowance | |

Check: 1.15 + 0.55 + 0.50 + 1.85 + 0.50 + 1.35 + 0.75 + 0.35 = **7.00**.
Artifact subtotal: 0.35 + 0.25 + 0.40 + 0.15 + 0.18 + 0.25 = **1.58 pp**, leaving 5.42 pp of prose.
Method at 1.85 is the largest block, matching the exemplar invariant (1.3-3.0 pp).

**Overflow order** (cut in this order; never adjust layout):
1. Algorithm 1 -> supplementary (-0.25). The four run-in headings already carry the loop.
2. Ablation paragraph -> P1's four-line form, name the dimensions and defer (-0.30).
3. Related work -> P1's late-and-short form, moved to just before the conclusion (-0.15).
4. Table 2 -> replaced by one sentence with the eight-of-eight sweep (-0.18).
5. Learning curves -> supplementary (-0.15).
Never cut: Fig. 1, Table 1, Fig. 3, the A3 dissociation sentence.

---

## 4. FIGURE AND TABLE SELECTION FOR THE MAIN PAPER

Five artifacts. Each justified by the scientific question it answers.

| artifact | asset | pp | the question it answers |
|---|---|---|---|
| **Fig. 1** architecture | `architecture.pdf` (769x402 pt, 1.92:1, `figure*` at top of the method page) | 0.35 | *What are the moving parts and in what order do they run?* The method is a four-stage loop; prose alone would cost more than 0.35 pp to make the order legible. Cluster memory already removed from the asset, consistent with §2. |
| **Table 1** | CoC Table 7 verbatim (`table*`) | 0.40 | *Does it win, where, and by how much?* 10 settings x 7 arms + $N_i$ + Init SR in one dense table. One dense table beats three thin ones. Bold = best, declared in the caption. |
| **Fig. 2** learning curves | `learning_curves.pdf` (1792x335 pt, 5.35:1) | 0.15 | *Is the margin an endpoint artefact or does it hold over the budget?* Five panels for 0.15 pp is the best value in the paper: the aspect ratio makes it nearly free. Shows Push-T separating from about the fifth demonstration and GridWorld finishing bunched, which is why GridWorld (image) is among the smallest margins. |
| **Table 2** | CoC Table 8 verbatim | 0.18 | *Are DISEIL's demonstrations individually more informative?* Chosen over `info_gain.pdf` because the exact per-setting numbers support the eight-of-eight sweep row by row, and Diff-DAgger is the one baseline that gates on the very quantity reported. |
| **Fig. 3** knockout summary | `knockout_summary.pdf` (= `F3_knockout_summary.pdf`) | 0.25 | *Which component buys the result?* **42 numbers in a single column**: 7 knockouts x 3 settings x (delta, margin retained). Dark cells mark margin retained $< 0$, so the honest finding that A3, A8 and A6 fall beneath their own baseline is legible at a glance rather than buried in prose. |

**Rejected, with reasons:**

- **`teaser.png` -> dropped entirely, not even supplementary.** Inspected. It is a hand-drawn
  four-panel *pipeline* (policy fails -> LLM prescribes -> expert demonstrates -> policy improves) in
  portrait orientation. It fails the page-1 test from the style analysis ("a page-1 figure must earn
  its 0.3 pp by carrying the **contrast**, not the architecture"), it duplicates Fig. 1's job, and its
  centre panel reads "LLM prescribes demos", which is exactly the claim §2 forbids. Three of the four
  exemplars have no page-1 figure. Buying 0.30 pp of prose is strictly better.
- **`allocation_ladder.pdf` (F1) -> supplementary.** Its A3 and A8 bars are already in Fig. 3, and it
  carries only 12 numbers to Fig. 3's 42. Its unique content is the A2 arm, which is one prose
  sentence (89.1 / 82.3 / 80.0 against DISEIL's 91.3 / 96.1 / 88.6). Keeping both would put A3 in two
  artifacts, violating "no result appears twice".
- **`gain_without_allocation.pdf` (F2) -> supplementary.** Strictly a subset of F1, which is itself a
  subset of Fig. 3 on the overlapping arms.
- **`confidence.pdf` -> supplementary.** $r = 0.82$ is an *observed property, not a mechanism*: no
  experiment gates on it and it is measured on DISEIL runs only. It does not advance a beat, and first
  on the style analysis's own cut list.
- **`info_gain.pdf` -> supplementary.** Displaced by Table 2; a result may not appear in both forms.
- **`failure_modes.pdf` -> supplementary.** Illustrative of the partition, not evidence for it. Fig. 3
  and Table 2 carry the evidence.

---

## 5. THE MAIN / SUPPLEMENTARY SPLIT

**Stays in the main paper** (the minimum that convinces a reviewer):

| item | why it cannot move |
|---|---|
| Eq. 4, the three-part acquisition rule | the framing equation; the gap is unstatable without it |
| Eqs. 1-3, 5-11 | Eq. 5 (first crossing, not peak), Eq. 6 (descriptor), Eq. 7 (partition), Eq. 8 (size constraint), Eq. 10 (feasibility) are the method. Eq. 11 stays as one sentence **with its honest flag** |
| Algorithm 1 | verbatim from the CoC; first on the overflow list, so it is the release valve |
| Table 1 (CoC Table 7) | the result |
| Fig. 2 learning curves | the result is not an endpoint artefact |
| Table 2 (CoC Table 8) | the first half of the dissociation |
| Fig. 3 knockouts | the second half, and the compact summary the spec requires |
| A3 + info-gain dissociation, in prose | **the argument of the paper** |
| A2 and A11, one sentence each | the two objections a reviewer raises unprompted |
| Lift-at-ceiling; A4/A5 are small; the ten-settings caveat; Wipe (image) has not plateaued; Eq. 11 unexercised; the loss is Diff-DAgger's | honesty the CoC already established and the spec forbids softening |

**Moves to `supplementary.pdf`** (strengthens the paper, never required to understand the method):

| item | why it moves |
|---|---|
| A9, A10, A12, A13 (design choices) | they justify settings, not the claim. A10 (six dimensions, inverted U) and A12 (silhouette beats every fixed $k$ by 4.1) are the strongest and get the first supplementary section |
| A14-A18 (diagnostics) | they measure properties of the running system, not the contribution. A16's bridging discrepancy and A15's non-claim are carried verbatim |
| F1, F2, F4-F13, `confidence.pdf`, `info_gain.pdf`, `failure_modes.pdf` | superseded, or displaced by a denser artifact |
| per-task descriptor table (CoC Table 4) | reference material |
| prompts, KAG examples, taxonomies | reproducibility, not argument |
| hyperparameters, $N_i$ sweep, retrain cadence detail | reproducibility |
| A18 cost tables | one limitation sentence in the main text carries the range; the tables do not fit |
| per-setting ablation numbers, failure cases, qualitative examples | volume |
| the 168-184 records-per-cell range | **outstanding item in the CoC; do not quote it in the main text and do not resolve it by assumption** |

---

## 6. SUPPLEMENTARY OUTLINE (`supplementary.tex`, separate PDF)

- **A. Implementation.** Policies per modality, R3M on the image branch, retraining cadence, the $N_i$
  data-scaling sweep and the starting-competence band, seeds and round accounting (180 per GridWorld
  setting, 100 per robot setting).
- **B. Task and descriptor detail.** The five tasks, the experts, and CoC Table 4 (the six components
  of $\phi$ per task). `failure_modes.pdf`.
- **C. Baselines.** The five gates as instantiations of the template; Stagger defined as a control
  implemented here, with no citation and never labelled a DAgger method.
- **D. Design-choice ablations.** A9 (context set), A10 (descriptor width, F7), A12 (silhouette vs
  fixed $k$), A13 (citation count, **with the note that the single-citation arm is confounded, not
  null**). Figures F7, F11.
- **E. Full knockout results.** A1-A8 per setting, F1 `allocation_ladder`, F2, F4, F5, F6.
- **F. Diagnostics.** A14 (purity, **scored against the reasoning model's own labels**), A15 (F12,
  cluster-count distribution), A16 (**the GridWorld bridging discrepancy, reported as active**), A17
  (F13, failures over budget), A18 (cost; **P1 and P5 rows must not be read against one another and
  token counts are not comparable across rows**).
- **G. Confidence.** `confidence.pdf`, the ten per-setting correlations, and the two limits.
- **H. Prompts, taxonomies and KAG examples.**
- **I. Extended limitations and the outstanding items.**

---

## 7. THE ABLATIONS THAT EARN MAIN-PAPER SPACE

Three, in one figure and two paragraphs (0.42 pp inside §7's 0.75).

1. **A3, the partition knockout, with the information-gain dissociation. Non-negotiable.**
   Largest damage of any knockout ($-2.2 / -4.1 / -6.8$, mean $-4.37$ points; margin retained
   $-53.2$ %), falling beneath its own best baseline on Push-T (92.0 vs 94.1) and Door (81.8 vs 84.2).
   Meanwhile information gain **does not fall**: $+0.02 / +0.16 / -0.13$, mean $+0.02$.
   *Why this and not another:* it is the only result that separates the two hypotheses a reviewer holds
   after Table 2 (are the demonstrations better, or is the budget better spread?), and it answers
   against the reading that flatters us. Greedy worst-loss selection collects demonstrations that are
   individually informative and jointly redundant. Nothing else in the paper can do this work.
   *Stated with it:* A3 knocks out the allocation **stack**, not the partition in isolation; the
   cleaner variant (descriptor kept, random partition into $k$ groups) was not run and is future work.

2. **A2, the uniform-random allocation control.** One sentence: 89.1 / 82.3 / 80.0 against DISEIL's
   91.3 / 96.1 / 88.6, below the strongest gated baseline on both robot settings.
   *Why:* it settles the "any failure replay would do" objection, which a reviewer raises unprompted
   and which would otherwise sink the paper. It costs one sentence because Fig. 3 already carries A8,
   the companion objection ("your fallback heuristic is doing the work", $-3.27$ points).

3. **A11, the budget sweep.** One sentence, no figure: margin $+9.07$ at $B{=}10$, $+2.87$ at $B{=}20$,
   $+2.83$ at $B{=}40$; DISEIL's own rate rises with budget and the margin shrinks because the baseline
   catches up from a lower start.
   *Why:* it is the sample-efficiency claim in the title, and it is the beat-8 sentence: allocation is
   worth most when the budget is smallest. **Includes the retraction**: DISEIL at $B{=}10$ does not
   match the strongest baseline at $B{=}20$.

**The compact summary, carried entirely by Fig. 3, one clause each and no discussion:** A6 ($-2.37$,
fallback rate 27-35 % of rounds), A4 and A5 ($-1.33$ each), A7 ($-1.27$), A1 ($-0.73$).
This ordering is itself the honest statement that **the language components are not the source of the
advantage**, and stating it in the main paper is a strength, not a concession. It also sets up the
practical reading: a deployment that cannot afford the reasoning stack can delete it, keep the
geometric clustering and the deterministic heuristic, and still beat every baseline, at about 1.33
points.

**Why not the others:** A9, A10, A12, A13 justify settings rather than the claim (a reviewer who
doubts six dimensions is not a reviewer who doubts the contribution); A14-A18 measure properties of
the running system; A1 is priced at 0.73 points and §2 declines to sell the component it tests. All
appear in full in the supplementary.
