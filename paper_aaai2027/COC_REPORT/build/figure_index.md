# Figure index — CoC report (Suyog Khanal, s226137394)

Every asset below was opened and read visually. Axis labels, legends and box text are transcribed
from the rendered figure, not inferred. Aspect ratio is given as width:height from the PDF media box.

**Naming rule applied throughout:** the method is **DISEIL**. Two existing figures still render the
dead name and are listed as blocking defects. "Mode" is used only for failure modes; observation
modality (state / image) is never called a mode.

---

## 1. Teaser diagram

- **Path:** `/weka/s226137394/DmNfull/paper_aaai2027/figures/Teaser_Diagram.pdf`
- **Media box:** 720.88 x 549.24 pt. **Aspect (as stored):** 1.31:1 landscape.
  **Aspect of the upright content:** 0.76:1 portrait (the drawing sits sideways on the page).
- **Type:** raster scan of a hand drawing (embedded JPEG, 3010 x 2094 px, ~274 ppi). Produced by CamScanner.
- **What it shows** (reading the drawing upright, four panels, clockwise from top-left):
  1. *"policy keeps failing"* — a robot arm at a table with a cube, drawn mid-wobble with motion
     lines, a "?!" speech mark, and three faded ghost arms each struck through with an X: repeated
     failed attempts at the same manipulation.
  2. *"LLM prescribes demos"* — a brain-shaped character labelled LLM operating a printing press
     that emits a paper scroll: the language model writing out demonstration configurations.
  3. *"expert gives the demonstration"* — a human holding a tablet teleoperates the arm along a
     dashed arc to place the cube; an arrow leads down to *"new demo added"* and a sheet of paper.
  4. *"update the policy"* / *"policy improved"* — the sheet feeds back into a table with a small
     neural-network glyph inside a circular-arrow loop, and the arm now grasps the cube cleanly.
- **Recommended placement:** Aim 1 introduction, first page, full text width, immediately after the
  abstract. It is the visual statement of the loop before any notation is introduced.
- **Caption:** *"The demonstration-distillation loop. A policy trained on a small demonstration set
  fails repeatedly on the same configurations. A language model reads those failures and prescribes
  the configuration of the next demonstration to collect. The expert supplies one demonstration at
  that configuration, the demonstration is added to the training set, and the policy is retrained.
  Each round of the loop spends a single unit of the demonstration budget."*
- **Defects to fix:**
  - **(blocking for typesetting)** The drawing is rotated 90 degrees relative to the page and the
    page rotation flag is 0, so `\includegraphics` will place it sideways. Either bake in the
    rotation (`gs`/`qpdf` rewrite) or pass `angle=90` and re-check the bounding box. A baked-in fix
    is safer, because the CoC will be assembled more than once.
  - The scan carries a grey cast and a blue-white edge tint on one side, and is slightly skewed.
    De-skew and threshold to clean black-on-white, or redraw as vector art. At present it is the
    only raster figure in the report and it will look it.
  - PDF metadata reads `Title: CamScanner 06-07-2026 18.29`, `Author: CamScanner`. Strip it.
  - The drawing contains no DISEIL wordmark, so no naming fix is needed here.

---

## 2. Architecture diagram (Aim 1, updated)

- **Path:** `/weka/s226137394/DmNfull/paper_aaai2027/figures/Architectural Diagram.pdf`
- **Media box:** 769.92 x 402 pt. **Aspect:** 1.92:1. Vector text (searchable), with small embedded
  robot renders.
- **What it shows** — every box, transcribed:
  - **Initial set of Expert demos** (database glyph) -> **Train Policy** (network glyph) ->
    **Policy Rollout** *on N heldout episodes* (pink robot).
  - -> **Flag Uncertainty at t\*** (orange warning box). This is the query gate: it selects the
    timestep t\* at which the policy is least certain.
  - From t\*, two branches leave:
    - **Perception branch:** three rendered frames labelled **Start Frame**, **t\* Frame**,
      **End Frame** feed (dotted arrows) into **Vision LLM** -> **Reasoning LLM**
      *root-cause analysis*.
    - **Geometric branch:** a long lavender arrow labelled **"Geometric Descriptor at t\*"** runs
      across the top of the figure into a scatter panel showing three coloured point clouds
      (blue crosses, green circles, purple diamonds) -> **Cluster Engine** *k failure modes*.
  - **Cluster Memory** (database glyph) feeds the Cluster Engine by a dashed arrow.
  - **KAG Grounding** *geometry facts* feeds the Reasoning LLM by a dashed arrow.
  - Reasoning LLM and Cluster Engine both feed **Prescription LLM** *prescribe the demo
    configurations P*.
  - Prescription LLM -> **Policy Rollout on P** (second pink robot). From it, an orange dashed
    return arrow labelled **"Solvable => Revise P"** runs back into the Prescription LLM.
  - Policy Rollout on P -> (green arrow) **Expert Demo** -> **Add demo** (green) into the dataset
    database -> **Update Policy** *on the the augmented dataset* -> **next round** (teal arrow) back
    into Policy Rollout.
- **Consistency notes for the writing (important):**
  - The figure is **consistent with correction A10**: clustering is driven by the *geometric*
    descriptor at t\*, not by a visual embedding. The vision-and-language path feeds *root-cause
    analysis*, not the clustering step. Describe it that way; do not reintroduce the frozen-R3M/PCA
    branch, which does not appear in this figure.
  - The figure labels the clusters **"k failure modes"**, which matches the reserved sense of "mode".
  - **The figure draws only ONE of the two checks.** The policy-solvability loop
    ("Solvable => Revise P") is drawn explicitly. The Equation-10 feasibility check is **not** drawn
    as a loop: KAG Grounding appears only as a one-way dashed input into the Reasoning LLM, with no
    violation-feedback arrow and no revision arrow. The brief requires both checks to be presented as
    distinct mechanisms, and requires the figure to be described exactly as drawn. Those two
    requirements collide here. Recommended resolution: **add a second feedback arrow to the figure**
    (Prescription LLM -> feasibility check against KAG constraints -> "Infeasible => Revise P"), so
    that the drawing carries both loops and the prose can describe it as drawn. If the figure is
    frozen, the prose must say plainly that the KAG supplies the constraints used by the feasibility
    check and that only the solvability loop is depicted.
- **Recommended placement:** opening of the Aim-1 method chapter, full text width, referenced from
  the first paragraph of the framework description and again from the algorithm box.
- **Caption:** *"The DISEIL framework. A policy is trained on an initial expert demonstration set and
  rolled out on held-out episodes. A query gate flags the timestep t\* of greatest policy
  uncertainty. Two descriptions of that timestep are formed: a geometric descriptor, which the
  cluster engine partitions into k failure modes against a memory of past clusters, and a
  vision-and-language description of the start, t\* and end frames, which a reasoning model turns
  into a root-cause account grounded in geometry facts retrieved from the knowledge-augmented graph.
  A prescription model proposes the configuration P of the next demonstration. The proposal is
  screened before any expert time is spent: if the current policy can already solve P, the
  prescription carries no information and P is revised. The expert then demonstrates the surviving
  configuration, the demonstration is added to the dataset, and the policy is retrained for the next
  round."*
- **Defects to fix:**
  - **Typo, confirmed in the text layer:** the Update Policy box reads *"on the the augmented
    dataset"*. Delete the duplicated "the".
  - *"on N heldout episodes"* -> *"on N held-out episodes"*.
  - Missing feasibility-check loop (see the consistency note above).

---

## 3. Learning curves, all five tasks

- **Path:** `/weka/s226137394/DmNfull/paper_aaai2027/figures/all_5_task_comparison.pdf`
- **Media box:** 1792.33 x 334.84 pt. **Aspect:** 5.35:1 — extremely wide.
- **What it shows:** suptitle *"Success rate vs demonstrations added (mean +/- std)"*. Five panels
  side by side, each with x-axis *"Number of demonstrations added"* (0 to 20, ticks every 2.5) and
  y-axis *"Success rate"*. Mean lines with shaded standard-deviation bands. Panel titles, left to
  right, exactly as printed:
  1. **GridWorld 5x5 (image)** — y from ~0.45 to ~0.92; all methods rise together and finish bunched
     around 0.88-0.92.
  2. **Push-T (state)** — y from ~0.45 to ~0.97; the ours-line separates upward from about 5
     demonstrations and stays clear of the DAgger family, with DiffDAgger closest.
  3. **Lift (state)** — every method saturates at 1.0 by roughly 5 demonstrations; the panel is flat
     thereafter.
  4. **Door (state)** — y from ~0.57 to ~0.97.
  5. **Wipe (image)** — y from ~0.42 to ~0.96, with wide bands throughout.
  Shared legend beneath all panels, in order: SafeDAgger, DropoutDAgger, EnsembleDAgger,
  ThriftyDAgger, Stagger, **DISTIL (Ours)** [must become DISEIL], DiffDAgger.
- **Recommended placement:** first results figure of the Aim-1 results chapter, full text width. If
  the 5.35:1 ratio is too wide for the body text block, reflow as 2 rows x 3 panels rather than
  shrinking to illegibility.
- **Caption:** *"Success rate against the number of demonstrations added, for the five tasks. Lines
  are means over seeds and shaded bands are one standard deviation (nine seeds for GridWorld, five
  for the robot tasks). The DAgger-family baselines and Diff-DAgger are shown against DISEIL. Lift
  saturates at a perfect success rate within a few demonstrations for every method and therefore
  separates nothing."*
- **Defects to fix:**
  - **(blocking)** The legend reads **"DISTIL (Ours)"**. The name is dead. Regenerate the figure with
    **"DISEIL (Ours)"**.
  - **Setting mismatch to resolve in the prose.** The brief nominates GridWorld (image), Push-T
    (state) and **Door (image)** as the three primary settings, but this figure plots **Door
    (state)**. Either the figure is regenerated for the image modality of Door, or the results text
    states explicitly that the five-task curves are shown for the modality printed in each panel
    title and that the Door *image* setting is reported separately. Do not silently describe this
    panel as the image setting.
  - Lift must be flagged in the caption and the text as uninformative (no headroom, no variance).
    The caption above already does so.
  - Axis-label font is small relative to the panel width; increase it when regenerating.

---

## 4. Confidence against success-rate change

- **Path:** `/weka/s226137394/DmNfull/paper_aaai2027/figures/confidence_vs_success.pdf`
- **Media box:** 536 x 401 pt. **Aspect:** 1.34:1.
- **What it shows:** a scatter plot, orange filled circles. x-axis **"LLM Confidence Score"**, from
  50% to 100%, gridlines every 10%. y-axis **"Δ Success Rate"**, from -0.2 to about +0.45, with a
  dotted horizontal reference line at 0. An inset box top-left reports **r = 0.86** and **n = 152**.
  The cloud rises from left to right: below roughly 55% confidence the points sit at or under zero;
  above roughly 70% they are almost all positive and reach +0.4.
- **Recommended placement:** Aim-1 results, in the analysis of prescription quality, as a
  single-column figure beside the text.
- **Caption:** *"The confidence the language model reports in a prescription against the change in
  policy success rate that the prescribed demonstration produces (n = 152 prescriptions, Pearson
  r = 0.86). Delta success rate is the change in the policy's success rate on the round-level
  rollout evaluation, measured before and after the round. Prescriptions issued with low confidence
  return nothing, and several cost a little; prescriptions issued with high confidence return most
  of the gain."*
- **Notes and defects:**
  - This is the figure where **Delta-SR first appears** (the y-axis reads "Δ Success Rate"). Per the
    brief, define it at this first appearance: the change in the policy's success rate on the
    round-level rollout evaluation. The caption above carries the definition.
  - The caption must state which settings the 152 prescriptions are pooled from. The figure itself
    does not say, and the reader cannot recover it. Get this from the workbook.
  - Report r as a correlation, not as evidence of a causal mechanism.

---

## 5. Information-gain box plot

- **Path:** `/weka/s226137394/DmNfull/paper_aaai2027/figures/info_gain_boxplot.pdf`
- **Media box:** 679 x 350 pt. **Aspect:** 1.94:1.
- **What it shows:** six box plots, orange medians, black whiskers, black filled outlier points.
  y-axis reads **"pre-finetune policy loss on the chosen/prescribed demo"**, 0 to 14. x-axis
  categories, rotated, left to right: SafeDAgger, DropoutDAgger, EnsembleDAgger, ThriftyDAgger,
  Stagger, **DISTIL (Ours)** [must become DISEIL]. Medians read approximately: SafeDAgger 2.1,
  DropoutDAgger 2.25, EnsembleDAgger 1.0, ThriftyDAgger 1.15, Stagger 1.5, ours 2.7. The ours-box is
  the highest and has the longest upper tail, with outliers reaching about 13.4.
- **Recommended placement:** Aim-1 results, immediately after the information-gain argument, as a
  single-column or 2/3-width figure.
- **Caption:** *"Information gain of the acquired demonstration, measured as the policy's per-step
  loss on that demonstration before the policy is retrained on it. A high pre-retrain loss means the
  demonstration lies in a region the policy has not learned. Suboptimal or invalid demonstrations are
  excluded by construction, because every prescription passes the feasibility check and every
  demonstration comes from the expert, so a high loss identifies genuinely novel data. DISEIL selects
  demonstrations with a higher pre-retrain loss than the DAgger-family baselines."*
- **Defects to fix:**
  - **(blocking)** x-axis category reads **"DISTIL (Ours)"**. Regenerate with **DISEIL**.
  - **Diff-DAgger is absent** while Stagger is present. Stagger is the GridWorld-only baseline and
    Diff-DAgger is the robot-only baseline, so this box plot appears to be GridWorld data. The figure
    never says. State the setting in the caption, and if the intent was to pool across settings then
    the missing Diff-DAgger column is a genuine omission that must be added.
  - The y-axis label uses the slash construction "chosen/prescribed". Settle on one term for one
    concept, per the style rule. Suggest "pre-retrain policy loss on the acquired demonstration",
    which also matches the wording of the information-gain claim.

---

## 6. Failure-mode clusters on Push-T

- **Path:** `/weka/s226137394/DmNfull/paper_aaai2027/figures/clustering_modes_pushT.pdf`
- **Media box:** 648 x 648 pt. **Aspect:** 1:1 square.
- **What it shows:** a 3 x 3 grid of rendered Push-T frames (white arm, red T-block, grey T-shaped
  goal outline, wooden table). Rows are the three discovered failure modes, labelled down the left
  edge in the row colour:
  - **M0, not-well-aligned** (blue) — the block is at the goal but rotationally wrong; per-panel
    annotations read *θerr 162°, contact 0.07m*; *θerr 162°, contact 0.06m*; *θerr 169°, contact 0.08m*.
  - **M1, no-contact** (orange) — the arm is away from the block, which has not been moved to the
    goal; annotations *θerr 104°, contact 0.11m*; *θerr 120°, contact 0.09m*; *θerr 102°, contact 0.08m*.
  - **M2, badly-rotated** (green) — the block is pushed but left at a large orientation error and far
    from contact; annotations *θerr 84°, contact 0.15m*; *θerr 37°, contact 0.17m*; *θerr 69°, contact 0.18m*.
  Each row therefore shows three sampled members of one cluster, with the two geometric quantities
  that separate the clusters printed above each frame.
- **Recommended placement:** Aim-1 method chapter, alongside the partition step, or as the first
  qualitative figure of the results. Square aspect suits a single column.
- **Caption:** *"The three failure modes discovered on Push-T by clustering the geometric descriptor
  at the flagged timestep. Each row shows three rollouts assigned to one mode, annotated with the
  block's orientation error and the distance between the end-effector and the block. The partition
  recovers behaviourally distinct failures: the block is delivered to the goal but rotationally wrong
  (M0), the arm never makes contact (M1), and the block is moved but left badly rotated and abandoned
  (M2). The clusters are found from geometry alone, without a visual embedding."*
- **Notes and defects:**
  - No naming defect: the figure carries no wordmark.
  - The row labels use M0/M1/M2 for failure modes, which is the reserved and correct sense of "mode".
    Keep the prose aligned; never write "mode" for state-or-image modality anywhere near this figure.
  - The mode names are hyphenated adjectives ("not-well-aligned"). In prose, spell them out.
  - Panel annotation text is small; enlarge if the figure is placed at single-column width.
  - This figure is direct support for correction A10 (geometric clustering, image modality included).
    Cite it there.

---

## 7. Aim-2 architecture (extracted, new asset)

- **Source:** `/weka/s226137394/DmNfull/paper_aaai2027/COC_REPORT/paper2_preview.pdf` (the single page
  *is* the figure).
- **Extracted to:**
  - `/weka/s226137394/DmNfull/paper_aaai2027/COC_REPORT/figures_generated/aim2_architecture.pdf`
    (vector, cropped to a tight bounding box with a 6 pt margin, 814 x 373 pt, **aspect 2.18:1**)
  - `/weka/s226137394/DmNfull/paper_aaai2027/COC_REPORT/figures_generated/aim2_architecture.png`
    (300 dpi raster, for any non-LaTeX use)
  Text remained vector through the crop, and the render was checked against the original.
- **Caption (exactly as required):** *"Proposed architecture for Aim 2"*
- **What it shows** — every major component, transcribed:
  - **IMITATION LOOP** (the part inherited from Aim 1, drawn along the top left):
    **Seed Expert Demos** *initial dataset* -> **Train Policy** *BC / diffusion* -> **Policy Rollout**
    *held-out* -> **Flag Uncertainty t\*** *diffusion-loss* (orange box, the same query gate as Aim 1).
  - **TRAJECTORY (V + A)** — two input boxes at the top right: **Frames (V)** *start / t\* / end* and
    **Actions (A)** *executed seq*. Both feed by dashed arrows into the new block.
  - **SINGLE LANGUAGE-GROUNDED SELECTOR** — a dashed violet enclosure holding four stacked components,
    and this is the Aim-2 contribution. Inside, top to bottom:
    - **REVERSE VLA** *V + A -> Language*. A vision-language-action model runs backwards: instead of
      mapping vision and language to actions, it maps the executed frames and actions back into
      language.
    - **Captions** *trajectory / action / failure* — the language description the reverse model emits.
    - **Language Skill Memory** *what's been taught / coverage* (cylinder/database glyph) — a store,
      in language, of what the policy has already been taught, from which coverage is computed.
    - **Unified LLM** *coverage-gap reason + prescribe* — a single model that both reasons about the
      coverage gap and issues the prescription.
    Internal arrows: REVERSE VLA -> Captions -> Language Skill Memory -> (edge labelled **coverage**)
    -> Unified LLM. An orange dashed arrow labelled **failure query** leaves Captions, runs down the
    right-hand side outside the enclosure, and re-enters the Unified LLM.
  - **PRESCRIBE + LEARN** (bottom row, right to left): Unified LLM -> (edge labelled **prescribe**) ->
    **Expert Demo** *prescribed config* -> **Add ONE demo** *augment dataset* -> a long teal edge
    labelled **next round** that returns to Train Policy, closing the loop.
- **How it extends Aim 1 (for the report's Aim-2 chapter):** Aim 1 uses a pipeline of specialised
  components at the flagged timestep — a vision model, a separate reasoning model, a geometric cluster
  engine with its cluster memory, a knowledge-augmented graph, and a prescription model. Aim 2
  collapses that pipeline into a **single language-grounded selector**. The geometric descriptor and
  the cluster engine are replaced by language: a reverse vision-language-action model captions what
  the policy actually did, those captions accumulate in a language skill memory, and coverage is
  measured in that language space rather than in a hand-designed 6-D descriptor space. One unified
  model then reasons about the coverage gap and prescribes. The outer loop is unchanged, which is the
  point: the same query gate, the same one-demonstration-per-round budget, the same retraining step.
  Aim 2 swaps the selector, not the protocol, and that is what makes it a clean successor to Aim 1.
- **Defects to fix:**
  - The enclosure title **"SINGLE LANGUAGE-GROUNDED SELECTOR"** is overrun by the dashed arrow that
    enters from Frames (V); the text layer extracts as `LANGUAGEIGROUNDED`. Nudge the label or route
    the arrow.
  - The figure is a preview drawn for the second paper. Before it goes into the CoC, confirm no dead
    method name appears anywhere in it (none does at present) and confirm the caption is set to the
    exact required string.

---

## 8. A2I2 logo (cover page)

- **Path:** `/weka/s226137394/DmNfull/paper_aaai2027/COC_REPORT/A2I2_Logo_Stacked_2025_Keyline.png`
- **What it shows:** the Deakin keyline lockup, black on transparent/white. Left: the Deakin shield
  in a circular keyline, with the wordmark **DEAKIN UNIVERSITY**. A vertical rule. Right, stacked over
  three lines: **DEAKIN / APPLIED ARTIFICIAL / INTELLIGENCE INITIATIVE**.
- **Aspect:** wide horizontal lockup, roughly 5:1.
- **Recommended placement:** cover page of the CoC report, centred above or below the title block. Do
  not stretch it; scale proportionally and leave clear space around the keyline.
- **Caption:** none. A cover mark takes no caption or figure number.

---

## 9. Compulsory training status

- **Path:** `/weka/s226137394/DmNfull/paper_aaai2027/COC_REPORT/Compulsory Training Status.png`
- **What it shows:** a screenshot of the candidate's compulsory-training panel. Heading *"Compulsory
  Training"*. Three rows, each with a green completion bar, and a completion date:
  | Item | Date shown |
  |---|---|
  | Research Integrity Training | 02-DEC-25 |
  | Research Induction | 10-DEC-25 |
  | HDR Respectful Behaviour | 02-DEC-25 |
- **Aspect:** roughly 6:1, a wide strip.
- **Recommended placement:** the HDR-training section of the CoC, as a small figure, or transcribed
  into a table in the body with the screenshot moved to the appendix. A transcribed table reads
  better in a formal report than a screenshot; the screenshot is the evidence.
- **Caption:** *"Status of the compulsory higher-degree-research training, as recorded in the
  candidate's training record."*
- **Defects to note:** it is a browser screenshot, complete with a "Top" hyperlink at the right edge.
  Crop that away. Resolution is low; if it must appear in the body, re-take it at a higher zoom.

---

## 10. Certificate — Research Integrity

- **Path:** `/weka/s226137394/DmNfull/paper_aaai2027/COC_REPORT/Research_Integrity_Deakin_Safety_and_Research_Integrity_Training_KHANAL.pdf`
- **Format:** 1 page, A4 (595.28 x 841.89 pt), portrait, aspect 0.71:1.
- **What it certifies:** a Deakin University *Certificate of completion*, presented to **SUYOG
  KHANAL**, for the successful completion of **Research Integrity**, on **Monday, 1 December 2025**.
  Footer carries `deakin.edu.au` and the CRICOS provider code 00113B.
- **Recommended placement:** appendix, HDR-training evidence, one certificate per page or two to a page.
- **Caption:** *"Certificate of completion, Research Integrity, 1 December 2025."*

## 11. Certificate — Respect at Deakin (HDR module)

- **Path:** `/weka/s226137394/DmNfull/paper_aaai2027/COC_REPORT/Certificate_of_Completion_-_Respect_at_Deakin_HDR_Respect_at_Deakin_-_Graduate_Research_and_Supervision_Module_KHANAL.pdf`
- **Format:** 1 page, A4, portrait, aspect 0.71:1.
- **What it certifies:** a Deakin University *Certificate of completion*, presented to **SUYOG
  KHANAL**, for the successful completion of the **Respect at Deakin - HDR Module** (the graduate
  research and supervision module), on **Monday, 1 December 2025**.
- **Recommended placement:** appendix, HDR-training evidence.
- **Caption:** *"Certificate of completion, Respect at Deakin higher-degree-research module,
  1 December 2025."*

## 12. SSC900 Academic Writing result

- **Path:** `/weka/s226137394/DmNfull/paper_aaai2027/COC_REPORT/SSC900 Academic Writing Result.pdf`
- **Format:** 1 page, portrait.
- **What it certifies:** this is **not a certificate**. It is a Deakin University **Statement of
  Results**, carrying an explicit disclaimer that it is unofficial. It records:
  - Student Name **SUYOG KHANAL**, Student ID **226137394**, Course **F975 Doctor of Philosophy**.
  - FAR972 PH D RESEARCH (2 credit load), 2025/HDR-Q4, grade **CE**
  - FAR972 PH D RESEARCH (2 credit load), 2026/HDR-Q1, grade **CE**
  - FAR972 PH D RESEARCH (2 credit load), 2026/HDR-Q2, grade **CE**
  - **SSC900 ACADEMIC WRITING AND COMMUNICATION**, 2026/TRI-1, grade **UP**
  - Generated on 13/07/2026.
- **Recommended placement:** appendix, HDR-training evidence, next to the two certificates.
- **Caption:** *"Statement of results recording completion of SSC900 Academic Writing and
  Communication in trimester 1, 2026."*
- **Defects to note:**
  - Expand the grade codes in the body text at first use rather than leaving **CE** and **UP** bare;
    give the ungraded-pass sense of UP for SSC900 and the continuing-enrolment sense of CE for the
    research units, and cite the university's results key, which the document itself links.
  - The document is marked unofficial. Say so, or obtain the academic transcript if the CoC panel
    expects formal evidence.

---

## Cross-cutting defects, ranked

1. **Dead method name still rendered in two figures.** `all_5_task_comparison.pdf` (legend) and
   `info_gain_boxplot.pdf` (x-axis category) both print **DISTIL (Ours)**. These must be regenerated
   as **DISEIL (Ours)** before the CoC is compiled. This is the single highest-priority fix, because
   the naming rule is absolute and these are two of the report's headline figures.
2. **Architecture typo:** *"on the the augmented dataset"* in the Update Policy box. Also
   *"heldout"* -> *"held-out"*.
3. **Architecture draws only the solvability loop.** The Equation-10 feasibility check has no
   feedback-and-revision arrow; the knowledge-augmented graph appears only as a one-way input. Add the
   second loop, or describe the figure exactly as drawn and carry the feasibility mechanism in prose.
4. **Teaser is sideways** in the PDF (page rotation flag 0), and is an uncleaned phone scan with
   CamScanner metadata.
5. **Door modality mismatch.** The five-task curves show Door (state); the nominated primary setting
   is Door (image). Resolve in the figure or state it plainly in the text.
6. **Two figures do not declare the settings they aggregate over:** `confidence_vs_success.pdf`
   (n = 152 prescriptions, pooled from what?) and `info_gain_boxplot.pdf` (Stagger present,
   Diff-DAgger absent, which suggests GridWorld). Recover both from the workbook and put them in the
   captions.
7. **Training-record date discrepancy.** The certificates state 1 December 2025; the training-status
   screenshot records 02-DEC-25 for the same two items. Quote one source consistently and note the
   record date if both are shown.
8. **No certificate supplied for Research Induction** (10-DEC-25 in the status screenshot). If the
   panel expects a certificate per item, obtain it.
