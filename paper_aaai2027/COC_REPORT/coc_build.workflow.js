export const meta = {
  name: 'coc-report-build',
  description: 'Build the Confirmation of Candidature report: repo map, sample-CoC structure, ablation analysis, D5 SLURM compute, A13 statistics, figure generation, then narrative, sectioned writing, multi-critic QA loop and final validation',
  phases: [
    { title: 'Recon' },
    { title: 'Assets' },
    { title: 'Narrative' },
    { title: 'Write' },
    { title: 'QA' },
    { title: 'Validate' },
  ],
}

const ROOT   = '/weka/s226137394/DmNfull'
const COC    = ROOT + '/paper_aaai2027/COC_REPORT'
const BUILD  = COC + '/build'
const SEC    = BUILD + '/sections'
const GFIG   = COC + '/figures_generated'
const XLSX   = COC + '/ablations_results/DISTIL_ablation_results.xlsx'
const FIGS   = ROOT + '/paper_aaai2027/figures'
const DOCS   = ROOT + '/clean_working_with ablations'
const REPORT = COC + '/CoC_Report.md'
const PAPER  = ROOT + '/paper_aaai2027/draft/paper.tex'

// ---------------------------------------------------------------------------
const FACTS = `AUTHORITATIVE PROJECT FACTS — obey exactly; never contradict; never invent numbers.

IDENTITY / NAMING (hard):
 - The method is now **DISEIL**. The name DISTIL is DEAD: it must appear NOWHERE in the report (nor PACE, nor P4). Internal code names (p4_top3_rotate, p4_subtask) are code identifiers only and must never appear in prose.
 - Aim-1 paper title: "Demonstration Distillation for Sample-Efficient Imitation Learning".
 - The acronym must be shown to originate from the title by taking ONE letter from each word of the title, in order:
     **D**emonstration  d**I**stillation  for  **S**ample-**E**fficient  **I**mitation  **L**earning   ->   D-I-S-E-I-L
   That is: D from "Demonstration", I from "dIstillation" (its second letter), S from "Sample", E from "Efficient", I from "Imitation", L from "Learning".
   In the abstract, at first mention, bold exactly those six letters and nothing else, so the reader sees the derivation (supervisor's bolding instruction). Do NOT bold the acronym letters in the title or in headings. Do NOT use any other derivation (in particular, do not take "IS" as a block from "DIStillation").

STUDENT / DOCUMENT:
 - Suyog Khanal, s226137394. A2I2, Deakin University. Supervisors: A/Prof Santu Rana; Dr Arun Kumar Anjanapura Venkatesh.
 - Thesis title: "Leveraging Large Language Models for Sample-Efficient Imitation Learning".
 - Candidature start 13 Nov 2025; CoC date 13 Aug 2026; thesis submission Nov 2028.
 - Aim1 -> AAAI 2027 main track (submitted Jul 2026). Aim2 -> CoRL 2027. Aim3 -> CoRL 2028.
 - NO PAGE LIMIT. Depth over brevity. One unified PhD programme: Aim1 -> Aim2 -> Aim3, each leading naturally to the next.

TASKS / SETTINGS:
 - 5 tasks x 2 observation modalities = 10 SETTINGS. A "setting" = one task under one observation modality (state or image). Never call modality a "mode"; "mode" is reserved for FAILURE modes (clusters). Sweep this collision everywhere (supervisor).
 - Tasks: GridWorld 5x5 (discrete, 3 obstacles, start->goal, human expert; A*/BFS are ONLY the feasibility/path-validity checker, never the expert), Push-T (ManiSkill), Lift / Wipe / Door (RoboSuite, UR5/UR5e).
 - Policies are ANY function f_theta with a per-step loss: GridWorld image = pure CNN, GridWorld state = MLP, robot tasks = state/image diffusion policies. The framework is policy-agnostic; never say it is "for diffusion policies".
 - Budget: the framework works under ANY fixed/restricted budget B. B=20 is the INSTANCE validated (supervisor principle #1: symbol in method/algorithm, concrete value only in Setup). D=1 demonstration per round is likewise the tested instance (A12 justifies it).
 - Seeds: 9 seeds GridWorld, 5 seeds robot tasks (state the counts; do not fabricate a uniform count).
 - Baselines: SafeDAgger, DropoutDAgger, EnsembleDAgger, ThriftyDAgger, Stagger (GridWorld only), Diff-DAgger (robot only). In comparison tables these MUST be explicitly labelled as the DAgger family (author instruction).

SCIENTIFIC CORRECTIONS (binding, from the ablation workbook):
 - A10 SUPERSEDES the old Eq.7 image branch: clustering is GEOMETRIC for EVERY run (state and image alike). The "frozen R3M embedding + PCA" branch is OUT OF DATE and must NOT appear. The descriptor is 6-D: robot [p_x,p_y,sin(theta),cos(theta),rho,delta]; GridWorld [agent cell (2), signed offset (2), progress, Manhattan]. A10 shows an inverted-U in silhouette peaking at 6-D.
 - LIFT IS UNINFORMATIVE for every ablation (100.0 +/- 0.0: no headroom, no variance). NEVER read a null result on Lift as evidence about any mechanism. Say so explicitly.
 - A13: sigma is MIS-SCALED for narrow-reset tasks (Door, GridWorld); the kernel is degenerate there, and on Lift it is masked by the ceiling. A per-task sigma (a fraction of each task's reset range) would make the memory function everywhere. Report this honestly as a limitation, not as a virtue.
 - A13 cross-check: the lambda=0 column MUST reproduce A1_Memory_Off.
 - EQUATION 10 is now the FEASIBILITY VERIFICATION mechanism. Describe the workflow as: the LLM proposes a prescription -> constraints are retrieved from the KAG -> feasibility check -> if constraints are violated, feedback is returned to the LLM -> revised prescription, until feasible. The KAG stores explicit environmental constraints as structured key-value knowledge (workspace bounds, reachability, object/spawn ranges, controller limits).
 - The UPDATED architecture figure additionally shows a POLICY-SOLVABILITY loop: Prescription LLM -> "Policy Rollout on P" -> if the CURRENT policy can already solve P, the prescription is uninformative and P is revised ("Solvable => Revise P"). Describe the figure exactly as drawn. These are two distinct checks (KAG-constraint feasibility; policy solvability) - present both, do not conflate them.
 - INFORMATION GAIN discussion must now include STARTING PERFORMANCE and the INITIAL DEMONSTRATIONS: explain why the initial demonstration count was chosen to place each task's starting success rate inside a target initial success-rate range (enough competence for meaningful rollout failures, low enough to leave headroom for the budget to matter). Information gain = the policy's per-step loss on a newly acquired demonstration measured BEFORE retraining on it. High pre-retrain loss means either (a) the demonstration covers an underrepresented region, or (b) it is suboptimal/invalid. (b) is ruled out by construction: prescriptions pass the feasibility check and the demonstrations come from the expert. So high pre-retrain loss identifies genuinely novel, underrepresented data. This is a CLAIM with an argument, not a hypothesis.

ABLATION SCOPE (author instruction):
 - The CoC discusses REPRESENTATIVE studies only, on three primary settings: GridWorld (image), Push-T (state), Door (image). All other cells/ablations are retained for supplementary/rebuttal and should be mentioned as such, not tabulated in full.
 - The Excel workbook is the SOURCE OF TRUTH for every number. Do NOT force every ablation into a table: choose the clearest presentation (line plot, grouped/stacked bar, scatter, heatmap, radar, or a table only when exact numbers matter).
 - Every reported ablation must discuss: motivation, setup, findings, why some combinations work, why others fail, implications, limitations, and its influence on the final DISEIL framework. Interpret scientifically; do not merely restate numbers.

SUPERVISOR FEEDBACK (apply all EXCEPT items that only concern AAAI page limits or conference formatting):
 - Frame everything as a GENERAL framework; concrete values are one validated instance.
 - The Method must contain only the novelty. Standard material (BC objective, dataset aggregation, generic query-gate template, silhouette criterion, A*/BFS, the Diff-DAgger rule) belongs in Background/Related Work or is explicitly flagged "we follow standard practice". Cite standard algorithms you merely use (A*, BFS, silhouette).
 - Abstract everything swappable: present clustering as a generic step C instantiated here as agglomerative clustering (K-means etc. would also serve).
 - Define every term before use; re-expand abbreviations at first use in the body.
 - Do not open a paragraph with a bare "This is ...".
 - Baselines: describe qualitatively; do NOT reproduce their specific hyperparameters.
 - Algorithm: atomic steps, readable standalone; loop header "for r = 1 to B" (symbolic, not B=20).
 - Define Delta-SR where it first appears: the change in the policy's success rate on the ROUND-LEVEL rollout evaluation.
 - No semicolons in headings.

STYLE: follow ${COC}/Non-AI content.md strictly. Formal academic tone. No AI vocabulary (delve, pivotal, crucial, robust, leverage, showcase, underscore, Moreover/Furthermore as filler), minimal em dashes, no negative parallelism ("not just X but Y"), no rule-of-three padding, no copula avoidance, no vague "-ing" tack-on clauses, no elegant variation (one term per concept), no formulaic conclusions. Vary sentence rhythm. Every non-obvious claim is cited or cut. Never fabricate citations, DOIs or statistics.`

// ============================== PHASE 1: RECON ==============================
phase('Recon')

const recon = await parallel([
  // --- 1. Repository Inspection Agent ---
  () => agent(`You are the REPOSITORY INSPECTION AGENT for a PhD Confirmation of Candidature (CoC) report.
${FACTS}
Inspect the repository at ${ROOT} and build a knowledge map of everything the CoC can use. Cover:
 - ${DOCS}/ : read 00_START_HERE, 01_METHOD_DISTIL, 02_DESIGN_CHANGES_THIS_RUN, 03_TASKS_AND_ENVS, 04_CODEBASE_MAP_AND_CONSOLIDATION, 05_ABLATIONS, 06_PROMPTS, 07_KAG, 08_ORCHESTRATION_2HPC, 09_REPRODUCIBILITY_AND_AGGREGATION, supervisor_ablation_ask.txt.
 - ${ROOT}/distil/ (the consolidated module: engine_cli.py, matrix.py, p4/, envs.py, experts.py, config.py) — how a run is launched, what telemetry is written.
 - The Aim-1 paper source ${PAPER} and ${ROOT}/paper_aaai2027/context/ (results_data.md, equations.tex, litreview.md, references.bib, kag_ur5_bounds.md, dossier_*.md).
 - Figures at ${FIGS}/ and any experiment result folders.
EXTRACT AND RECORD, with exact file paths and verbatim quotes where useful:
 (a) REPRESENTATIVE PROMPTS actually used (VLM prompt, reasoning prompt, prescription prompt) — the CoC must include representative prompts, so quote them.
 (b) REPRESENTATIVE KAG EXAMPLES — the CoC must include them; show the KAG as STRUCTURED KEY-VALUE environmental constraints (workspace bounds, spawn ranges, reachability, controller limits). Quote a real KAG document (e.g. Push-T and one UR5 task).
 (c) CLUSTER NAMING: how discovered failure modes get their names/labels (the CoC must explain cluster naming).
 (d) The exact DISEIL loop as implemented, initial demonstration counts per task, starting success rates, budget, seeds.
 (e) Anything about how initial demonstration counts were chosen relative to a target starting success-rate range.
WRITE ${BUILD}/repo_map.md (a structured knowledge map with paths + quotes). Return a 12-line summary.`, {label:'repo-map', phase:'Recon', effort:'high'}),

  // --- 2. CoC Structure Agent ---
  () => agent(`You are the CoC STRUCTURE AGENT. Two real Deakin HDR Confirmation-of-Candidature reports are provided as REFERENCE ONLY:
 ${COC}/other_students_coc_sample_ref/COC_Report_Hung_Du-151024.pdf
 ${COC}/other_students_coc_sample_ref/ConfirmationofCandidatureReport.pdf
Read BOTH (the Read tool renders PDFs). Also read ${COC}/COC_REPORT_INSTRUCTIONS.md.
${FACTS}
Infer, WITHOUT copying or paraphrasing any sentence from them: the expected chapter/section structure, ordering, front matter and cover-page conventions, academic register, depth of technical detail, how the literature review is organised, how methodology and results are presented, how the project plan and Gantt chart are presented, figure/table/caption conventions, and reference style.
Then produce a DETAILED SECTION PLAN for OUR report: an ordered list of sections and subsections, each with (i) purpose, (ii) target depth (approx word count; remember there is no page limit and depth is wanted), (iii) which repository assets/figures/tables it uses, (iv) what must be argued. The plan must realise: cover page (A2I2 logo), executive summary/abstract, introduction and research vision, shared background + literature review, Aim 1 (full: motivation, gap, method, architecture, implementation, experiments, results, ablations, limitations), Aim 2 (Reverse-VLA), Aim 3 (original, ambitious, extends Aim 2, completes the story), research-programme coherence, project plan + Gantt, HDR training, conclusion, references.
WRITE ${BUILD}/coc_structure.md. Return the section list with target word counts.`, {label:'coc-structure', phase:'Recon', effort:'high'}),

  // --- 3. Ablation Analysis Agent ---
  () => agent(`You are the ABLATION ANALYSIS AGENT. Read EVERY sheet of ${XLSX} (the README sheet explains all others). Use pandas.
${FACTS}
Produce a complete ablation dossier. For the three PRIMARY settings (GridWorld image, Push-T state, Door image) analyse in depth; for other cells summarise only.
Cover the knockouts A1 (memory off), A2 (random allocation on robots), A3 (clustering off), A4 (LLM vs heuristic), A5 (VLM off), A6 (KAG off), A7 (bridging off), A8 (fallback only), A9 (context-set composition), A10 (descriptor dimensionality, scored by SILHOUETTE), A11, A12 (D per round), A13 (memory constants), A14 (k selection), A15 (cited episodes); and diagnostics D1 (cluster purity), D2 (k* distribution), D3 (bridge split), D4 (failure count), D5 (compute), S1 (sign test).
For EACH: motivation, setup, findings, WHY good combinations work and WHY poor ones fail, implications, limitations, and its influence on the final DISEIL framework. Interpret scientifically.
Be intellectually honest about the uncomfortable results: A4 (LLM vs heuristic) and A5 (VLM off) show SMALL gaps — state plainly what that does and does not license us to claim. A3 shows info gain stays HIGH while success rate DROPS: this is the evidence that gain without allocation is redundant. Lift is at ceiling and is uninformative everywhere.
For EACH ablation RECOMMEND the best PRESENTATION FORMAT (line plot / grouped bar / stacked bar / scatter / heatmap / radar / table) and justify; tables only where exact numbers matter. Produce the exact data series each figure needs.
WRITE ${BUILD}/ablation_dossier.md (analysis + per-ablation presentation recommendation + data series). Return a 15-line summary incl. the list of figures to generate.`, {label:'ablation-analysis', phase:'Recon', effort:'high'}),

  // --- 4. Experiment Execution Agent (D5) — longest lead time ---
  () => agent(`You are the EXPERIMENT EXECUTION AGENT. The D5_Compute sheet of ${XLSX} is EMPTY and instructs: "Run 1 job per task to compute and fill in these matrix."
Required rows (5): Door/state, Push-T/image, Wipe/image, Door/image, GridWorld 5x5/image.
Required columns: Baseline s/round, DISEIL s/round, VLM tokens/round, LLM tokens/round, Overhead x, KAG token contribution, Reasoning LLM tokens/round.
${FACTS}
STEP 1 — RECON (do this before submitting anything): read "${DOCS}/08_ORCHESTRATION_2HPC.md", "${DOCS}/09_REPRODUCIBILITY_AND_AGGREGATION.md", "${DOCS}/SETUP_HPC2.md", "${DOCS}/00_START_HERE.md", and ${ROOT}/distil/engine_cli.py, ${ROOT}/distil/matrix.py, ${ROOT}/distil/config.py, plus the telemetry writer. Determine EXACTLY how to launch one short run per setting that records per-round wall-clock and VLM/LLM token counts, and how the baseline per-round wall-clock is measured. Check ${ROOT}/.env for the model/API configuration (never print secret values). Check the SLURM partitions with sinfo.
STEP 2 — DECIDE AND REPORT FEASIBILITY. A few rounds per setting is enough to get seconds/round and tokens/round; you do NOT need a full 20-round budget. Prefer the smallest run that yields a trustworthy per-round measurement.
STEP 3 — EXECUTE. If feasible, submit the SLURM jobs (sbatch), record job IDs, and poll squeue. Wait for completion, polling periodically, for as long as is reasonable within your session. Collect the telemetry outputs and COMPUTE the seven columns (Overhead x = DISEIL s/round / Baseline s/round).
STEP 4 — RECORD. Write ${BUILD}/d5_compute.md containing: the exact commands/scripts used, job IDs, per-setting measured values, the completed D5 matrix as a markdown table, and any caveats (e.g. measured over k rounds, shared GPU node). ALSO write the same values to ${BUILD}/d5_compute.csv.
STEP 5 — If (and only if) the jobs CANNOT be run (missing dependency, no GPU allocation, credentials unavailable), do NOT fabricate any number. Instead write ${BUILD}/d5_compute.md documenting precisely why, what was attempted, and the exact commands the author must run. Mark every cell UNMEASURED.
NEVER invent timing or token numbers. Return: feasibility verdict, job IDs (if any), and the D5 table (or the blocked-reason).`, {label:'d5-compute', phase:'Recon', effort:'high'}),

  // --- 5. Statistics ("Scientist") Agent — the A13 embedded instruction ---
  () => agent(`You are the STATISTICS ("Scientist") AGENT. The A13 sheet of ${XLSX} embeds a statistical-analysis instruction. Execute it exactly.
${FACTS}
The A13 sheet sweeps each memory constant one at a time: gamma in {0.3,0.5,0.6,0.7,0.9} (paper value 0.6); sigma in {0.02,0.04,0.06,0.1,0.2} (paper 0.06); lambda in {0.0,0.5,1.0,2.0,4.0} (paper 1.0). Each is measured on the 10 task x modality conditions (matched blocks).
For EACH hyperparameter sweep:
 1. Friedman test (scipy.stats.friedmanchisquare) across the swept values, treating the 10 conditions as matched blocks. Report chi-square and p.
 2. Post-hoc pairwise Wilcoxon signed-rank tests (scipy.stats.wilcoxon, paired, two-sided) comparing the paper's chosen value against every other value.
 3. Holm-Bonferroni correction WITHIN each hyperparameter's family only (statsmodels.stats.multitest or implement manually). Do NOT correct across hyperparameters.
 4. Average rank per value across the 10 conditions (rank 1 = best per row).
 5. Flag any hyperparameter where the chosen value has the best average rank but does NOT reach corrected significance against neighbours: report as "directionally best but not statistically distinguishable" rather than overstating.
OUTPUT: for each hyperparameter a markdown table (value | avg_rank | wilcoxon_p_vs_chosen | holm_corrected_p), the Friedman chi2 and p, a one-sentence plain-English verdict, and a paste-ready paragraph for the report.
Also run the S1_SignTest sheet's analysis (sign test / Wilcoxon over the 10 paired means, DISEIL vs best baseline) and report it.
CROSS-CHECK: verify the lambda=0 column of A13 reproduces A1_Memory_Off; report any disagreement as a data-integrity flag.
SAVE: ${BUILD}/stats_results.csv (one row per hyperparameter-value pair: hyperparameter_name, value, avg_rank, wilcoxon_p, holm_p, friedman_chi2, friedman_p) and ${BUILD}/stats_report.md (tables + verdicts + paste-ready paragraphs).
Use real computation only. Never fabricate a p-value. Return the three verdicts + the S1 result.`, {label:'stats-scientist', phase:'Recon', effort:'high'}),
])
log(`Recon complete: ${recon.filter(Boolean).length}/5 agents returned`)

// ============================== PHASE 2: ASSETS =============================
phase('Assets')

const assets = await parallel([
  // --- Figure generation from the workbook ---
  () => agent(`You are the FIGURE GENERATION AGENT. Build publication-quality figures for the CoC from the ablation workbook.
${FACTS}
Read ${BUILD}/ablation_dossier.md (it recommends a presentation format and the data series for each ablation) and ${XLSX} (source of truth), plus ${BUILD}/stats_report.md.
Generate figures with matplotlib (Agg backend) into ${GFIG}/ as PDF (vector) AND a PNG preview. Requirements:
 - Focus on the three PRIMARY settings (GridWorld image, Push-T state, Door image); use other cells only where a figure genuinely needs them (e.g. the A10 silhouette curve, the A13 sweeps, the A1/A3 knockout comparison).
 - Do NOT force tables into figures and do not turn everything into a bar chart: honour the dossier's recommended format per ablation.
 - Suggested set (adjust to the dossier): (i) knockout summary — margin retained vs full DISEIL across A1/A3/A4/A5/A6/A7/A8 for the 3 primary settings (grouped bar or slope/dumbbell); (ii) A3 "gain != success" — info gain vs success rate (scatter or twin-axis) showing gain stays high while SR drops; (iii) A10 silhouette vs descriptor dimensionality (line plot, inverted U, peak at 6-D); (iv) A13 memory-constant sweeps gamma/sigma/lambda (line plots with the paired-seed-noise band, annotating INERT vs LIVE cells); (v) A12 demonstrations-per-round D in {1,2,3} (grouped bar or line); (vi) A2 random-allocation control vs Diff-DAgger vs DISEIL on robots; (vii) D3 bridge-vs-targeted split and D2 k* distribution (stacked bar / histogram); (viii) D1 cluster purity (heatmap).
 - Style: clean, colourblind-safe palette, no chartjunk, readable fonts (>=9pt at print size), axis labels with units, explicit legends, no title inside the figure (the caption carries it). Mark the ceiling on Lift wherever Lift appears, so no one over-reads a null.
 - Every number must come from the workbook. Never invent data.
WRITE ${BUILD}/figures_generated.md: for each figure — filename, what it shows, the data series used (with sheet + cell provenance), and a publication-quality caption. Return the list of generated files.`, {label:'figure-generation', phase:'Assets', effort:'high'}),

  // --- Existing figure indexing + Aim-2 architecture extraction ---
  () => agent(`You are the FIGURE INTEGRATION AGENT for existing assets.
${FACTS}
Read (visually, with the Read tool) and index these:
 - ${FIGS}/Teaser_Diagram.pdf  (the UPDATED teaser — use this one)
 - "${FIGS}/Architectural Diagram.pdf"  (the UPDATED architecture — use this one; it shows the policy-solvability loop "Solvable => Revise P", NOT the old infeasibility loop)
 - ${FIGS}/all_5_task_comparison.pdf  (learning curves for ALL FIVE tasks — the CoC must include these)
 - ${FIGS}/confidence_vs_success.pdf, ${FIGS}/info_gain_boxplot.pdf, ${FIGS}/clustering_modes_pushT.pdf
 - ${COC}/paper2_preview.pdf — EXTRACT the Aim-2 architecture figure from this PDF and save it as an image/PDF into ${GFIG}/aim2_architecture.pdf (or .png). It must be captioned exactly "Proposed architecture for Aim 2". Describe every major component so the report can explain how it extends Aim 1.
 - ${COC}/A2I2_Logo_Stacked_2025_Keyline.png (cover page), "${COC}/Compulsory Training Status.png", and the three training-certificate PDFs (SSC900 Academic Writing; Research Integrity; Respect at Deakin) — read them and record what each certifies (course, outcome, date) for the HDR-training section.
For each figure record: file path, exactly what it shows (read axis labels/legends off the figure), aspect ratio, recommended placement in the report, and a publication-quality caption.
Note any defect the author should fix (e.g. the architecture diagram's "on the the augmented dataset" typo).
WRITE ${BUILD}/figure_index.md. Return a summary line per asset.`, {label:'figure-index', phase:'Assets', effort:'high'}),

  // --- Citation / literature agent ---
  () => agent(`You are the LITERATURE + CITATION AGENT. The CoC needs a thorough, well-grounded literature review for EVERY aim.
${FACTS}
Start from the verified bibliography already in the repo: ${ROOT}/paper_aaai2027/context/references.bib (52 verified entries) and ${ROOT}/paper_aaai2027/context/litreview.md (per-paper cards: title, authors, venue, summary, relation). Also read ${COC}/paper2_reverse_vla_concept.md (its appendix lists Aim-2 references, several flagged as unverified 2025/26 arXiv items).
Produce a literature plan covering:
 (a) SHARED BACKGROUND: imitation learning, behaviour cloning, covariate shift, DAgger and the query-efficient DAgger family, diffusion policies, uncertainty estimation, active learning / coreset & diversity selection, LLM/VLM for robotics, LLM failure reasoning.
 (b) AIM 1 (DISEIL): position against the DAgger family and demonstration-selection work.
 (c) AIM 2 (Reverse VLA): forward VLAs (RT-2, OpenVLA, Octo), language-as-intermediate (RT-H, ECoT), trajectory captioning, inverse dynamics / latent action, demonstration/dataset selection (STRAP, data quality in IL), REFLECT.
 (d) AIM 3: literature that supports an ambitious long-horizon extension.
RULES: prefer the 52 already-verified entries. If you propose a NEW reference, you must verify it exists (search/fetch) and record title, authors, venue, year, arXiv ID/DOI, URL. NEVER fabricate a citation. Explicitly list any reference from the Aim-2 concept doc that you could NOT verify, and recommend dropping it.
WRITE: ${BUILD}/literature_plan.md (per-section citation map: claim -> citekey) and ${BUILD}/references_coc.bib (the full bibliography for the CoC, merging the verified 52 with any newly verified entries).
Return: counts (reused / newly verified / rejected-unverifiable) and the list of rejected ones.`, {label:'citations', phase:'Assets', effort:'high'}),
])
log(`Assets complete: ${assets.filter(Boolean).length}/3 agents returned`)

// ============================== PHASE 3: NARRATIVE ==========================
phase('Narrative')
await agent(`You are the RESEARCH NARRATIVE AGENT. Build the single connective spine for the whole CoC so it reads as ONE PhD programme, not three papers.
${FACTS}
Read: ${BUILD}/coc_structure.md, ${BUILD}/repo_map.md, ${BUILD}/ablation_dossier.md, ${BUILD}/figure_index.md, ${BUILD}/literature_plan.md, ${COC}/paper2_reverse_vla_concept.md, ${COC}/COC_REPORT_INSTRUCTIONS.md, ${COC}/SUPERVISOR_PAPER_FEEDBACK.txt.
Produce the narrative spine:
 - The central research vision: leveraging LLMs for sample-efficient imitation learning — the through-line from "expert demonstrations are the binding cost" to each aim.
 - AIM 1 (DISEIL): under a restricted budget, decide WHICH failures to correct and HOW each corrective demonstration is placed. Its honest residual weakness, established by our own ablations: the selector is dataset-blind and stateless (it reasons about the current round's failures, with only a geometric cluster memory), and A4/A5 show the LLM/VLM contribute less than the allocation machinery.
 - AIM 2 (Reverse VLA): this weakness IS the motivation. Give the selector language-indexed dataset self-awareness (coverage memory) so a demonstration is spent on a genuine coverage gap, not a locally uncertain state. Show explicitly how Aim 1's cluster memory + KAG are subsumed/generalised.
 - AIM 3: devise an ORIGINAL, ambitious long-term aim that extends Aim 2 and completes the story. It must follow from Aim 2's residual limitation the way Aim 2 follows from Aim 1's. Develop it fully (motivation, gap, proposed approach, novelty, evaluation strategy, risks). Make it credible and connected, not a wish list.
 - Explicit "how Aim N leads to Aim N+1" bridging passages.
WRITE ${BUILD}/narrative_spine.md, including the Aim-3 proposal in full. Return a 15-line summary of the spine and the Aim-3 title + one-paragraph pitch.`, {label:'narrative', phase:'Narrative', effort:'high'})

// ============================== PHASE 4: WRITE ==============================
phase('Write')
const SECTIONS = [
  {k:'00_front',      t:'Cover page + title page + table of contents + executive summary/abstract',
   d:`Cover page per the instructions: A2I2 logo (${COC}/A2I2_Logo_Stacked_2025_Keyline.png) at the top, Deakin University, A2I2, "Confirmation of Candidature Report", thesis title, student name, student ID, supervisors, candidature start date (13 Nov 2025), CoC date (13 Aug 2026). Then a table of contents, then an executive summary/abstract of the whole programme. The abstract introduces DISEIL with the acronym letters bolded so its derivation from the title is visible.`},
  {k:'01_intro',      t:'Introduction and research vision',
   d:'The problem (expert demonstrations are the binding cost of imitation learning), the central vision, the three aims and how they compose, the contributions to date, and a roadmap of the document. Re-expand all abbreviations at first use here.'},
  {k:'02_background', t:'Background and literature review',
   d:'Shared technical background and a thorough literature review: imitation learning and behaviour cloning, covariate shift, DAgger and the query-efficient DAgger family (SafeDAgger, DropoutDAgger, EnsembleDAgger, ThriftyDAgger, Diff-DAgger, Stagger), diffusion policies, uncertainty estimation, active learning / coreset and diversity selection, LLMs and VLMs for robotics, LLM failure reasoning and self-correction. THIS is where the standard material lives (BC objective, dataset aggregation, generic query-gate template, silhouette criterion, A*/BFS) per supervisor principle #2 — so that the Aim-1 method section contains only our novelty.'},
  {k:'03_aim1_method',t:'Aim 1 — Demonstration Distillation (DISEIL): motivation, gap, method',
   d:`Motivation and problem statement, the research gap, then the DISEIL framework. Present it as a GENERAL framework (budget B, D demonstrations per round; concrete values only in Setup). Formal method with equations, using the notation from ${ROOT}/paper_aaai2027/context/equations.tex but CORRECTED per A10: clustering is geometric for every run (NO R3M/PCA branch). Present clustering as a generic step instantiated here as agglomerative clustering. Cover: uncertainty flagging at t*, VLM perception of start/t*/end frames, KAG-grounded root-cause reasoning, the 6-D geometric descriptor, failure-mode clustering, the cluster memory, the prescription (targeted correction vs bridging), CLUSTER NAMING (explain how discovered modes are named), and EQUATION 10 as the FEASIBILITY VERIFICATION mechanism (LLM proposes -> KAG constraints retrieved -> feasibility check -> feedback on violation -> revised prescription until feasible), with the KAG holding explicit structured key-value environmental constraints. Also describe the policy-solvability check shown in the architecture figure ("Solvable => Revise P"). Include the architecture figure (updated) and the teaser (updated), an atomic, readable Algorithm (loop "for r = 1 to B"), and REPRESENTATIVE PROMPTS and REPRESENTATIVE KAG EXAMPLES (quoted from the repo, per ${BUILD}/repo_map.md).`},
  {k:'04_aim1_exp',   t:'Aim 1 — experiments, results and analysis',
   d:`Setup (tasks, the 10 settings = 5 tasks x 2 modalities, policy instantiations: GridWorld image = CNN, GridWorld state = MLP, robots = diffusion; baselines described qualitatively and labelled explicitly as the DAgger family; B=20, D=1, seeds 9/5; the initial demonstration counts and starting success rates, and WHY the initial demonstration count was chosen to land the starting success rate inside a target range). Main results table (all 10 settings, DAgger-family labelled). The five-task learning curves (${FIGS}/all_5_task_comparison.pdf). The information-gain analysis, rewritten per the FACTS: pre-retrain loss, the two-readings argument, feasibility+expert ruling out the bad reading, therefore novelty from underrepresented regions — now ALSO tied to starting performance and the initial demonstrations. Define Delta-SR at first use (round-level rollout evaluation). Confidence-vs-improvement analysis (r = 0.82-0.89) explaining WHY the prescription confidence is trustworthy in-round: the success-rate signal arrives late (only after collection, retraining and re-rollout), whereas the confidence is reported blind at prescription time. Discovered failure modes (clustering figure). Note Lift's ceiling.`},
  {k:'05_aim1_abl',   t:'Aim 1 — ablation studies',
   d:`From ${BUILD}/ablation_dossier.md, ${BUILD}/stats_report.md, ${BUILD}/figures_generated.md and ${BUILD}/d5_compute.md. REPRESENTATIVE studies only, on the three primary settings (GridWorld image, Push-T state, Door image); state explicitly that the remaining ablations and cells are retained for the supplementary material. Use the recommended presentation format per ablation (figures where trends matter, tables only for exact comparisons). For each: motivation, setup, findings, why some combinations work and others fail, implications, limitations, influence on the final DISEIL framework. Be honest about A4/A5 (small gaps) and about Lift's ceiling. Include the A13 statistics (Friedman/Wilcoxon/Holm/avg-rank) and the sigma mis-scaling finding as an honest limitation. Include the D5 compute cost (or its measured status).`},
  {k:'06_aim1_lim',   t:'Aim 1 — limitations, current progress and status',
   d:'Honest limitations (dataset-blind and stateless selector; hand-designed descriptor; per-round reasoning cost; sigma mis-scaling; Lift ceiling; the modest marginal value of the LLM/VLM shown by A4/A5). Current progress and status: submitted to AAAI 2027 main track, July 2026. This section must set up Aim 2 — the residual weakness IS Aim 2\'s motivation.'},
  {k:'07_aim2',       t:'Aim 2 — Reverse VLA: coverage-aware demonstration selection',
   d:`From ${COC}/paper2_reverse_vla_concept.md and the narrative spine. Motivation (the gap Aim 1 leaves), research gap, proposed methodology (the (Vision+Action)->Language captioner; the language-indexed, competence-weighted coverage memory; the single grounded selector that collapses Aim 1's three stateless LLM calls plus the clustering stack), expected contributions, novelty and honest positioning against nearby work, evaluation strategy (benchmarks, metrics D@tau and success-vs-demos AUC, baselines, ablations incl. the matched-information ablation that tests whether language does causal work), risks and mitigations, and an explicit subsection on the relationship to Aim 1. Include the Aim-2 architecture figure extracted from ${COC}/paper2_preview.pdf, captioned exactly "Proposed architecture for Aim 2", and explain each major component and how it extends Aim 1. Target venue CoRL 2027.`},
  {k:'08_aim3',       t:'Aim 3 — long-term vision',
   d:'The ORIGINAL Aim 3 developed in the narrative spine. It must extend Aim 2 naturally, be ambitious, and complete the research story. Full treatment: motivation, gap, proposed approach, novelty, evaluation strategy, risks, and its relationship to Aims 1 and 2. Target venue CoRL 2028. It is acceptable and expected that this is future work beyond current implementation.'},
  {k:'09_plan',       t:'Research programme coherence, project plan and Gantt chart',
   d:`First a short section arguing the coherence of the programme (how Aim 1 -> Aim 2 -> Aim 3 compose into one thesis). Then the detailed project plan: Aim 1 (AAAI 2027 main track, submitted July 2026), Aim 2 (CoRL 2027; abstract and full paper late May 2027; conference Oct-Nov 2027), Aim 3 (CoRL 2028; late May 2028; conference early Nov 2028), thesis submission November 2028. Then a professional HDR-style GANTT CHART covering the full candidature (Nov 2025 - Nov 2028) with milestones: literature review, problem formulation, methodology development, implementation, experimentation, evaluation, paper writing, conference submissions, revisions, Aim 1/2/3 completion, thesis writing, thesis submission, examination preparation. Render the Gantt as a Markdown-embeddable figure: generate it with matplotlib into ${GFIG}/gantt_chart.pdf (+ .png) and reference it; also give a milestone table. Ensure the chart aligns with the publication timeline and the Nov 2028 submission.`},
  {k:'10_training',   t:'HDR training and professional development',
   d:`Summarise the completed HDR training from the certificates indexed in ${BUILD}/figure_index.md: SSC900 Academic Writing; Deakin Safety and Research Integrity training; Respect at Deakin (Graduate Research and Supervision module); and the Compulsory Training Status. State what each certifies and how it supports the candidature. Do not invent dates or results that are not in the documents.`},
  {k:'11_conclusion', t:'Conclusion',
   d:'Draw the programme together: what has been achieved (Aim 1, submitted), what is next (Aims 2 and 3), and why the trajectory is coherent and feasible within the timeline. End on a concrete next step, not a formulaic flourish.'},
]

await parallel(SECTIONS.map(s => () => agent(
`You are a specialist SECTION WRITER for a PhD Confirmation of Candidature report. Write ONLY the section: "${s.t}".
${FACTS}
Read first: ${BUILD}/narrative_spine.md (the connective spine — obey it), ${BUILD}/coc_structure.md (target depth and conventions), ${BUILD}/repo_map.md, ${BUILD}/ablation_dossier.md, ${BUILD}/stats_report.md, ${BUILD}/figures_generated.md, ${BUILD}/figure_index.md, ${BUILD}/literature_plan.md, ${BUILD}/d5_compute.md, and ${COC}/Non-AI content.md (style rules — obey strictly).
SECTION BRIEF: ${s.d}
FORMAT: Markdown. Use ## / ### headings (the assembler will fix levels). Reference figures with standard Markdown image syntax and RELATIVE paths from ${COC} (e.g. ![...](figures_generated/xxx.pdf) or ![...](../figures/xxx.pdf)), each with a numbered, self-contained caption. Cite with [@citekey] using keys that exist in ${BUILD}/references_coc.bib. Tables in Markdown. Equations in LaTeX ($...$ / $$...$$).
DEPTH: there is no page limit. Write with the depth of a strong PhD proposal chapter. Do not pad, but do not compress away substance.
HONESTY: every number must come from the authoritative sources. Never invent a statistic, citation or result. If something is unmeasured, say so plainly.
WRITE your section to ${SEC}/${s.k}.md. Return a 4-line summary plus your word count.`,
  {label:`write:${s.k}`, phase:'Write', effort:'high'}
)))

// Assemble
await agent(`You are the ASSEMBLY AGENT. Merge the sections into the single CoC report.
${FACTS}
Read every file in ${SEC}/ in filename order (00_front .. 11_conclusion) plus ${BUILD}/narrative_spine.md.
Produce ONE Markdown document at ${REPORT}:
 - Correct, consistent heading hierarchy throughout (# for the report title / chapters, ## sections, ### subsections).
 - Continuous, correct numbering of ALL figures, tables and equations across the whole document, and fix every cross-reference to match.
 - Insert the bridging passages from the narrative spine so Aim 1 -> Aim 2 -> Aim 3 read as one programme.
 - Remove cross-section duplication (the sections were written in parallel; expect repeated background). Keep the best instance.
 - A References section at the end, generated from ${BUILD}/references_coc.bib in a consistent style, containing every cited key and nothing uncited.
 - A table of contents matching the final headings.
Do not delete substance to save space; there is no page limit.
Return: total word count, figure count, table count, and the final section list.`, {label:'assemble', phase:'Write', effort:'high'})

// ============================== PHASE 5: QA =================================
phase('QA')
const CRIT = {type:'object', additionalProperties:false, properties:{
  ok:{type:'boolean'},
  issues:{type:'array', items:{type:'object', additionalProperties:false, properties:{
    severity:{type:'string', enum:['critical','major','minor']},
    location:{type:'string'}, problem:{type:'string'}, fix:{type:'string'}
  }, required:['severity','location','problem','fix']}}
}, required:['ok','issues']}

for (let round = 1; round <= 2; round++) {
  const crits = await parallel([
    () => agent(`You are the RESULTS VERIFICATION AGENT (QA round ${round}). Read ${REPORT}.
${FACTS}
Check EVERY number in the report against the authoritative sources: ${XLSX} (all sheets), ${ROOT}/paper_aaai2027/context/results_data.md, ${BUILD}/stats_results.csv, ${BUILD}/d5_compute.md, ${ROOT}/paper_aaai2027/context/kag_ur5_bounds.md.
Flag: any number that does not match; any invented statistic or p-value; any claim stronger than the data supports (especially A4/A5 small gaps, and ANY inference drawn from Lift, which is at ceiling); any ablation reported outside the three primary settings as if it were primary; any surviving reference to the dead R3M/PCA clustering branch; any place where B=20 or D=1 is presented as part of the framework rather than as the validated instance.
Return the structured verdict.`, {label:`qa-results-${round}`, phase:'QA', schema:CRIT, effort:'high'}),

    () => agent(`You are the CITATION CHECKING AGENT (QA round ${round}). Read ${REPORT} and ${BUILD}/references_coc.bib.
${FACTS}
Verify: every [@citekey] resolves to a bib entry; every bib entry is cited; no fabricated references (spot-check the riskiest, especially any 2025/26 arXiv items from the Aim-2 concept doc — if a reference cannot be verified, it must be removed); claims attributed to the right source; standard algorithms that are merely used (A*, BFS, silhouette, agglomerative clustering, DAgger, diffusion policy) are cited.
Return the structured verdict.`, {label:`qa-citations-${round}`, phase:'QA', schema:CRIT, effort:'high'}),

    () => agent(`You are the CONSISTENCY AGENT (QA round ${round}). Read ${REPORT}.
${FACTS}
Verify: DISEIL used everywhere and DISTIL/PACE/P4 nowhere; the acronym derivation from the title is shown once, with bolding, in the abstract; "mode" is used only for failure modes and "modality"/"setting" for state/image (no collision); figure/table/equation numbering is continuous and every cross-reference resolves; the updated teaser and updated architecture are the ones used; the five-task learning curves are present; the Aim-2 architecture figure is captioned exactly "Proposed architecture for Aim 2"; the Gantt chart is present and aligns with the Nov 2028 thesis submission; the aims form one connected programme (Aim 1 -> Aim 2 -> Aim 3 bridges present); Equation 10 is presented as feasibility verification with KAG key-value constraints; cluster naming is explained; representative prompts and representative KAG examples are included; no section contradicts another.
Return the structured verdict.`, {label:`qa-consistency-${round}`, phase:'QA', schema:CRIT, effort:'high'}),

    () => agent(`You are the WRITING QUALITY (HUMANIZATION) AGENT (QA round ${round}). Read ${REPORT} and ${COC}/Non-AI content.md.
${FACTS}
Flag every violation of the non-AI style rules with an exact excerpt: AI vocabulary (delve, pivotal, crucial, key, robust, seamless, comprehensive, nuanced, leverage, foster, showcase, underscore, highlight), filler connectives (Moreover/Furthermore/Additionally/Notably/"It is worth noting"), em-dash overuse, negative parallelism ("not just X but Y", "not X but rather Y"), rule-of-three padding, copula avoidance (serves as / stands as / represents / boasts / features), vague "-ing" tack-on clauses, elegant variation (renaming the same concept), formulaic conclusions ("In conclusion", "Despite its promise", "continues to evolve"), chatbot filler, and uniform sentence rhythm.
Also flag any passage that reads as promotional rather than scientific.
Return the structured verdict (each issue = one excerpt + its fix).`, {label:`qa-style-${round}`, phase:'QA', schema:CRIT, effort:'high'}),
  ])

  const all = crits.filter(Boolean).flatMap(c => c.issues || [])
  const blocking = all.filter(i => i.severity !== 'minor')
  log(`QA round ${round}: ${all.length} issues (${blocking.length} critical/major)`)
  if (!all.length) break

  await agent(`You are the REVISION AGENT (QA round ${round}). Four specialist reviewers audited ${REPORT}.
${FACTS}
Fix EVERY critical and major issue, and every minor issue that is cheap to fix. Edit ${REPORT} in place with targeted Edit calls (never rewrite the whole file with Write). Do not weaken correct content; do not delete substance for brevity (no page limit). If a reviewer is wrong, leave the text and note why in your return message.
ISSUES (JSON): ${JSON.stringify(all)}
Return a bullet list mapping each issue to the fix applied (or a reasoned rebuttal).`, {label:`qa-revise-${round}`, phase:'QA', effort:'high'})
}

// ============================== PHASE 6: VALIDATE ===========================
phase('Validate')
const GATE = {type:'object', additionalProperties:false, properties:{
  passed:{type:'boolean'}, words:{type:'integer'}, figures:{type:'integer'}, tables:{type:'integer'}, refs:{type:'integer'},
  checklist:{type:'array', items:{type:'object', additionalProperties:false, properties:{
    item:{type:'string'}, pass:{type:'boolean'}, note:{type:'string'}
  }, required:['item','pass','note']}},
  outstanding:{type:'string'}, summary:{type:'string'}
}, required:['passed','words','figures','tables','refs','checklist','outstanding','summary']}

const gate = await agent(`You are the FINAL VALIDATION AGENT for the CoC report at ${REPORT}.
${FACTS}
Run the author's mandatory checklist; for each item report pass/fail with evidence (grep, or the location in the document):
 1. DISEIL used everywhere.  2. DISTIL (and PACE/P4) removed entirely.  3. Updated teaser used.  4. Updated architecture used.  5. Learning curves for all five tasks included.  6. Information-gain discussion updated (pre-retrain loss argument + starting performance + initial demonstrations + why the initial demonstration count was chosen for the target starting success-rate range).  7. Initial-demonstration discussion added.  8. Representative prompts included.  9. Representative KAG examples included (structured key-value environmental constraints).  10. Equation 10 updated to feasibility verification (LLM proposes -> KAG constraints -> feasibility check -> feedback on violation -> revise until feasible).  11. Cluster naming explained.  12. Humanized writing applied (spot-check for AI tells; report the em-dash count and any flagged vocabulary).  13. References verified (every citation resolves; nothing fabricated).  14. Cross-references consistent (figures/tables/equations numbered continuously; all refs resolve).
ALSO verify: comparison tables label the DAgger family explicitly; ablations are confined to the three primary settings with the rest deferred to supplementary; Lift's ceiling caveat present; the Gantt chart and Nov-2028 thesis submission present; the cover page has the A2I2 logo and all required fields; the three aims connect.
Then write ${COC}/COC_BUILD_REPORT.md: the checklist with evidence, document statistics (words, figures, tables, references), the D5 compute status, and a clear list of anything the AUTHOR must still supply or decide.
Return the structured verdict.`, {label:'final-validation', phase:'Validate', schema:GATE, effort:'high'})

return { gate }
