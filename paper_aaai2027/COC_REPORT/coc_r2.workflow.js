export const meta = {
  name: 'coc-revision-round2',
  description: 'Algorithm float, cluster-memory reframed as a configurable feature (Eq 8a + study A12 removed), abstract to one page, figure deletions/resizing/placement, table 8 reduced, then renumber, rebuild and audit',
  phases: [ { title: 'Content' }, { title: 'Renumber' }, { title: 'QA' } ],
}

const COC    = '/weka/s226137394/DmNfull/paper_aaai2027/COC_REPORT'
const BUILD  = COC + '/build'
const V2      = BUILD + '/v2'
const GFIG   = COC + '/figures_generated'
const XLSX   = COC + '/ablations_results/DISTIL_ablation_results.xlsx'
const REPORT = COC + '/CoC_Report.md'
const TT     = '/home/s226137394/.TinyTeX/bin/x86_64-linux'

const CORE = `THE REPORT IS GENERATED. ${REPORT} is assembled from ${V2}/*.md by ${BUILD}/assemble.py.
=> EDIT THE CHAPTER FILES IN ${V2}/, then run \`python3 ${BUILD}/assemble.py\`, then \`bash ${COC}/build_pdf.sh\`.
NEVER hand-edit ${REPORT} (assemble.py overwrites it). Use targeted Edit calls.

Method is DISEIL. Never DISTIL/PACE/P4. "A2I2" is banned (write the institute in full).
Numbers come only from ${XLSX} and ${COC}/../context/results_data.md. NEVER invent one.
Style: ${COC}/Non-AI content.md — plain headings, no wordplay, no AI vocabulary, minimal em dashes.
build_pdf.sh HARD-FAILS on: a Greek glyph vanishing, a pandoc-generated "Figure N:" caption, the banned acronym, or a table overflowing. It must still pass.

CURRENT NUMBERING (before this round's deletions):
 Fig1 teaser | Fig2 architecture | Fig3 aim2 arch | Fig4 clustering modes | Fig5 aggregate significance |
 Fig6 all-5-task curves | Fig7 info-gain boxplot | Fig8 confidence | Fig9 allocation ladder |
 Fig10 gain-without-allocation | Fig11 grounding/feasibility | Fig12 prescription+VLM | Fig13 bridging |
 Fig14 knockout summary | Fig15 descriptor dim | Fig16 context/selection | Fig17 cluster count |
 Fig18 budget sweep | Fig19 memory constants | Fig20 failures over budget | Fig21 gantt | Fig22 training status |
 Figs A.1-A.3 certificates.
 Tables: 7 = success rate, 8 = per-demo information gain, 9 = memory-constant sweep, 10 = cluster purity, 11 = per-round cost.
 Eq 8a = Gaussian memory penalty P_mem (gamma, sigma_mem). Eq 8b = target selection (argmax of mean loss minus lambda*P_mem).`

// ===================== PHASE 1: CONTENT (parallel, disjoint files) =====================
phase('Content')
await parallel([

  // --- 04_aims.md : algorithm float + memory reframing + Fig 2 placement ---
  () => agent(`You are the METHOD agent. Edit ONLY ${V2}/04_aims.md.
${CORE}

TASK 1 — ALGORITHM 1 AS A REAL ALGORITHM (supervisor item H5).
Algorithm 1 is currently a fenced VERBATIM code block of ~34 steps. It renders as monospace code, not a typeset algorithm, and it is far too long.
 - Add \\usepackage{algorithm} and \\usepackage{algpseudocode} to ${BUILD}/preamble.tex. Install if needed: \`${TT}/tlmgr install algorithms algorithmicx\`. Do a trivial compile to prove they work BEFORE editing the report.
 - Replace the verbatim block with a RAW LATEX float (pandoc passes raw LaTeX through):
     \\begin{algorithm}[t]\\caption{DISEIL}\\label{alg:diseil}\\begin{algorithmic}[1] ... \\end{algorithmic}\\end{algorithm}
 - SHORT: target 14-18 \\State lines, not 34. ATOMIC: one action per line; never cram two actions into a step. Loop header symbolic: \\For{$r = 1$ to $B$}.
 - It must convey the four stages standalone: perceive (flag $t^\\star$, describe the failure), partition (cluster failures into modes), prioritise (choose the target mode), prescribe (propose, verify against the environment model, revise until feasible), then collect $D$ demonstrations, aggregate, retrain at the per-task cadence.
 - Reuse the surrounding notation exactly. No prose or commentary lines inside the algorithm.
 - If prose nearby merely restates the algorithm step by step, trim it.

TASK 2 — RECAST THE CLUSTER MEMORY AS A CONFIGURABLE, TASK-DEPENDENT FEATURE (author's instruction; this is a real scientific softening and it must be done honestly).
 - DELETE Equation 8a entirely (the Gaussian penalty $P_{\\mathrm{mem}}$ with $\\gamma$ and $\\sigma_{\\mathrm{mem}}$). Do not relabel another equation 8a.
 - SIMPLIFY Equation 8b so target selection no longer carries the memory term in the core method: the round targets the highest-mean-loss mode among the near-dominant clusters. Renumber the remaining equations continuously (there must be no 8a/8b split left).
 - In prose, describe the memory as what it is: a CONFIGURABLE, TASK-DEPENDENT component that can be switched on or off. It becomes active when a task produces RECURRING failure clusters, where it rotates the budget away from modes already corrected. In environments without such recurrence it costs negligible overhead and changes performance very little. State this plainly; do NOT use the word "optional" (that word is banned elsewhere in this project for the two prescription arms), and do NOT oversell the memory as a headline contribution.
 - EVIDENCE for that claim, use it in one or two sentences, no plots: the memory-off knockout (study A1) costs only about 0.5 to 1.2 percentage points; and the kernel is inert in most rounds because the candidate set of near-dominant clusters is a singleton in 56 to 84 per cent of rounds, so the target is returned regardless of the memory term. (Source: ${BUILD}/sigma_calibration/sigma_report.md — read it and quote it accurately; do not invent a number.)
 - Remove every remaining claim that treats the memory (or $\\sigma$) as a load-bearing part of the contribution. The contribution is the pairing of failure-mode partitioning with a feasibility-verified prescription.

TASK 3 — FIGURE PLACEMENT. Figure 2 (the DISEIL architecture, Architectural_Diagram.pdf) must appear IMMEDIATELY AFTER THE FIRST PARAGRAPH of the Aim 1 Methodology subsection (4.1.3). Move it there.

Return: the number of algorithm lines, the new equation numbering in this chapter, and how the memory is now framed (quote your sentence).`, {label:'method-04', phase:'Content', effort:'high'}),

  // --- 05_progress.md : delete A12 study + Fig 5 + Fig 19 + Table 9, renumber A's, cut Table 8 ---
  () => agent(`You are the PROGRESS/ABLATIONS agent. Edit ONLY ${V2}/05_progress.md.
${CORE}

TASK 1 — DELETE THE MEMORY-CONSTANT STUDY (now A12) ENTIRELY.
The cluster memory is being recast as a configurable, task-dependent feature and $\\sigma$ is being removed from the method (see the method chapter). A sensitivity sweep of $\\gamma$, $\\sigma$ and $\\lambda$ is therefore orphaned.
 - Delete study A12 (memory constants): its discussion, its FIGURE 19 (F10_memory_constants) and its TABLE 9 (memory-constant sweep).
 - RENUMBER the remaining ablations: A13->A12, A14->A13, A15->A14, A16->A15, A17->A16, A18->A17, A19->A18. Update EVERY mention in text, captions and tables, including figure captions that name a study (e.g. "study A17" for bridging, "study A16" for cluster count).
 - KEEP study A1 (memory off). REFRAME it: it is now the evidence that the memory is a low-impact, task-dependent component (0.5 to 1.2 percentage points when switched off), consistent with the configurable-feature framing. Do not present it as a knockout of a headline contribution.

TASK 2 — DELETE FIGURE 5 (F14_aggregate_significance, "paired margin over the strongest baseline"). The author judges it redundant with Table 7 (final held-out success rate). Remove the figure and its discussion, and make sure the point it made is carried by the Table 7 discussion instead.

TASK 3 — TABLE 8 (per-demonstration information gain): show ONLY Diff-DAgger and DISEIL. Drop the SafeDAgger / DropoutDAgger / EnsembleDAgger / ThriftyDAgger / Stagger columns. Keep the three ablation settings. Every cell on one line (write "2.81±2.09", never "2.81 ± 2.09" — spaces give LaTeX break points and that is what broke the tables before). Adjust the surrounding text so it no longer discusses the dropped columns.

TASK 4 — FIGURE 10 (F2_gain_without_allocation): the author wants only the FIRST panel. Remove the second panel (information gain with clustering removed) from the discussion; the figure itself is being regenerated as a single panel by the figure agent. Ensure the caption and the text describe one panel only.

TASK 5 — Remove any surviving text that treats $\\sigma$ or the memory constants as load-bearing.

Return: the ablation renumbering you applied, and confirmation that Figure 5, Figure 19 and Table 9 are gone.`, {label:'progress-05', phase:'Content', effort:'high'}),

  // --- 00_front.md : abstract ---
  () => agent(`You are the ABSTRACT agent. Edit ONLY ${V2}/00_front.md.
${CORE}
 - The Abstract must fit on ONE page and must contain NO CITATIONS (no [n] markers at all).
 - Keep the bolded DISEIL acronym derivation at first mention: **D**emonstration d**I**stillation for **S**ample-**E**fficient **I**mitation **L**earning.
 - It must still state: the problem (expert demonstrations are the binding cost under a restricted budget), what DISEIL does (perceive failures, partition into modes, prioritise, prescribe a feasibility-verified demonstration), that the learner is any policy with a per-step loss, the scale of the evaluation (five tasks, two observation modalities, against DAgger-family baselines), the headline outcome, and the shape of the programme (three aims).
 - Do NOT promise the cluster memory as a headline contribution (it is being recast as a configurable, task-dependent component).
 - Plain academic prose. No AI vocabulary, no rule-of-three padding.
Return the word count and confirm zero citation markers.`, {label:'abstract-00', phase:'Content', effort:'high'}),

  // --- 01_intro.md + 07_appendix.md : figure sizing/placement ---
  () => agent(`You are the FIGURE PLACEMENT agent. Edit ${V2}/01_intro.md and ${V2}/07_appendix.md.
${CORE}
 - FIGURE 1 (the teaser, Teaser_Diagram_rot.pdf) must be SMALL and must land on PAGE 5 of the built PDF. Reduce its width (pandoc image attributes, e.g. ![](path){width=45%}) and/or wrap it in a raw LaTeX figure float with explicit placement. Then BUILD and CHECK which page it actually lands on with pdftotext/Read, and iterate on the width/placement until it is on page 5. Do not guess; verify.
 - FIGURES A.1, A.2 and A.3 (the three certificates, in the appendix) must be SMALLER, and each figure must sit on the SAME PAGE as its own caption. Wrap each in a raw LaTeX figure float with the caption inside the float (so LaTeX cannot separate them), sized to roughly 0.42\\textheight or a width that keeps figure+caption together. BUILD and VERIFY visually with the Read tool that no certificate is separated from its caption.
 - You may need to coordinate: another agent is deleting Figure 5 and Figure 19, so figure NUMBERS will shift. Do not renumber anything yourself; a later agent does the global renumbering. Refer to figures by their FILE names in any comment you leave.
Return: the page Figure 1 lands on, and confirmation that each certificate sits with its caption.`, {label:'figplace', phase:'Content', effort:'high'}),

  // --- figures: regenerate F2 single-panel; retire F10 and F14 ---
  () => agent(`You are the FIGURE ENGINEER. Edit ${GFIG}/make_figures.py and regenerate.
${CORE}
 - F2_gain_without_allocation.pdf (report Figure 10): REDRAW AS A SINGLE PANEL. Keep the FIRST panel (success rate with clustering removed) and DELETE the second panel (information gain with clustering removed). Keep the grouped-bar style used by F1_allocation_ladder.
 - RETIRE F10_memory_constants.pdf (the memory-constant sweep) — the study is deleted. Remove its generating code.
 - RETIRE F14_aggregate_significance.pdf (the paired-margin figure) — the figure is deleted. Remove its generating code.
 - Keep the existing rules: no orange TEXT, no prose/verdict annotations inside any figure, three ablation settings only (GridWorld image, Push-T state, Door image), no Lift. NOTE: coloured MARKS (lines, markers, bar fills) are fine and must NOT be recoloured — the author has confirmed only orange captions and in-figure text were the concern.
 - Regenerate all figures, then VIEW the regenerated F2 with the Read tool to confirm it is a single panel.
Return the list of figures regenerated and retired.`, {label:'figures', phase:'Content', effort:'high'}),
])

// ===================== PHASE 2: RENUMBER + BUILD =====================
phase('Renumber')
const B = {type:'object', additionalProperties:false, properties:{
  pages:{type:'integer'}, figures:{type:'integer'}, tables:{type:'integer'},
  broken_refs:{type:'integer'}, abstract_pages:{type:'number'}, teaser_page:{type:'integer'},
  built:{type:'boolean'}, summary:{type:'string'}
}, required:['pages','figures','tables','broken_refs','abstract_pages','teaser_page','built','summary']}

const built = await agent(`You are the RENUMBER + BUILD agent. The other agents deleted Figure 5, Figure 19 and Table 9, and renumbered the ablation studies. Now make the document globally consistent and build it.
${CORE}
1. RENUMBER FIGURES continuously across ALL chapter files in ${V2}/: with old Figures 5 and 19 deleted, the remaining figures must run 1..N with no gaps, and every in-text reference ("Figure 12", "see Figure 17", etc.) must be updated to match. The appendix certificates keep the A.1/A.2/A.3 scheme.
2. RENUMBER TABLES continuously: Table 9 (memory constants) is deleted, so tables must run 1..M with no gaps and every reference updated.
3. RENUMBER EQUATIONS continuously: Eq 8a is deleted and 8b simplified, so there must be no 8a/8b split; equations run 1..K with no gaps and every \\tag and every in-text reference updated.
4. Verify the ABLATION renumbering (A1..A18, no A19, no D-series) is consistent in text, captions and tables.
5. Run \`python3 ${BUILD}/assemble.py\` then \`bash ${COC}/build_pdf.sh\` (it hard-fails on Greek loss, pandoc captions, the banned acronym and table overflow).
6. VERIFY IN THE PDF (use pdftotext and the Read tool, not the markdown):
   - the Abstract occupies at most ONE page and contains ZERO citation markers,
   - Figure 1 (teaser) lands on page 5 and is small,
   - Figure 2 (architecture) sits immediately after the first paragraph of the Aim-1 Methodology subsection,
   - the certificates each sit on the same page as their caption,
   - Algorithm 1 renders as a proper algorithm float (ruled caption, numbered lines, bold keywords) and is short,
   - no reference points at a deleted figure/table/equation, and no figure/table number is skipped,
   - the memory is described as a configurable, task-dependent component and no memory-constant plot or table survives.
   Fix anything that fails and rebuild.
Return the structured verdict (broken_refs must be 0).`, {label:'renumber-build', phase:'Renumber', schema:B, effort:'high'})
log(`Build: ${built?.pages}pp, figs=${built?.figures}, tables=${built?.tables}, broken refs=${built?.broken_refs}`)

// ===================== PHASE 3: QA =====================
phase('QA')
const Q = {type:'object', additionalProperties:false, properties:{
  passed:{type:'boolean'}, pages:{type:'integer'},
  items:{type:'array', items:{type:'object', additionalProperties:false, properties:{
    id:{type:'string'}, pass:{type:'boolean'}, evidence:{type:'string'}
  }, required:['id','pass','evidence']}},
  outstanding:{type:'string'}
}, required:['passed','pages','items','outstanding']}

const qa = await agent(`You are the AUDITOR for revision round 2. Verify the built PDF ${COC}/CoC_Report.pdf against the author's ten items. Report pass/fail WITH EVIDENCE (page number, grep count, or what you SAW in the figure/page).
 1. Algorithm 1 is a proper, SHORT algorithm float with atomic steps (open the page and look).
 2. Orange MARKS were explicitly allowed; only orange captions/in-figure TEXT had to go. Confirm no orange text remains; do not flag coloured marks.
 3. The cluster memory is presented as a configurable, task-dependent component; Equation 8a (the Gaussian penalty) is GONE; the memory-constant study, its figure and its table are GONE; no memory-constant plots survive; the memory is not sold as a headline contribution.
 4. Abstract is at most ONE page and contains ZERO citations.
 5. The teaser figure is small and lands on page 5.
 6. The architecture figure sits immediately after the first paragraph of the Aim-1 Methodology subsection.
 7. The old aggregate-significance figure is deleted and its point is carried by the success-rate table.
 8. The per-demonstration information-gain table shows ONLY Diff-DAgger and DISEIL.
 9. The gain-without-allocation figure is a SINGLE panel (no information-gain panel).
 10. The three certificates are smaller and each sits on the same page as its caption.
ALSO check nothing regressed: numbering continuous with no gaps and no broken references; ablations A1..A18 with no D-series; the banned acronym absent; Greek glyphs intact; no table overflowing; DISEIL naming clean.
Fix trivial failures yourself and rebuild. Write ${COC}/SUPERVISOR_REVISION_REPORT.md with the item-by-item table for BOTH rounds and the final document statistics.
Return the structured verdict.`, {label:'audit-r2', phase:'QA', schema:Q, effort:'high'})

return { built, qa }
