export const meta = {
  name: 'distil-rewrite',
  description: 'Rebrand PACE->DISTIL, insert real results/figures/tables, fix framing, compile with tectonic, and trim to <=9 pages with zero overfull boxes',
  phases: [
    { title: 'Figures' },
    { title: 'Rewrite' },
    { title: 'CompileFix' },
    { title: 'Verify' },
    { title: 'Final' },
  ],
}

const DIR   = '/weka/s226137394/DmNfull/paper_aaai2027'
const CTX   = DIR + '/context'
const DRAFT = DIR + '/draft/paper.tex'
const FIGS  = DIR + '/draft/figures'
const TT    = '/home/s226137394/.TinyTeX/bin/x86_64-linux'
const COMPILE = `cd ${DIR}/draft && ${TT}/pdflatex -interaction=nonstopmode paper.tex && ${TT}/bibtex paper && ${TT}/pdflatex -interaction=nonstopmode paper.tex && ${TT}/pdflatex -interaction=nonstopmode paper.tex`
// If pdflatex reports "! LaTeX Error: File \`<name>.sty' not found", install it with:
//   ${TT}/tlmgr install <package>   (package name usually = sty name; algorithms provides algorithm/algorithmic)

const FACTS = `AUTHORITATIVE facts file (read it FIRST, numbers must match it exactly): ${CTX}/results_data.md
Figure manifest (read after Figures phase writes it): ${CTX}/figures_manifest.md
Other context: ${CTX}/dossier_method.md, ${CTX}/dossier_baselines.md, ${CTX}/dossier_experiments.md, ${CTX}/equations.tex, ${CTX}/litreview.md, ${CTX}/references.bib, ${CTX}/aaai_format.md (AAAI rules; forbidden packages/commands).
BANNED STRINGS in the final paper (case-insensitive): "PACE", "P4", "\\PH{", "optional", "optionally", "planned", "upcoming", "pending", "placeholder", "StackCube", "PlugCharger", "PickCube". Also banned: any acronym-style stage branding like "Perceive -> Assess -> Choose -> Execute"; describe the loop functionally instead (uncertainty flagging, VLM failure perception on start/t*/end frames, KAG-grounded root-cause reasoning, failure-mode clustering with cluster memory, LLM demonstration prescription with a feasibility/re-prescribe loop, expert demo collection, policy update).
Do NOT invent statistics that are not in results_data.md (no p-values, no significance tests, no per-seed tests).`

// ---------------- Phase 1: read the actual figures ----------------
phase('Figures')
await agent(`You are the FIGURE-ANALYSIS agent for the DISTIL paper (title: "DISTIL: Demonstration Distillation for Sample-Efficient Imitation Learning").
Read each staged figure with the Read tool (it renders PDFs/PNGs visually): ${FIGS}/teaser.png, ${FIGS}/architecture.pdf, ${FIGS}/comparison_baselines.pdf, ${FIGS}/confidence_vs_success.pdf, ${FIGS}/info_gain_boxplot.pdf, ${FIGS}/clustering_modes_pusht.pdf.
Also read ${CTX}/results_data.md for context.
For EACH figure write to ${CTX}/figures_manifest.md: (1) exactly what it shows (read titles, axis labels, legends OFF the figure — especially the y-axis label/units of info_gain_boxplot.pdf and WHICH three tasks appear in comparison_baselines.pdf); (2) approximate aspect ratio and a sizing recommendation (single-column \\includegraphics[width=\\columnwidth] vs two-column figure* with width=\\textwidth) — architecture.pdf is MANDATED as figure* full-width, never cropped or squashed; teaser.png goes on page 1 in the Introduction; (3) a publication-quality \\caption draft (concise, states the takeaway, mentions task/modality where relevant: confidence_vs_success and info_gain_boxplot are GridWorld 5x5 IMAGE setting; clustering_modes is Push-T IMAGE setting); (4) a \\label suggestion (fig:teaser, fig:arch, fig:comparison, fig:conf, fig:infogain, fig:clusters).
Return a 6-line summary (one per figure).`, {label:'figure-analysis', phase:'Figures'})

// ---------------- Phase 2: the big rewrite ----------------
phase('Rewrite')
await agent(`You are the LEAD REWRITER. Transform ${DRAFT} (currently a paper about "PACE") into the paper "DISTIL: Demonstration Distillation for Sample-Efficient Imitation Learning". Edit the file IN PLACE. This is a major revision; be thorough and globally consistent.
${FACTS}

CHANGES (all mandatory):
1. TITLE + NAME: title above; the method is DISTIL (ours). Purge every PACE mention and any four-stage acronym branding.
2. NARRATIVE SPINE: expert demonstrations are costly, and querying the expert at every rollout miss is wasteful. Under a FIXED demonstration budget (20 demos here), what matters is WHICH failures get corrected and HOW each corrective demo is placed. DISTIL distills the budget into maximally informative corrective demos: it flags uncertain rollouts, perceives failures with a vision LLM (start / high-loss t* / end frames), does KAG-grounded root-cause reasoning, clusters failures into k modes (with cross-round cluster memory), and has an LLM prescribe the corrective demo with a feasibility check and re-prescribe loop; the expert demo is added and the policy retrained. Match ${FIGS}/architecture.pdf.
3. POLICY-AGNOSTIC: the learner is ANY policy f_theta with a per-step training loss; instantiations: GridWorld 5x5 image = pure CNN policy, GridWorld state = MLP policy, robot tasks = diffusion policies (state & image). Rewrite equations that assumed a diffusion policy so the core method uses a generic per-step loss ell_t (BCE for the classifier policies, diffusion loss for diffusion policies) — keep the unified baseline-query-predicate framework (those predicates legitimately reference their own signals; Diff-DAgger's diffusion-loss rule applies on robot tasks only).
4. HYBRID, NOT OPTIONAL: the which-failure selection arm and the bridge/where-demo placement arm are BOTH integral defaults of one hybrid mechanism. The banned-word list enforces this.
5. REAL RESULTS: no placeholders anywhere. Build exactly TWO result tables from results_data.md: Table A (MAIN, two-column table* with booktabs): final SR at the 20-demo budget, all 5 tasks x 2 modalities x 7 methods, mean+-std, DISTIL bold, dashes for Stagger on robot tasks and Diff-DAgger on GridWorld, footnote the seed counts (9 seeds discrete GridWorld, 5 seeds continuous robot tasks); choose orientation (methods as columns vs rows) for best column fit. Table B (secondary, may be single-column or table*): per-demonstration information gain mean+-std from results_data.md Table B; put the confidence-success Pearson correlations (r = 0.82-0.89, per-cell values in results_data.md) in prose or the caption, NOT a third table. DELETE the old ablation table and any Q-to-target metric machinery.
6. FIGURES: wire in all six per ${CTX}/figures_manifest.md — teaser.png early in the Introduction on page 1; architecture.pdf as an uncropped full-width figure*; comparison_baselines.pdf, confidence_vs_success.pdf, info_gain_boxplot.pdf, clustering_modes_pusht.pdf in Experiments/analysis. Every figure referenced from the text via \\ref. Remove references to the deleted stub figures (lc_push_image, qual_push).
7. EXPERIMENTS REWRITE: protocol = 20-demo budget for every task; 9-seeded (GridWorld) / 5-seeded (robot) runs; held-out SR as primary metric; per-demo information gain and confidence-success correlation as analysis. Report and interpret the actual numbers (DISTIL best in all 10 cells; sensible callouts: Push-T state 96.1 vs Diff-DAgger 90.7; Wipe image 95.3 vs 89.6 and +25.7pp over SafeDAgger; Door image 99.2). Include the honest nuance from results_data.md about info gain vs its allocation across failure modes. Do NOT invent statistics.
8. TRIM HARD: previous build was 14 PDF pages; target <= 9 pages TOTAL including references (~7 body + ~2 refs). DELETE both appendices (fold at most 2-3 sentences of the baseline-predicate summary into Method), cut redundant framing, compress Related Work moderately (keep the \\cite coverage — aim to retain at least ~45 of the 52 cite keys; list any keys you drop). Keep the math spine: notation, generic IL objective, unified query predicate, DISTIL's clustering/selection/prescription equations, and the ONE algorithm box (trim the second if redundant).
9. AAAI COMPLIANCE + WIDTH DISCIPLINE: template preamble unchanged; no banned packages/commands; equations must fit a single column (break with align/split; use \\resizebox only for tables if needed); tables must not exceed their column/text width.
Return: a 10-line change summary + the list of dropped cite keys (if any).`, {label:'lead-rewrite', phase:'Rewrite', effort:'high'})

// ---------------- Phase 3: compile-and-fix loop ----------------
phase('CompileFix')
const FIX_SCHEMA = {type:'object', additionalProperties:false, properties:{
  compiled:{type:'boolean'}, pages:{type:'integer'}, overfull_count:{type:'integer'},
  worst_overfull_pt:{type:'number'}, actions:{type:'string'}, done:{type:'boolean'}
}, required:['compiled','pages','overfull_count','worst_overfull_pt','actions','done']}

let clean = false
for (let i = 1; i <= 4 && !clean; i++) {
  const r = await agent(`You are the LATEX BUILD-FIXER (iteration ${i}/4) for the DISTIL paper.
CRITICAL TOOL RULE: modify ${DRAFT} ONLY with targeted Edit calls (exact string replacement). NEVER overwrite the whole file with Write — full-file writes are permission-blocked and will fail.
${FACTS}
1. Compile: run \`${COMPILE}\` with Bash (allow up to 10 minutes). If compilation ERRORS, read the error in the output/paper.log and fix it: for a missing .sty run \`${TT}/tlmgr install <package>\` (the 'algorithms' package provides algorithm/algorithmic; 'newtx'+'fontaxes' provide newtxtext); for LaTeX source errors fix ${DRAFT}; then recompile until it builds.
2. Read ${DIR}/draft/paper.log: extract total page count ("Output written on paper.pdf (N pages") and EVERY "Overfull \\\\hbox"/"Overfull \\\\vbox" with its badness in pt and source line.
3. Fix all overfull boxes > 1pt: break/realign equations (align, split, aligned; shorten with \\! spacing or introduce line breaks at binary relations), narrow tables (abbreviate headers, \\setlength{\\tabcolsep}, \\small or \\footnotesize inside tables, or transpose), never let content cross the column gutter or bleed off the page. Do NOT use forbidden packages/commands (no geometry, no \\addtolength on page dims).
4. Page budget: if total pages > 9, TRIM prose (never numbers/tables/figures/citations) — tighten Experiments and Related Work first — and recompile. If pages <= 8, do NOT pad.
5. Also fix any "Citation ... undefined" or missing-figure warnings.
Iterate compile-fix within your session until: builds cleanly, pages <= 9, zero overfull boxes > 1pt — or you have made your best 3 attempts.
Return the structured verdict (done=true only if all three conditions hold).`, {label:`build-fix-${i}`, phase:'CompileFix', schema:FIX_SCHEMA})
  log(`Build iter ${i}: compiled=${r.compiled} pages=${r.pages} overfull=${r.overfull_count} worst=${r.worst_overfull_pt}pt done=${r.done}`)
  clean = r.done
}

// ---------------- Phase 4: verification panel ----------------
phase('Verify')
const FID_SCHEMA = {type:'object', additionalProperties:false, properties:{
  ok:{type:'boolean'},
  issues:{type:'array', items:{type:'object', additionalProperties:false, properties:{
    kind:{type:'string'}, location:{type:'string'}, detail:{type:'string'}, fix:{type:'string'}
  }, required:['kind','location','detail','fix']}}
}, required:['ok','issues']}
const REV_SCHEMA = {type:'object', additionalProperties:false, properties:{
  satisfied:{type:'boolean'}, overall:{type:'string'},
  issues:{type:'array', items:{type:'object', additionalProperties:false, properties:{
    severity:{type:'string', enum:['critical','major','minor']},
    section:{type:'string'}, problem:{type:'string'}, fix:{type:'string'}
  }, required:['severity','section','problem','fix']}}
}, required:['satisfied','overall','issues']}

const [fid, rev] = await parallel([
  () => agent(`You are the DATA-FIDELITY & COMPLIANCE checker for ${DRAFT}.
${FACTS}
Verify exhaustively and report EVERY violation: (a) every numeric value in tables and prose matches ${CTX}/results_data.md exactly (SR percentages, stds, info-gain values, Pearson r values, budget=20, seeds 9/5); (b) zero occurrences of the banned strings (grep case-insensitively; for "P4" ignore hits inside \\cite keys or the bibliography only); (c) all six figures included with correct files, referenced via \\ref, architecture as figure*; (d) every \\cite key exists in ${CTX}/references.bib; (e) title is exactly "DISTIL: Demonstration Distillation for Sample-Efficient Imitation Learning"; (f) no invented statistics; (g) Stagger appears only for GridWorld, Diff-DAgger only for robot tasks; (h) no forbidden LaTeX packages/commands.
Return the structured verdict.`, {label:'fidelity-check', phase:'Verify', schema:FID_SCHEMA, effort:'high'}),
  () => agent(`You are a top-tier AAAI AREA-CHAIR reviewing the DISTIL paper. Read ${DRAFT} (and ${CTX}/results_data.md for what the data supports).
Critique: does the budget-constrained framing land in the abstract/intro? Is DISTIL's mechanism clear and matched to the architecture figure? Are claims supported by the reported numbers without overreach? Is the writing tight enough for the page limit? Do figures/tables carry the argument? Be specific; do NOT rewrite.
Return the structured verdict.`, {label:'ac-review', phase:'Verify', schema:REV_SCHEMA, effort:'high'}),
])
const fixes = [
  ...(fid && !fid.ok ? fid.issues.map(x => ({src:'fidelity', ...x})) : []),
  ...(rev && !rev.satisfied ? rev.issues.filter(x => x.severity !== 'minor').map(x => ({src:'reviewer', ...x})) : []),
]
log(`Verify: fidelity ok=${fid?.ok} (${fid?.issues?.length ?? 'n/a'} issues), reviewer satisfied=${rev?.satisfied} (${rev?.issues?.length ?? 'n/a'} issues), applying ${fixes.length} fixes`)
if (fixes.length) {
  await agent(`You are the LEAD REWRITER applying verification fixes to ${DRAFT} in place.
CRITICAL TOOL RULE: modify ${DRAFT} ONLY with targeted Edit calls (exact string replacement). NEVER overwrite the whole file with Write — full-file writes are permission-blocked and will fail.
${FACTS}
Fix every issue below exactly; do not alter correct numbers; keep page discipline (if you add text, cut elsewhere). Then recompile with \`${COMPILE}\` and confirm: builds, <= 9 pages, no overfull > 1pt.
Issues (JSON): ${JSON.stringify(fixes)}
Return a bullet list mapping issue -> fix, plus final page count and overfull count from paper.log.`, {label:'apply-fixes', phase:'Verify', effort:'high'})
}

// ---------------- Phase 5: final gate ----------------
phase('Final')
const FINAL_SCHEMA = {type:'object', additionalProperties:false, properties:{
  pages:{type:'integer'}, overfull_count:{type:'integer'}, banned_hits:{type:'integer'},
  figures_ok:{type:'boolean'}, numbers_ok:{type:'boolean'}, summary:{type:'string'}
}, required:['pages','overfull_count','banned_hits','figures_ok','numbers_ok','summary']}
const gate = await agent(`You are the FINAL GATE. Recompile with \`${COMPILE}\`, then verify from scratch: page count from paper.log; overfull boxes > 1pt; banned-string grep of ${DRAFT} (list of banned strings in the facts below); all six figures present in the PDF build (no missing-file warnings); spot-check 10 random numbers in the tables against ${CTX}/results_data.md.
${FACTS}
Write a concise build report to ${DIR}/REPORT.md (pages, overfull, checks passed, anything the author must still do). Return the structured verdict.`, {label:'final-gate', phase:'Final', schema:FINAL_SCHEMA, effort:'high'})
return { gate }
