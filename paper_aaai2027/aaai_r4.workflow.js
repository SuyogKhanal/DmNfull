export const meta = {
  name: 'aaai-round4-changes',
  description: 'Teaser on page 1, Lift ceiling made legible, Table 2 GridWorld rows dropped, all statistical testing removed from paper+supplementary+CoC, Table 1 dashes; rebuild all three PDFs and audit',
  phases: [ { title: 'Edit' }, { title: 'Build' }, { title: 'Audit' } ],
}

const ROOT = '/weka/s226137394/DmNfull/paper_aaai2027'
const OUT  = ROOT + '/overleaf_aaai'
const COC  = ROOT + '/COC_REPORT'
const CH   = OUT + '/_changes_r4.md'
const TT   = '/home/s226137394/.TinyTeX/bin/x86_64-linux'
const BUILD_PAPER = `cd ${OUT} && ${TT}/pdflatex -interaction=nonstopmode main_paper.tex && ${TT}/bibtex main_paper; ${TT}/pdflatex -interaction=nonstopmode main_paper.tex && ${TT}/pdflatex -interaction=nonstopmode main_paper.tex && ${TT}/pdflatex -interaction=nonstopmode supplementary.tex && ${TT}/bibtex supplementary; ${TT}/pdflatex -interaction=nonstopmode supplementary.tex && ${TT}/pdflatex -interaction=nonstopmode supplementary.tex`
const BUILD_COC = `cd ${COC} && python3 build/assemble.py && bash build_pdf.sh`

const CORE = `BINDING CHANGE LIST: ${CH}. Read it in full first. Also binding: ${OUT}/PAPER_SPEC.md.
Method is DISEIL. Never DISTIL/PACE/P4. Never invent a number.
Main paper: 7 content pages MAX + <= 2 reference pages. NEVER fix length with layout, spacing, margins,
font size, \\vspace or \\setlength. That is a desk reject. Cut prose or move material instead.
pdflatex only (aaai2027.sty refuses XeLaTeX). No literal unicode Greek: use $\\sigma$, not the character.
No em-dashes. No hanging words (a paragraph must not end with one short word alone on a line).
Style: ${COC}/Non-AI content.md.
Build the paper + supplementary with: ${BUILD_PAPER}
The CoC report is GENERATED from ${COC}/build/v2/*.md by ${COC}/build/assemble.py; edit those chapter
files, never CoC_Report.md, then rebuild with: ${BUILD_COC}`

// ======================= EDIT (parallel, disjoint files) =======================
phase('Edit')
await parallel([

  // --- main paper: all five items ---
  () => agent(`You are the PAPER EDITOR. Apply change items 1, 2, 3 and 5, and the main-paper half of item 4, to the AAAI paper.
${CORE}
Files: ${OUT}/sec/*.tex (and ${OUT}/main_paper.tex only if a package is genuinely needed).

ITEM 4 FIRST (it frees the space the teaser needs): delete every statistical claim from
${OUT}/sec/01_abstract_intro.tex (~lines 22-23 and ~90) and ${OUT}/sec/04_experiments.tex (~lines 47,
132-138, 150): p-values, sign test, Wilcoxon, paired t / t(4), "significance", and the whole
"collapse to five task means / the paired difference is positive on every task" framing. The abstract
must lose its p-value sentence completely. What survives is the plain claim the table already shows:
DISEIL holds the best mean in all ten settings. Do NOT substitute a new hedge or a new statistical
claim. Keep the non-statistical caveats (Lift ceiling; A4/A5 small gaps) in plain words.

ITEM 1: insert figures/teaser.png on PAGE 1, immediately after the FIRST paragraph of the
Introduction. Small is preferred. Use the .png (the .pdf variants carry a hidden CamScanner
watermark). Give it a caption that states the takeaway, not a description. After building, OPEN the
rendered PDF and CONFIRM it is on page 1 and reads after the first paragraph; iterate on size and
float placement (not on spacing commands) until it does.

ITEM 2: make the Lift ceiling legible. Author-provided facts, use exactly and do not extrapolate:
Lift (state) ThriftyDAgger reaches 100% only after the 17th demonstration while DISEIL reaches 100%
after the 9th; Lift (image) DISEIL reaches 100% after the 17th. State this where Lift is discussed and
make Table 1 readable on the point (a caption note is acceptable; a new column is NOT, because
demos-to-ceiling is known only for those cells). The point: a tied final number is not a tie in sample
efficiency, because the ceiling hides how fast the budget got there.

ITEM 3: Table 2 compares Diff-DAgger against DISEIL, and Diff-DAgger does not run on GridWorld, so its
GridWorld rows carry no comparison. Remove those rows and fix the caption and any prose that counts
the table's rows.

ITEM 5: replace every "n/a" cell in Table 1 with a plain dash (\`--\`). Keep the caption's explanation
that a dash means the method does not apply (Diff-DAgger on GridWorld; Stagger on the robot tasks).

Then BUILD and report the exact content page count (pages before the References section). It MUST be
<= 7.0. If it is over, cut prose; never adjust layout.
Return: the page count, where the teaser landed, and a bullet per item.`, {label:'paper', phase:'Edit', effort:'high'}),

  // --- supplementary: statistics removal ---
  () => agent(`You are the SUPPLEMENTARY EDITOR. Apply the supplementary half of change item 4.
${CORE}
File: ${OUT}/supplementary.tex (~line 339 mentions Wilcoxon and Friedman, but sweep the WHOLE file).
Remove every statistical test and p-value: sign test, Wilcoxon signed-rank, paired t / t(4), Friedman,
Holm-Bonferroni, "statistically significant"/"significance", and any "collapsed to five task means"
framing. If a section exists only to report statistical testing, delete the section and fix the
surrounding cross-references and section numbering.
Do not replace them with a new statistical claim. The supplementary keeps its ablations, tables,
hyperparameters, prompts and KAG examples; only the testing goes.
The supplementary has no page limit, so nothing else needs to move.
Return a bullet list of what you removed and the new section list.`, {label:'supp', phase:'Edit', effort:'high'}),

  // --- CoC report: statistics removal ---
  () => agent(`You are the CoC EDITOR. Apply the CoC half of change item 4 to the Confirmation of Candidature report.
${CORE}
The author wants the rank tests, statistical tests and p-values gone from the CoC report as well.
Files (edit these, NOT CoC_Report.md): ${COC}/build/v2/05_progress.md (about 11 hits),
${COC}/build/v2/06_plan.md (1), ${COC}/build/v2/00_front.md (1). Sweep all chapters to be sure.
Remove: p-values, sign test, Wilcoxon signed-rank, paired t / t(4), Friedman, Holm-Bonferroni,
"statistically significant"/"significance", and the "collapse to five task means / the paired
difference is positive on every task" framing. If a passage or subsection exists only to report
statistical testing, remove it and repair the surrounding prose, cross-references and numbering so
nothing dangles.
What survives: DISEIL holds the best mean in all ten settings, stated plainly, plus the
non-statistical caveats (the Lift ceiling, the A4/A5 small gaps, the fact that the two modalities of a
task share the expert and reset distribution, which may be said in plain words with no test attached).
Do NOT introduce a new statistical claim and do NOT soften an honest caveat.
Then REBUILD the CoC: ${BUILD_COC} (the build script hard-fails on Greek loss, double captions, the
banned acronym, a scanner watermark and table overflow; it must still pass).
Return: what you removed per file, and the new CoC page count.`, {label:'coc', phase:'Edit', effort:'high'}),
])

// ======================= BUILD =======================
phase('Build')
const B = {type:'object', additionalProperties:false, properties:{
  built:{type:'boolean'}, content_pages:{type:'number'}, total_pages:{type:'integer'},
  supp_pages:{type:'integer'}, coc_pages:{type:'integer'}, teaser_page:{type:'integer'},
  undefined_refs:{type:'integer'}, summary:{type:'string'}
}, required:['built','content_pages','total_pages','supp_pages','coc_pages','teaser_page','undefined_refs','summary']}

const built = await agent(`You are the BUILD engineer. Rebuild everything and measure.
${CORE}
1. Build the paper and supplementary: ${BUILD_PAPER}
2. Confirm the CoC still builds: ${BUILD_COC}
3. MEASURE from the rendered PDFs: main-paper content pages (pages before References), total pages,
   supplementary pages, CoC pages, which page the teaser landed on, undefined refs/citations.
4. ENFORCE: content <= 7.0 pages and total <= 9. If over, report exactly which section exceeds its
   budget in ${OUT}/_plan.md. Do NOT touch layout.
5. Verify no statistical language survived anywhere: grep the three built PDFs' text layers for
   "p =", "p-value", "sign test", "Wilcoxon", "Friedman", "Holm", "significan", "t(4)". Report the
   counts; they must be 0 (except an unrelated ordinary use of the word "significant" in plain prose,
   which you should flag rather than silently allow).
Return the structured verdict.`, {label:'build', phase:'Build', schema:B, effort:'high'})
log(`Build: content=${built?.content_pages}pp total=${built?.total_pages}pp supp=${built?.supp_pages}pp coc=${built?.coc_pages}pp teaser=p${built?.teaser_page}`)

// ======================= AUDIT =======================
phase('Audit')
const A = {type:'object', additionalProperties:false, properties:{
  passed:{type:'boolean'},
  checks:{type:'array', items:{type:'object', additionalProperties:false, properties:{
    id:{type:'string'}, pass:{type:'boolean'}, evidence:{type:'string'}
  }, required:['id','pass','evidence']}},
  outstanding:{type:'string'}
}, required:['passed','checks','outstanding']}

const audit = await agent(`You are the AUDITOR for round 4. Verify every change item against the RENDERED PDFs, with evidence.
${CORE}
Read ${OUT}/main_paper.pdf, ${OUT}/supplementary.pdf and ${COC}/CoC_Report.pdf (the Read tool renders PDFs; open the pages).
Check, each with evidence:
 1. The teaser is on PAGE 1 of the main paper and sits after the first Introduction paragraph.
 2. The Lift ceiling is legible: the body states ThriftyDAgger reaches 100% on Lift (state) only after
    the 17th demonstration while DISEIL reaches it after the 9th, and that DISEIL reaches 100% on Lift
    (image) after the 17th; Table 1 is readable on this point. Numbers exactly as given, nothing
    extrapolated to other methods or settings.
 3. Table 2 has no GridWorld rows and its caption matches.
 4. Table 1 uses dashes, not "n/a", and the caption explains what a dash means.
 5. NO statistical testing survives in ANY of the three documents: zero p-values, sign test, Wilcoxon,
    paired t / t(4), Friedman, Holm, "significance", or "collapsed to five task means" framing.
    Report the grep counts per document.
 6. The abstract has no p-value sentence and still does not discuss ablations.
 7. Content <= 7.0 pages, total <= 9; supplementary is a separate PDF; the CoC still builds.
 8. No layout manipulation was introduced: diff ${OUT}/aaai2027.sty against /tmp/kit27/AuthorKit27/aaai2027.sty
    (must be identical) and grep the sources for \\vspace, \\setlength, \\hspace and body-text \\small.
 9. No em-dashes; DISEIL throughout; no literal unicode Greek; every \\cite resolves.
Fix trivial failures yourself and rebuild. Write ${OUT}/ROUND4_REPORT.md with the check table.
Return the structured verdict.`, {label:'audit', phase:'Audit', schema:A, effort:'high'})

return { built, audit }
