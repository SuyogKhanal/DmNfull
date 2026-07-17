export const meta = {
  name: 'aaai-paper-diseil',
  description: 'Transform the CoC report into a submission-ready AAAI paper: recon, narrative plan, write main+supplementary, compile, then a 7-agent adversarial review loop until no major weaknesses remain',
  phases: [
    { title: 'Recon' },
    { title: 'Plan' },
    { title: 'Write' },
    { title: 'Build' },
    { title: 'Review' },
    { title: 'Final' },
  ],
}

const ROOT = '/weka/s226137394/DmNfull/paper_aaai2027'
const OUT  = ROOT + '/overleaf_aaai'
const COC  = ROOT + '/COC_REPORT'
const V2   = COC + '/build/v2'
const SPEC = OUT + '/PAPER_SPEC.md'
const TT   = '/home/s226137394/.TinyTeX/bin/x86_64-linux'
const BUILD = `cd ${OUT} && ${TT}/pdflatex -interaction=nonstopmode main_paper.tex && ${TT}/bibtex main_paper; ${TT}/pdflatex -interaction=nonstopmode main_paper.tex && ${TT}/pdflatex -interaction=nonstopmode main_paper.tex && ${TT}/pdflatex -interaction=nonstopmode supplementary.tex && ${TT}/bibtex supplementary; ${TT}/pdflatex -interaction=nonstopmode supplementary.tex && ${TT}/pdflatex -interaction=nonstopmode supplementary.tex`

const CORE = `THE BINDING SPEC IS ${SPEC}. Read it in full before anything else. Every rule in it is mandatory.

Non-negotiables you will be audited on:
 - 7 CONTENT PAGES MAX + up to 2 pages of references. The ONLY legal way to fit is to write less.
   NEVER touch margins, columns, spacing, font size, \\vspace, \\setlength, or aaai2027.sty. That is a desk reject.
 - pdflatex ONLY (aaai2027.sty refuses XeLaTeX). NO literal unicode Greek: use $\\sigma$, never the character.
 - Anonymous: \\usepackage[submission]{aaai2027}, author = "Anonymous Submission".
 - Supplementary is a SEPARATE PDF (content appendices count against the page limit).
 - Every \\cite key must exist in ${OUT}/references.bib. NEVER fabricate a citation or a number.
 - Method is DISEIL. Never DISTIL/PACE/P4.
 - Style: ${COC}/Non-AI content.md. No em-dashes. No hanging words (a paragraph must not end with one
   short word alone on a line). No marketing language, no filler, no AI phrasing. The abstract must NOT
   discuss ablations.
 - The paper is Aim 1 (DISEIL) ONLY. No Aim 2/Aim 3, no candidature material.

Working dir: ${OUT} (already staged: aaai2027.sty, aaai2027.bst, references.bib with 102 verified entries,
figures/ with space-free names, latexmkrc). Build both PDFs with:
  ${BUILD}
Primary source (truth): ${V2}/*.md. Secondary (wording only, STALE, says DISTIL): ${ROOT}/draft/paper.tex.
Style exemplars only: ${ROOT}/past_papers_aaai/*.pdf.`

// ============================== RECON ==============================
phase('Recon')
const recon = await parallel([
  () => agent(`You are the CONTENT EXTRACTION analyst. Mine the CoC report for everything the AAAI paper needs.
${CORE}
Read the primary source thoroughly: ${V2}/03_gap_rq.md (gap + research questions), ${V2}/02_background.md (related work), ${V2}/04_aims.md section 4.1 ONLY (problem formulation, DISEIL methodology, Algorithm 1, equations), ${V2}/05_progress.md (setup, results Tables 7 and 8, all ablations A1..A18, the statistics, the limitations).
Produce ${OUT}/_recon_content.md capturing, precisely and with the exact numbers:
 1. The scientific contribution in one paragraph, then as 3 crisp bullets.
 2. The problem, the gap, and the key insight, each in 2-3 sentences.
 3. The method: every equation with its exact LaTeX (Greek in math mode), the notation table, the algorithm, and what each stage does. Flag which parts are standard (BC objective, dataset aggregation, silhouette, A*/BFS, the Diff-DAgger loss rule) versus the novelty, because standard material belongs in Related Work / preliminaries and NOT presented as ours.
 4. Table 7 verbatim (success rate, mean+-SE, Ni, Init SR) and Table 8 verbatim (information gain).
 5. The statistics exactly as the CoC states them (collapsed task-level test: paired t(4)=4.10, p=0.015 two-sided; sign test p=0.031 one-sided; and WHY the ten settings are not ten independent experiments).
 6. Every ablation A1..A18: one line each with its number, what it knocks out, and its result. Mark which 2-3 are strong enough to earn space in a 7-page paper and which go to supplementary.
 7. The honest caveats that must survive into the paper: Lift at ceiling; A4/A5 small gaps; cluster memory as a configurable task-specific component; the confidence figure r=0.82 n=180.
 8. The limitations.
Return a 15-line summary including the 3 contribution bullets.`, {label:'extract', phase:'Recon', effort:'high'}),

  () => agent(`You are the AAAI STYLE analyst. Study what an accepted AAAI paper actually looks like.
${CORE}
Read all four accepted papers in ${ROOT}/past_papers_aaai/ (the Read tool renders PDFs). Do NOT copy any content.
Extract, concretely:
 1. Section structure and the page budget each section gets in a 7-page paper (measure: how much space do intro / related work / method / experiments / ablations / conclusion actually occupy?).
 2. How the first page is used: does the abstract+intro carry a figure? How large? Where does the teaser sit?
 3. How results are presented: table density, how many figures, how captions are written (do they state the takeaway or describe the image?).
 4. Sentence-level register: how claims are hedged, how contributions are stated, how related work is positioned without disparaging.
 5. How ablations are compressed into a main paper and what gets deferred.
 6. Concrete anti-patterns to avoid that these papers never do.
Produce ${OUT}/_recon_style.md with a per-section page budget summing to 7.0, and a short "house style" checklist the writers must follow.
Return the page budget table.`, {label:'style', phase:'Recon', effort:'high'}),
])
log(`Recon done: ${recon.filter(Boolean).length}/2`)

// ============================== PLAN ==============================
phase('Plan')
await agent(`You are AGENT 1, the PRINCIPAL RESEARCH SCIENTIST: a senior researcher with AAAI, NeurIPS, ICML, ICLR and CVPR papers. Decide what this paper IS.
${CORE}
Read ${OUT}/_recon_content.md and ${OUT}/_recon_style.md.
Produce ${OUT}/_plan.md:
 1. THE STORY in eight beats: Problem -> Gap -> Insight -> Method -> Experiments -> Evidence -> Limitations -> Conclusion. One sentence per beat. This is the spine every section must serve.
 2. THE CONTRIBUTION as a reviewer would want it: what is genuinely new here? Be honest. The novelty is the pairing of failure-mode partitioning with a feasibility-verified LLM prescription under a fixed budget, not "we used an LLM". Do not oversell.
 3. SECTION PLAN with a hard page budget summing to <= 7.0, matching the style analysis. For each section: purpose, what it argues, its length, and which figure/table it carries.
 4. FIGURE AND TABLE SELECTION for the main paper: at most what fits. Justify each by the scientific question it answers. Everything else goes to the supplementary. Available assets are listed in the spec.
 5. THE MAIN/SUPPLEMENTARY SPLIT: an explicit list of what stays and what moves, with reasons.
 6. SUPPLEMENTARY OUTLINE.
 7. The 2-3 ablations that earn main-paper space, and why those and not the others.
Be ruthless: 7 pages is the binding constraint and the CoC has 70+ pages of material.
Return the story beats and the page budget.`, {label:'plan', phase:'Plan', effort:'high'})

// ============================== WRITE ==============================
phase('Write')
const SECTIONS = [
  {k:'01_abstract_intro', t:'Abstract + Introduction',
   d:'The abstract states the problem, the method, the setting and the headline result. It must NOT discuss ablations. Bold the DISEIL acronym letters once at first mention. The introduction motivates the fixed-budget problem, states why query-gated DAgger methods fail to answer it, gives the insight, and lists contributions. Carries the architecture or teaser figure per the plan.'},
  {k:'02_related', t:'Related Work',
   d:'Position against interactive imitation learning and the DAgger family, diffusion policies, uncertainty-based querying, LLM/VLM for robotics, and demonstration selection. This is where the STANDARD material lives (BC objective, dataset aggregation, silhouette, A*/BFS, the Diff-DAgger loss rule), so the Method contains only the novelty. Compress hard; cite precisely.'},
  {k:'03_method', t:'Method',
   d:'Problem formulation (any policy f_theta with a per-step loss; budget B and D as symbols, concrete values only in Experiments), then DISEIL: uncertainty flagging at t*, VLM failure perception, KAG-grounded root-cause reasoning, geometric failure-mode partitioning, prescription (targeted vs bridging) verified against the environment model with re-prescription. Include the architecture figure and a SHORT algorithm block with atomic steps. All Greek in math mode. The cluster memory is a configurable task-specific component, not a headline claim.'},
  {k:'04_experiments', t:'Experiments and Results',
   d:'Setup (5 tasks x 2 modalities = 10 settings; policies: CNN/MLP on GridWorld, state and image diffusion policies on the robot tasks; B=20, D=1; 5 seeds robot / 9 GridWorld; Ni and Init SR), baselines described qualitatively as the DAgger family, then the main results table (Table 7 verbatim) and the learning curves. Report the aggregate claim with the CoC statistics EXACTLY (collapsed task-level: paired t(4)=4.10, p=0.015; sign test p=0.031) and state why ten settings are not ten independent experiments. Then the information-gain evidence and the confidence result (r=0.82, n=180). Note the Lift ceiling.'},
  {k:'05_ablation_limits', t:'Ablations, Limitations and Conclusion',
   d:'Only the 2-3 ablations the plan selected, each justifying a design decision, pointing to the supplementary for the rest. Then honest limitations (A4/A5 small gaps, the hand-designed descriptor, the per-round reasoning cost, the Lift ceiling) and a conclusion that closes the story. At most one forward-looking clause; no Aim 2/3 section.'},
]
await parallel(SECTIONS.map(s => () => agent(
`You are AGENT 1, the PRINCIPAL RESEARCH SCIENTIST, writing ONE section of the AAAI paper: "${s.t}".
${CORE}
Read ${OUT}/_plan.md (the story spine and the page budget: OBEY THEM), ${OUT}/_recon_content.md (the facts and exact numbers), ${OUT}/_recon_style.md (house style).
SECTION BRIEF: ${s.d}
Write LaTeX fragments only (no preamble, no \\begin{document}); the assembler wires them together.
 - Cite with \\cite{key} using keys that exist in ${OUT}/references.bib. Never invent one.
 - Figures: \\includegraphics{figures/<name>} with paths relative to the project root; captions state the scientific takeaway, not a description of the image.
 - Every number must match ${OUT}/_recon_content.md exactly.
 - Respect your page budget. Write like a professor, not an assistant: no em-dashes, no hanging words, no filler, no marketing, no generic phrasing. Every sentence carries new information and the section must lead into the next.
WRITE to ${OUT}/sec/${s.k}.tex. Return a 4-line summary and your estimated column-inches.`,
  {label:`write:${s.k}`, phase:'Write', effort:'high'}
)))

await agent(`You are AGENT 1 writing the SUPPLEMENTARY document.
${CORE}
Read ${OUT}/_plan.md (the main/supplementary split) and ${OUT}/_recon_content.md.
Write a COMPLETE standalone ${OUT}/supplementary.tex: same aaai2027 class, \\usepackage[submission]{aaai2027}, title "Supplementary Material: DISEIL: Demonstration Distillation for Sample-Efficient Imitation Learning", author "Anonymous Submission". It has no page limit.
Include everything the plan moved out of the main paper: the complete ablation suite A1..A18 with per-setting tables and figures (assets are in ${OUT}/figures/), hyperparameters, implementation details, representative prompts, representative KAG examples (structured key-value environmental constraints), extended derivations, the compute cost, failure cases, qualitative examples and additional discussion.
It must strengthen the paper without being required to understand the method. Cross-reference the main paper by section name, never by a \\ref into another document.
Return the supplementary's section list.`, {label:'write:supp', phase:'Write', effort:'high'})

// ============================== BUILD ==============================
phase('Build')
const BSCHEMA = {type:'object', additionalProperties:false, properties:{
  built:{type:'boolean'}, content_pages:{type:'number'}, total_pages:{type:'integer'},
  supp_pages:{type:'integer'}, undefined_refs:{type:'integer'}, overfull:{type:'integer'},
  summary:{type:'string'}
}, required:['built','content_pages','total_pages','supp_pages','undefined_refs','overfull','summary']}

const built = await agent(`You are the ASSEMBLY + BUILD engineer.
${CORE}
1. Write ${OUT}/main_paper.tex: start from the kit template /tmp/kit27/AuthorKit27/AnonymousSubmission2027.tex, keep its preamble structure and the \\usepackage[submission]{aaai2027} line, set the title and "Anonymous Submission", then \\input the section fragments from ${OUT}/sec/ in order. End with \\bibliographystyle{aaai2027} and \\bibliography{references}. Add only packages the kit permits (algorithm/algorithmic, booktabs, amsmath, graphicx are fine).
2. BUILD both documents: ${BUILD}
   If a package is missing, install it with ${TT}/tlmgr install <pkg>.
3. MEASURE precisely, from the PDF and the .log:
   - content pages = pages before the References section starts;
   - total pages, supplementary pages, undefined references/citations, overfull hboxes.
4. ENFORCE the limit: content MUST be <= 7.0 pages. If over, do NOT touch layout, spacing, margins or font size. Instead report exactly how much is over and WHICH section is over its budget from ${OUT}/_plan.md. Do not fix it by cutting evidence; report it and let the review loop cut prose.
5. Verify: no literal unicode Greek slipped in; every \\cite resolves; every figure loads.
Return the structured verdict.`, {label:'build', phase:'Build', schema:BSCHEMA, effort:'high'})
log(`Build: content=${built?.content_pages}pp total=${built?.total_pages}pp supp=${built?.supp_pages}pp undef=${built?.undefined_refs}`)

// ============================== REVIEW LOOP ==============================
phase('Review')
const RSCHEMA = {type:'object', additionalProperties:false, properties:{
  verdict:{type:'string', enum:['accept','minor','major','reject']},
  issues:{type:'array', items:{type:'object', additionalProperties:false, properties:{
    severity:{type:'string', enum:['critical','major','minor']},
    location:{type:'string'}, problem:{type:'string'}, fix:{type:'string'}
  }, required:['severity','location','problem','fix']}}
}, required:['verdict','issues']}

const REVIEWERS = [
  {k:'story', n:'AGENT 2, STORY CONSISTENCY REVIEWER', f:`You are a demanding AAAI reviewer seeing this work for the FIRST time. Be brutal; do not accept weak writing. Check logical flow, motivation, transitions, consistency, contribution alignment, the problem statement, the hypothesis and the conclusions. Detect weak arguments, unsupported claims, disconnected paragraphs, inconsistent terminology, missing motivation, and any section that does not lead into the next. Ask of every paragraph: why does this problem matter, why do existing methods fail, why is this idea necessary, why does it work, why should I believe it? If a paragraph does not answer one of those, say so.`},
  {k:'technical', n:'AGENT 3, TECHNICAL VERIFICATION REVIEWER', f:`Verify equations, the algorithm, symbols, notation, assumptions, experiment descriptions and implementation details against the primary source ${V2}/04_aims.md and ${V2}/05_progress.md. Every symbol must be defined before use and used consistently. Check the algorithm is correct and matches the prose. Check that standard machinery is not presented as novel. Check every number against ${OUT}/_recon_content.md. Check the statistics are stated exactly as the CoC states them and are not overclaimed.`},
  {k:'citation', n:'AGENT 4, CITATION VERIFICATION AGENT', f:`This is the most important audit. For EVERY \\cite: verify the key exists in ${OUT}/references.bib; verify the cited work actually supports the claim it is attached to (read the bib entry and, where the claim is specific, check it against ${COC}/build/references_coc.bib and the litreview cards at ${COC}/context/litreview.md if present); verify relevance and placement. Flag unsupported claims, incorrect citations, misleading references and citation mismatches. NEVER accept a fabricated or invented reference. Report any claim that needs a citation and lacks one.`},
  {k:'aiwriting', n:'AGENT 5, AI WRITING DETECTION AGENT', f:`Critique the prose as if detecting AI-generated text, using ${COC}/Non-AI content.md. Identify repetitive sentence structures, predictable wording, excessive transitions, generic expressions, obvious LLM patterns, unnatural paragraph rhythm, robotic phrasing, em-dashes, rule-of-three padding, copula avoidance, and hanging words at paragraph ends. Quote the exact offending text and give a rewrite. The target is prose indistinguishable from an experienced human researcher.`},
  {k:'format', n:'AGENT 6, AAAI FORMATTING AGENT', f:`Verify compliance with the AAAI template: the style file is UNMODIFIED and no layout/spacing/margin/font manipulation has been introduced anywhere (this is a desk-reject check, be thorough: grep for \\vspace, \\setlength, \\hspace, negative spacing, \\small on body text, geometry changes); caption style; algorithm formatting; bibliography formatting; equation formatting; figure and table placement and sizing; anonymity. Read the built PDF pages, do not judge from the source alone.`},
  {k:'submission', n:'AGENT 7, SUBMISSION COMPLIANCE AGENT', f:`Verify: content pages <= 7.0 (count them in the PDF: pages before References); references within 2 pages; supplementary is a separate PDF; anonymous submission compliance; figure quality and legibility at print size; both PDFs compile; no missing references, unresolved citations, missing labels, broken figure links; no overfull boxes that affect appearance; no LaTeX warnings that affect appearance. Reject internally until every issue is fixed. Report the exact page counts you measured.`},
]

let round = 0, clean = false
while (round < 3 && !clean) {
  round += 1
  const revs = await parallel(REVIEWERS.map(r => () => agent(
`You are ${r.n}, reviewing the DISEIL AAAI submission (review round ${round} of at most 3).
${CORE}
Read the BUILT PDFs ${OUT}/main_paper.pdf and ${OUT}/supplementary.pdf (the Read tool renders PDFs; several checks can only be judged in the rendered PDF), and the source in ${OUT}/sec/, ${OUT}/main_paper.tex, ${OUT}/supplementary.tex.
YOUR MANDATE: ${r.f}
Be specific: every issue needs an exact location and a concrete fix. Do NOT edit any file. Severity: critical = would reject over this; major = must fix; minor = polish.
Return the structured verdict.`,
    {label:`${r.k}-r${round}`, phase:'Review', schema:RSCHEMA, effort:'high'}
  )))
  const all = revs.filter(Boolean).flatMap(r => r.issues || [])
  const blocking = all.filter(i => i.severity !== 'minor')
  const verdicts = revs.filter(Boolean).map(r => r.verdict)
  log(`Round ${round}: verdicts=${verdicts.join(',')} | ${all.length} issues (${blocking.length} blocking)`)
  clean = blocking.length === 0
  if (clean) break

  await agent(`You are AGENT 1, the PRINCIPAL RESEARCH SCIENTIST, responding to review round ${round}. Six reviewers audited the paper.
${CORE}
Fix EVERY critical and major issue, and every cheap minor one. Edit the fragments in ${OUT}/sec/, ${OUT}/supplementary.tex and ${OUT}/main_paper.tex with targeted Edit calls.
ISSUES (JSON): ${JSON.stringify(all)}
Rules while fixing:
 - If a reviewer is wrong, leave the text and say why in your return message. Do not make a change that damages correctness to satisfy a reviewer.
 - The page limit is absolute. If a fix adds text, cut text elsewhere. NEVER fix a page overflow with spacing, margins, font size or any layout change: cut prose or move material to the supplementary.
 - Never weaken an honest caveat (the Lift ceiling, the A4/A5 small gaps, the collapsed statistical claim) to make the paper look stronger.
Then REBUILD: ${BUILD}
Confirm content pages <= 7.0 and that no citation or reference broke.
Return a bullet list mapping each issue to the fix applied or the reasoned rebuttal.`, {label:`revise-r${round}`, phase:'Review', effort:'high'})
}

// ============================== FINAL ==============================
phase('Final')
const FSCHEMA = {type:'object', additionalProperties:false, properties:{
  submission_ready:{type:'boolean'}, content_pages:{type:'number'}, total_pages:{type:'integer'},
  supp_pages:{type:'integer'}, figures:{type:'integer'}, tables:{type:'integer'}, citations:{type:'integer'},
  checks:{type:'array', items:{type:'object', additionalProperties:false, properties:{
    id:{type:'string'}, pass:{type:'boolean'}, evidence:{type:'string'}
  }, required:['id','pass','evidence']}},
  outstanding:{type:'string'}
}, required:['submission_ready','content_pages','total_pages','supp_pages','figures','tables','citations','checks','outstanding']}

const gate = await agent(`You are the FINAL SUBMISSION GATE for the DISEIL AAAI paper.
${CORE}
Rebuild from clean (${BUILD}) and verify EVERY item with evidence, reading the rendered PDFs:
 1. content pages <= 7.0 (state the exact count and where References begins); references <= 2 pages.
 2. supplementary is a separate, compiling PDF.
 3. aaai2027.sty is byte-identical to the kit's (diff it against /tmp/kit27/AuthorKit27/aaai2027.sty) and NO layout/spacing/margin/font manipulation exists anywhere in the source.
 4. anonymous submission compliance.
 5. zero undefined citations/references, zero missing figures, no appearance-affecting overfull boxes.
 6. every \\cite key exists in references.bib; no fabricated reference.
 7. no literal unicode Greek in the source; DISEIL used throughout, DISTIL/PACE/P4 absent.
 8. no em-dashes; the abstract does not discuss ablations.
 9. every number matches ${OUT}/_recon_content.md; the honest caveats (Lift ceiling, A4/A5, collapsed statistics) are present and not softened.
 10. figures legible at print size; captions state a takeaway.
Then write ${OUT}/SUBMISSION_REPORT.md with the check table, the page accounting, the main/supplementary split, and anything the author must still decide.
Also write ${OUT}/README.md: how to upload to Overleaf (set compiler to pdfLaTeX; XeLaTeX is refused by the style file), what each file is, and that the sources of truth are the CoC chapters.
Return the structured verdict.`, {label:'final-gate', phase:'Final', schema:FSCHEMA, effort:'high'})

return { built, gate, rounds: round }
