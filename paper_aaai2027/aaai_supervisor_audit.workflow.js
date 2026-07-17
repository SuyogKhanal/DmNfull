export const meta = {
  name: 'aaai-supervisor-feedback-audit',
  description: 'Audit the DISEIL AAAI paper against the supervisor feedback written on the original DISTIL draft, fix every item that still applies, rebuild and verify',
  phases: [ { title: 'Audit' }, { title: 'Fix' }, { title: 'Verify' } ],
}

const ROOT = '/weka/s226137394/DmNfull/paper_aaai2027'
const OUT  = ROOT + '/overleaf_aaai'
const COC  = ROOT + '/COC_REPORT'
const FB   = COC + '/SUPERVISOR_PAPER_FEEDBACK.txt'
const TT   = '/home/s226137394/.TinyTeX/bin/x86_64-linux'
const BUILD = `cd ${OUT} && ${TT}/pdflatex -interaction=nonstopmode main_paper.tex && ${TT}/bibtex main_paper; ${TT}/pdflatex -interaction=nonstopmode main_paper.tex && ${TT}/pdflatex -interaction=nonstopmode main_paper.tex && ${TT}/pdflatex -interaction=nonstopmode supplementary.tex && ${TT}/bibtex supplementary; ${TT}/pdflatex -interaction=nonstopmode supplementary.tex && ${TT}/pdflatex -interaction=nonstopmode supplementary.tex`

const CORE = `CONTEXT. The supervisor's feedback at ${FB} was written about the FIRST AAAI draft (${ROOT}/draft/paper.pdf, the old "DISTIL" paper). The paper has since been rewritten from the CoC report and is now ${OUT}/main_paper.pdf (method renamed DISEIL). That feedback was NEVER explicitly applied to the new paper: it is the most directly relevant document available, because it is line-level feedback on an AAAI submission of this exact work.

BINDING SPECS: ${OUT}/PAPER_SPEC.md and ${OUT}/_changes_r4.md (round 4 is the LATEST author decision and overrides anything older, including the supervisor's, where they conflict).

HARD CONSTRAINTS (unchanged):
 - Main paper: 7.0 content pages MAX (currently 6.89, so about 0.1 page of headroom), total <= 9.
   NEVER fix length with layout, spacing, margins, font size, \\vspace or \\setlength. Cut prose or move
   material to the supplementary instead. aaai2027.sty must stay byte-identical to the kit.
 - pdflatex only; no literal unicode Greek; anonymous submission; every \\cite key must exist.
 - No em-dashes. No hanging words. Method is DISEIL, never DISTIL/PACE/P4.
 - The abstract must not discuss ablations.
 - NO statistical testing anywhere (round-4 decision): no p-values, sign test, Wilcoxon, paired t,
   Friedman, Holm, "significance", or the collapsed-five-task-means framing. Do NOT reintroduce any of
   these even if the supervisor's feedback implies a statistical claim.
 - Never invent a number; every value traces to the CoC.
Build with: ${BUILD}`

// ============================ AUDIT ============================
phase('Audit')
const ASCHEMA = {type:'object', additionalProperties:false, properties:{
  applicable:{type:'integer'}, satisfied:{type:'integer'}, violated:{type:'integer'}, obsolete:{type:'integer'},
  items:{type:'array', items:{type:'object', additionalProperties:false, properties:{
    id:{type:'string'},
    requirement:{type:'string'},
    status:{type:'string', enum:['satisfied','violated','obsolete','superseded']},
    evidence:{type:'string'},
    fix:{type:'string'}
  }, required:['id','requirement','status','evidence','fix']}}
}, required:['applicable','satisfied','violated','obsolete','items']}

const audit = await agent(`You are the SUPERVISOR-FEEDBACK AUDITOR.
${CORE}
Read ${FB} IN FULL. It is dense, line-level feedback. Enumerate EVERY discrete requirement in it (there are roughly 35), including but not limited to:
 - the five load-bearing principles (general framework not a fixed instance; Method contains only the novelty; abstract everything swappable, e.g. a generic clustering step instantiated as agglomerative; read adversarially; space discipline);
 - abstract: define "setting" before use (a setting = one task under one observation modality; 5 tasks x 2 modalities = 10 settings; N baselines per setting), expand and bold the acronym at first mention, bold only in the abstract and not the title, break the long final sentence, fix the "prescription confidences that predict realized policy improvement" phrasing, do not enumerate every framework component;
 - Figure 1 teaser caption should not foreground "an LLM" if the title does not;
 - introduction: re-expand abbreviations at first use; do not open a paragraph with a bare "This is ..."; do NOT reference the method figure from the introduction; a figure must sit on the page of its first reference or the next; stop repeating the concrete budget; fix the mode/modality collision ("failure modes" vs observation "modality") and sweep the whole paper;
 - related work: expand it; it is the destination for material pulled out of Method;
 - method: separate ours from standard. The BC objective, the per-step loss as an uncertainty signal (Diff-DAgger's idea), dataset aggregation, the generic query-gate template and Diff-DAgger's own query rule are NOT ours and must not sit under Method as if they were. Pull "baseline query rules" out of Method into an Experiments "Baselines" subsection. Drop the baselines' specific hyperparameters. Add a "for completeness we restate the common template" framing. Cite the standard algorithms actually used (A*/BFS, silhouette, agglomerative clustering);
 - algorithm: must read as an algorithm, not a compressed method section; short; atomic steps; symbolic loop header "for r = 1 to B", not the concrete value; the demonstrations-per-round choice stated deliberately;
 - experiments: the research questions should say "budget B" with the concrete value only in Setup; the KAG/workspace-bound equation is GENERAL and one task is only a parameterisation of it, so do not write that the equation "gives the Push-T mechanism"; define the success-rate delta where it first appears (change in success rate on the round-level rollout evaluation; note "round-level", not "ground level");
 - figures/captions/headings: trim over-long captions; no semicolons in headings; remove dangling single words at the end of a paragraph or caption because they waste a whole line;
 - length: 7 pages of content + up to 2 of references, and point to the supplementary for extra experiments.
For EACH requirement, examine the CURRENT paper (read the RENDERED ${OUT}/main_paper.pdf and the source ${OUT}/sec/*.tex, ${OUT}/main_paper.tex) and classify:
 - **satisfied** (already done: give the evidence),
 - **violated** (still applies and the paper breaks it: give the exact location and a concrete fix),
 - **obsolete** (referred to something in the old draft that no longer exists),
 - **superseded** (a later author decision in ${OUT}/_changes_r4.md or PAPER_SPEC.md overrides it: say which).
Be exact and sceptical. Do not mark something satisfied without evidence from the current paper. Do not edit any file.
WRITE ${OUT}/_supervisor_audit.md with the full item table.
Return the structured verdict.`, {label:'audit', phase:'Audit', schema:ASCHEMA, effort:'high'})
log(`Audit: ${audit?.applicable} applicable, ${audit?.satisfied} satisfied, ${audit?.violated} VIOLATED, ${audit?.obsolete} obsolete`)

// ============================ FIX ============================
phase('Fix')
const violated = (audit?.items || []).filter(i => i.status === 'violated')
if (violated.length) {
  await agent(`You are the PRINCIPAL RESEARCH SCIENTIST. Apply the supervisor's outstanding feedback to the AAAI paper.
${CORE}
The auditor found ${violated.length} requirements from the supervisor's feedback that the current paper still violates.
VIOLATED ITEMS (JSON): ${JSON.stringify(violated)}
Fix every one, editing ${OUT}/sec/*.tex (and ${OUT}/supplementary.tex where an item moves material there).
Rules while fixing:
 - The page limit is absolute and there is only ~0.1 page of headroom. Anything you add must be paid for by cutting elsewhere or by moving material to the supplementary. NEVER use a layout, spacing or font change to make room.
 - Do not reintroduce statistical testing (round-4 decision) even if a fix seems to invite it.
 - Do not weaken an honest caveat (the Lift ceiling and its demos-to-ceiling reading, the small A4/A5 gaps) to satisfy a stylistic point.
 - If a fix would damage the paper or contradict a later author decision, skip it and say why in your return message rather than making the paper worse.
 - Preserve the DISEIL naming, the acronym bolding in the abstract, and the teaser on page 1.
Then REBUILD: ${BUILD}
Confirm content pages <= 7.0 and that no citation or reference broke.
Return a bullet list mapping each item to the fix applied, or to a reasoned skip.`, {label:'fix', phase:'Fix', effort:'high'})
} else {
  log('No violated items: nothing to fix.')
}

// ============================ VERIFY ============================
phase('Verify')
const VSCHEMA = {type:'object', additionalProperties:false, properties:{
  passed:{type:'boolean'}, content_pages:{type:'number'}, total_pages:{type:'integer'},
  remaining_violations:{type:'integer'},
  checks:{type:'array', items:{type:'object', additionalProperties:false, properties:{
    id:{type:'string'}, pass:{type:'boolean'}, evidence:{type:'string'}
  }, required:['id','pass','evidence']}},
  outstanding:{type:'string'}
}, required:['passed','content_pages','total_pages','remaining_violations','checks','outstanding']}

const verify = await agent(`You are the VERIFIER. Re-check the supervisor's feedback against the REBUILT paper.
${CORE}
Read ${OUT}/_supervisor_audit.md (the item table) and the rebuilt ${OUT}/main_paper.pdf.
For every item previously marked violated, confirm it is now satisfied, with evidence from the rendered PDF. Report any that remain.
ALSO re-confirm nothing regressed: content pages <= 7.0 and total <= 9; aaai2027.sty byte-identical to /tmp/kit27/AuthorKit27/aaai2027.sty; zero \\vspace/\\setlength/\\hspace/geometry in the sources; no statistical testing anywhere; no em-dashes; DISEIL throughout with no DISTIL leakage; teaser still on page 1; acronym still bolded in the abstract; abstract still free of ablations; 0 undefined citations or references.
Append the verification table to ${OUT}/_supervisor_audit.md.
Return the structured verdict.`, {label:'verify', phase:'Verify', schema:VSCHEMA, effort:'high'})

return { audit, verify }
