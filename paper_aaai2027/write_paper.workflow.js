export const meta = {
  name: 'aaai-paper-writer',
  description: 'Write the AAAI-2027 paper: content -> narrative -> draft, then reviewer<->drafter and AI-checker<->humanizer loops, then citation fact-check and final assembly',
  phases: [
    { title: 'Structure' },
    { title: 'Story' },
    { title: 'Draft' },
    { title: 'ReviewLoop' },
    { title: 'DeAILoop' },
    { title: 'CiteCheck' },
    { title: 'Assemble' },
  ],
}

// ---------------------------------------------------------------------------
// PREREQUISITE context files (must exist in CTX before running this workflow):
//   aaai_format.md            - AAAI-2027 format/style/limits          (from the user)
//   litreview.md              - related-works synthesis + per-paper cards + claim->cite map
//   references.bib            - all bibtex entries                     (external paper-reading agent)
//   dossier_method.md         - P4-LLM internals                      (paper-grounding run)
//   dossier_baselines.md      - baseline decision rules               (paper-grounding run)
//   dossier_experiments.md    - tasks/policies/metrics/protocol       (paper-grounding run)
//   equations.tex             - shared LaTeX equation set + notation   (paper-grounding run)
//   results_placeholders.md   - \PH{} placeholder scheme (results not final yet)
// Run with: Workflow({ scriptPath: '/weka/s226137394/DmNfull/paper_aaai2027/write_paper.workflow.js' })
// ---------------------------------------------------------------------------

const DIR   = '/weka/s226137394/DmNfull/paper_aaai2027'
const CTX   = DIR + '/context'
const SEC   = DIR + '/sections'
const DRAFT = DIR + '/draft/paper.tex'
const BIB   = CTX + '/references.bib'
const KIT   = CTX + '/aaai_author_kit'

const COMMON = `Shared context files you MUST read first:
 - AAAI LaTeX template to BUILD ON: ${KIT}/AnonymousSubmission2027.tex  (style ${KIT}/aaai2027.sty, bib style aaai2027)
 - AAAI format cheat-sheet:        ${CTX}/aaai_format.md
 - Method/baseline/exp dossiers:  ${CTX}/dossier_method.md, ${CTX}/dossier_baselines.md, ${CTX}/dossier_experiments.md
 - Equations (use these symbols): ${CTX}/equations.tex
 - Related work + per-paper cards + claim->cite map: ${CTX}/litreview.md
 - Bibliography (cite keys):      ${CTX}/references.bib
 - Results placeholders:          ${CTX}/results_placeholders.md

METHOD NAME (authoritative): our method is PACE = Perceive -> Assess -> Choose -> Execute.
 Stage mapping: Perceive = VLM perception of rollout failures + failure-descriptor featurization; Assess = clustering the failures into failure modes; Choose = diversity / k-center (coreset) selection of which failures to correct; Execute = prescribe the corrective reset / sub-task-entry scenario, collect the expert demo, retrain.
 NOTE: the context files sometimes still say "P4-LLM" or use old stage names (Partition/Prioritize/Prescribe) -- that is the SAME method. Always write "PACE" and the PACE stage names.

HARD RULES:
 - Use the AAAI-2027 kit UNMODIFIED: \\usepackage[submission]{aaai2027}; natbib \\cite/\\citep/\\citet; \\bibliographystyle{aaai2027}; \\bibliography{references}. NEVER use hyperref/geometry/fullpage/multicol/wrapfig or any forbidden package, and no \\newpage/\\clearpage/\\pagestyle (full list in aaai_format.md).
 - Anonymous submission: author block = "Anonymous Submission", empty affiliations; anonymize self-citations.
 - ~7 pages of body (references + optional reproducibility checklist do NOT count).
 - MATHEMATICAL style: define PACE formally with the notation/labels from equations.tex; include ONE \\begin{algorithm}...\\end{algorithm} (algorithmic) box for the PACE loop; use \\begin{align}/equation environments liberally in the Method section.
 - TABLES (booktabs): at least (1) a task-suite summary table, (2) the main results table(s) from results_placeholders.md, (3) an ablation table -- each with a \\caption and referenced via \\ref.
 - EVERY numeric result MUST be a \\PH{...} placeholder (never invent a number). EVERY \\cite{key} MUST exist in references.bib.
 - The five tasks are exactly: toy 5x5 grid, Push (ManiSkill PushT), Lift (RoboSuite UR5e), Wipe (RoboSuite UR5), Door (RoboSuite UR5). Do NOT mention StackCube/PlugCharger/PickCube.`

const REVIEW_SCHEMA = { type:'object', additionalProperties:false, properties:{
  satisfied:{type:'boolean'},
  overall:{type:'string'},
  issues:{type:'array', items:{type:'object', additionalProperties:false, properties:{
    severity:{type:'string', enum:['critical','major','minor']},
    section:{type:'string'}, problem:{type:'string'}, fix:{type:'string'}
  }, required:['severity','section','problem','fix']}}
}, required:['satisfied','overall','issues']}

const AICHECK_SCHEMA = { type:'object', additionalProperties:false, properties:{
  clean:{type:'boolean'}, ai_score:{type:'number'},
  flagged:{type:'array', items:{type:'object', additionalProperties:false, properties:{
    section:{type:'string'}, excerpt:{type:'string'}, why:{type:'string'}
  }, required:['section','excerpt','why']}}
}, required:['clean','ai_score','flagged']}

const CITE_SCHEMA = { type:'object', additionalProperties:false, properties:{
  all_ok:{type:'boolean'},
  issues:{type:'array', items:{type:'object', additionalProperties:false, properties:{
    type:{type:'string', enum:['missing_bib','unused_bib','wrong_meta','unsupported_claim','orphan_cite','duplicate']},
    location:{type:'string'}, detail:{type:'string'}, fix:{type:'string'}
  }, required:['type','location','detail','fix']}}
}, required:['all_ok','issues']}

// ---------- Structure: 7-page content creator, fanned out by section ----------
phase('Structure')
const SECTIONS = [
  {key:'abstract',    brief:'Abstract + a crisp one-paragraph contributions list.'},
  {key:'intro',       brief:'Introduction: the query-efficiency problem in interactive IL, the gap in DAgger-family querying, the P4-LLM idea, explicit contributions, and a teaser of results (placeholders only).'},
  {key:'related',     brief:'Related Work: adapt the thematic synthesis in litreview.md (IIL/DAgger family; diffusion & generative IL; uncertainty-based querying; LLM/VLM for robot planning & code/reward; LLM/VLM failure reflection & self-correction; active learning coreset/diversity; visual reps; sims & benchmarks). Position P4-LLM against each thread.'},
  {key:'method',      brief:'Method: preliminaries + PACE (Perceive/Assess/Choose/Execute). Present it FORMALLY with the symbols/labels from equations.tex, include a numbered \\\\begin{algorithm} box for the PACE loop, and where possible present the baselines as special cases of one shared querying framework.'},
  {key:'experiments', brief:'Experiments: setup (five tasks T1-T5, policies per modality, baselines, metrics, 5 seeds, active-loop protocol) + main results table and learning-curve/ablation callouts, all with \\PH{} placeholders.'},
  {key:'conclusion',  brief:'Conclusion + limitations + future work.'},
]
await parallel(SECTIONS.map(s => () => agent(
  `You are the 7-PAGE CONTENT CREATOR, writing ONLY the "${s.key}" section of an AAAI-2027 paper.
${COMMON}
Task: ${s.brief}
Be technically precise and complete; prioritize correct content over prose polish (a later agent polishes voice). Reference equations by their labels, cite with \\cite{key} keys that exist in references.bib, and express every quantitative result as a \\PH{...} macro. Respect this section's fair share of the ~7-page body budget.
WRITE the section (LaTeX body, no preamble) to ${SEC}/${s.key}.md. Return a 3-line summary plus your estimated column-inches.`,
  {label:`content:${s.key}`, phase:'Structure'}
)))

// ---------- Story: narrative spine ----------
phase('Story')
await agent(`You are the STORY-CONNECTING agent. Read every file in ${SEC}/ (abstract, intro, related, method, experiments, conclusion) and:
${COMMON}
Produce a narrative spine that makes the paper read as ONE connected argument: the through-line problem -> insight -> method -> evidence; the promises the intro makes that each later section must pay off; and concrete transition sentences to insert between and within sections. Flag any contradiction, any term used before it is defined, and any dangling/unsupported claim.
WRITE to ${CTX}/narrative_spine.md. Return a 5-line summary.`, {label:'story', phase:'Story'})

// ---------- Draft: assemble a natural, research-grade LaTeX draft ----------
phase('Draft')
await agent(`You are the PAPER-DRAFTING agent. Assemble the FULL paper into one polished, natural, research-grade LaTeX file, built on the official AAAI-2027 template.
${COMMON}
Also read the narrative spine: ${CTX}/narrative_spine.md.
Start from ${KIT}/AnonymousSubmission2027.tex: copy its preamble and the title/author scaffolding UNCHANGED (keep \\usepackage[submission]{aaai2027}; author = "Anonymous Submission"), then replace the body with our paper. Merge ${SEC}/*.md into one coherent document: insert the spine's transitions, unify tense/voice/notation, remove cross-section redundancy, make sure every intro promise is paid off, wire in the equations, the PACE \\begin{algorithm} box, the required booktabs tables, and \\cite keys. Keep every \\PH{} placeholder intact. End with \\bibliographystyle{aaai2027} and \\bibliography{references}. Target ~7 body pages.
WRITE the full compilable document to ${DRAFT}. Return a 6-line summary and a self-estimated page count.`, {label:'draft-v1', phase:'Draft'})

// ---------- ReviewLoop: reviewer <-> drafter (iterate) ----------
phase('ReviewLoop')
const MAX_REVIEW = 3
for (let i = 0; i < MAX_REVIEW; i++) {
  const rv = await agent(`You are a TOP-TIER AAAI AREA-CHAIR / REVIEWER. Read the current draft at ${DRAFT}.
${COMMON}
Critique it as for a real AAAI review: method soundness, clarity, novelty framing vs related work, experimental rigor (results are placeholders, so judge the DESIGN and the claims it will support), math correctness/consistency with equations.tex, structure, and whether each stated contribution is actually supported. Be specific and harsh. Do NOT rewrite; only critique.
Return the structured verdict.`, {label:`review-${i+1}`, phase:'ReviewLoop', schema:REVIEW_SCHEMA, effort:'high'})
  const blocking = rv.issues.filter(x => x.severity !== 'minor')
  log(`Review ${i+1}: satisfied=${rv.satisfied}, blocking=${blocking.length}, minor=${rv.issues.length - blocking.length}`)
  if (rv.satisfied || blocking.length === 0) break
  await agent(`You are the PAPER-DRAFTING agent. A reviewer critiqued the draft at ${DRAFT}. Revise it IN PLACE to resolve every issue below, without breaking \\PH{} placeholders or \\cite keys and staying within the AAAI page budget.
${COMMON}
Reviewer overall: ${rv.overall}
Issues (JSON): ${JSON.stringify(rv.issues)}
Read ${DRAFT} (and dossiers as needed), apply fixes, and WRITE the revised full document back to ${DRAFT}. Return a bullet list mapping each issue -> what you changed.`, {label:`revise-${i+1}`, phase:'ReviewLoop'})
}

// ---------- DeAILoop: AI-detector <-> humanizer (iterate) ----------
phase('DeAILoop')
const MAX_AI = 2
for (let i = 0; i < MAX_AI; i++) {
  const ai = await agent(`You are an AI-WRITING DETECTOR. Read ${DRAFT}. Identify passages that read as AI-generated: formulaic hedging, "In this paper we propose", tricolon/list-of-three padding, empty transitions ("Moreover,", "Furthermore,", "Additionally,"), uniform sentence rhythm, vague intensifiers, and over-signposting. Score overall AI-likeness 0-100 (lower = more human). Do NOT rewrite; only flag with exact excerpts and the section they live in.
Return the structured verdict.`, {label:`ai-check-${i+1}`, phase:'DeAILoop', schema:AICHECK_SCHEMA, effort:'high'})
  log(`AI-check ${i+1}: score=${ai.ai_score}, flagged=${ai.flagged.length}`)
  if (ai.clean || ai.flagged.length === 0) break
  await agent(`You are the NON-AI-SOUNDING (humanizing) writer. Rewrite ONLY the flagged passages in ${DRAFT} so they read like a human researcher: vary sentence length, cut filler transitions and hedges, prefer concrete verbs, keep full technical precision, and preserve meaning plus every \\cite / \\PH / equation reference. Do not touch unflagged text.
Flagged (JSON): ${JSON.stringify(ai.flagged)}
Read ${DRAFT}, edit in place, WRITE back to ${DRAFT}. Return a 4-line summary.`, {label:`humanize-${i+1}`, phase:'DeAILoop'})
}

// ---------- CiteCheck: citation fact-check <-> drafter ----------
phase('CiteCheck')
const cite = await agent(`You are the CITATION FACT-CHECK agent. Cross-check the draft ${DRAFT} against the bibliography ${BIB} and the per-paper cards in ${CTX}/litreview.md.
Verify: every \\cite{key} resolves to an entry in references.bib; every bib entry is cited at least once (else flag unused); each cited claim is actually supported by that paper's card (no misattribution); years/authors named in prose match the bib; there are no duplicate entries. Do NOT rewrite prose; report issues with exact fixes.
Return the structured verdict.`, {label:'cite-check', phase:'CiteCheck', schema:CITE_SCHEMA, effort:'high'})
log(`Citation check: all_ok=${cite.all_ok}, issues=${cite.issues.length}`)
if (!cite.all_ok && cite.issues.length) {
  await agent(`You are the PAPER-DRAFTING agent. Fix these citation issues in ${DRAFT} (and in ${BIB} if an entry must be added or de-duplicated). Do not alter technical claims except to correct a misattributed citation.
Issues (JSON): ${JSON.stringify(cite.issues)}
Return a bullet list of the fixes applied.`, {label:'cite-fix', phase:'CiteCheck'})
}

// ---------- Assemble: final consistency + page budget + author to-do ----------
phase('Assemble')
const final = await agent(`You are the FINAL ASSEMBLY agent. Read ${DRAFT}. Do a final consistency pass: estimate page count vs the ~7-page body target (flag over/under and suggest per-section trims/expansions), verify the abstract matches the paper, list every remaining \\PH{} placeholder so the author knows exactly which results to fill, and produce a short author to-do list. Do NOT change technical content.
WRITE the checklist to ${DIR}/FINAL_CHECKLIST.md and return it.`, {label:'assemble', phase:'Assemble', effort:'high'})

return { final }
