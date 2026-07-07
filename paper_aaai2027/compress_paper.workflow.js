export const meta = {
  name: 'pace-paper-compress',
  description: 'Compress the PACE AAAI-2027 draft (~13-15pp) to ~7pp by cutting repeated framing while preserving all math/tables/algorithms/citations/placeholders',
  phases: [ { title: 'Compress' }, { title: 'Verify' } ],
}

const DIR   = '/weka/s226137394/DmNfull/paper_aaai2027'
const CTX   = DIR + '/context'
const DRAFT = DIR + '/draft/paper.tex'
const BIB   = CTX + '/references.bib'

const RULES = `PRESERVE VERBATIM (never delete): every \\begin{equation}/align block, the \\begin{algorithm} box(es), every \\begin{table}/\\begin{figure}, every \\PH{...} result macro, and every \\cite/\\citep/\\citet key, \\label and \\ref. You may ONLY cut or rewrite PROSE. Keep meaning and technical precision. Obey AAAI rules (no forbidden packages, no \\newpage/\\clearpage/\\pagestyle). The method name is PACE (Perceive -> Assess -> Choose -> Execute).`

// ---------- Compress: one section at a time, edit in place (sequential = no write races) ----------
phase('Compress')
const CUTS = [
  {name:'Abstract',      target:'about 200 words',
   notes:'One dense paragraph. Keep the PACE expansion, the which-to-query-vs-where-to-place framing, implemented-vs-planned tasks, and the all-placeholders disclaimer.'},
  {name:'Introduction',  target:'about 1100 words',
   notes:'State the which-vs-where thesis ONCE and briefly, as the contribution. Do NOT develop the formal unifying view here (it belongs in Method). Keep the explicit contributions list. Delete restatements and motivational padding.'},
  {name:'Related Work',  target:'about 750 words',
   notes:'One tight paragraph per theme; keep every \\cite; compress sentence-level paper summaries into clauses.'},
  {name:'Method',        target:'about 1800 words of prose (equations and the algorithm box are separate and stay)',
   notes:'This section OWNS the single canonical statement that the baselines answer WHEN to query while PACE adds WHICH failures and WHERE to place the corrective demo -- state it exactly once. Keep every equation, the algorithm box, and all notation; you may merge only trivially duplicative equations. Move any long T1 grid-world Perceive walkthrough into a new \\section*{Appendix} placed just before the bibliography (an appendix does not count toward the 7-page body).'},
  {name:'Experiments',   target:'about 2200 words of prose (all tables/figures/callouts stay)',
   notes:'Keep ALL tables, figures, and every \\PH result macro and its callouts. Do NOT restate the which-vs-where thesis. Compress the Metrics and Seeds prose; remove the twice-told free-resets-vs-prescribed-where narration. Replace the rhetorical phrase that uses \\PH{placeholder} with plain words such as "a red placeholder macro" -- the \\PH macro is only for real result cells.'},
  {name:'Conclusion',    target:'about 500 words',
   notes:'Tighten limitations and future work; introduce no new claims or numbers.'},
]
for (const c of CUTS) {
  await agent(`You are a precise LaTeX COPY-EDITOR compressing ONE section of an AAAI-2027 paper to hit the page limit.
${RULES}
Open ${DRAFT}. Find the section "${c.name}" (its \\section / \\section* heading, or the abstract environment). Rewrite ONLY that section's prose down to ${c.target}. Cut guidance: ${c.notes}
Edit the file IN PLACE (use Edit); leave the preamble and every other section untouched.
Return: the section name, its approx word count BEFORE and AFTER, and an explicit list of any \\cite / \\PH / \\label you removed (this list should normally be EMPTY; if non-empty, justify each).`, {label:`compress:${c.name}`, phase:'Compress'})
  log(`Compressed ${c.name}`)
}

// ---------- Verify: inventory guard + flow check ----------
phase('Verify')
const INV_SCHEMA = {type:'object', additionalProperties:false, properties:{
  ok:{type:'boolean'},
  counts:{type:'object', additionalProperties:true},
  losses:{type:'array', items:{type:'object', additionalProperties:false, properties:{
    kind:{type:'string'}, identifier:{type:'string'}, note:{type:'string'}
  }, required:['kind','identifier','note']}}
}, required:['ok','counts','losses']}

const FLOW_SCHEMA = {type:'object', additionalProperties:false, properties:{
  reads_ok:{type:'boolean'},
  issues:{type:'array', items:{type:'object', additionalProperties:false, properties:{
    location:{type:'string'}, problem:{type:'string'}, fix:{type:'string'}
  }, required:['location','problem','fix']}}
}, required:['reads_ok','issues']}

const INV = await agent(`You are the INVENTORY-GUARD. Read ${DRAFT} and ${BIB}. Verify the compression lost NOTHING important.
Expected baseline (pre-compression): 137 unique \\PH result macros (one rhetorical \\PH{placeholder} was intentionally removed, so 137 not 138), 52 distinct \\cite keys all resolving in references.bib, 4 tables, 2 algorithm boxes, 2 figures, ~25 equation/align blocks.
Report current counts for each; list ANY \\PH result macro, \\cite key, \\label, table, figure, algorithm, or equation now MISSING vs expected; and any dangling \\ref whose \\label no longer exists.`, {label:'inventory-guard', phase:'Verify', schema:INV_SCHEMA, effort:'high'})
log(`Inventory ok=${INV.ok}, losses=${INV.losses.length}`)

const FLOW = await agent(`You are a careful READER. Read ${DRAFT} end to end after aggressive compression. Check the paper still reads as ONE connected argument: no sentence refers to cut material ("as discussed above", "recall that", a \\ref to a removed item), the abstract still matches the body, transitions are intact, and no passage became choppy/telegraphic or newly robotic from the cuts. Do NOT rewrite; report issues with exact locations and one-line fixes.`, {label:'flow-check', phase:'Verify', schema:FLOW_SCHEMA, effort:'high'})
log(`Flow reads_ok=${FLOW.reads_ok}, issues=${FLOW.issues.length}`)

return { inventory: INV, flow: FLOW }
