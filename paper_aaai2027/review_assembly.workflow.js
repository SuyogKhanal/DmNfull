export const meta = {
  name: 'distil-review-assembly',
  description: 'Apply author content fixes + de-AI pass, then run a 3-reviewer assembly vs response agent for 3-4 rounds with memory, full transparency log, and per-round draft snapshots',
  phases: [
    { title: 'Ground' },
    { title: 'Revise' },
    { title: 'ReviewLoop' },
    { title: 'Final' },
  ],
}

const DIR    = '/weka/s226137394/DmNfull/paper_aaai2027'
const CTX    = DIR + '/context'
const DRAFT  = DIR + '/draft/paper.tex'
const LOG    = DIR + '/REVIEW_LOG.md'
const DRAFTS = DIR + '/drafts'
const SCRATCH = '/tmp/claude-497625/-weka-s226137394-DmNfull/9b435ef3-b0d3-428c-abd3-81b374b5a08f/scratchpad'
const TT     = '/home/s226137394/.TinyTeX/bin/x86_64-linux'
const COMPILE = `cd ${DIR}/draft && ${TT}/pdflatex -interaction=nonstopmode paper.tex && ${TT}/bibtex paper; ${TT}/pdflatex -interaction=nonstopmode paper.tex && ${TT}/pdflatex -interaction=nonstopmode paper.tex`

const GUARD = `NON-NEGOTIABLE GUARDRAILS (every agent that touches ${DRAFT}):
 - Modify ${DRAFT} ONLY with targeted Edit calls (exact string replacement). NEVER overwrite the whole file with Write (permission-blocked).
 - Every number must match ${CTX}/results_data.md (plus ${CTX}/kag_ur5_bounds.md for the UR5 KAG bounds). NEVER invent statistics, p-values, sample counts, or hyperparameters.
 - AAAI-2027 compliance per ${CTX}/aaai_format.md: unmodified template preamble, no forbidden packages/commands.
 - After ANY edit session: recompile with \`${COMPILE}\`, confirm <= 9 total pages and 0 overfull boxes > 1pt (check draft/paper.log); fix before finishing.
 - Banned everywhere (word-boundary): PACE, P4 (outside \\cite keys), placeholder, optional/optionally, planned, upcoming, pending, StackCube, PlugCharger, PickCube. Also banned: describing A* or BFS as the GridWorld EXPERT (they are feasibility checkers only; the expert is a human); the string "n=152" (or n = 152) in prose; any claim that per-demo info gain is a "hypothesis".`

const STYLE = `NON-AI PROSE RULES (from the author's style skill; apply strictly):
 - Kill AI vocabulary: delve, tapestry, testament, vibrant, pivotal, crucial, key (as adjective), landscape, realm, underscore, boasts, garner, intricate, interplay, meticulous, seamless, robust, nuanced, multifaceted, comprehensive, rich, profound, groundbreaking, holistic, leverage, foster, highlight, showcase, enhance, "It is worth noting", Moreover/Furthermore/Additionally as default connectives.
 - EM DASHES: the author wants none to very minimal. Replace "---" constructions with commas, periods, parentheses, or restructured sentences. Target: at most 2-3 em dashes in the whole paper.
 - No negative parallelism ("not just X, but Y", "not X, but rather Y") as rhetorical flourish; state the fact directly.
 - No rule-of-three padding; use the real number of items.
 - Copula avoidance: "serves as/stands as/represents/functions as" -> "is"; "boasts/features/offers" -> "has".
 - No vague "-ing" tack-ons ("..., highlighting the importance of..."); delete or replace with a concrete fact.
 - No elegant variation: one term per concept, repeated (the policy, the prescription, the budget).
 - No formulaic conclusions ("Despite its promise... continues to evolve").
 - Uneven sentence rhythm; plain verbs; concrete nouns; specific numbers; claims cited or cut.
 - Keep ALL technical content, math, \\cite keys, table/figure numbers intact while rewriting prose.`

// ============ Phase 1: ground the UR5 KAG bounds from the author's repo ============
phase('Ground')
await agent(`You are the KAG-BOUNDS agent. The paper states the demonstration-prescription perturbation bounds (delta_max / theta_max, i.e. max positional and rotational perturbation for prescribed start states) for Push-T and GridWorld, but not for the three UR5 RoboSuite tasks (Lift, Wipe, Door). Recover the real values.
1. Clone the author's repo: \`git clone --depth 1 https://github.com/SuyogKhanal/diff-dagger-ur5 ${SCRATCH}/diff-dagger-ur5\` (read-only; NEVER push).
2. Search it for KAG ground documents and prescription/perturbation bounds for Lift, Wipe, Door: grep for kag, KAG, perturb, delta_max, theta_max, spawn, workspace, bounds, ResetSpec, and look for json/yaml under kag/, config/, p4*/ directories. Also read the local template ${'/weka/s226137394/DmNfull/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_rl_robo/p4/kag'}/PushT-v1.json for the expected KAG structure and ${'/weka/s226137394/DmNfull/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_rl_robo'}/config.yaml (p4.subtask.perturb_max_xy / perturb_max_theta) for the local bound-naming pattern.
3. For each of Lift, Wipe, Door: if a KAG doc / bound values EXIST in the repo, extract the exact delta_max (xy, metres) and theta_max (radians) values and record the source file path. If a task's KAG doc is MISSING, author one following the PushT-v1.json structure, deriving workspace/spawn bounds from that task's actual config files in the repo (never invent numbers with no source; every value must trace to a repo file); save authored docs BOTH into the cloned repo's kag directory AND to ${CTX}/kag_ur5/<Task>.json so they persist.
4. Write ${CTX}/kag_ur5_bounds.md: a small table (task | delta_max | theta_max | source: found-at-path OR authored-from-paths) plus 2-3 sentences per task describing its KAG content (objects, controller, workspace). State clearly which docs were FOUND vs AUTHORED.
Return the table as your final message.`, {label:'kag-bounds', phase:'Ground', effort:'high'})

// ============ Phase 2: author content fixes, then de-AI pass, then snapshot ============
phase('Revise')
await agent(`You are the CONTENT REVISER applying the author's mandated changes to ${DRAFT}. Read first: ${DRAFT}, ${CTX}/results_data.md, ${CTX}/kag_ur5_bounds.md, ${CTX}/figures_manifest.md.
${GUARD}

APPLY ALL of these (author's instructions, binding):
1. INFO GAIN IS A CLAIM, NOT A HYPOTHESIS. Current text hedges the per-demonstration information gain as a hypothesis. Replace with this argument, stated plainly: each round, before fine-tuning on the newly collected demonstration, we evaluate the current policy's per-step loss on that demonstration (the pre-finetune loss). A high pre-finetune loss can mean only two things: (a) the demonstration covers a region the current policy has not learned, i.e. it is diverse data from an underrepresented region, or (b) the demonstration is suboptimal or invalid. Case (b) is ruled out by construction: every prescription passes the feasibility check before any demonstration is collected, and the demonstrations themselves are produced by the expert. Therefore high pre-finetune loss identifies demonstrations from underrepresented regions, and the per-demonstration information gain measures exactly the diversity a method's chosen demonstrations add. Wire this into the metric definition (Table 2 / info-gain discussion) and the Q2 analysis.
2. GRIDWORLD EXPERT IS A HUMAN. Nowhere may A* or BFS be called the expert. A*/BFS appear ONLY as the feasibility check for prescribed grid layouts (path-validity). The demonstrations on GridWorld were provided by a human expert. Fix every occurrence (grep for A*, A\\textsuperscript, BFS, astar, shortest-path expert).
3. DESCRIBE THE GRIDWORLD. Where the task suite is introduced, describe the 5x5 grid: the agent moves from a start cell to a goal cell with four discrete moves, and the grid contains three obstacles; success is reaching the goal. (Three obstacles; do not carry over any other obstacle count.)
4. CONFIDENCE PASSAGE REWRITE. Delete the sentence explaining n=152 via deterministic-fallback exclusion entirely, and remove "n=152" from prose anywhere (the figure already shows it). Replace the passage with the author's argument: why should the LLM prescription be trusted in each round? The success-rate feedback arrives late: only after the demonstration is collected, the policy retrained, and new rollouts evaluated. The prescription confidence score is reported at prescription time, blind to that future outcome and to the success-rate trajectory. Figure~(conf) shows this blind, self-reported confidence correlates with the realized policy improvement (r between 0.82 and 0.89 across all ten settings), so a confident prescription is evidence, available immediately, that the round will help.
5. STAGGER / DIFF-DAGGER ASYMMETRY. Add a short factual justification where baselines are introduced: Diff-DAgger's query rule is the diffusion policy's own training loss, so it requires a diffusion policy; the GridWorld learners are discrete-action CNN/MLP classifiers, so Diff-DAgger does not apply there, and Stagger (one uniformly chosen corrective demonstration per round) is the matched control on GridWorld. For why Stagger is absent on the robot tasks, give a factual protocol-grounded reason (e.g. Stagger's random pick assumes any visited state can be instantiated as a demonstration start, which holds on the grid but not for contact-rich manipulation states, where Diff-DAgger is the loss-native control) — verify your stated reason against ${CTX}/dossier_baselines.md and do not fabricate experiments.
6. FIX THE DISTILLATION METAPHOR. The paper currently says DISTIL "distills a fixed budget into the most informative corrections". Wrong direction. The correct story everywhere (abstract, teaser caption, intro, method opener): out of the pool of ALL possible corrective demonstrations, DISTIL distills the few most informative ones, the purest corrections, and spends the fixed budget only on those. Update every occurrence of the metaphor.
7. UR5 KAG BOUNDS. Where the prescription bounds (delta_max/theta_max) are given for Push-T and GridWorld, add the corresponding exact values for Lift, Wipe, and Door from ${CTX}/kag_ur5_bounds.md, with the same notation. Keep it to one or two sentences.
8. LIMITATIONS REWRITE. Remove any limitation about missing ablations (ablations go to supplementary material). Replace with real limitations grounded in the method (e.g. dependence on VLM/LLM API quality, per-round wall-clock cost of the reasoning pipeline, KAG authoring effort per task family, single-demonstration-per-round granularity).
9. ONLINE/OFFLINE BRIDGE, LOUD AND CLEAR. Add this as an explicit, prominent claim (intro contribution list AND a short discussion paragraph): DAgger-family methods are online interactive imitation learning (an expert must be on call during training: corrections are collected while the policy rolls out); classical behavior cloning is offline (collect demonstrations, then train). DISTIL bridges the two. Its prescriptions are standing artifacts: the loop can run its selection and bridging analysis on-policy, then pause; the prescribed start configurations remain valid, and the expert can return days later to record the prescribed demonstrations in one batch before the loop resumes. Expert time is decoupled from policy training time. Make this loud but factual.
Then recompile (${COMPILE}); confirm <= 9 pages, 0 overfull. Return a numbered list mapping each item 1-9 to the edit(s) you made.`, {label:'content-revise', phase:'Revise', effort:'high'})

await agent(`You are the HUMAN-PROSE EDITOR. Do a full de-AI pass over ${DRAFT} prose.
${STYLE}
${GUARD}
Method: read the paper section by section. For each section, find every violation of the style rules (AI vocabulary, em dashes "---", negative parallelism, rule-of-three padding, copula avoidance, "-ing" tack-ons, elegant variation, formulaic transitions) and rewrite the sentence with targeted Edit calls. Preserve all math, numbers, \\cite keys, refs, and technical meaning exactly. Vary sentence rhythm; prefer plain verbs. Do not shorten so aggressively that content is lost; this is a style pass, not a trim.
When done: run the self-check (grep the draft for each flagged word and for "---"; count remaining em dashes, target <= 3), recompile (${COMPILE}), confirm <= 9 pages and 0 overfull.
Then snapshot the pre-review draft: \`mkdir -p ${DRAFTS}/round_0_pre_review && cp ${DIR}/draft/paper.tex ${DIR}/draft/paper.pdf ${DRAFTS}/round_0_pre_review/\`.
Return: counts of fixes per rule category, remaining em-dash count, final page count.`, {label:'de-ai-pass', phase:'Revise', effort:'high'})

// ============ Phase 3: reviewer assembly vs response agent, 3-4 rounds with memory ============
phase('ReviewLoop')
const REV_SCHEMA = {type:'object', additionalProperties:false, properties:{
  satisfied:{type:'boolean'},
  score:{type:'number', description:'1-10 accept confidence'},
  review_text:{type:'string', description:'the review as prose, like a real AAAI review (150-400 words)'},
  resolved:{type:'array', items:{type:'object', additionalProperties:false, properties:{
    id:{type:'string'}, comment:{type:'string', description:'explicit acknowledgment of how it was addressed'}
  }, required:['id','comment']}},
  unresolved:{type:'array', items:{type:'object', additionalProperties:false, properties:{
    id:{type:'string'}, reason:{type:'string'}
  }, required:['id','reason']}},
  new_issues:{type:'array', items:{type:'object', additionalProperties:false, properties:{
    id:{type:'string'}, severity:{type:'string', enum:['critical','major','minor']},
    section:{type:'string'}, problem:{type:'string'}, suggested_fix:{type:'string'}
  }, required:['id','severity','section','problem','suggested_fix']}}
}, required:['satisfied','score','review_text','resolved','unresolved','new_issues']}

const REVIEWERS = [
  {key:'R1', name:'Method & soundness', focus:`math and algorithm correctness; whether every claim is backed by the mechanism; logical validity of the info-gain argument (pre-finetune loss -> underrepresented region, given feasibility check + expert-collected demos); whether the online/offline-bridge claim is stated precisely and honestly; notation consistency; whether the unified query-predicate framing is fair to the baselines`},
  {key:'R2', name:'Experiments & evidence', focus:`protocol completeness (budget 20, 9/5 seeds, held-out eval); fairness and clarity of the Stagger/Diff-DAgger asymmetry justification; whether numbers in text match the tables; whether figures support the claims made about them; overclaiming vs the data (means vs per-seed spreads); missing experimental details a reviewer would demand`},
  {key:'R3', name:'Presentation & style', focus:`clarity and flow for a reader who has not seen the code; AAAI format compliance; abstract quality; whether the paper reads as human-written (apply these rules: ${'no AI vocabulary (delve/pivotal/robust/leverage/moreover-furthermore chains), minimal em dashes, no negative parallelism, no rule-of-three padding, no copula avoidance, no -ing tack-ons, no elegant variation, no formulaic conclusions'}); figure/table captions self-contained; section balance within 9 pages`},
]

// per-reviewer memory of open issues, carried across rounds
const open = { R1: [], R2: [], R3: [] }
let round = 0
let allSatisfied = false

while (round < 4 && !(allSatisfied && round >= 3)) {
  round += 1
  const reviews = await parallel(REVIEWERS.map(rv => () => agent(
`You are Reviewer ${rv.key} (${rv.name}) on an AAAI-2027 program committee, reviewing the paper draft at ${DRAFT} (round ${round} of at most 4). You are critical, specific, and fair: praise what is good in one sentence, then dig for real problems in your focus area: ${rv.focus}.
Ground truth for numbers: ${CTX}/results_data.md and ${CTX}/kag_ur5_bounds.md. Review history (read it): ${LOG}.
YOUR OPEN ISSUES FROM PRIOR ROUNDS (verify each against the CURRENT draft): ${JSON.stringify(open[rv.key])}
For each prior issue decide: resolved (say explicitly HOW the revision addressed it, acknowledge good fixes plainly) or unresolved (why the fix is insufficient). Then raise NEW issues you find in the current draft (id format "${rv.key}-r${round}-<n>"). Do not repeat an issue you already marked resolved. Severity honestly: critical = would reject over this; major = must fix; minor = polish.
satisfied=true only if nothing critical/major remains open in your area. Do NOT edit any file.
Return the structured verdict.`,
    {label:`${rv.key}-round${round}`, phase:'ReviewLoop', schema:REV_SCHEMA, effort:'high'}
  )))

  const [r1, r2, r3] = reviews
  const byKey = { R1: r1, R2: r2, R3: r3 }
  for (const k of ['R1','R2','R3']) {
    const r = byKey[k]
    if (!r) continue
    const stillOpen = r.unresolved.map(u => ({...open[k].find(o => o.id === u.id), id: u.id, reason: u.reason}))
    open[k] = [...stillOpen, ...r.new_issues]
  }
  allSatisfied = ['R1','R2','R3'].every(k => byKey[k] && byKey[k].satisfied)
  const totalOpen = open.R1.length + open.R2.length + open.R3.length
  log(`Round ${round}: scores R1=${r1?.score} R2=${r2?.score} R3=${r3?.score}; satisfied=${allSatisfied}; open issues=${totalOpen}`)

  await agent(`You are the AUTHOR RESPONSE agent for round ${round}. Three reviewers just reviewed ${DRAFT}. Their full verdicts (JSON): ${JSON.stringify({R1:r1, R2:r2, R3:r3})}
${GUARD}
${STYLE}
Do, in order:
1. APPEND to ${LOG} a section "## Round ${round}" containing, per reviewer: their score, their review_text verbatim (quoted), their acknowledgments of resolved issues (the 'resolved' comments, so the reader sees the reviewers reacting to earlier fixes), then each unresolved/new issue with its id, and directly under each issue your point-by-point AUTHOR RESPONSE: what you changed (quote the new text briefly) or a reasoned rebuttal if you decline (rebuttal allowed only with evidence from results_data.md / the method dossiers; never rebut to dodge work).
2. FIX the draft: apply edits for every critical and major issue (and minors when cheap). Follow the style rules; keep numbers faithful; keep the online/offline-bridge claim and the author's mandated framings intact (info-gain claim argument, human GridWorld expert, distill-from-the-pool metaphor, three obstacles, no n=152 prose).
3. RECOMPILE (${COMPILE}); ensure <= 9 pages, 0 overfull > 1pt; fix if not.
4. SNAPSHOT: \`mkdir -p ${DRAFTS}/round_${round} && cp ${DIR}/draft/paper.tex ${DIR}/draft/paper.pdf ${DRAFTS}/round_${round}/\` and write ${DRAFTS}/round_${round}/CHANGES.md summarizing what changed vs the previous snapshot (bullet per issue id).
If reviewers were fully satisfied and no critical/major issues remain, steps 2-3 may be minimal (log acknowledgments, verify build) but ALWAYS do steps 1 and 4.
Return: issues fixed / rebutted counts, final page count, overfull count.`, {label:`author-response-${round}`, phase:'ReviewLoop', effort:'high'})
}

// ============ Phase 4: final gate + archive ============
phase('Final')
const GATE_SCHEMA = {type:'object', additionalProperties:false, properties:{
  pages:{type:'integer'}, overfull_count:{type:'integer'}, banned_hits:{type:'integer'},
  em_dashes:{type:'integer'}, rounds_run:{type:'integer'}, all_satisfied:{type:'boolean'}, summary:{type:'string'}
}, required:['pages','overfull_count','banned_hits','em_dashes','rounds_run','all_satisfied','summary']}
const gate = await agent(`You are the FINAL GATE for the DISTIL paper after ${round} review rounds (final satisfaction: ${allSatisfied}).
${GUARD}
1. Recompile (${COMPILE}). Verify: page count <= 9; overfull boxes; word-boundary banned-string grep (list in the guardrails, including A*/BFS-as-expert phrasing and "n=152"/"n = 152" in prose); em-dash ("---") count in ${DRAFT} (target <= 3); all 6 figures load; spot-check 12 numbers across both tables and the analysis prose against ${CTX}/results_data.md; confirm the mandated framings are present (info-gain claim argument, human GridWorld expert + A*/BFS feasibility-only, three obstacles, distill-from-the-pool metaphor, Stagger/Diff-DAgger justification, UR5 KAG bounds, online/offline bridge in both intro and discussion).
2. Archive the final version: \`mkdir -p ${DRAFTS}/final && cp ${DIR}/draft/paper.tex ${DIR}/draft/paper.pdf ${DRAFTS}/final/\`.
3. APPEND to ${LOG} a closing section "## Outcome" with rounds run, final scores trajectory, and what remains open (should be nothing critical/major).
4. Rewrite ${DIR}/REPORT.md as the final build report (pages, overfull, checks, snapshot inventory of ${DRAFTS}).
Return the structured verdict.`, {label:'final-gate', phase:'Final', schema:GATE_SCHEMA, effort:'high'})
return { gate, rounds: round, allSatisfied }
