export const meta = {
  name: 'aaai-round5-abstract',
  description: 'Rewrite the abstract to the supervisor spec via a 3-draft judge panel, then sweep the whole paper for undefined forward references, comma chains, jargon and underselling; rebuild and verify',
  phases: [ { title: 'Abstract' }, { title: 'Sweep' }, { title: 'Verify' } ],
}

const ROOT = '/weka/s226137394/DmNfull/paper_aaai2027'
const OUT  = ROOT + '/overleaf_aaai'
const COC  = ROOT + '/COC_REPORT'
const CH   = OUT + '/_changes_r5.md'
const TT   = '/home/s226137394/.TinyTeX/bin/x86_64-linux'
const BUILD = `cd ${OUT} && ${TT}/pdflatex -interaction=nonstopmode main_paper.tex && ${TT}/bibtex main_paper; ${TT}/pdflatex -interaction=nonstopmode main_paper.tex && ${TT}/pdflatex -interaction=nonstopmode main_paper.tex`

const CORE = `BINDING: ${CH} (round 5, the supervisor on the abstract). Read it in full first. Also binding, and NOT to be undone: ${OUT}/PAPER_SPEC.md, ${OUT}/_changes_r4.md.

HIS TWO GOVERNING RULES:
 1. A reader must NEVER have to go back to a line already read. Any phrase that only resolves by
    re-reading an earlier sentence is broken.
 2. Assume NO expertise. The reader may come from any field. Plain words, or explain on the spot.

HARD CONSTRAINTS (unchanged, all still enforced):
 - Content <= 7.0 pages (currently 6.87), total <= 9. NEVER use layout, spacing, margins or font size
   to fit. Cut prose or move material instead. aaai2027.sty stays byte-identical to the kit.
 - pdflatex only; no literal unicode Greek; anonymous; every \\cite key must exist; never invent a number.
 - No em-dashes. No hanging words. DISEIL throughout, never DISTIL/PACE/P4.
 - NO statistical testing anywhere: no p-values, sign test, Wilcoxon, paired t, Friedman, Holm,
   "significance", or collapsed-task-means framing. Do not reintroduce any of it.
 - The abstract must not discuss ablations, and must keep the acronym expansion with the bolded
   letters at first mention: \\textbf{D}emonstration d\\textbf{I}stillation for
   \\textbf{S}ample-\\textbf{E}fficient \\textbf{I}mitation \\textbf{L}earning.
 - Style: ${COC}/Non-AI content.md.
Build with: ${BUILD}`

// ===================== ABSTRACT: 3 drafts -> judge -> synthesise =====================
phase('Abstract')
const ANGLES = [
  {k:'A', a:`Open on the COST. The strongest plain fact: an expert demonstration is expensive and its cost does not fall with more compute. Lead with the economic constraint, then the decision it forces.`},
  {k:'B', a:`Open on the DECISION. Existing methods answer only one question, when to interrupt the learner. Two other questions are left to chance. Lead with the missing decision, then who makes it.`},
  {k:'C', a:`Open on the RESULT-SHAPED problem. Under a fixed number of demonstrations, what a demonstration contains decides everything. Lead with the binding budget, then the gap.`},
]
await parallel(ANGLES.map(x => () => agent(
`You are a senior AAAI author rewriting ONE candidate abstract for the DISEIL paper. Draft ${x.k}.
${CORE}
The current abstract is ${OUT}/sec/01_abstract_intro.tex lines 1-20. The supervisor rejected it: the
first sentence runs four lines on commas and sounds casual; "the other two decisions" and "takes the
flag as given" force the reader back; one sentence carries about seven sentences of content; the
descriptor detail does not belong; "observation modalities" is jargon; it undersells the result.

YOUR ANGLE: ${x.a}

REQUIREMENTS (from ${CH}, all mandatory):
 - Short, direct, ACADEMIC sentences. No sentence should need re-reading. Break the comma chains.
 - Nothing may refer backwards to a term not yet introduced. If you use a word like "flag" or
   "template", introduce it in the same breath or do not use it.
 - Name the two decisions PLAINLY, in their own sentence: which failure to correct, and where the
   demonstration begins.
 - No "six-dimensional", no "geometric descriptor", no pipeline enumeration.
 - Use "uncertain", never "unreliable", for the flagged step. If you describe it, describe it plainly:
   an episode runs start to end; somewhere the policy first becomes uncertain; an episode that does
   not achieve the task is a failed episode.
 - "across five tasks, with image and state observations" (NOT "two observation modalities").
 - "a budget of 20 demonstrations" (numeral).
 - "against the best baselines" (NOT the DAgger-family enumeration).
 - DO NOT UNDERSELL: DISEIL reaches the HIGHEST mean success rate in every setting (a tie is still
   highest; do not write "best or joint-best").
 - The margin: "on average about 3 percentage points above the strongest baseline" (the measured
   value is 2.80; state it as an approximate average in percentage points).
 - Keep the bolded acronym expansion at first mention.
 - No ablations. No statistics. No em-dashes. About 150-170 words.
Write ONLY the abstract, as a LaTeX fragment (\\begin{abstract} ... \\end{abstract}).
WRITE it to ${OUT}/_abs_${x.k}.tex and also return the full text.`,
  {label:`abstract-${x.k}`, phase:'Abstract', effort:'high'}
)))

const JUDGE = {type:'object', additionalProperties:false, properties:{
  winner:{type:'string'}, rationale:{type:'string'}, final_abstract:{type:'string'}
}, required:['winner','rationale','final_abstract']}

const judged = await agent(`You are the SUPERVISOR, judging three candidate abstracts by his own standard.
${CORE}
Read ${OUT}/_abs_A.tex, ${OUT}/_abs_B.tex, ${OUT}/_abs_C.tex.
Judge each HARSHLY against his rules, in this order of importance:
 1. Does the reader ever have to go back to a line already read? Any undefined forward reference kills it.
 2. Is the first sentence short, direct and academic, or does it ramble across lines on commas?
 3. Could a reader from another field (neuroscience, management) follow it? Any unexplained jargon?
 4. Does any sentence carry several sentences of content joined by commas?
 5. Does it advertise the result or hedge it? ("best or joint-best" is hedging; "highest" is correct.)
 6. Does it obey: no descriptor detail, no ablations, no statistics, numeral 20, "image and state
    observations", "against the best baselines", "about 3 percentage points", bolded acronym, no em-dashes?
Pick the strongest, then SYNTHESISE the final abstract: take the winner as the base and graft the best
sentences from the other two. The result must beat all three. Keep it to about 150-170 words.
WRITE the final abstract to ${OUT}/_abs_final.tex, then EDIT ${OUT}/sec/01_abstract_intro.tex to
replace lines 1-20 (the \\begin{abstract}...\\end{abstract} block) with it. Leave the Introduction
that follows untouched for now.
Return the winner, the rationale, and the final abstract text.`, {label:'judge', phase:'Abstract', effort:'high'})
log(`Abstract: winner=${judged?.winner}`)

// ===================== SWEEP: measured per-file sentence surgery =====================
phase('Sweep')
const FILES = [
  {f:'03_method.tex',            n:16, note:'WORST FILE. Includes a 103-word/13-comma sentence and an 85-word/7-comma sentence.'},
  {f:'05_ablation_limits.tex',   n:8,  note:'Includes a 55-word/6-comma and a 51-word/4-comma sentence.'},
  {f:'04_experiments.tex',       n:6,  note:'Includes one very long run (check whether it is a table block rather than prose).'},
  {f:'01_abstract_intro.tex',    n:6,  note:'The ABSTRACT here is already rewritten and MUST NOT be touched. Fix only the Introduction below it.'},
  {f:'02_related.tex',           n:4,  note:'Includes a 65-word/5-comma enumeration of the five gated baselines.'},
]
await parallel(FILES.map(x => () => agent(
`You are a senior AAAI author performing SENTENCE SURGERY on ONE file: ${OUT}/sec/${x.f}.
${CORE}

THE AUTHOR'S OWN VERDICT, which is why this pass exists: "there are multiple places in the main paper
where sentence length is so much because of commas, where even I was getting lost. Imagine, me, the
author of the paper, getting lost there." If the author loses the thread, a reviewer stopped reading
earlier. This is the priority fix of round 5.

MEASURED EVIDENCE: ${OUT}/_long_sentences.json lists every flagged sentence per file, with its word and
comma count. Your file has ${x.n} flagged. ${x.note}
Read that JSON, take YOUR file's entries, and rewrite EVERY one of them.

HOW TO FIX (this is prose surgery, not deletion):
 - Split each comma chain into short, direct sentences. One idea per sentence. If a sentence needs a
   second pass to parse, it is still broken.
 - Keep every fact, number, citation, symbol and equation. Nothing may be lost or invented. Where a
   flagged "sentence" is really a display equation with prose wrapped around it, fix the PROSE around
   it and leave the mathematics alone.
 - No undefined forward references: no "the other two", "the flag", "the template", "as given", "the
   former/the latter", and no sentence opening on a bare "This"/"That"/"These" that points at a
   previous clause. Every reference must resolve on first read.
 - No unexplained jargon: the reader may be from any field. Prefer plain words, or define briefly in
   place ("observation modality" -> "image and state observations"; also check "robot-gated",
   "query gate", "on-policy", "covariate shift", "held-out", "roll-out").
 - "unreliable" -> "uncertain" wherever it describes the flagged step.
 - Do not undersell our own result: "best or joint-best" -> "highest". Keep every honest caveat.
 - No em-dashes. No hanging words (no paragraph ending with one short word alone on a line).
 - Do NOT touch the abstract (already rewritten and approved this round).

LENGTH: splitting sentences usually costs a little space and there is only ~0.13 page of headroom
across the whole paper. Aim to be NET NEUTRAL OR SHORTER on your file: shorter sentences let you delete
connective padding, so compress as you split. NEVER use layout, spacing or font changes.

Edit the file in place with targeted Edit calls. Then report, per flagged sentence, its before/after
word and comma counts.
Return: how many you rewrote, the worst before/after pair, and your file's net line change.`,
  {label:`surgery:${x.f}`, phase:'Sweep', effort:'high'}
)))

await agent(`You are the PRINCIPAL RESEARCH SCIENTIST. The per-file surgery is done. Now do the cross-cutting pass and rebuild.
${CORE}
The abstract is already rewritten and the per-file sentence surgery is complete. Now fix what only a
whole-paper view can catch, across ${OUT}/sec/*.tex:

1. UNDEFINED FORWARD REFERENCES (his primary complaint). Sweep for every phrase that only resolves by
   re-reading: "the other two", "the flag", "the template", "as given", "the same construction",
   "the former/the latter", and sentences opening on a bare "This"/"That"/"These" whose referent is a
   previous sentence's clause. Rewrite each so it resolves in place, on first read.
2. COMMA CHAINS. Find sentences carrying several independent clauses joined by commas and split them
   into direct sentences. A reader must not need a second pass to parse a sentence.
3. JARGON WITHOUT EXPLANATION. The reader may be from any field. Either use plain words or define
   briefly on first use: "observation modality" (prefer "image and state observations"),
   "robot-gated", "query gate", "on-policy", "covariate shift", "held-out", "roll-out".
   Do not bloat: a short appositive is enough.
4. "unreliable" -> "uncertain" wherever it describes the flagged step. Where the paper describes the
   flagged step, make the picture plain: an episode runs from start to end; somewhere in it the policy
   first becomes uncertain; an episode that does not achieve the task is a failed episode.
5. UNDERSELLING. Remove defensive hedges around our own result where the data supports the plain
   claim ("best or joint-best" -> "highest"). Ensure the Lift tie is presented as OUR win in one line
   near the table: DISEIL reaches 100% after the 9th demonstration on Lift (state) where ThriftyDAgger
   needs the 17th. Keep every honest caveat (the Lift ceiling reading, the small A4/A5 gaps): this is
   about not hedging what we won, never about hiding what we did not.
6. Keep the concrete budget as the numeral 20 where it is stated, and keep B as the symbol in the
   method and algorithm.
7. RE-MEASURE. After the surgery, re-run the detector over the edited files and report how many
   sentences still have >=30 words AND >=3 commas (it was 28 before this round, worst 103w/13c).
   Rewrite any that survived. The target is zero sentences that need a second pass to parse.
Then REBUILD (${BUILD}) and confirm content <= 7.0 pages. Anything you add must be paid for by cutting
elsewhere. NEVER touch layout.
Return: a count of fixes per category, the surviving long-sentence count, the page count, and any place
you judged a hedge to be an honest caveat that must stay.`, {label:'sweep', phase:'Sweep', effort:'high'})

// ===================== VERIFY =====================
phase('Verify')
const V = {type:'object', additionalProperties:false, properties:{
  passed:{type:'boolean'}, content_pages:{type:'number'}, total_pages:{type:'integer'},
  checks:{type:'array', items:{type:'object', additionalProperties:false, properties:{
    id:{type:'string'}, pass:{type:'boolean'}, evidence:{type:'string'}
  }, required:['id','pass','evidence']}},
  abstract_text:{type:'string'}, outstanding:{type:'string'}
}, required:['passed','content_pages','total_pages','checks','abstract_text','outstanding']}

const v = await agent(`You are the VERIFIER for round 5. Judge the REBUILT paper as the supervisor would.
${CORE}
Read the rendered ${OUT}/main_paper.pdf (page 1 especially) and the sources.
Check each of his twelve abstract points with evidence:
 1. first sentence is short, direct, academic (quote it; give its word count);
 2. the two decisions are named plainly in their own sentence;
 3. no "flag" (or any term) used before it is introduced;
 4. no second "the other two" style reference;
 5. no "six-dimensional" / "geometric descriptor" in the abstract;
 6. no sentence in the abstract carries several sentences of content on commas (report the longest
    sentence's word count);
 7. "uncertain" not "unreliable";
 8. "image and state observations", not "two observation modalities";
 9. the numeral "20";
 10. "against the best baselines";
 11. NOT underselling: "highest", not "best or joint-best";
 12. "about 3 percentage points" phrasing.
Then the paper-wide sweep, which is the AUTHOR'S PRIORITY this round (he reported getting lost in the
main paper's comma-chained sentences, and 28 sentences of >=30 words with >=3 commas were measured,
the worst at 103 words and 13 commas). Re-run that measurement over the rebuilt sources and report:
 - how many sentences still have >=30 words AND >=3 commas, and quote the three worst with counts;
 - how many undefined forward references remain ("the other two", "the flag", "the template", "as
   given", bare "This/That" openings), quoting any;
 - how many unexplained jargon terms remain, quoting any.
The bar: no sentence in the paper should need a second pass to parse.
Then confirm nothing regressed: content <= 7.0, total <= 9, aaai2027.sty byte-identical to
/tmp/kit27/AuthorKit27/aaai2027.sty, zero \\vspace/\\setlength/\\hspace, no statistics, no em-dashes,
no ablations in the abstract, acronym still bolded, DISEIL throughout, 0 undefined citations.
Fix trivial failures yourself and rebuild. Write ${OUT}/ROUND5_REPORT.md with the check table and the
final abstract quoted in full.
Return the structured verdict, including the final abstract text.`, {label:'verify', phase:'Verify', schema:V, effort:'high'})

return { judged, v }
