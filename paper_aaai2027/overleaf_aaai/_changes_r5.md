# Round-5 changes: the supervisor on the abstract (and the same faults paper-wide). Binding.

He read the built PDF. His verdict on the abstract is that a reader gets lost, and that the writing
sounds like someone talking casually rather than an academic paper. Every point below is verified
against the current source.

## The governing rule he gave
**A reader must never have to go back to a line they have already read.** If a phrase only makes
sense once you re-read an earlier sentence, it is broken. He says we make this mistake repeatedly.

**Second rule: assume no expertise.** The paper may be read by someone from neuroscience, from
management, from any field. Do not assume the reader already knows the area. Terms must be plain or
explained on the spot.

## Abstract: rewrite it (these are his specific objections)
1. **The first sentence is off.** It runs four lines, chained by commas, and reads informally
   ("whose cost does not fall when compute is bought or a simulator is downloaded"). It is not a
   powerful opening. Make it a short, direct, academic statement.
2. **"fixes the other two decisions"** -> he asked: *which* other two? They are named mid-sentence
   between commas, so the reader loses them. Name them plainly, in their own sentence.
3. **"which takes the flag as given"** -> *what flag?* The word "flag" never appears before this line
   (the template sentence says "hand control to the expert at the first crossing"). This is the
   go-back-and-infer fault in its purest form. Either introduce the term first or do not use it.
4. **"decides the other two"** -> same fault again, second occurrence.
5. **Drop the descriptor detail.** No "six-dimensional", no "geometric descriptor" in the abstract.
   It is implementation, not the claim.
6. **The DISEIL sentence is one sentence carrying about seven sentences of content**, running from
   "We present DISEIL..." to "...before any expert time is spent", held together by commas. Break it
   into short, direct sentences.
7. **"the step where the policy first becomes unreliable"** -> the correct notion is **uncertain**,
   not unreliable. And explain it plainly: an episode runs from its start to its end; somewhere in it
   the policy first becomes uncertain; if the episode does not achieve the task it is a failed
   episode. Do not assume the reader knows what any of this means.
8. **"five tasks under two observation modalities"** -> "observation modalities" is jargon a general
   reader will not parse. Write it plainly, e.g. **"across five tasks, with image and state
   observations"**.
9. **"a budget of twenty demonstrations"** -> use the numeral: **"a budget of 20 demonstrations"**
   (the numeral also buys character space).
10. **"against the query-efficient DAgger family and a uniform-random allocation control"** ->
    say **"against the best baselines"**.
11. **STOP UNDERSELLING.** "attains the best **or joint-best** mean held-out success rate" hedges our
    own result because one baseline ties us on one task. That is not how you advertise your method.
    State that DISEIL reaches the **highest** mean success rate in every setting. A tie is still the
    highest, so this is accurate and not an overclaim.
    Where a baseline matches us (Lift), that is handled in ONE line after the table, and it is a win
    for us: DISEIL reaches 100% after the **9th** demonstration on Lift (state) where ThriftyDAgger
    needs the **17th**. Same endpoint, far fewer expert calls. Do not hedge the headline to
    accommodate a tie that we win on efficiency.
12. **"by a mean of 2.80 points"** -> say it as an average and approximate, in percentage points:
    **"on average about 3 percentage points above the strongest baseline"**. Held-out success rate is
    measured in per cent, so "points" alone is imprecise.

## Paper-wide sweep (he says these faults recur, not only in the abstract)
Apply the same two rules to the WHOLE main paper, and to the supplementary where it is cheap:
 - **No undefined forward references.** Sweep for phrases that only resolve by re-reading: "the other
   two", "the flag", "the template", "as given", "the same construction", bare "this"/"that" openings.
   Every referring phrase must be resolvable on first read, in place.
 - **Break the comma chains.** Find sentences that carry several independent clauses joined by commas
   and split them. Target: no sentence should need to be re-read to be parsed.
 - **No unexplained jargon.** "observation modality", "robot-gated", "query gate", "on-policy",
   "covariate shift", "held-out": either use plain words or define on first use, briefly.
 - **"unreliable" -> "uncertain"** wherever it describes the flagged step.
 - **Do not undersell.** Remove defensive hedges around our own result where the data supports the
   plain claim. Keep every honest caveat (the Lift ceiling reading, the small A4/A5 gaps); this is
   about not hedging what we actually won.

## Constraints that still bind (unchanged)
- Content <= 7.0 pages (currently 6.87, ~0.13 headroom), total <= 9. NEVER fix length with layout,
  spacing, margins or font size. aaai2027.sty stays byte-identical to the kit.
- pdflatex only; no literal unicode Greek; anonymous; every \cite key must exist; never invent a number.
- No em-dashes. No hanging words. DISEIL throughout.
- NO statistical testing anywhere (round 4): no p-values, sign test, Wilcoxon, paired t, Friedman,
  Holm, "significance", or collapsed-task-means framing. Do not reintroduce any of it.
- The abstract must not discuss ablations.
- Keep the acronym expansion with the bolded letters at first mention in the abstract.

## Facts the abstract may use
5 tasks; image and state observations; 10 settings; budget of 20 demonstrations; DISEIL has the
highest mean held-out success rate in all 10; average margin over the strongest baseline in each
setting is 2.80 points -> state as "about 3 percentage points"; Lift tie: DISEIL at the 9th
demonstration vs ThriftyDAgger at the 17th.
