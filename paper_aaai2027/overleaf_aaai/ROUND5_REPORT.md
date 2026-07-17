# Round-5 verification report

Verdict: **PASS**. All twelve abstract points met, the paper-wide comma-chain fault is cleared
(28 -> 0), and no hard constraint regressed.

Built with pdflatex only. `main_paper.pdf`: **9 pages total**, **6.847 content pages**
(References begin on page 7 at 84.7% down the text block), **0 undefined citations/references**.

---

## 1. The twelve abstract points

| # | His point | Pass | Evidence |
|---|---|---|---|
| 1 | First sentence short, direct, academic | YES | "In interactive imitation learning, a learner practises a task and an expert takes over to demonstrate the right actions." = **19 words**, one comma, declarative. The old four-line comma chain ("whose cost does not fall when compute is bought...") is gone. |
| 2 | The two decisions named plainly in their own sentence | YES | "Two decisions are then left to chance: which failed episode to correct, and where the demonstration begins." A standalone sentence naming both. |
| 3 | No term used before it is introduced | YES | "flag" **does not appear** in the abstract (0 occurrences). "failed episode" is now *defined before its first use*: "An episode that runs from start to end without achieving the task is a failed episode." |
| 4 | No second "the other two" style reference | YES | "the other two" = 0 occurrences paper-wide. The single back-reference is "which decides both", now sitting **directly adjacent** to the sentence naming the two decisions. |
| 5 | No "six-dimensional" / "geometric descriptor" in abstract | YES | Both = 0 occurrences in the abstract. |
| 6 | No sentence carrying several sentences of content on commas | YES | Abstract is **12 sentences**; **longest = 25 words**. The old seven-clause DISEIL sentence is split into four. Max commas in any one sentence = 4, and that sentence is a 24-word list ("five tasks, with image and state observations, ten settings in all, at a budget of 20 demonstrations, against the best baselines"), which parses in one pass. |
| 7 | "uncertain" not "unreliable" | YES | "...the first step of an episode where the policy becomes **uncertain**." "unreliable" = 0 in the abstract. Explained plainly on the spot per his instruction: episode runs start to end; policy first becomes uncertain; no task achieved = failed episode. |
| 8 | "image and state observations" not "two observation modalities" | YES | "We evaluate on five tasks, with **image and state observations**, ten settings in all". `modalit*` = **0 occurrences paper-wide**. |
| 9 | The numeral "20" | YES | "at a budget of **20** demonstrations". |
| 10 | "against the best baselines" | YES | Verbatim: "**against the best baselines**." The DAgger-family / uniform-random-control enumeration is gone from the abstract. |
| 11 | Not underselling: "highest", not "best or joint-best" | YES | "DISEIL reaches the **highest** mean success rate in all ten". "joint-best" = 0 occurrences. The Lift tie is handled in one line after the table (9th vs ThriftyDAgger's 17th demonstration), not by hedging the headline. |
| 12 | "about 3 percentage points" | YES | "on average **about 3 percentage points** above the strongest baseline." The literal "2.80" no longer appears in the abstract. |

Also confirmed: **no ablations in the abstract** (`ablation` = 0 occurrences), and the acronym
expansion keeps its bolded letters at first mention.

---

## 2. Paper-wide sweep (the author's priority this round)

Measured over `sec/*.tex` with LaTeX stripped, display environments and headings treated as
sentence boundaries, math and citations reduced to single tokens. **317 sentences.**

### Comma chains
| metric | before | now |
|---|---|---|
| sentences with >=30 words AND >=3 commas | 28 | **0** |
| worst offender | 103 words / 13 commas | **none exists** |

There is no sentence left that meets the threshold, so there is no "three worst" to quote. For
reference, the three **longest** sentences remaining are all single-clause and parse in one pass:

- 35 words / **1 comma** (`04_experiments.tex`): "Every method receives $B = 20$ expert demonstrations at $D = 1$ per round and retrains on a shared cadence $m$, from scratch every round on GridWorld ($m = 1$) and on every fourth acquired demonstration on the robot tasks ($m = 4$)."
- 33 words / **1 comma** (`02_related.tex`): "Labelling every visited state is expensive, so query-efficient variants add a query gate: a test that scores each visited state and hands control to the expert at the first score above a threshold."
- 33 words / **0 commas** (`05_ablation_limits.tex`): "Removing the allocation stack leaves per-demonstration information gain untouched while success falls by a mean of 4.37 points across the three settings ablated: the demonstrations stayed informative and stopped covering the failure distribution."

The most comma-heavy sentence in the whole paper is 24 words with 4 commas (the abstract's
evaluation list). Nothing requires a second pass.

### Undefined forward references: **0**
`the other two`, `the flag`, `the template`, `as given`, `the same construction`: **zero hits**.

Bare `This`/`That` sentence openings: **zero**. Three `This`/`That` openings survive, and all three
carry an explicit noun that names the referent from the immediately preceding sentence, which is the
prescribed fix pattern, not the fault:
- "**That drift** is called covariate shift" (names "errors compound" and defines the term on the spot)
- "**That band** is where a fixed budget can be spent well or badly" (names "a round-zero success rate near 50 per cent")
- "**This knockout** removes the whole allocation stack" (names the knockout just described)

### Unexplained jargon: **0**
Every term on his list is either eliminated or defined in place on first use:

| term | status |
|---|---|
| "observation modality" | **eliminated** (0 hits). Written as "with state observations and with image observations". |
| "robot-gated" | **eliminated** (0 hits). |
| "on-policy" | **eliminated** (0 hits). |
| "covariate shift" | **defined on first use**, Introduction: "The learner's own small errors then carry it into states it never saw, where errors compound. That drift is called covariate shift." |
| "query gate" | **defined on first use**, Related Work: "a query gate: a test that scores each visited state and hands control to the expert at the first score above a threshold." |
| "held-out" | **defined at both sites**: Table 1 caption "on held-out test episodes, meaning start states never trained on"; Setup "the held-out success rate: the fraction of episodes solved on start states never trained on". Not used in the abstract. |
| "unreliable" | 1 hit, in Related Work, describing vision-language models judging distance from pixels. This is **not** the flagged step, so the "unreliable -> uncertain" rule does not apply. |

---

## 3. Regression checks

| check | required | result |
|---|---|---|
| content pages | <= 7.0 | **6.847** |
| total pages | <= 9 | **9** |
| `aaai2027.sty` | byte-identical to `/tmp/kit27/AuthorKit27/aaai2027.sty` | **identical** (md5 `1b1c792e2acfbbd8b672259e267b95d8`) |
| `\vspace` / `\setlength` / `\hspace` | zero | **zero** |
| statistical testing | none | **zero** (p-value, sign test, Wilcoxon, paired t, t(4), Friedman, Holm, "significance") |
| em-dashes | none | **zero** |
| literal unicode Greek | none | **zero** |
| ablations in abstract | none | **zero** |
| acronym bolded at first mention | yes | `\textbf{D}emonstration d\textbf{I}stillation for \textbf{S}ample-\textbf{E}fficient \textbf{I}mitation \textbf{L}earning` |
| DISEIL throughout | no DISTIL/PACE/P4 | **zero** hits; DISEIL x37 |
| undefined citations | 0 | **0** |

Note: the one `\footnotesize` in `04_experiments.tex` scopes a wide results **table**, not body text,
and predates this round. It is standard AAAI table practice and is not a layout fix for length.

---

## 4. Fix applied this round

One trivial rule-1 fault survived the rebuild and was fixed here. The definition sentence sat
*between* the two decisions and the phrase that refers to them, so "which decides both" needed the
reader to hold "two decisions" across an intervening line. The definition now precedes its own first
use, and "both" sits adjacent to what it names.

`sec/01_abstract_intro.tex`, before:
> ... becomes uncertain. **Two decisions are left to chance: which failed episode to correct, and where the demonstration begins. A failed episode runs from start to end without achieving the task.** We present DISEIL (...), which decides both.

after:
> ... becomes uncertain. **An episode that runs from start to end without achieving the task is a failed episode. Two decisions are then left to chance: which failed episode to correct, and where the demonstration begins.** We present DISEIL (...), which decides both.

Rebuilt: content pages unchanged at 6.847 (the added words were absorbed into existing lines).

---

## 5. The final abstract, in full

> In interactive imitation learning, a learner practises a task and an expert takes over to
> demonstrate the right actions. Every takeover costs expert time, which compute cannot reduce.
> Under a fixed budget, what each demonstration contains decides the result. Existing methods decide
> only when to take over. They call the expert at the first step of an episode where the policy
> becomes uncertain. An episode that runs from start to end without achieving the task is a failed
> episode. Two decisions are then left to chance: which failed episode to correct, and where the
> demonstration begins. We present **DISEIL** (**D**emonstration d**I**stillation for
> **S**ample-**E**fficient **I**mitation **L**earning), which decides both. After each round of
> practice, DISEIL groups the failed episodes into modes of failure. For each mode it prescribes
> where a new demonstration should begin, and checks the start is reachable before any expert time
> is spent. We evaluate on five tasks, with image and state observations, ten settings in all, at a
> budget of 20 demonstrations, against the best baselines. DISEIL reaches the highest mean success
> rate in all ten, on average about 3 percentage points above the strongest baseline.

12 sentences. Longest 25 words. No sentence needs a second pass.

---

## 6. Outstanding

Nothing blocking. Two judgement calls left standing, both defensible and neither a round-5 fault:

1. The body still uses the exact figure "2.80 percentage points" in the Introduction contribution
   list and in Experiments. His point 12 governs the **abstract**, where "about 3 percentage points"
   now stands. The body is the right place for the exact number, and the spec forbids inventing or
   re-rounding numbers.
2. `\footnotesize` on the Table 1 float. Standard AAAI table practice, present before this round,
   not used to buy page length.
