---
name: non-ai-content
description: >
  Style rules that make generated or edited text avoid the tells of AI writing —
  puffery, "AI-vocabulary" words (delve, tapestry, testament, vibrant, pivotal…),
  negative parallelisms ("not just X, but Y"), the rule of three, copula-avoidance,
  "-ing" tack-on clauses, em-dash / boldface / Title-Case overuse, formulaic
  "challenges and future prospects" conclusions, chatbot filler, and fabricated
  citations. Use when drafting or rewriting academic, technical, or general prose
  that must read as human-written.
source: Adapted from "Wikipedia:Signs of AI writing" (https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)
---

# Non-AI Content — write like a human, not a chatbot

Apply these rules when writing or editing prose. They are prohibitions with fixes,
not suggestions. When a rule conflicts with clarity, keep clarity — but the default
is: **cut the tell, state the fact plainly.**

## How to use
- Load as a skill, or paste at the top of a drafting/editing prompt.
- After drafting, run the **Self-check** at the bottom and fix every hit before returning text.

---

## 1. Kill the "AI vocabulary"
Do **not** reach for these words as default connective tissue. Delete them or replace
with the plain word. They are the single strongest tell.

- **Flagged words:** delve, tapestry, testament, vibrant, pivotal, crucial, key,
  landscape, realm, underscore(s), boasts, bolstered, garner, intricate/intricacies,
  interplay, meticulous, enduring, valuable, seamless, robust, nuanced, multifaceted,
  comprehensive, rich, profound, groundbreaking, renowned, holistic.
- **Flagged connectives:** Additionally, Moreover, Furthermore, Notably, Importantly,
  "It is worth noting that", "It is important to note".
- **Flagged verbs of vague action:** enhance, foster, highlight, showcase, leverage,
  align with, resonate with, exemplify, underpin.

**Fix:** say the concrete thing. "delves into" → "examines"; "plays a pivotal role in"
→ "matters for" / "controls"; "leverages" → "uses"; "Additionally, X" → "X" or "Also, X".

## 2. No puffery, promotion, or editorializing
Do not sell the subject. Drop marketing adjectives and value judgments unless a cited
source makes the claim.
- **Avoid:** "nestled in the heart of", "breathtaking", "state-of-the-art", "cutting-edge",
  "a diverse array of", "stands as a vibrant…", "commitment to excellence".
- **Fix:** report what a thing *is* and *does*; let the reader judge importance.

## 3. No inflated significance / legacy / "broader trends"
Do not append sentences about why something is historically important, symbolic, or
part of a larger movement unless that is the actual point and it is sourced.
- **Avoid:** "stands as a testament to", "underscores the importance of", "left an
  indelible mark", "reflects a broader shift toward", "set the stage for", "a pivotal
  turning point", "an evolving landscape".
- **Fix:** delete the significance sentence, or replace it with one concrete, attributable fact.

## 4. Be specific — no vague attribution or overgeneralization
Do not attribute claims to unnamed authorities or inflate how many sources agree.
- **Avoid:** "Experts argue", "Observers have noted", "Industry reports suggest",
  "Some critics say", "studies show", "it is widely believed", "several sources".
- **Fix:** name the source and cite it (`\cite{key}`), give the number, or cut the claim.
  In an academic paper, every non-obvious claim gets a real citation or is removed.

## 5. Sentence patterns to eliminate
- **Negative parallelism** — do not use the "correcting a misconception" cadence:
  - "not only X, but also Y" · "It's not just X, it's Y" · "not X, but rather Y" ·
    "X rather than Y" (as a rhetorical flourish) · "no A, no B, just C".
  - **Fix:** state Y directly. "It is not just a classifier, but a full framework" →
    "It classifies inputs and also plans actions."
- **Rule of three** — do not default to three-item lists / three stacked adjectives for
  rhythm ("fast, robust, and scalable"; "adjective, adjective, adjective").
  - **Fix:** use the number of items the content actually has (often one or two).
- **Copula avoidance** — do not dodge "is/are" with inflated verbs.
  - "serves as / stands as / functions as / represents / marks" → **"is"**.
  - "boasts / features / offers" → **"has"**.
- **"-ing" tack-on clauses** (superficial analysis) — do not glue a vague participial
  comment onto a sentence: "…, highlighting its significance", "…, reflecting a broader
  trend", "…, fostering community", "…, further enhancing X".
  - **Fix:** delete the clause, or make it a concrete separate sentence with a fact.
- **Elegant variation** — do not rename the same thing every sentence to avoid repetition
  ("the model" → "the framework" → "the system" → "the approach"). Repeat the plain noun.

## 6. Formatting discipline
- **Em dashes:** use sparingly. Prefer commas, periods, or parentheses. Do not use
  em dashes as an all-purpose connector.
- **Boldface:** do not bold terms for emphasis mid-prose. Reserve bold for genuine
  defined terms or real UI labels.
- **Headings:** use sentence case ("Related work"), not Title Case ("Related Work And
  Prior Art"). Don't skip heading levels; don't put a horizontal rule right before a heading.
- **Inline-header bullet lists:** avoid the "**Term**: description" bullet pattern when
  prose or a plain sentence works. Don't convert every paragraph into a bulleted list.
- **Quotes/symbols:** use straight quotes `"` `'` (unless the venue wants curly). No emoji
  as separators or emphasis. Don't use tables where a sentence or short list is clearer.

## 7. No formulaic conclusions
Do not end sections with the stock arc.
- **Avoid:** "In conclusion / Overall / In summary…"; "Despite its promise, X faces
  several challenges."; "With ongoing research, X continues to evolve."; "Challenges and
  Future Directions" filler.
- **Fix:** end on a concrete result, a specific limitation, or the next concrete step.
  If there is nothing to add, stop.

## 8. Remove chatbot / collaborative filler
This is text a human author would never write in a document.
- **Avoid:** "Certainly!", "I hope this helps", "Great question", "As an AI language
  model", "Here is…", "Feel free to…", "Let me know if…", "As of my last update",
  knowledge-cutoff disclaimers, offers to revise, restating the prompt back.
- **Fix:** delete entirely. Deliver the content only.

## 9. Never fabricate facts, citations, or artifacts
- Do not invent citations, DOIs, ISBNs, page numbers, URLs, author lists, dates, or
  statistics. If a value is unknown, say so or leave it out — never guess a plausible one.
- Verify DOIs/links resolve to the intended source; don't cite a book without page/locator
  when a specific claim needs one.
- Strip tool/generation artifacts entirely: `oaicite`, `oai_citation`, `contentReference`,
  `turn0search0`, `:::`, `attribution`, `grok_card`, stray `**`/`*` markdown left in the text,
  "[1m]", placeholder brackets like `[insert X]`.

## 10. What human, non-AI writing looks like (do this)
- Plain verbs, concrete nouns, specific numbers, named sources.
- Uneven rhythm: mix short and long sentences; don't make every paragraph the same shape.
- Claims are either common knowledge, argued from evidence, or cited — never asserted with a vague authority.
- One clear term per concept, repeated, not varied for flourish.
- Say less. Cut any sentence that only adds tone, significance, or hedging.

---

## Self-check (run before returning any prose)
1. Search the draft for every word in §1 and §7 — remove or replace each hit.
2. Any "not just… but", "not X but Y", "X rather than Y"? Rewrite as a direct statement (§5).
3. Any sentence ending in a comma + "-ing" comment? Delete or make it a concrete fact (§5).
4. Count three-item lists and stacked adjectives — trim to the real count (§5).
5. Replace "serves as / boasts / features" with "is / has" (§5).
6. Em dashes, bolded terms, Title-Case headings, "**Term**:" bullets — reduce to necessity (§6).
7. Any conclusion using "In conclusion / Despite its promise / continues to evolve"? Rewrite (§7).
8. Any chatbot filler or knowledge-cutoff/hedging line? Delete (§8).
9. Every citation/number/DOI/link real and verified? No generation artifacts left? (§9)
10. Read one paragraph aloud: does it sound like a person explaining, or a brochure? Fix the brochure.
