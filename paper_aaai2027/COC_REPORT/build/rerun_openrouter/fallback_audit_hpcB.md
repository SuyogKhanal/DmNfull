# Fallback audit — HPC-B (rohan)

Per run: did every round actually use the OpenRouter LLM, or did the deterministic
geometric rule choose the demonstration?

**How each round is classified** (derived from `distil/p4/loop.py`):

| class | meaning |
|---|---|
| `llm_rounds` | LLM called, tokens > 0, prescription issued |
| `budget_free` | round never reached the planner (no usable failures, or the final budget-reached round). The LLM is legitimately not called. **Not** a fallback. |
| `fallback` | **INVALID**: the LLM was not called at all (0 tokens), *or* its output was unparseable so `planner.py:243-248` fell back to `geometric_select`/`geometric_bridge` |
| `escalated` | the LLM *did* prescribe, but the expert could not collect that demo, so DISEIL's designed in-round escalation (`infeasible_attempts=4`) took the nearest-untried failure. **Method-intrinsic, not a fault.** |

Demo provenance is read from the `[collect aN] ... choice=` lines in each run's `run.log`;
it does **not** exist in `result.json` (which records `mode` but never `choice`). A run whose
`run.log` is missing or truncated is failed, not passed — absence of evidence is not evidence.

| task | mod | seed | job | rounds | llm | budget-free | **fallback** | demos | llm-prescribed | escalated | geometric | VLM tok | LLM tok | valid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Lift | state | 1 | 3398450 | 17 | 10 | 7 | **0** | 10 | 10 | 0 | 0 | 14,078 | 50,222 | YES |
| Lift | state | 2 | 3399115 | 15 | 7 | 8 | **0** | 7 | 7 | 0 | 0 | 14,081 | 32,596 | YES |
| Lift | state | 3 | 3399116 | 15 | 7 | 8 | **0** | 7 | 7 | 0 | 0 | 8,698 | 27,430 | YES |
| Lift | state | 4 | 3399117 | 17 | 7 | 10 | **0** | 7 | 7 | 0 | 0 | 8,635 | 22,848 | YES |
| Lift | state | 5 | 3399118 | 8 | 4 | 4 | **0** | 4 | 4 | 0 | 0 | 8,605 | 20,160 | YES |
| Lift | image | 1 | 3399119 | — | — | — | — | — | — | — | — | — | — | **UNRUN** |
| Lift | image | 2 | 3399120 | — | — | — | — | — | — | — | — | — | — | **UNRUN** |
| Lift | image | 3 | 3399121 | — | — | — | — | — | — | — | — | — | — | **UNRUN** |
| Lift | image | 4 | 3399122 | — | — | — | — | — | — | — | — | — | — | **UNRUN** |
| Lift | image | 5 | 3399123 | — | — | — | — | — | — | — | — | — | — | **UNRUN** |
| Wipe | state | 1 | 3399124 | — | — | — | — | — | — | — | — | — | — | **UNRUN** |
| Wipe | state | 2 | 3399125 | — | — | — | — | — | — | — | — | — | — | **UNRUN** |
| Wipe | state | 3 | 3399126 | — | — | — | — | — | — | — | — | — | — | **UNRUN** |
| Wipe | state | 4 | 3399127 | — | — | — | — | — | — | — | — | — | — | **UNRUN** |
| Wipe | state | 5 | 3399128 | — | — | — | — | — | — | — | — | — | — | **UNRUN** |
| Wipe | image | 1 | 3399129 | — | — | — | — | — | — | — | — | — | — | **UNRUN** |
| Wipe | image | 2 | 3399130 | — | — | — | — | — | — | — | — | — | — | **UNRUN** |
| Wipe | image | 3 | 3399131 | — | — | — | — | — | — | — | — | — | — | **UNRUN** |
| Wipe | image | 4 | 3399132 | — | — | — | — | — | — | — | — | — | — | **UNRUN** |
| Wipe | image | 5 | 3399133 | — | — | — | — | — | — | — | — | — | — | **UNRUN** |

**Totals over 5 completed runs:** fallback rounds = **0**, geometric-chosen demos = **0**, escalated demos = 0, VLM tokens = 54,097, LLM tokens = 153,256.

No OpenRouter 429s, rate-limit errors or truncated responses were observed in any run log.

