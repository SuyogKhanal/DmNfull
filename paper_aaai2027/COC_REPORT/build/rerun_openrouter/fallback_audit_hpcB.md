# Fallback audit — HPC-B (rohan)

Per run: did every round actually use the OpenRouter LLM, or did the deterministic geometric
rule choose the demonstration?

| class | meaning |
|---|---|
| `llm` | LLM called, tokens > 0, prescription issued |
| `budget-free` | round never reached the planner (no usable failures, or the final budget-reached round). The LLM is legitimately not called. **Not** a fallback. |
| **`fallback`** | **INVALID**: the LLM was not called at all (0 tokens), *or* its output was unparseable so `planner.py:243-248` fell back to `geometric_select`/`geometric_bridge`. |
| `escalated` | the LLM *did* prescribe, but the expert could not collect that demo, so DISEIL's designed in-round escalation (`infeasible_attempts=4`, `loop.py:161`) took the nearest-untried failure. **Method-intrinsic, not a fault.** |

Demo provenance is read from the `[collect aN] ... choice=` lines of each `run.log`; it does **not**
exist in `result.json`, which records `mode` but never `choice`. A run whose `run.log` is missing or
truncated is failed, not passed — absence of evidence is not evidence of correctness.

| task | mod | seed | job | rounds | llm | budget-free | **fallback** | demos | llm-prescribed | escalated | geometric | VLM tok | LLM tok | valid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Lift | state | 1 | 3398450 | 17 | 10 | 7 | **0** | 10 | 10 | 0 | 0 | 14,078 | 50,222 | YES |
| Lift | state | 2 | 3399115 | 15 | 7 | 8 | **0** | 7 | 7 | 0 | 0 | 14,081 | 32,596 | YES |
| Lift | state | 3 | 3399116 | 15 | 7 | 8 | **0** | 7 | 7 | 0 | 0 | 8,698 | 27,430 | YES |
| Lift | state | 4 | 3399117 | 17 | 7 | 10 | **0** | 7 | 7 | 0 | 0 | 8,635 | 22,848 | YES |
| Lift | state | 5 | 3399118 | 8 | 4 | 4 | **0** | 4 | 4 | 0 | 0 | 8,605 | 20,160 | YES |
| Lift | image | 1 | 3399119 | 22 | 20 | 2 | **0** | 20 | 20 | 0 | 0 | 55,310 | 117,116 | YES |
| Lift | image | 2 | 3399120 | 21 | 20 | 1 | **0** | 20 | 20 | 0 | 0 | 63,938 | 130,893 | YES |
| Lift | image | 3 | 3399121 | 21 | 20 | 1 | **0** | 20 | 20 | 0 | 0 | 61,865 | 122,532 | YES |
| Lift | image | 4 | 3399122 | 21 | 20 | 1 | **0** | 20 | 20 | 0 | 0 | 57,584 | 140,597 | YES |
| Lift | image | 5 | 3399123 | 22 | 21 | 1 | **0** | 20 | 20 | 0 | 0 | 49,875 | 121,216 | YES |
| Wipe | state | 1 | 3399124 | 21 | 20 | 1 | **0** | 20 | 20 | 0 | 0 | 58,384 | 135,220 | YES |
| Wipe | state | 2 | 3399125 | 22 | 21 | 1 | **0** | 20 | 19 | 1 | 0 | 57,325 | 141,166 | YES |
| Wipe | state | 3 | 3399126 | 21 | 20 | 1 | **0** | 20 | 20 | 0 | 0 | 55,323 | 131,330 | YES |
| Wipe | state | 4 | 3399127 | 21 | 20 | 1 | **0** | 20 | 20 | 0 | 0 | 61,724 | 140,983 | YES |
| Wipe | state | 5 | 3399128 | 21 | 20 | 1 | **0** | 20 | 19 | 1 | 0 | 56,361 | 124,179 | YES |
| Wipe | image | 1 | 3399129 | — | — | — | — | — | — | — | — | — | — | **UNRUN** |
| Wipe | image | 2 | 3399130 | — | — | — | — | — | — | — | — | — | — | **UNRUN** |
| Wipe | image | 3 | 3399131 | — | — | — | — | — | — | — | — | — | — | **UNRUN** |
| Wipe | image | 4 | 3399132 | — | — | — | — | — | — | — | — | — | — | **UNRUN** |
| Wipe | image | 5 | 3399133 | — | — | — | — | — | — | — | — | — | — | **UNRUN** |

**Totals over 15 completed runs:** fallback rounds = **0**, demos chosen by the geometric fallback = **0**, escalated demos = 2 (method-intrinsic), VLM tokens = 631,786, LLM tokens = 1,458,488.

Every acquired demonstration in every completed run traces to a live OpenRouter prescription.
No 429s, rate-limit errors or truncated responses appear in any run log.

