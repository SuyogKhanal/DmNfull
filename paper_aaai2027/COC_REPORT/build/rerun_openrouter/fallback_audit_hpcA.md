# Fallback audit — DISEIL OpenRouter re-run (HPC-A)
Per run: rounds that used the LLM vs. rounds that fell back to the deterministic geometric planner.
**A run is VALID iff `fallback_rounds == 0`** — i.e. the LLM ran on every round it was called for. Any run that fell back to the deterministic planner is INVALID: it is not DISEIL, and it is discarded and re-run, never reported.

`budget_free` rounds are rounds in which the policy produced no usable failures, so there was nothing to prescribe and the LLM was correctly not called. Those are not fallbacks and do not invalidate a run.

**A budget shortfall is not a fallback and does not invalidate a run.** The loop stops when the policy yields no usable failures for 4 consecutive rounds (`distil/p4/loop.py:110-113`): the policy has saturated and the remaining budget is *unspendable*, because there is no failure left to prescribe a correction for. This is existing method behaviour, and the PUBLISHED runs contain it too — published Door (state) seed 4 acquired only 11/20. Such seeds are reported with their shortfall and their reason, and are included in the mean; excluding them would make the re-run non-comparable to the numbers it is meant to reproduce. `saturation_patience` was NOT touched.

All runs were launched with `DISEIL_STRICT_LLM=1`, under which an unavailable or failing LLM **aborts** the run rather than degrading to the fallback. A run that completed therefore cannot contain a fallback round; the table below is the post-hoc confirmation of that, not a substitute for it.

---

## Door (state)

Arm `full` · B=20 · D=1 · held-out 100 eps · VLM `qwen/qwen3-vl-30b-a3b-instruct` · LLM `qwen/qwen3-32b`

| seed | demos acquired | rounds | LLM-active | budget-free | **fallback** | VLM tokens | LLM tokens | final SR | valid (LLM ran) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 20/20 | 30 | 26 | 4 | **0** | 47,199 | 126,826 | 0.96 | YES |
| 2 | 16/20 ⚠️ | 23 | 17 | 6 | **0** | 42,626 | 108,163 | 0.96 | YES |
| 3 | 20/20 | 32 | 23 | 9 | **0** | 43,739 | 126,149 | 0.98 | YES |
| 4 | 20/20 | 48 | 31 | 17 | **0** | 47,166 | 139,842 | 0.95 | YES |
| 5 | 20/20 | 36 | 28 | 8 | **0** | 47,134 | 132,848 | 0.98 | YES |

**Cell total fallback rounds: 0.** 5/5 completed seeds are valid (the LLM ran on every round it was called for).

**Final held-out success rate: 96.6 ± 1.3** over 5 seeds.

⚠️ Budget shortfall (policy saturated, remaining budget unspendable) on seed(s) **2** — marked ⚠️ above. Not a fallback; the LLM ran throughout. Reported, not corrected.

---

## Door (image)

**UNRUN** — no result file. Not reported, not invented.

---

## PushT (state)

**UNRUN** — no result file. Not reported, not invented.

---

## PushT (image)

**UNRUN** — no result file. Not reported, not invented.
