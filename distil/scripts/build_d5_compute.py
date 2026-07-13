"""Build the D5_Compute matrix from measured telemetry. Invents nothing.

Estimator (identical for DISEIL and the baseline, so the ratio is honest):

  s/round      mean of the per-round `sec` field over rounds >= 1. Round 0 is dropped
               because it carries the one-off initial training (8000 steps vs 4000)
               and bootstrap collection sits outside the round loop entirely.
  VLM tok/rnd      mean over LLM-active rounds of tokens.by_stage.vlm.total
  LLM tok/rnd      mean of (analysis.total + decision.total)  -- the reasoning-LLM stages
  Reasoning tok/rnd mean of tokens.reasoning (hidden thinking tokens billed in the
                   completion, reported by the endpoint under
                   completion_tokens_details.reasoning_tokens)
  KAG tok/rnd      mean of kag_calls * tokens(kag_block); tokens(kag_block) is measured
                   per task by a paired prompt-token diff against the serving tokenizer
                   (distil/scripts/measure_kag_tokens.py -> kag_tokens.json)
  Overhead x       DISEIL s/round / Baseline s/round

Usage:
  python distil/scripts/build_d5_compute.py
"""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path("/weka/s226137394/DmNfull")
COMPUTE = ROOT / "distil/results/_compute"
RESULTS = ROOT / "distil/results"
BUILD = ROOT / "paper_aaai2027/COC_REPORT/build"

# The five D5 rows. 'status' = measurable | blocked (with reason).
ROWS = [
    ("Door", "state", "Door / state"),
    ("PushT", "image", "Push-T / image"),
    ("Wipe", "image", "Wipe / image"),
    ("Door", "image", "Door / image"),
    ("GridWorld", "image", "GridWorld 5x5 / image"),
]

BLOCKED = {
    ("PushT", "image"): (
        "Push-T is not implemented in the DISEIL module: `distil.config.get_config` "
        "raises AssertionError (task absent from TASKS), there is no Push-T env, expert "
        "or KAG graph, and `distil/matrix.py` marks both Push-T cells RUNNABLE=False "
        "(\"vendoring pending\"). The only Push-T code lives in the superseded fork at "
        "/weka/s226137394/diff-dagger, whose image path is the frozen-R3M embedding "
        "pipeline that A10 explicitly retires. Measuring there would report the cost of "
        "a code path the paper no longer describes."),
}
# NOTE: (GridWorld, image) was blocked here until the image policy + the GridWorld
# SafeDAgger arm were wired (gridworld/{policy,baselines}.py, run.py). It is now
# runnable — distil/scripts/run_gridworld_d5.sbatch — so it stays a normal row and
# reports PENDING until both leaves land under results/_compute/GridWorld/image/.


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _sec_per_round(res) -> float | None:
    secs = [r.get("sec") for r in res.get("history", []) if r.get("sec") is not None]
    return st.mean(secs[1:]) if len(secs) > 1 else None


def _token_stats(res, kag_tokens_per_call: int | None):
    """Per-round token means over the rounds where the LLM actually ran."""
    vlm, llm, reas, kag, tot = [], [], [], [], []
    for r in res.get("history", []):
        t = r.get("tokens") or {}
        bs = t.get("by_stage")
        if not bs or not t.get("total"):
            continue
        v = bs.get("vlm", {}).get("total", 0)
        a = bs.get("analysis", {}).get("total", 0)
        d = bs.get("decision", {}).get("total", 0)
        vlm.append(v)
        llm.append(a + d)
        reas.append(t.get("reasoning", 0))
        tot.append(t.get("total", 0))
        if kag_tokens_per_call is not None:
            kag.append(int(t.get("kag_calls", 0)) * kag_tokens_per_call)
    if not tot:
        return None
    return {
        "rounds_with_llm": len(tot),
        "vlm": st.mean(vlm), "llm": st.mean(llm),
        "reasoning": st.mean(reas), "kag": st.mean(kag) if kag else None,
        "total": st.mean(tot),
    }


def main():
    kag_json = _load(BUILD / "kag_tokens.json") or {"tasks": {}}
    # tokens(kag_block) per KAG-bearing call: analysis and decision agree to <1%, so
    # take the mean of the two measured deltas.
    kag_per_call = {}
    for task, rec in kag_json.get("tasks", {}).items():
        kag_per_call[task] = round(
            (rec["analysis_kag_delta"] + rec["decision_kag_delta"]) / 2)

    out_rows, notes = [], []
    for task, mod, label in ROWS:
        if (task, mod) in BLOCKED:
            out_rows.append({"setting": label, "status": "UNMEASURED",
                             "baseline_s_per_round": "UNMEASURED",
                             "diseil_s_per_round": "UNMEASURED",
                             "vlm_tokens_per_round": "UNMEASURED",
                             "llm_tokens_per_round": "UNMEASURED",
                             "overhead_x": "UNMEASURED",
                             "kag_tokens_per_round": "UNMEASURED",
                             "reasoning_llm_tokens_per_round": "UNMEASURED",
                             "source": "blocked"})
            notes.append(f"- **{label}** — UNMEASURED. {BLOCKED[(task, mod)]}")
            continue

        d = _load(COMPUTE / task / mod / "full" / "seed1" / "result.json")
        b = _load(COMPUTE / task / mod / "safe" / "seed1" / "result.json")
        if d is None or b is None:
            out_rows.append({"setting": label, "status": "PENDING",
                             **{k: "PENDING" for k in
                                ("baseline_s_per_round", "diseil_s_per_round",
                                 "vlm_tokens_per_round", "llm_tokens_per_round",
                                 "overhead_x", "kag_tokens_per_round",
                                 "reasoning_llm_tokens_per_round")},
                             "source": "job incomplete"})
            continue

        ds = _sec_per_round(d)
        bs = _sec_per_round(b)
        tk = _token_stats(d, kag_per_call.get(task))
        row = {
            "setting": label, "status": "measured",
            "baseline_s_per_round": round(bs, 1) if bs else None,
            "diseil_s_per_round": round(ds, 1) if ds else None,
            "vlm_tokens_per_round": round(tk["vlm"]) if tk else None,
            "llm_tokens_per_round": round(tk["llm"]) if tk else None,
            "overhead_x": round(ds / bs, 2) if (ds and bs) else None,
            "kag_tokens_per_round": (round(tk["kag"]) if tk and tk["kag"] else None),
            "reasoning_llm_tokens_per_round": round(tk["reasoning"]) if tk else None,
            "source": (f"DISEIL {len(d['history'])} rnds / baseline(SafeDAgger) "
                       f"{len(b['history'])} rnds; tokens over "
                       f"{tk['rounds_with_llm'] if tk else 0} LLM-active rounds"),
        }
        out_rows.append(row)

    BUILD.mkdir(parents=True, exist_ok=True)
    cols = ["setting", "baseline_s_per_round", "diseil_s_per_round",
            "vlm_tokens_per_round", "llm_tokens_per_round", "overhead_x",
            "kag_tokens_per_round", "reasoning_llm_tokens_per_round",
            "status", "source"]
    with open(BUILD / "d5_compute.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out_rows:
            w.writerow({c: r.get(c, "") for c in cols})

    print(json.dumps({"rows": out_rows, "kag_tokens_per_call": kag_per_call},
                     indent=2, default=str))
    print(f"\n[write] {BUILD/'d5_compute.csv'}")


if __name__ == "__main__":
    main()
