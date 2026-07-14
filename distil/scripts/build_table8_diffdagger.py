"""Rebuild the per-round wall-clock + token cost table (paper Table 8 / workbook sheet
D5_Compute) with DIFF-DAGGER as the baseline instead of SafeDAgger.

Reuses distil/scripts/parse_d5_stages.py (parse_full + the new parse_diffdagger) — every
second reported here traces to a printed, timestamped event in a run that COMPLETED on
this cluster. Nothing is estimated or interpolated; anything unavailable is UNMEASURED.

Scope: Door/state, Door/image, Wipe/image. Both protocols:
    P1 = the run's FIRST round (round 0).
    P5 = mean +/- sample SD over the LLM-active rounds (0..4).

THE BUDGET ASYMMETRY (measured, not assumed): DISEIL adds ONE successful demo per round
(nd=1), Diff-DAgger adds interventions_per_round=4. The loop stops at
final_demos = n_init + budget. So at BUDGET=5 the two arms do NOT run the same number of
rounds: DISEIL runs 5 LLM-active rounds (0..4), Diff-DAgger reaches the stop threshold
after only 2 (log: "[stop] reached final_demos=9"). A 5-round baseline spread therefore
CANNOT come from a BUDGET=5 Diff-DAgger run. We run Diff-DAgger a second time at
BUDGET=20, which with Nd=4 yields exactly rounds 0..4, and use that for the P5 block.
Both are reported.

Outputs -> paper_aaai2027/COC_REPORT/build/table8_diffdagger/
"""
from __future__ import annotations

import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_d5_stages import parse_full, parse_diffdagger   # noqa: E402

ROOT = Path("/weka/s226137394/DmNfull")
COMPUTE = ROOT / "distil/results/_compute"
OUT = ROOT / "paper_aaai2027/COC_REPORT/build/table8_diffdagger"
KAG_TOKENS = json.loads((ROOT / "paper_aaai2027/COC_REPORT/build/kag_tokens.json").read_text())

SETTINGS = [("Door", "state"), ("Door", "image"), ("Wipe", "image")]
LLM_ACTIVE = 5                       # rounds 0..4

# Arms on disk. 'full' = DISEIL (already existed). The two diffdagger dirs are new.
ARM_DIRS = {
    "full":            lambda t, m: COMPUTE / t / m / "full" / "seed1",
    "diffdagger_b5":   lambda t, m: COMPUTE / t / m / "diffdagger" / "seed1",
    "diffdagger_b20":  lambda t, m: COMPUTE / t / m / "diffdagger_b20" / "seed1",
}


def _load(task, mod, arm):
    d = ARM_DIRS[arm](task, mod)
    log, res = d / "run.log", d / "result.json"
    if not log.exists():
        return {"status": "MISSING", "dir": str(d), "rounds": []}
    parser = parse_full if arm == "full" else parse_diffdagger
    rounds = parser(log)
    rj = json.loads(res.read_text()) if res.exists() else None
    if rj:
        for i, h in enumerate(rj["history"]):
            if i < len(rounds):
                # DISEIL: result.json 'sec' is the authoritative round wall-clock.
                # Diff-DAgger: _wrap_history writes no 'sec' -> stays None, and we fall
                # back to round_s_from_log (a logged-timestamp derivation).
                rounds[i]["sec_result_json"] = h.get("sec")
                rounds[i]["eval_success"] = h.get("eval_success")
                if arm == "full":
                    rounds[i]["tokens"] = h.get("tokens") or {}
    return {
        "status": "COMPLETE" if rj else "NO result.json (run did not finish)",
        "dir": str(d.relative_to(ROOT)),
        "completed": rj is not None,
        "final_success": (rj or {}).get("final_success"),
        "wall_sec": (rj or {}).get("wall_sec"),
        "rounds": rounds,
    }


def round_s(r):
    """Round wall-clock. Prefer result.json 'sec' (DISEIL); else the log derivation."""
    v = r.get("sec_result_json")
    return float(v) if v is not None else (
        float(r["round_s_from_log"]) if r.get("round_s_from_log") is not None else None)


def kag_per_round(task, tok):
    """Exact KAG prompt-token contribution for one round:
        n_analysis_calls * analysis_kag_delta + n_decision_calls * decision_kag_delta
    Deltas are measured (paired with/without-KAG prompt-token diff, max_tokens=1) and
    live in build/kag_tokens.json."""
    k = KAG_TOKENS["tasks"].get(task)
    if not k or not tok:
        return None
    bs = tok.get("by_stage") or {}
    na = (bs.get("analysis") or {}).get("calls", 0)
    nd = (bs.get("decision") or {}).get("calls", 0)
    return na * k["analysis_kag_delta"] + nd * k["decision_kag_delta"]


def stage_split(r, arm):
    """Seconds, split into what BOTH arms pay vs what is arm-specific."""
    tr, ev = r.get("train_s"), r.get("eval_s")
    shared = (tr + ev) if (tr is not None and ev is not None) else None
    if arm == "full":
        scr, llm, pre = r.get("screen_s"), r.get("llm_s"), r.get("prescribe_s")
        spec = sum(v for v in (scr, llm, pre) if v is not None) if scr is not None else None
        return dict(train_s=tr, eval_s=ev, shared_s=shared, screen_s=scr,
                    analysis_prescribe_s=((llm or 0) + (pre or 0)) if llm is not None else None,
                    llm_s=llm, prescribe_s=pre, specific_s=spec, gate_s=None)
    gate = r.get("gate_s")
    return dict(train_s=tr, eval_s=ev, shared_s=shared, screen_s=None,
                analysis_prescribe_s=None, llm_s=None, prescribe_s=None,
                specific_s=gate, gate_s=gate)


def agg(vals):
    """mean +/- SAMPLE SD (ddof=1). Returns (mean, sd, n). sd=None when n<2."""
    v = [float(x) for x in vals if x is not None]
    if not v:
        return None, None, 0
    return (st.mean(v), (st.stdev(v) if len(v) > 1 else None), len(v))


def fmt(m, s, nd=1):
    if m is None:
        return "UNMEASURED"
    if s is None:
        return f"{m:.{nd}f}"
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {}
    for task, mod in SETTINGS:
        data[f"{task}/{mod}"] = {a: _load(task, mod, a) for a in ARM_DIRS}

    (OUT / "raw_stage_spans.json").write_text(json.dumps(data, indent=1, default=str))

    csv_rows = []
    md = []

    for task, mod in SETTINGS:
        key = f"{task}/{mod}"
        full = data[key]["full"]
        dd5 = data[key]["diffdagger_b5"]
        dd20 = data[key]["diffdagger_b20"]

        for proto in ("P1", "P5"):
            # ---- pick the rounds each protocol looks at ----
            if proto == "P1":
                f_rounds = full["rounds"][:1]
                b_rounds = dd5["rounds"][:1]           # round 0 = the FIRST round, both arms
                b_arm, b_src = "diffdagger (BUDGET=5)", dd5
            else:
                f_rounds = full["rounds"][:LLM_ACTIVE]           # 0..4, LLM-active
                # Preferred baseline: BUDGET=20 (Nd=4 -> exactly rounds 0..4).
                # If that run is not available on HARDWARE-MATCHED nodes, fall back to the
                # matched BUDGET=5 run's rounds rather than silently substituting a
                # cross-GPU-class number. Wall-clock is only comparable within a GPU class.
                if dd20["rounds"]:
                    b_rounds = dd20["rounds"][:LLM_ACTIVE]
                    b_arm, b_src = "diffdagger (BUDGET=20, 5 rounds)", dd20
                else:
                    b_rounds = dd5["rounds"][:LLM_ACTIVE]
                    b_arm, b_src = ("diffdagger (BUDGET=5 FALLBACK — only 2 rounds; the "
                                    "hardware-matched BUDGET=20 run did not schedule)", dd5)

            f_sp = [stage_split(r, "full") for r in f_rounds]
            b_sp = [stage_split(r, "dd") for r in b_rounds]

            f_round = [round_s(r) for r in f_rounds]
            b_round = [round_s(r) for r in b_rounds]

            fm, fs, fn = agg(f_round)
            bm, bs_, bn = agg(b_round)
            over = (fm / bm) if (fm and bm) else None

            f_shared = agg([x["shared_s"] for x in f_sp])
            b_shared = agg([x["shared_s"] for x in b_sp])
            f_screen = agg([x["screen_s"] for x in f_sp])
            f_anpr = agg([x["analysis_prescribe_s"] for x in f_sp])
            f_spec = agg([x["specific_s"] for x in f_sp])
            b_gate = agg([x["gate_s"] for x in b_sp])

            # reasoning-only add-on, computed PER ROUND where both arms have a value,
            # then aggregated (not mean-minus-mean), so the SD is meaningful.
            n_pair = min(len(f_sp), len(b_sp))
            addon = [f_sp[i]["specific_s"] - b_sp[i]["specific_s"]
                     for i in range(n_pair)
                     if f_sp[i]["specific_s"] is not None and b_sp[i]["specific_s"] is not None]
            am, as_, an = agg(addon)

            # ---- tokens (DISEIL only; Diff-DAgger = 0 by construction) ----
            toks = [r.get("tokens") or {} for r in f_rounds]
            vlm = agg([(t.get("by_stage", {}).get("vlm") or {}).get("total") for t in toks])
            llm = agg([((t.get("by_stage", {}).get("analysis") or {}).get("total", 0)
                        + (t.get("by_stage", {}).get("decision") or {}).get("total", 0)) or None
                       for t in toks])
            rsn = agg([t.get("reasoning") for t in toks])
            kag = agg([kag_per_round(task, t) for t in toks])

            csv_rows.append(dict(
                setting=key, protocol=proto,
                n_rounds_diseil=fn, n_rounds_diffdagger=bn,
                diffdagger_s_per_round=bm, diffdagger_s_sd=bs_,
                diseil_s_per_round=fm, diseil_s_sd=fs,
                overhead_x=over,
                shared_train_eval_s_diseil=f_shared[0], shared_train_eval_s_diseil_sd=f_shared[1],
                shared_train_eval_s_diffdagger=b_shared[0], shared_train_eval_s_diffdagger_sd=b_shared[1],
                diseil_screening_s=f_screen[0], diseil_screening_s_sd=f_screen[1],
                diseil_analysis_prescription_s=f_anpr[0], diseil_analysis_prescription_s_sd=f_anpr[1],
                diseil_specific_s=f_spec[0], diseil_specific_s_sd=f_spec[1],
                diffdagger_gate_s=b_gate[0], diffdagger_gate_s_sd=b_gate[1],
                reasoning_only_addon_s=am, reasoning_only_addon_s_sd=as_,
                vlm_tokens_per_round=vlm[0], vlm_tokens_sd=vlm[1],
                llm_tokens_per_round=llm[0], llm_tokens_sd=llm[1],
                reasoning_llm_tokens_per_round=rsn[0], reasoning_llm_tokens_sd=rsn[1],
                kag_token_contribution_per_round=kag[0], kag_token_contribution_sd=kag[1],
                diffdagger_vlm_tokens=0, diffdagger_llm_tokens=0,
                diffdagger_reasoning_tokens=0, diffdagger_kag_tokens=0,
                baseline_arm_dir=b_src.get("dir"), diseil_dir=full.get("dir"), baseline_source=b_arm,
                baseline_status=b_src.get("status"), diseil_status=full.get("status"),
            ))

    with (OUT / "table8.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        for r in csv_rows:
            w.writerow(r)
    print(f"[write] {OUT/'table8.csv'} ({len(csv_rows)} rows)")
    json.dump(csv_rows, (OUT / "table8_rows.json").open("w"), indent=1, default=str)

    # console summary
    for r in csv_rows:
        print(f"\n--- {r['setting']} {r['protocol']} "
              f"(DISEIL n={r['n_rounds_diseil']}, Diff-DAgger n={r['n_rounds_diffdagger']})")
        print(f"    Diff-DAgger s/round : {fmt(r['diffdagger_s_per_round'], r['diffdagger_s_sd'])}")
        print(f"    DISEIL      s/round : {fmt(r['diseil_s_per_round'], r['diseil_s_sd'])}")
        print(f"    Overhead x          : "
              f"{r['overhead_x']:.3f}" if r['overhead_x'] else "    Overhead x          : UNMEASURED")
        print(f"    shared train+eval   : DISEIL {fmt(r['shared_train_eval_s_diseil'],r['shared_train_eval_s_diseil_sd'])}"
              f" | DD {fmt(r['shared_train_eval_s_diffdagger'],r['shared_train_eval_s_diffdagger_sd'])}")
        print(f"    DISEIL screen       : {fmt(r['diseil_screening_s'], r['diseil_screening_s_sd'])}")
        print(f"    DISEIL analysis+pre : {fmt(r['diseil_analysis_prescription_s'], r['diseil_analysis_prescription_s_sd'])}")
        print(f"    DD gate             : {fmt(r['diffdagger_gate_s'], r['diffdagger_gate_s_sd'])}")
        print(f"    reasoning-only ADDON: {fmt(r['reasoning_only_addon_s'], r['reasoning_only_addon_s_sd'])}")
        print(f"    tokens vlm/llm/rsn/kag: {fmt(r['vlm_tokens_per_round'],r['vlm_tokens_sd'],0)}"
              f" / {fmt(r['llm_tokens_per_round'],r['llm_tokens_sd'],0)}"
              f" / {fmt(r['reasoning_llm_tokens_per_round'],r['reasoning_llm_tokens_sd'],0)}"
              f" / {fmt(r['kag_token_contribution_per_round'],r['kag_token_contribution_sd'],0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
