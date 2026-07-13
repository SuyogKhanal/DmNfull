"""Reconstruct per-round STAGE SPANS for the RoboSuite D5 runs from run.log timestamps.

Additive / read-only: reads run.log + result.json + telemetry/*.jsonl, writes a new
side-file. Touches no pipeline code and no existing output schema.

DISEIL round (distil/run.py) prints, in order:
    [train] ... -> [calibrate] ... -> [eval] ... -> [screen] ... ->
    [distil-llm ep*] x N -> [distil-llm] N failures analysed -> [collect a0]
so the round decomposes as
    train  = train_line   -> calibrate_line
    eval   = calibrate    -> eval_line
    screen = eval_line    -> screen_line          (40-ep failure screen; policy rollout)
    llm    = screen_line  -> "failures analysed"  (clustering + VLM + reasoning LLM)
    presc  = analysed     -> collect_line         (prescription + feasibility/expert solve)
and round wall-clock = train_line -> collect_line, which we CHECK against
result.json history[i]["sec"].

SafeDAgger round prints:
    [train] ... -> [eval] SR= -> [safe] intervention ...
    train  = round_start -> eval_line
    eval   = ...                (SR line is printed at the END of eval)
    gate   = eval_line   -> intervention_line     (rollout until the 1st intervention)
The safe arm prints no [calibrate], so train and eval cannot be separated from the log;
they are reported jointly as `train+eval`.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

COMPUTE = Path("/weka/s226137394/DmNfull/distil/results/_compute")
TS = re.compile(r"^\[(\d\d):(\d\d):(\d\d)\]\s*(.*)$")


def _lines(p: Path):
    """-> [(t_or_None, msg)] in file order. Untimestamped lines (the `===== safe Round N`
    headers) are kept, with t=None, because they are the only round anchor the safe arm
    prints."""
    out = []
    prev = None
    day = 0
    for raw in p.read_text(errors="replace").splitlines():
        m = TS.match(raw)
        if not m:
            out.append((None, raw.strip()))
            continue
        h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
        t = h * 3600 + mi * 60 + s
        if prev is not None and t < prev - 3600:  # midnight rollover
            day += 1
        prev = t
        out.append((t + day * 86400, m.group(4)))
    return out


def parse_full(p: Path):
    """-> list of dicts, one per DISEIL round."""
    L = _lines(p)
    rounds, cur = [], None
    for t, msg in L:
        if t is None:
            continue
        if "[train]" in msg and "trajs" in msg:
            if cur is not None:
                rounds.append(cur)
            cur = {"t_train": t}
        elif cur is None:
            continue
        elif "train step" in msg:
            cur["t_last_step"] = t
        elif "[calibrate]" in msg:
            cur["t_calib"] = t
        elif "[eval] pure-policy" in msg:
            cur["t_eval"] = t
        elif re.search(r"\[screen\] \d+/\d+ failures found", msg):
            cur["t_screen"] = t
        elif "failures analysed" in msg:
            cur["t_llm"] = t
            m = re.search(r"tokens=(\d+)", msg)
            if m:
                cur["log_tokens"] = int(m.group(1))
        elif "[collect a0]" in msg:
            cur["t_collect"] = t
    if cur is not None:
        rounds.append(cur)

    out = []
    for i, r in enumerate(rounds):
        d = {"round": i}
        if "t_calib" in r:
            d["train_s"] = r["t_calib"] - r["t_train"]
        if "t_last_step" in r and "t_calib" in r:
            # the conformal-threshold pass, printed by the [calibrate] line
            d["calibrate_s"] = r["t_calib"] - r["t_last_step"]
        if "t_eval" in r and "t_calib" in r:
            d["eval_s"] = r["t_eval"] - r["t_calib"]
        if "t_screen" in r and "t_eval" in r:
            d["screen_s"] = r["t_screen"] - r["t_eval"]
        if "t_llm" in r and "t_screen" in r:
            d["llm_s"] = r["t_llm"] - r["t_screen"]
        if "t_collect" in r and "t_llm" in r:
            d["prescribe_s"] = r["t_collect"] - r["t_llm"]
        if "t_collect" in r:
            d["round_s_from_log"] = r["t_collect"] - r["t_train"]
        d["log_tokens"] = r.get("log_tokens")
        out.append(d)
    return out


def parse_safe(p: Path):
    """The safe arm prints NO `[train] N trajs` line and its `===== safe Round N =====`
    header carries NO timestamp, so a round starts at the last timestamped line before
    that header (the blank line the previous round emits) and ends at its `[safe]
    intervention` line. Checked against result.json `sec`."""
    L = _lines(p)
    hdr = re.compile(r"=====\s*safe Round (\d+)")
    rounds, cur, last_t = [], None, None
    for t, msg in L:
        if t is None:
            m = hdr.search(msg)
            if m:
                if cur is not None:
                    rounds.append(cur)
                cur = {"round": int(m.group(1)), "t_start": last_t}
            continue
        last_t = t
        if cur is None:
            continue
        if "train step" in msg:
            cur["t_last_step"] = t
        elif "[eval] SR=" in msg:
            cur["t_eval"] = t
        elif "[safe] intervention" in msg:
            cur["t_gate"] = t
    if cur is not None:
        rounds.append(cur)

    out = []
    for r in rounds:
        d = {"round": r["round"]}
        s = r.get("t_start")
        if s is None:
            out.append(d)
            continue
        if "t_last_step" in r:
            d["train_s"] = r["t_last_step"] - s
        if "t_eval" in r and "t_last_step" in r:
            d["eval_s"] = r["t_eval"] - r["t_last_step"]
        if "t_gate" in r and "t_eval" in r:
            # rollout until the 1st intervention: the safe arm's ONLY screening cost
            d["gate_s"] = r["t_gate"] - r["t_eval"]
        if "t_gate" in r:
            d["round_s_from_log"] = r["t_gate"] - s
        out.append(d)
    return out


def tokens_from_telemetry(seed_dir: Path):
    """Per-round token dicts, straight from telemetry/round_*.jsonl (works on
    in-flight runs, where result.json does not exist yet)."""
    tel = seed_dir / "telemetry"
    out = {}
    if not tel.is_dir():
        return out
    for f in sorted(tel.glob("round_*.jsonl")):
        for line in f.read_text().splitlines():
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("kind") == "llm_decision" and e.get("tokens"):
                out[e["round"]] = e["tokens"]
    return out


def main():
    settings = [("Door", "state"), ("Door", "image"), ("Wipe", "image")]
    report = {}
    for task, mod in settings:
        key = f"{task}/{mod}"
        rec = {}
        for arm, parser in (("full", parse_full), ("safe", parse_safe)):
            sd = COMPUTE / task / mod / arm / "seed1"
            log = sd / "run.log"
            if not log.exists():
                rec[arm] = {"status": "missing"}
                continue
            stages = parser(log)
            res = sd / "result.json"
            rj = json.loads(res.read_text()) if res.exists() else None
            if rj:
                for i, h in enumerate(rj["history"]):
                    if i < len(stages):
                        stages[i]["sec_result_json"] = h.get("sec")
                        if arm == "full":
                            stages[i]["tokens"] = h.get("tokens") or {}
                        stages[i]["eval_success"] = h.get("eval_success")
                        stages[i]["n_screen_failures"] = h.get("n_screen_failures")
            if arm == "full":
                tk = tokens_from_telemetry(sd)
                for s in stages:
                    if not s.get("tokens") and s["round"] in tk:
                        s["tokens"] = tk[s["round"]]
                        s["tokens_source"] = "telemetry (run in flight)"
            rec[arm] = {
                "status": "complete" if rj else "IN FLIGHT (no result.json yet)",
                "n_rounds_logged": len(stages),
                "rounds": stages,
            }
        report[key] = rec

    out = Path("/weka/s226137394/DmNfull/distil/results/_compute/d5_stage_spans.json")
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\n[write] {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
