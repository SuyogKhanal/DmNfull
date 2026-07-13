"""D5 compute telemetry — per-round wall-clock + per-call LLM/VLM token usage.

PURPOSE (paper D5_Compute table): measure, for ONE round of a Push-T arm,
  * the round's total wall-clock,
  * the DISEIL-specific reasoning wall-clock (rollout analysis + VLM + reasoning
    LLM + prescription + feasibility/expert-solve),
  * the exact prompt/completion token usage of every VLM and LLM call, with the
    model name and whether the KAG block was present in that prompt.

DESIGN CONSTRAINTS (this repo is shared):
  * COMPLETELY INERT unless the environment variable ``D5_TELEMETRY=1`` is set.
    ``install()`` returns immediately otherwise, so every existing run is
    byte-identical to before this file existed.
  * When active it only WRAPS functions (call-through, return value untouched)
    and appends to a NEW side file ``<work_dir>/telemetry/d5_events.jsonl``.
    No prompt, threshold, default, control-flow branch or existing output schema
    is modified anywhere.
  * Every emit is defensive: a telemetry failure can never crash a run.

The token numbers are the ones the serving endpoint itself reports: the fork's
``llm_clients._responses_call`` already returns ``prompt_tokens`` /
``completion_tokens`` from the Responses-API ``usage`` object — they were simply
being thrown away by the caller. We record them; we do not compute them.

Event schema (one JSON object per line):
  {"kind":"round_start", "round":n, "ts":t}
  {"kind":"span",  "stage":"rollout_and_detect"|"analyze_and_prescribe"|
                            "collect_prescribed_demos"|"train_policy"|
                            "evaluate_heldout"|"collect_initial_demos"|
                            "iil_episode",
                   "round":n, "ts_start":t0, "ts_end":t1, "dur_s":d}
  {"kind":"llm_call", "stage":"VLM"|"Reasoning"|"Plain", "model":str,
                   "effort":str, "prompt_tokens":int, "completion_tokens":int,
                   "total_tokens":int, "kag_in_prompt":bool, "kag_chars":int,
                   "completion_text":str, "round":n, "ts_start":t0,
                   "ts_end":t1, "dur_s":d}

``completion_text`` is stored so the hidden-thinking ("reasoning") token count can
be recovered EXACTLY post-hoc: the Qwen proxy strips ``<think>…</think>`` from the
returned text but vLLM still bills those tokens inside ``output_tokens``, so
    reasoning_tokens = completion_tokens - tokens(completion_text)
with the serving tokenizer (see distil/scripts/build_pusht_d5_row.py). Nothing is
estimated here.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

ENV_FLAG = "D5_TELEMETRY"

_S: Dict[str, Any] = {
    "path": None,        # Path to d5_events.jsonl
    "arm": None,         # method name
    "round": 0,          # current round (set by IterationRecord.__init__ hook)
    "kag_probe": "",     # first chars of the rendered KAG block (substring probe)
    "kag_chars": 0,
    "installed": False,
    "lock": threading.Lock(),
}


def enabled() -> bool:
    return os.environ.get(ENV_FLAG, "0") == "1"


# ---- writer ------------------------------------------------------------
def _emit(rec: Dict[str, Any]) -> None:
    p = _S["path"]
    if p is None:
        return
    rec.setdefault("arm", _S["arm"])
    rec.setdefault("round", _S["round"])
    try:
        with _S["lock"]:
            with open(p, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        pass


def _timed(stage: str, fn):
    """Return a call-through wrapper that records a wall-clock span."""
    def wrapped(*a, **kw):
        t0 = time.time()
        try:
            return fn(*a, **kw)
        finally:
            t1 = time.time()
            _emit({"kind": "span", "stage": stage, "ts_start": t0,
                   "ts_end": t1, "dur_s": round(t1 - t0, 3)})
    wrapped.__name__ = getattr(fn, "__name__", stage)
    wrapped._d5_wrapped = True          # idempotence guard
    return wrapped


def _flat_prompt_text(messages: Any) -> str:
    """Concatenate the TEXT parts of a Responses-API message list. Image parts
    (base64 data URIs) are skipped — we only need it for the KAG substring probe."""
    out: List[str] = []
    try:
        for m in messages or []:
            c = m.get("content") if isinstance(m, dict) else None
            if isinstance(c, str):
                out.append(c)
            elif isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and part.get("type") in (
                            "input_text", "text", "output_text"):
                        out.append(str(part.get("text", "")))
    except Exception:
        pass
    return "\n".join(out)


# ---- install -----------------------------------------------------------
def install(work_dir: str, method: str) -> None:
    """Arm the telemetry for one arm. No-op unless D5_TELEMETRY=1."""
    if not enabled() or _S["installed"]:
        return

    tel = Path(work_dir) / "telemetry"
    try:
        tel.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    _S["path"] = tel / "d5_events.jsonl"
    _S["arm"] = str(method)
    _S["installed"] = True

    # -- 0. Make the fork importable before patching it. In the real run this has
    # ALREADY happened (orchestrator._common imports ..envs.env_setup, which calls
    # bootstrap_fork_path() at module import), so this is a no-op — it exists only so
    # that install() cannot silently no-op if it is ever called before that import.
    # bootstrap_fork_path() is idempotent and this whole path is D5_TELEMETRY-only.
    try:
        from .envs.env_setup import bootstrap_fork_path
        bootstrap_fork_path()
    except Exception as exc:
        _emit({"kind": "install_warn", "what": "fork_path", "err": str(exc)})

    # -- 1. LLM/VLM token usage (the numbers the endpoint already reports) --
    try:
        from diffdagger.main_analysis import llm_clients as LC

        if not getattr(LC._responses_call, "_d5_wrapped", False):
            _orig_call = LC._responses_call

            def _call(model, messages, *, max_output_tokens, effort, label):
                t0 = time.time()
                try:
                    out = _orig_call(model, messages,
                                     max_output_tokens=max_output_tokens,
                                     effort=effort, label=label)
                except Exception as exc:
                    _emit({"kind": "llm_call", "stage": str(label),
                           "model": str(model), "effort": str(effort),
                           "error": f"{type(exc).__name__}: {exc}",
                           "ts_start": t0, "ts_end": time.time()})
                    raise
                t1 = time.time()
                probe = _S["kag_probe"]
                flat = _flat_prompt_text(messages) if probe else ""
                pt = int(out.get("prompt_tokens", 0) or 0)
                ct = int(out.get("completion_tokens", 0) or 0)
                _emit({"kind": "llm_call", "stage": str(label),
                       "model": str(model), "effort": str(effort),
                       "prompt_tokens": pt, "completion_tokens": ct,
                       "total_tokens": pt + ct,
                       "kag_in_prompt": bool(probe and probe in flat),
                       "kag_chars": int(_S["kag_chars"]),
                       "max_output_tokens": int(max_output_tokens),
                       "completion_text": out.get("text", ""),
                       "ts_start": t0, "ts_end": t1,
                       "dur_s": round(t1 - t0, 3)})
                return out

            _call._d5_wrapped = True
            LC._responses_call = _call
    except Exception as exc:                      # fork not importable → skip
        _emit({"kind": "install_warn", "what": "llm_clients", "err": str(exc)})

    # -- 2. DISEIL round structure (p4_subtask / p4_top3 engine) --
    try:
        from diffdagger.main_pipeline import pipeline as PL

        if not getattr(PL.IterationRecord.__init__, "_d5_wrapped", False):
            _orig_ir = PL.IterationRecord.__init__

            def _ir_init(self, iteration, work_dir_):
                _S["round"] = int(iteration)
                _emit({"kind": "round_start", "round": int(iteration),
                       "ts": time.time()})
                return _orig_ir(self, iteration, work_dir_)
            _ir_init._d5_wrapped = True
            PL.IterationRecord.__init__ = _ir_init

        if not getattr(PL.LLMGuidedDAggerPipeline._init_llm_clients,
                       "_d5_wrapped", False):
            _orig_init = PL.LLMGuidedDAggerPipeline._init_llm_clients

            def _init(self):
                r = _orig_init(self)
                kag = getattr(self, "_kag_formatted", "") or ""
                _S["kag_probe"] = kag.strip()[:300]
                _S["kag_chars"] = len(kag)
                return r
            _init._d5_wrapped = True
            PL.LLMGuidedDAggerPipeline._init_llm_clients = _init

        for meth, stage in (("_analyze_and_prescribe", "analyze_and_prescribe"),
                            ("_collect_prescribed_demos",
                             "collect_prescribed_demos")):
            f = getattr(PL.LLMGuidedDAggerPipeline, meth)
            if not getattr(f, "_d5_wrapped", False):
                setattr(PL.LLMGuidedDAggerPipeline, meth, _timed(stage, f))
    except Exception as exc:
        _emit({"kind": "install_warn", "what": "pipeline", "err": str(exc)})

    # -- 3. sim_bridge stages (the DISEIL engine imports these at call time) --
    try:
        from diffdagger.main_pipeline import sim_bridge as SB
        for name, stage in (("train_policy", "train_policy"),
                            ("evaluate_heldout", "evaluate_heldout"),
                            ("rollout_and_detect", "rollout_and_detect"),
                            ("collect_initial_demos_shared",
                             "collect_initial_demos")):
            f = getattr(SB, name, None)
            if f is not None and not getattr(f, "_d5_wrapped", False):
                setattr(SB, name, _timed(stage, f))
    except Exception as exc:
        _emit({"kind": "install_warn", "what": "sim_bridge", "err": str(exc)})

    # -- 4. baseline arm (iil_baselines binds these names at MODULE import) --
    try:
        from .selection import iil_baselines as IIL
        for name, stage in (("train_policy", "train_policy"),
                            ("evaluate_heldout", "evaluate_heldout"),
                            ("collect_initial_demos_shared",
                             "collect_initial_demos"),
                            ("_run_one_episode_iil", "iil_episode")):
            f = getattr(IIL, name, None)
            if f is not None and not getattr(f, "_d5_wrapped", False):
                setattr(IIL, name, _timed(stage, f))
    except Exception as exc:
        _emit({"kind": "install_warn", "what": "iil_baselines", "err": str(exc)})

    _emit({"kind": "install", "method": method, "work_dir": str(work_dir),
           "ts": time.time()})
