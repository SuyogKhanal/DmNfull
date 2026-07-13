"""DISTIL multi-stage LLM decision client — OpenRouter (Chat Completions).

Replaces the old local-vLLM-proxy / Responses-API client (02_..md #3). Talks to
OpenRouter's OpenAI-compatible Chat Completions endpoint (stable; best-documented
for the `reasoning` + `image_url` params). Three roles (06_PROMPTS.md):

  A. VLM  (perception, effort LOW)   : qwen/qwen3-vl-30b-a3b-instruct
        reads the failure's start / t_flag / end frames as image_url parts ->
        a concrete spatial description. Non-thinking Instruct variant.
  B. ANALYSIS (effort HIGH)          : qwen/qwen3-32b
        classifies root_cause + phase, KAG-grounded, strict JSON.
  C. DECISION (effort HIGH)          : qwen/qwen3-32b
        SELECT ep# / BRIDGE ep#,ep# + a REQUIRED confidence score (02_..md #6).

`qwen/qwen3-32b` is a binary thinking-on/off model with no native effort tiers, so
we make low<->high differ by an explicit reasoning-token budget:
  high -> reasoning={"max_tokens": N_high, "exclude": True}   (exclude keeps content clean JSON)
  low  -> reasoning={"enabled": False}                        (no-think, fast/cheap aggregator)
We also strip any inline <think>...</think> before json parse (the Qwen gotcha,
06_..md) as belt-and-suspenders.

decide() returns a structured dict (label + confidence + per-stage analyses +
token usage + resolved models) so the loop can log everything (golden rule 3/4).
make_llm() returns None when no OpenRouter base_url/key is configured, so the loop
degrades to the geometric planner.
"""
from __future__ import annotations

import base64
import json
import os
import random
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import prompts as P
from .kag import load_kag_text

# reasoning-token budgets (05_..md llm_effort is an ablation knob; overridable via env)
_N_HIGH = int(os.environ.get("DISTIL_REASONING_HIGH", "4096"))
_OVERALL_MAX = int(os.environ.get("OAI_MAX_OUTPUT_TOKENS", "16384"))


@lru_cache(maxsize=1)
def _oai_client():
    from openai import OpenAI
    key = (os.environ.get("OPENROUTER_API_KEY")
           or os.environ.get("OPENAI_API_KEY") or "missing-key")
    kwargs: Dict[str, Any] = {"api_key": key}
    if os.environ.get("OPENAI_BASE_URL"):
        kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
    kwargs["timeout"] = float(os.environ.get("OAI_SDK_TIMEOUT", "120"))
    kwargs["max_retries"] = int(os.environ.get("OAI_SDK_MAX_RETRIES", "0"))  # we retry ourselves
    return OpenAI(**kwargs)


def _retryable():
    try:
        import openai
        return (openai.APIConnectionError, openai.APITimeoutError,
                openai.RateLimitError, openai.InternalServerError)
    except Exception:
        return (Exception,)


def _oai_retry(fn):
    max_attempts = int(os.environ.get("OAI_MAX_ATTEMPTS", "4"))
    base, cap = 1.8, 30.0
    last = None
    for i in range(max_attempts):
        try:
            return fn()
        except _retryable() as e:
            last = e
            if i == max_attempts - 1:
                raise
            time.sleep(min(cap, base ** i) + random.uniform(0, 0.75))
    raise last


def _data_url(path: str) -> str:
    ext = Path(path).suffix.lstrip(".").lower() or "png"
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(txt: str) -> str:
    return _THINK_RE.sub("", txt or "").strip()


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    text = _strip_think(text)
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    i = text.find("{")
    while i != -1:
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[i:j + 1])
                    except Exception:
                        break
        i = text.find("{", i + 1)
    return None


# 'CONFIDENCE: 82 — one-line rationale'  (accept -, —, :, or nothing before rationale)
_CONF_RE = re.compile(r"CONFIDENCE\s*[:=]?\s*(\d{1,3})\s*(?:[-—:]\s*(.*))?", re.IGNORECASE)


def parse_confidence(text: str):
    """Return (confidence:int|None in [0,100], rationale:str)."""
    if not text:
        return None, ""
    m = _CONF_RE.search(_strip_think(text))
    if not m:
        return None, ""
    c = max(0, min(100, int(m.group(1))))
    rat = (m.group(2) or "").strip()[:200]
    return c, rat


class _Usage:
    """Token accounting. Keeps the original aggregate keys (prompt/completion/total/
    calls) AND a per-stage breakdown + reasoning tokens, for the compute table
    (09_..md Tier-4 #6: sec/round and tokens/round, split VLM vs reasoning LLM).

    Stage is derived from the call label: 'vlm_ep7' -> vlm, 'analysis_ep7' ->
    analysis, 'decision' -> decision. OpenRouter reports the reasoning-token count
    under usage.completion_tokens_details.reasoning_tokens even when the reasoning
    text is excluded from the content, so the hidden thinking cost is observable.
    """

    STAGES = ("vlm", "analysis", "decision")

    def __init__(self):
        self.prompt = self.completion = self.total = self.calls = 0
        self.reasoning = 0
        self.by_stage = {s: {"prompt": 0, "completion": 0, "total": 0,
                             "reasoning": 0, "calls": 0} for s in self.STAGES}

    def add(self, u, stage: str = None):
        if u is None:
            return
        p = int(getattr(u, "prompt_tokens", 0) or 0)
        c = int(getattr(u, "completion_tokens", 0) or 0)
        t = int(getattr(u, "total_tokens", 0) or 0)
        det = getattr(u, "completion_tokens_details", None)
        r = int(getattr(det, "reasoning_tokens", 0) or 0) if det is not None else 0
        self.prompt += p
        self.completion += c
        self.total += t
        self.reasoning += r
        self.calls += 1
        b = self.by_stage.get(stage)
        if b is not None:
            b["prompt"] += p
            b["completion"] += c
            b["total"] += t
            b["reasoning"] += r
            b["calls"] += 1

    def asdict(self):
        return {"prompt": self.prompt, "completion": self.completion,
                "total": self.total, "calls": self.calls,
                "reasoning": self.reasoning,
                "by_stage": {s: dict(v) for s, v in self.by_stage.items()}}


class DistilLLM:
    """Multi-stage VLM+reasoning decision head over OpenRouter Chat Completions."""

    def __init__(self, vlm_model: str, llm_model: str, use_vlm: bool = True,
                 use_kag: bool = True, max_tokens: int = _OVERALL_MAX):
        self.vlm_model = vlm_model
        self.llm_model = llm_model
        self.use_vlm = bool(use_vlm)      # vlm=off ablation (05_..md)
        self.use_kag = bool(use_kag)      # kag=off ablation (05_..md)
        self.max_tokens = int(max_tokens)

    # ── low-level call ───────────────────────────────────────────────────────
    def _chat(self, model: str, messages, *, effort: str, usage: _Usage,
              dump: Optional[Dict] = None, label: str = "") -> str:
        extra: Dict[str, Any] = {}
        if effort == "high":
            extra["reasoning"] = {"max_tokens": _N_HIGH, "exclude": True}
        elif effort == "low":
            extra["reasoning"] = {"enabled": False}
        # effort == "none" (VLM instruct): send nothing.

        def _do():
            return _oai_client().chat.completions.create(
                model=model, messages=messages, max_tokens=self.max_tokens,
                extra_body=extra or None)
        r = _oai_retry(_do)
        usage.add(getattr(r, "usage", None),
                  stage=(label.split("_")[0] if label else None))
        content = (r.choices[0].message.content or "") if r.choices else ""
        content = _strip_think(content)
        if dump is not None:
            dump[label] = {"model": model, "effort": effort,
                           "messages": messages, "response": content}
        return content

    # ── Stage A: VLM (low/none effort) ───────────────────────────────────────
    def _vlm(self, task_desc: str, f: Dict[str, Any], usage: _Usage,
             dump: Optional[Dict], task: str = None) -> str:
        if not self.use_vlm:
            return "(VLM disabled — geometric descriptor + root-cause only)"
        frames = f.get("frames", {})
        roles = [r for r in ("start", "high_loss", "end")
                 if frames.get(r) and os.path.exists(frames[r])]
        if not roles:
            return "(no frames available)"
        parts: List[Dict[str, Any]] = [
            {"type": "text", "text": P.vlm_prompt(task_desc, roles, f.get("t_star", 0), task=task)}]
        for r in roles:
            parts.append({"type": "text", "text": f"[{r} frame]"})
            parts.append({"type": "image_url", "image_url": {"url": _data_url(frames[r])}})
        msgs = [{"role": "system", "content": P.vlm_system(task)},
                {"role": "user", "content": parts}]
        txt = self._chat(self.vlm_model, msgs, effort="none", usage=usage,
                         dump=dump, label=f"vlm_ep{f.get('ep_id')}")
        return txt or "(VLM returned no description)"

    # ── Stage B: analysis (high effort) ──────────────────────────────────────
    def _analysis(self, task_desc: str, kag: str, vlm_report: str, usage: _Usage,
                  dump: Optional[Dict], ep_id, enums=None) -> Dict[str, str]:
        rc, ph = enums if enums else (None, None)
        msgs = [{"role": "system", "content": P.ANALYSIS_SYSTEM},
                {"role": "user", "content": P.analysis_prompt(task_desc, kag, vlm_report,
                                                              root_causes=rc, phases=ph)}]
        txt = self._chat(self.llm_model, msgs, effort="high", usage=usage,
                         dump=dump, label=f"analysis_ep{ep_id}")
        obj = _extract_json(txt) or {}
        return {"root_cause": obj.get("root_cause", "approach_failure"),
                "phase": obj.get("phase", "pre_grasp"),
                "rationale": str(obj.get("rationale", ""))[:200]}

    # ── Stage C: decision (high effort) — SELECT/BRIDGE + CONFIDENCE ─────────
    def _decision(self, task_desc: str, kag: str, members: List[Dict], bridge: bool,
                  usage: _Usage, dump: Optional[Dict]) -> str:
        msgs = [{"role": "system", "content": P.DECISION_SYSTEM},
                {"role": "user", "content": P.decision_prompt(task_desc, kag, members, bridge)}]
        return self._chat(self.llm_model, msgs, effort="high", usage=usage,
                          dump=dump, label="decision")

    def decide(self, task: str, failures: List[Dict[str, Any]], bridge_supported: bool,
               *, log_fn=None, prompt_dir: Optional[str] = None) -> Dict[str, Any]:
        """Run VLM->analysis per failure then one decision call. Returns a dict:
        {label, raw_decision, confidence, confidence_rationale, members[], tokens, models}."""
        task_desc = P.TASK_DESC.get(task, task)
        kag = load_kag_text(task) if self.use_kag else ""
        enums = P.enums_for(task)
        usage = _Usage()
        dump: Dict[str, Any] = {} if prompt_dir else None
        members = []
        for f in failures:
            vlm_report = self._vlm(task_desc, f, usage, dump, task=task)
            analysis = self._analysis(task_desc, kag, vlm_report, usage, dump, f.get("ep_id"), enums)
            members.append({**f, "vlm_report": vlm_report, "analysis": analysis})
            if log_fn:
                log_fn(f"    [distil-llm ep{f['ep_id']}] root_cause={analysis['root_cause']} "
                       f"phase={analysis['phase']}")
        raw = self._decision(task_desc, kag, members, bridge_supported, usage, dump)
        conf, conf_rat = parse_confidence(raw)
        # KAG accounting for the compute table: the KAG block is inserted verbatim into
        # every analysis prompt (one per analysed failure) and into the single decision
        # prompt. Its per-round token contribution is therefore
        #   kag_calls * tokens(kag_block),
        # where tokens(kag_block) is measured against the SERVING tokenizer by a paired
        # prompt-token diff (scripts/measure_kag_tokens.py). We log the multiplicand
        # (kag_calls) and the raw size here so the product is reproducible.
        tok = usage.asdict()
        tok["kag_calls"] = (len(members) + 1) if (self.use_kag and kag) else 0
        tok["kag_chars"] = len(kag) if self.use_kag else 0
        if prompt_dir:
            try:
                self._write_prompts(prompt_dir, dump, kag)
            except Exception as e:  # telemetry must never crash the loop
                if log_fn:
                    log_fn(f"    [distil-llm] prompt dump failed: {e}")
        return {"label": raw, "raw_decision": raw, "confidence": conf,
                "confidence_rationale": conf_rat,
                "members": [{"ep_id": m["ep_id"], "root_cause": m["analysis"]["root_cause"],
                             "phase": m["analysis"]["phase"],
                             "rationale": m["analysis"]["rationale"],
                             "vlm_report": m["vlm_report"]} for m in members],
                "tokens": tok,
                "models": {"vlm": self.vlm_model, "llm": self.llm_model,
                           "use_vlm": self.use_vlm, "use_kag": self.use_kag}}

    @staticmethod
    def _write_prompts(prompt_dir: str, dump: Dict, kag: str) -> None:
        d = Path(prompt_dir)
        d.mkdir(parents=True, exist_ok=True)
        for label, rec in (dump or {}).items():
            with open(d / f"{label}.txt", "w") as fh:
                fh.write(f"# model={rec['model']} effort={rec['effort']}\n\n")
                for msg in rec["messages"]:
                    fh.write(f"===== {msg['role'].upper()} =====\n")
                    c = msg["content"]
                    if isinstance(c, list):
                        for part in c:
                            if part.get("type") == "image_url":
                                fh.write("[image_url: <base64 data omitted>]\n")
                            else:
                                fh.write(part.get("text", "") + "\n")
                    else:
                        fh.write(str(c) + "\n")
                    fh.write("\n")
                fh.write("===== RESPONSE =====\n" + rec["response"] + "\n")


def make_llm(use_vlm: bool = True, use_kag: bool = True,
             max_tokens: Optional[int] = None) -> Optional[DistilLLM]:
    """Build the multi-stage client, or None if OpenRouter is not configured."""
    if not os.environ.get("OPENAI_BASE_URL"):
        return None
    if not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        return None
    vlm = os.environ.get("VLM_MODEL_NAME", "qwen/qwen3-vl-30b-a3b-instruct")
    llm = os.environ.get("LLM_MODEL_NAME", "qwen/qwen3-32b")
    mt = int(max_tokens or _OVERALL_MAX)
    return DistilLLM(vlm, llm, use_vlm=use_vlm, use_kag=use_kag, max_tokens=mt)
