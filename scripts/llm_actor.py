"""Per-step LLM action recommender for the McNemar `llm_actor` evaluation mode.

The active-loop pipeline (pipeline/reasoning.py, aggregator.py, etc.) runs
*after* an episode finishes — it analyses failures and prescribes future demos.
That post-hoc surface is the wrong one for `llm_actor` mode, which needs a
fresh action recommendation at every env step.

This module wraps a thin per-step LLM call that consumes the SAME profile
flag set as the offline pipeline (use_kag / use_rag / use_tkf / use_reasoning
/ use_plain_llm), so toggling P4 -> P5 -> P6 actually changes the prompt the
LLM sees and therefore the actions it emits. Used as a shielding override
on top of the diffusion policy: a valid 0/1/2/3 from the LLM is taken;
None / illegal / parse-error falls back to the policy's argmax.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.maze_env import ACTION_DELTAS, ACTION_NAMES, TILE_FIRE, TILE_WALL
from pipeline.kag_loader import load_and_format as load_kag_text


_ACTION_TOKEN_RE = re.compile(r'"action"\s*:\s*(?:"?(\d|null|UP|DOWN|LEFT|RIGHT)"?)', re.IGNORECASE)
_NAME_TO_ID = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}


def _legal_action(env, action: int) -> bool:
    """A move is legal if it stays on grid AND the target tile is not WALL/FIRE.

    Mirrors the env's step() rules: walls bounce (no progress), fires terminate
    with a large negative reward. We treat both as illegal so the LLM can't
    end the episode by stepping into fire.
    """
    if action not in (0, 1, 2, 3):
        return False
    dr, dc = ACTION_DELTAS[action]
    r, c = env.agent_pos[0] + dr, env.agent_pos[1] + dc
    H, W = env.grid.shape
    if not (0 <= r < H and 0 <= c < W):
        return False
    tile = env.grid[r, c]
    if tile == TILE_WALL or tile == TILE_FIRE:
        return False
    return True


def _ascii_grid_with_agent(env) -> str:
    return env.get_grid_image_description()


def _format_history(history: List[Tuple[Tuple[int, int], int]]) -> str:
    if not history:
        return "(no prior steps)"
    parts = []
    for pos, act in history[-8:]:
        parts.append(f"  at {tuple(pos)} -> {ACTION_NAMES.get(act, '?')}")
    return "\n".join(parts)


def _retrieve_rag_block(rag_bank, env, recent_history) -> str:
    """Best-effort RAG retrieval: query the bank with a textual description of
    the current cell + goal + fires. Returns "" when the bank is empty or
    retrieval errors (cold-start eval is a valid scenario)."""
    if rag_bank is None:
        return ""
    try:
        fires = list(zip(*(env.grid == TILE_FIRE).nonzero()))
        query = (
            f"agent at {tuple(env.agent_pos)} heading to goal {tuple(env.goal_pos)}; "
            f"fires at {[tuple(f) for f in fires]}"
        )
        block = rag_bank.retrieve(vision_report=query, end_frame_path=None)
        return block or ""
    except Exception as e:
        return f"[RAG retrieve failed: {e!r}]"


def _retrieve_tkf_block(tkf_index_dir: Optional[Path], env) -> str:
    """Best-effort TKF retrieval against the demo CLIP index.

    Returns "" if the index isn't built or the query path errors out — the
    caller already handles missing context gracefully.
    """
    if tkf_index_dir is None:
        return ""
    try:
        from pipeline.knowledge_fetcher import _load_index, _query_index
        index, meta = _load_index(tkf_index_dir)
        if index is None or not meta:
            return ""
        fires = list(zip(*(env.grid == TILE_FIRE).nonzero()))
        q = (
            f"agent at {tuple(env.agent_pos)} goal {tuple(env.goal_pos)} "
            f"fires {[tuple(f) for f in fires]}"
        )
        results = _query_index(index, meta, q, "openai/clip-vit-large-patch14", top_k=2)
        if not results:
            return ""
        lines = ["Similar past demos:"]
        for r in results[:2]:
            lines.append(
                f"  demo {r.get('demo_id','?')} sim={r.get('similarity',0.0):.2f} "
                f"start={r.get('start_pos','?')} goal={r.get('goal_pos','?')}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"[TKF retrieve failed: {e!r}]"


class LLMActor:
    """Per-step LLM-as-shielding-override action recommender."""

    def __init__(
        self,
        profile_flags: Dict,
        llm_cfg: Dict,
        kag_doc_path: Optional[str] = None,
        rag_bank=None,
        tkf_index_dir: Optional[str] = None,
    ):
        self.flags = dict(profile_flags or {})
        self.llm_cfg = dict(llm_cfg or {})
        self.model = self.llm_cfg.get("model", "gpt-5-nano-2025-08-07")
        self.reasoning_effort = self.llm_cfg.get("reasoning_effort", "high")
        self.max_tokens = int(self.llm_cfg.get("max_output_tokens", 16384))

        self._kag_text: str = ""
        if self.flags.get("use_kag") and kag_doc_path:
            try:
                self._kag_text = load_kag_text(kag_doc_path) or ""
            except Exception as e:
                print(f"[LLMActor] KAG load failed: {e!r}; continuing without KAG.")

        self._rag_bank = rag_bank if self.flags.get("use_rag") else None
        self._tkf_index_dir = Path(tkf_index_dir) if (self.flags.get("use_tkf") and tkf_index_dir) else None

        self._client = None
        self.fallback_count = 0
        self.call_count = 0

    def _oai(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def _build_prompt(self, env, recent_history) -> List[Dict]:
        ascii_grid = _ascii_grid_with_agent(env)
        fires = [list(p) for p in zip(*(env.grid == TILE_FIRE).nonzero())]
        sys_msg = (
            "You control a maze agent. Output ONLY JSON of the form "
            '{"action": <0|1|2|3|null>}. '
            "0=UP, 1=DOWN, 2=LEFT, 3=RIGHT. Use null to abstain only if every "
            "legal move is clearly suicidal. Goal: reach the GOAL tile without "
            "entering FIRE or WALL tiles."
        )
        blocks = [
            f"Grid (A=agent, G=goal, F=fire, █=wall, ·=free):\n{ascii_grid}",
            f"agent_pos={list(env.agent_pos)} goal_pos={list(env.goal_pos)} fires={fires}",
            f"Recent steps:\n{_format_history(recent_history)}",
        ]
        if self.flags.get("use_kag") and self._kag_text:
            blocks.append("Domain knowledge (KAG):\n" + self._kag_text)
        if self.flags.get("use_rag"):
            rag = _retrieve_rag_block(self._rag_bank, env, recent_history)
            if rag:
                blocks.append("Retrieved past failures (RAG):\n" + rag)
        if self.flags.get("use_tkf"):
            tkf = _retrieve_tkf_block(self._tkf_index_dir, env)
            if tkf:
                blocks.append("Trained knowledge (TKF):\n" + tkf)
        user = "\n\n".join(blocks) + '\n\nReturn only: {"action": <int>}'
        return [{"role": "system", "content": sys_msg},
                {"role": "user",   "content": user}]

    def _parse(self, text: str) -> Optional[int]:
        if not text:
            return None
        try:
            obj = json.loads(text.strip())
            v = obj.get("action") if isinstance(obj, dict) else None
            if isinstance(v, int) and 0 <= v <= 3:
                return v
            if isinstance(v, str):
                if v.upper() in _NAME_TO_ID:
                    return _NAME_TO_ID[v.upper()]
                if v.isdigit() and 0 <= int(v) <= 3:
                    return int(v)
            return None
        except Exception:
            pass
        m = _ACTION_TOKEN_RE.search(text)
        if not m:
            return None
        tok = m.group(1)
        if tok.lower() == "null":
            return None
        if tok.upper() in _NAME_TO_ID:
            return _NAME_TO_ID[tok.upper()]
        if tok.isdigit() and 0 <= int(tok) <= 3:
            return int(tok)
        return None

    def suggest_action(self, env, recent_history) -> Optional[int]:
        """Return 0/1/2/3 if the LLM emits a legal action; None to fall back."""
        self.call_count += 1
        try:
            client = self._oai()
            messages = self._build_prompt(env, recent_history)
            use_reasoning = bool(self.flags.get("use_reasoning"))
            kwargs = {
                "model": self.model,
                "input": messages,
                "max_output_tokens": self.max_tokens,
                "reasoning": {"effort": self.reasoning_effort if use_reasoning else "low"},
            }
            r = client.responses.create(**kwargs)
            text = r.output_text or ""
        except Exception as e:
            print(f"[LLMActor] LLM call failed: {e!r}; abstaining.")
            self.fallback_count += 1
            return None

        action = self._parse(text)
        if action is None:
            self.fallback_count += 1
            return None
        if not _legal_action(env, action):
            self.fallback_count += 1
            return None
        return action
