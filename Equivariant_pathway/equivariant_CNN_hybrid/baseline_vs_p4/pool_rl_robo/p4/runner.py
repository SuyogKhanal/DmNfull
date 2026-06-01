"""Thin text-only client for the local Qwen3-32B vLLM server.

The maze suite routes through a 2-backend proxy; here there is only ONE text
backend, so routing is unnecessary. The one proxy feature we need is stripping
Qwen3 ``<think>...</think>`` blocks (Qwen3 is a reasoning model and emits them
inline, which would break ``json.loads``). We also try to disable thinking via
``chat_template_kwargs`` and fall back gracefully if the server rejects it.

Reads ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` / ``LLM_MODEL_NAME`` from the
environment (set by run_pool_x.sh after the vLLM server is up).
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    if not text:
        return ""
    text = _THINK_RE.sub("", text)
    # Drop a dangling unclosed <think> ... (no closing tag).
    if "<think>" in text and "</think>" not in text:
        text = text.split("<think>", 1)[0]
    # If a closing tag survived without an opener, keep what follows it.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return text.strip()


def extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        pass
    s, e = text.find("{"), text.rfind("}")
    if 0 <= s < e:
        try:
            return json.loads(text[s:e + 1])
        except Exception:  # noqa: BLE001
            return None
    return None


class QwenClient:
    """Text + vision client. Routes by model name (through the proxy): text uses
    ``LLM_MODEL_NAME`` (qwen3-32b); image calls use ``VLM_MODEL_NAME``
    (qwen3-vl-32b). Pass ``images=[base64-png,...]`` to issue a VLM call."""

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 model: Optional[str] = None, vlm_model: Optional[str] = None,
                 vlm_base_url: Optional[str] = None, temperature: float = 0.0):
        from openai import OpenAI
        self.model = model or os.environ.get("LLM_MODEL_NAME", "qwen3-32b")
        self.vlm_model = vlm_model or os.environ.get("VLM_MODEL_NAME", "qwen3-vl-32b")
        self.temperature = float(temperature)
        self._no_think = True  # optimistic; flips off if the server rejects it
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "local-qwen")
        to = float(os.environ.get("OAI_SDK_TIMEOUT", "900"))
        mr = int(os.environ.get("OAI_SDK_MAX_RETRIES", "2"))
        text_base = base_url or os.environ.get("OPENAI_BASE_URL")
        self.client = OpenAI(base_url=text_base, api_key=api_key, timeout=to, max_retries=mr)
        # Vision calls go to a separate vLLM server when VLM_BASE_URL is set
        # (no proxy needed); otherwise reuse the text client (single-server/proxy).
        vlm_base = vlm_base_url or os.environ.get("VLM_BASE_URL")
        self.vlm_client = (OpenAI(base_url=vlm_base, api_key=api_key, timeout=to, max_retries=mr)
                           if vlm_base else self.client)

    @staticmethod
    def _user_content(user: str, images):
        if not images:
            return user
        content = [{"type": "text", "text": user}]
        for b64 in images:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "auto"}})
        return content

    def _create(self, messages, max_tokens: int, no_think: bool, model: str, client):
        kwargs = dict(model=model, messages=messages,
                      max_tokens=int(max_tokens), temperature=self.temperature)
        if no_think:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        return client.chat.completions.create(**kwargs)

    def chat(self, system: str, user: str, max_tokens: int = 2048,
             images=None, model: Optional[str] = None) -> str:
        client = self.vlm_client if images else self.client
        model = model or (self.vlm_model if images else self.model)
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": self._user_content(user, images)}]
        try:
            resp = self._create(messages, max_tokens, self._no_think, model, client)
        except Exception:  # noqa: BLE001 — server may reject extra_body
            self._no_think = False
            resp = self._create(messages, max_tokens, False, model, client)
        text = resp.choices[0].message.content or ""
        return strip_think(text)

    def chat_json(self, system: str, user: str, retries: int = 3,
                  max_tokens: int = 2048, images=None, model: Optional[str] = None) -> dict:
        last = ""
        for i in range(int(retries)):
            u = user if i == 0 else (user + "\n\nReturn ONLY a valid JSON object, no prose.")
            last = self.chat(system, u, max_tokens=max_tokens, images=images, model=model)
            obj = extract_json(last)
            if isinstance(obj, dict):
                return obj
        raise ValueError(f"no valid JSON after {retries} tries: {last[:200]!r}")
