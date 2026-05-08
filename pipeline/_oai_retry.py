"""Minimal retry shim for OpenAI Responses API calls.

Phase B per-episode chains call ``client.responses.create`` from many
threads concurrently (see pipeline_runner._run_phase_b_parallel). At
larger pool sizes that fan-out routinely brushes the per-org TPM
ceiling and the SDK raises ``openai.RateLimitError`` (HTTP 429). The
error message embeds OpenAI's own backoff hint, e.g. "Please try
again in 1.294s". This helper parses that hint, sleeps that long
plus a small jitter, and retries the call.

Used by reasoning.py, aggregator.py, vlm_analyser.py, and
knowledge_fetcher.py via a single ``call_with_retry`` wrapper. When
no rate-limit error is raised the wrapper is byte-equivalent to a
direct call.
"""
from __future__ import annotations

import random
import re
import time
from typing import Callable, TypeVar

T = TypeVar("T")

_HINT_RE = re.compile(r"try again in\s+([\d.]+)\s*(ms|s)\b", re.IGNORECASE)


def _parse_retry_after_seconds(exc: Exception, fallback: float) -> float:
    """Read OpenAI's "Please try again in Xms/s" hint from the error
    message. If absent, return the caller-supplied fallback."""
    msg = str(exc) if exc is not None else ""
    m = _HINT_RE.search(msg)
    if not m:
        return fallback
    n = float(m.group(1))
    unit = m.group(2).lower()
    return n / 1000.0 if unit == "ms" else n


def call_with_retry(
    fn: Callable[..., T],
    *args,
    max_retries: int = 6,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    label: str = "oai",
    **kwargs,
) -> T:
    """Call ``fn(*args, **kwargs)`` with retries on transient OpenAI
    errors (rate limits, connection blips, 5xx).

    On RateLimitError, sleep the duration the error message hints at
    (clamped) plus 100-500 ms jitter before retrying. On other
    transient errors, sleep an exponentially increasing fallback.
    Re-raises the last exception when ``max_retries`` is exhausted.
    """
    try:
        from openai import (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            RateLimitError,
        )
    except Exception:
        # SDK missing or unexpected shape — call once, let the caller
        # surface whatever it raises. Never silently swallow.
        return fn(*args, **kwargs)

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except RateLimitError as e:
            wait = _parse_retry_after_seconds(e, fallback=base_delay * (2 ** attempt))
            wait = min(wait + random.uniform(0.1, 0.5), max_delay)
            print(
                f"[{label}-retry] 429 rate-limit "
                f"(attempt {attempt + 1}/{max_retries + 1}); "
                f"sleeping {wait:.2f}s before retry"
            )
            time.sleep(wait)
            last_exc = e
        except (APIConnectionError, APITimeoutError, APIStatusError) as e:
            wait = min(
                base_delay * (2 ** attempt) + random.uniform(0.1, 0.5),
                max_delay,
            )
            print(
                f"[{label}-retry] transient API error "
                f"(attempt {attempt + 1}/{max_retries + 1}); "
                f"sleeping {wait:.2f}s: {e}"
            )
            time.sleep(wait)
            last_exc = e

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"[{label}-retry] exhausted retries without an exception")
