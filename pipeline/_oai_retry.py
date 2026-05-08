"""Robust OpenAI rate-limit handling for parallel Phase B calls.

The earlier per-call retry shim was bouncing against the org-wide TPM
ceiling (200k tokens/min for gpt-5-nano) because:

  * Phase B fans out N per-failure threads inside a single process.
  * The pool-size sweep launches 4 processes in parallel (one per
    pool size) — they all share the same OpenAI org and therefore
    the same TPM ceiling.
  * On 429, the per-call retry slept the hinted ~200ms and retried
    against a still-empty bucket, exhausting all retries.

This module solves the underlying problem by stacking three layers:

  Layer 1 — TOKEN BUCKET (process-global, thread-safe):
            paces outgoing calls so the process never sustains more
            than ``target_tpm`` tokens/min. Refill rate is
            target_tpm/60 tokens/sec. ``acquire(est_tokens)`` blocks
            until the bucket has enough and then debits it.

  Layer 2 — IN-FLIGHT SEMAPHORE (process-global):
            caps the number of concurrent ``responses.create`` calls
            the process can have in flight. Without this, Phase B
            launches 25 threads at once and even with a token bucket
            they all race to the API simultaneously.

  Layer 3 — SDK NATIVE RETRY:
            ``OpenAI(max_retries=N, timeout=T)`` wires automatic
            retries with the ``Retry-After`` header that OpenAI
            actually sets on 429 responses. This is more reliable
            than parsing the prose hint we were doing before.

  Layer 4 — APP-LEVEL FALLBACK RETRY (this module):
            catches the cases where the SDK exhausted its retries
            (e.g. burst that needed >SDK budget seconds to drain)
            and reschedules the call after a longer wait.

The token bucket is sized via three env vars so the same code works
both for a single-process p4_only run and the 4-way pool sweep:

    OAI_TPM_BUDGET   org-wide TPM ceiling (default 200000)
    OAI_TPM_SHARE    fraction of OAI_TPM_BUDGET this process may use
                     (default 1.0; the sweep sets 0.22 so 4 procs
                     share the ceiling with ~10% headroom)
    OAI_MAX_IN_FLIGHT  per-process concurrent-call cap (default 4)
    OAI_SDK_MAX_RETRIES  SDK-level retry budget (default 8)
    OAI_SDK_TIMEOUT      per-request timeout in seconds (default 120)

A single process targeting 0.22 * 200_000 = 44_000 TPM is a soft
ceiling; the bucket is approximate (input-token estimation is char-
count/4) so we pick a share that leaves headroom for the estimation
error.
"""
from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from typing import Callable, TypeVar

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Configuration (env-driven so the sweep can override per-process)            #
# --------------------------------------------------------------------------- #
def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "")
    try:
        return int(raw) if raw else default
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key, "")
    try:
        return float(raw) if raw else default
    except (TypeError, ValueError):
        return default


_TPM_BUDGET        = _env_int  ("OAI_TPM_BUDGET",       200_000)
_TPM_SHARE         = _env_float("OAI_TPM_SHARE",        1.0)
_MAX_IN_FLIGHT     = _env_int  ("OAI_MAX_IN_FLIGHT",    4)
_SDK_MAX_RETRIES   = _env_int  ("OAI_SDK_MAX_RETRIES",  8)
_SDK_TIMEOUT       = _env_float("OAI_SDK_TIMEOUT",      120.0)
_TARGET_TPM        = max(1_000, int(_TPM_BUDGET * _TPM_SHARE))


# --------------------------------------------------------------------------- #
# Layer 1: token bucket                                                       #
# --------------------------------------------------------------------------- #
class _TokenBucket:
    """Process-wide thread-safe token bucket for OpenAI TPM pacing.

    Refills at ``capacity / 60`` tokens per second up to capacity.
    ``acquire(n)`` blocks until n tokens are available and debits
    them. Returns total seconds slept so the caller can log throttle
    pressure.
    """

    def __init__(self, capacity_tpm: int):
        self.capacity = float(capacity_tpm)
        self.tokens = float(capacity_tpm)
        self.refill_rate = capacity_tpm / 60.0
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, n: int) -> float:
        n = max(1, int(n))
        if n > self.capacity:
            # Single request bigger than the whole minute budget — clamp it
            # to capacity so we don't deadlock; it'll still consume one
            # full minute of budget when it lands.
            n = int(self.capacity)
        slept_total = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self.last_refill
                self.tokens = min(
                    self.capacity,
                    self.tokens + elapsed * self.refill_rate,
                )
                self.last_refill = now
                if self.tokens >= n:
                    self.tokens -= n
                    return slept_total
                deficit = n - self.tokens
                wait = max(0.05, deficit / self.refill_rate) + random.uniform(0.05, 0.25)
            time.sleep(wait)
            slept_total += wait


_BUCKET: _TokenBucket = _TokenBucket(_TARGET_TPM)


# --------------------------------------------------------------------------- #
# Layer 2: in-flight semaphore                                                #
# --------------------------------------------------------------------------- #
_SEMAPHORE = threading.Semaphore(_MAX_IN_FLIGHT)


# --------------------------------------------------------------------------- #
# Layer 3: shared SDK client with native retry / timeout                      #
# --------------------------------------------------------------------------- #
_CLIENT_LOCK = threading.Lock()
_CLIENT = None


def make_client():
    """Return a process-shared ``OpenAI`` client configured with the
    SDK's native retry + timeout handling. Reused across all callers
    so retry counters share state across calls and the connection
    pool is shared.
    """
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is not None:
            return _CLIENT
        from openai import OpenAI
        _CLIENT = OpenAI(
            max_retries=_SDK_MAX_RETRIES,
            timeout=_SDK_TIMEOUT,
        )
    return _CLIENT


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
_HINT_RE = re.compile(r"try again in\s+([\d.]+)\s*(ms|s)\b", re.IGNORECASE)


def _parse_retry_after_seconds(exc: Exception, fallback: float) -> float:
    msg = str(exc) if exc is not None else ""
    m = _HINT_RE.search(msg)
    if not m:
        return fallback
    n = float(m.group(1))
    unit = m.group(2).lower()
    return n / 1000.0 if unit == "ms" else n


def _estimate_tokens(kwargs: dict) -> int:
    """Rough request-size estimate.

    output: max_output_tokens (defaults to 1000 if absent).
    input:  len(json(input)) // 4 — OpenAI's published rule of thumb.

    Slightly over-estimating is preferable to under-estimating since
    the bucket is a soft ceiling.
    """
    out_tokens = int(kwargs.get("max_output_tokens", 1000) or 1000)
    inp = kwargs.get("input")
    inp_chars = 0
    if inp is not None:
        try:
            inp_chars = len(json.dumps(inp, default=str))
        except Exception:
            inp_chars = len(str(inp))
    return out_tokens + max(100, inp_chars // 4)


# --------------------------------------------------------------------------- #
# Layer 4: app-level fallback retry                                           #
# --------------------------------------------------------------------------- #
def call_with_retry(
    fn: Callable[..., T],
    *args,
    max_retries: int = 12,
    base_delay: float = 2.0,
    max_delay: float = 120.0,
    label: str = "oai",
    **kwargs,
) -> T:
    """Invoke ``fn`` with the full pacing+retry stack.

    Order of operations per attempt:
      1. token bucket: block until enough TPM is available.
      2. in-flight semaphore: block until a concurrency slot opens.
      3. fn(*args, **kwargs)  — SDK does its own Retry-After loop.
      4. on RateLimitError: parse hint, sleep at least that long
         (clamped by max_delay), then retry — even if the SDK gave
         up. With 4 processes pinned to 22% share each there should
         not be a permanent ceiling, but fallback retries keep us
         alive through transient bursts.
    """
    try:
        from openai import (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            RateLimitError,
        )
    except Exception:
        return fn(*args, **kwargs)

    est_tokens = _estimate_tokens(kwargs)
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        slept_in_bucket = _BUCKET.acquire(est_tokens)
        if slept_in_bucket >= 0.5 and attempt == 0:
            print(
                f"[{label}-pace] throttled {slept_in_bucket:.1f}s before issue "
                f"(target_tpm={_TARGET_TPM}, est_tokens={est_tokens})"
            )

        with _SEMAPHORE:
            try:
                return fn(*args, **kwargs)
            except RateLimitError as e:
                hint = _parse_retry_after_seconds(
                    e, fallback=base_delay * (2 ** min(attempt, 6))
                )
                # Honour the hint, but never less than 1s and never more
                # than max_delay; jitter avoids thundering herd across
                # threads / processes.
                wait = min(max(1.0, hint) + random.uniform(0.5, 2.5), max_delay)
                print(
                    f"[{label}-retry] 429 rate-limit "
                    f"(attempt {attempt + 1}/{max_retries + 1}, "
                    f"hint={hint:.2f}s); sleeping {wait:.2f}s"
                )
                last_exc = e
            except (APIConnectionError, APITimeoutError, APIStatusError) as e:
                wait = min(
                    base_delay * (2 ** attempt) + random.uniform(0.1, 1.0),
                    max_delay,
                )
                print(
                    f"[{label}-retry] transient API error "
                    f"(attempt {attempt + 1}/{max_retries + 1}); "
                    f"sleeping {wait:.2f}s: {e}"
                )
                last_exc = e

        # Sleep AFTER releasing the semaphore so other threads can make
        # progress while this one waits out its backoff.
        time.sleep(wait)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"[{label}-retry] exhausted retries without an exception")


# --------------------------------------------------------------------------- #
# Diagnostics — printed once at import time so the slurm log records the     #
# actual budget the process is operating under.                                #
# --------------------------------------------------------------------------- #
print(
    f"[oai-retry] config: TPM_BUDGET={_TPM_BUDGET} "
    f"SHARE={_TPM_SHARE:.3f} -> TARGET_TPM={_TARGET_TPM} "
    f"MAX_IN_FLIGHT={_MAX_IN_FLIGHT} "
    f"SDK_MAX_RETRIES={_SDK_MAX_RETRIES} SDK_TIMEOUT={_SDK_TIMEOUT}s",
    flush=True,
)
