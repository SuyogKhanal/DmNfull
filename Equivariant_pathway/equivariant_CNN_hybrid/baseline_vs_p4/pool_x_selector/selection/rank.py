"""Highest-loss episode ranker for DAgger-style selection.

Verbatim ranking semantics from upstream ``baseline_budget.py``:

    primary:   -policy_loss   (higher loss first)
    secondary: -n_steps       (more off-policy meandering first on ties)
    tertiary:  jitter         (deterministic random tiebreak)

We expose it as a clean function so both the baseline selector and the
P4 sequential selector (when it wants to pick a single failure to focus on)
can call it.
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple


def rank_failures(
    failures: List[Dict], rng_seed: int,
) -> List[Tuple[float, float, float, Dict]]:
    """Return failures decorated with the (loss, steps, jitter) sort key.

    Each input dict must contain ``policy_loss`` and ``n_steps``. The
    returned list is sorted; the highest-priority failure is first. The
    fourth tuple element is the original record so the caller can pick
    a demo without losing context.
    """
    rng = random.Random(int(rng_seed))
    decorated: List[Tuple[float, float, float, Dict]] = []
    for rec in failures:
        jitter = rng.random()
        decorated.append((
            -float(rec.get("policy_loss", 0.0)),
            -float(rec.get("n_steps", 0)),
            jitter,
            rec,
        ))
    decorated.sort()
    return decorated


def top_one(failures: List[Dict], rng_seed: int) -> Dict | None:
    """Convenience: return only the single highest-priority failure
    record, or ``None`` if the input is empty.
    """
    ranked = rank_failures(failures, rng_seed)
    return ranked[0][3] if ranked else None
