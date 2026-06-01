"""Highest-discrepancy candidate ranker (continuous-control analogue of
pool_x_selector's ``selection/rank.py`` highest-loss ranker).

    primary:   -discrepancy   (larger ||a_nov - a_exp||_2 first)
    secondary: -reward_to_go  (more upside ahead first, on ties)
    tertiary:  jitter         (deterministic random tiebreak)
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional


def rank_candidates(records: List[Dict], rng_seed: int) -> List[Dict]:
    rng = random.Random(int(rng_seed))
    keyed = [((-float(r.get("discrepancy", 0.0)),
               -float(r.get("reward_to_go", 0.0)),
               rng.random()), r) for r in records]
    keyed.sort(key=lambda t: t[0])
    return [r for _k, r in keyed]


def top_one(records: List[Dict], rng_seed: int) -> Optional[Dict]:
    ranked = rank_candidates(records, rng_seed)
    return ranked[0] if ranked else None
