"""Fix 4 — diversity-aware candidate selection.

The set of failures the VLM/LLM analyse each round should COVER distinct modes,
not be three instances of the same high-loss failure. We seed with the worst
peak-loss failure and greedily add the failure farthest (max-min distance, in
standardized feature space) from those already selected. The dominant cluster's
representative is force-included first so the failure we anchor the demo on is
always analysed.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .descriptor import FailureDescriptor


def _standardize(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return (X - mu) / sd


def farthest_point_select(descs: List[FailureDescriptor], k: int,
                          *, force_first: int = -1) -> List[int]:
    """Return up to k global indices: optional forced index first, then the
    worst-peak-loss failure, then greedy max-min farthest points."""
    n = len(descs)
    if n == 0:
        return []
    k = max(1, min(int(k), n))
    feats = np.asarray([d.feature() for d in descs], dtype=float)
    Xs = _standardize(feats) if n > 1 else feats

    selected: List[int] = []
    if 0 <= force_first < n:
        selected.append(force_first)
    # seed with the worst peak-loss failure (always include the hardest)
    worst = max(range(n), key=lambda i: descs[i].peak_loss)
    if worst not in selected:
        selected.append(worst)

    while len(selected) < k and len(selected) < n:
        best_i, best_d = None, -1.0
        for i in range(n):
            if i in selected:
                continue
            d = min(float(np.linalg.norm(Xs[i] - Xs[j])) for j in selected)
            if d > best_d:
                best_i, best_d = i, d
        if best_i is None:
            break
        selected.append(best_i)
    return selected[:k]
