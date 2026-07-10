"""Cross-round prescribed-centroid memory + recency penalty (Suyog's Fix-1 doubt).

Because we retrain + re-rollout after EVERY single demo, the held-out failure
distribution shifts each round. Re-clustering fresh each round handles that, but
a persistent failure mode could be prescribed over and over. This memory records
the (round, centroid) we prescribed each round and applies a Gaussian recency
penalty so coverage ROTATES: among near-equal-dominance clusters, prefer the one
whose centroid is farthest from what we recently covered.

Persisted as JSON so it survives the (sequential) per-round process and is
auditable in the run dir.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional


class CentroidMemory:
    def __init__(self, path: str, *, gamma: float = 0.6, sigma: float = 0.06):
        self.path = Path(path)
        self.gamma = float(gamma)
        self.sigma = float(sigma)
        self.entries: List[Dict] = []
        self._load()

    def _load(self) -> None:
        try:
            if self.path.is_file():
                data = json.loads(self.path.read_text())
                self.entries = list(data.get("entries", []))
                self.gamma = float(data.get("gamma", self.gamma))
                self.sigma = float(data.get("sigma", self.sigma))
        except Exception:
            self.entries = []

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({
                "gamma": self.gamma, "sigma": self.sigma,
                "entries": self.entries}, indent=2, default=str))
        except Exception:
            pass

    def append(self, rnd: int, centroid_xyz, centroid_theta: float) -> None:
        self.entries.append({
            "round": int(rnd),
            "centroid_xyz": [float(v) for v in centroid_xyz],
            "centroid_theta": float(centroid_theta),
        })
        self._save()

    def recency_penalty(self, centroid_xyz, *, now: int) -> float:
        """Sum of gamma^(now-round) * exp(-||c - c_i||^2 / (2 sigma^2)) over the
        x,y plane. ~1 when c sits on a very recently-covered point, →0 otherwise."""
        if not self.entries:
            return 0.0
        cx, cy = float(centroid_xyz[0]), float(centroid_xyz[1])
        pen = 0.0
        two_s2 = 2.0 * self.sigma * self.sigma
        for e in self.entries:
            r = int(e.get("round", 0))
            w = self.gamma ** max(0, now - r)
            ex, ey = e["centroid_xyz"][0], e["centroid_xyz"][1]
            d2 = (cx - ex) ** 2 + (cy - ey) ** 2
            pen += w * math.exp(-d2 / two_s2)
        return float(pen)

    def select_target(self, clusters, dominant, descs, *, now: int, lam: float = 1.0):
        """Among clusters that tie the dominant on size (within 1 member), pick
        the one maximizing (mean_peak_loss - lam * recency_penalty(centroid)).
        Returns (chosen_cluster, debug_scores). Rotates coverage off recently
        prescribed regions while still favouring high-loss modes."""
        if not clusters:
            return dominant, []
        top_size = dominant.size
        cands = [c for c in clusters if c.size >= top_size - 1]
        if not cands:
            cands = [dominant]
        scores = []
        best, best_score = None, None
        for c in cands:
            pen = self.recency_penalty(c.centroid_xyz, now=now)
            score = c.mean_peak_loss - lam * pen
            scores.append({"label": c.label, "size": c.size,
                           "mean_peak_loss": round(c.mean_peak_loss, 6),
                           "recency_penalty": round(pen, 6),
                           "score": round(score, 6)})
            if best_score is None or score > best_score:
                best, best_score = c, score
        return best or dominant, scores
