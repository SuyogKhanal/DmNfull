"""GridWorld geometric failure descriptor (Eq 7 for GridWorld).

Anchored at t_flag (first uncertainty-threshold crossing). Features (geometric, no
pixels — identical for state and image modality, 02_..md #4):
    [ar, ac, dr=gr-ar, dc=gc-ac, rho (path progress), manhattan/2(gs-1)]
Satisfies the duck-type contract of distil/p4/clustering: cluster_feature(),
centroid_pos() (planar xy = agent cell), centroid_theta() (0 — no yaw), peak_loss
(here = the flag entropy), episode_id (the layout seed = scene identity for re-roll).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from .expert import build_grid, compute_distance_map, INF


@dataclass
class GWFailureDescriptor:
    episode_id: int                 # layout seed = scene identity (SELECT re-roll)
    seed: int
    t_star: int                     # t_flag anchor / takeover step
    T: int
    peak_loss: float                # flag entropy (self-uncertainty)
    layout: Dict[str, Any]          # {grid, start, goal, fires}
    agent_cell: List[int]           # agent (r,c) at t_flag
    exec_actions: List[int]         # policy actions (replay to t_flag for SELECT)

    def _dist(self):
        grid = build_grid(len(self.layout["grid"]), self.layout["fires"], self.layout["goal"],
                          wall_template=np.asarray(self.layout["grid"], dtype=np.int32))
        return compute_distance_map(grid, tuple(self.layout["goal"]))

    def feature(self) -> List[float]:
        gs = len(self.layout["grid"])
        ar, ac = self.agent_cell
        gr, gc = self.layout["goal"]
        dmap = self._dist()
        d_here = float(dmap[ar, ac])
        d_start = float(dmap[self.layout["start"][0], self.layout["start"][1]])
        if d_start >= INF or d_start <= 0 or d_here >= INF:
            rho = self.t_star / max(1, self.T)
        else:
            rho = 1.0 - d_here / d_start
        manh = abs(ar - gr) + abs(ac - gc)
        return [float(ar), float(ac), float(gr - ar), float(gc - ac),
                float(rho), manh / max(1.0, 2.0 * (gs - 1))]

    def cluster_feature(self) -> List[float]:
        return self.feature()

    def centroid_pos(self) -> List[float]:
        return [float(self.agent_cell[0]), float(self.agent_cell[1]), 0.0]

    def centroid_theta(self) -> float:
        return 0.0

    def digest(self) -> Dict[str, Any]:
        return {"episode_id": self.episode_id, "seed": self.seed, "t_star": self.t_star,
                "T": self.T, "peak_loss": round(self.peak_loss, 6),
                "agent_cell": list(self.agent_cell), "goal": list(self.layout["goal"]),
                "start": list(self.layout["start"]), "fires": self.layout["fires"],
                "feature": [round(v, 4) for v in self.feature()]}
