"""GridWorldHybridPlanner — the DISTIL decision engine for GridWorld.

Identical structure to distil/p4/planner.py (robosuite): cluster the screening
failures (shared silhouette-k), rotate the target via Eq-9 cluster memory, then the
LLM chooses SELECT (BFS-correct one failed layout from t_flag) or BRIDGE (a
middle-ground corridor placement between 2-3 cited failures). Reuses the task-agnostic
compression core verbatim; only the descriptor + collect primitives are GridWorld-
specific. Geometric fallback + in-round infeasibility escalation match the robot side.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..p4 import CentroidMemory, Telemetry, cluster_failures, farthest_point_select, parse_choice
from ..p4.planner import _single_cluster
from .collect import collect_bridge_gw, collect_select_gw


@dataclass
class GWLayoutSpec:
    mode: str                       # "select" | "bridge"
    choice: str                     # provenance tag
    desc: Optional[Any] = None      # SELECT: the target descriptor
    cited: List[Any] = field(default_factory=list)   # BRIDGE: cited descriptors
    cited_ids: List[int] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def digest(self) -> Dict[str, Any]:
        return {"mode": self.mode, "choice": self.choice, "cited": self.cited_ids,
                "sel_ep": self.desc.episode_id if self.desc else None,
                "provenance": self.provenance}


class GridWorldHybridPlanner:
    def __init__(self, work_dir: str, cfg: Dict[str, Any]):
        self.task = "GridWorld"
        self.cfg = dict(cfg or {})
        self.bridge_supported = bool(self.cfg.get("bridge", True))
        self.max_k = int(self.cfg.get("max_clusters", 4))
        self.cap = int(self.cfg.get("analyze_cap", 3))
        self.snap_eps = float(self.cfg.get("snap_eps", 1.0))    # cells: tight cluster => SELECT
        self.tele = Telemetry(work_dir)
        self.memory = CentroidMemory(str(Path(work_dir) / "telemetry" / "centroid_memory.json"),
                                     gamma=float(self.cfg.get("memory_gamma", 0.6)),
                                     sigma=float(self.cfg.get("memory_sigma", 0.06)))
        self.lam = float(self.cfg.get("memory_lambda", 1.0))
        self.allocation = self.cfg.get("allocation", "distil")
        self.clustering = self.cfg.get("clustering", "silhouette")
        self._rng = np.random.default_rng(int(self.cfg.get("_seed", 0)))
        self._rnd = -1
        self._descs: List[Any] = []
        self._cr = None
        self._target = None
        self._tried: set = set()

    def set_round(self, rnd: int, descs: List[Any]) -> None:
        self._rnd = rnd
        self._descs = descs
        self._tried = set()
        if self.clustering == "off":
            self._cr = _single_cluster(descs)
            self._target = self._cr.dominant
        else:
            self._cr = cluster_failures(descs, max_k=self.max_k)
            self._target, _scores = self.memory.select_target(
                self._cr.clusters, self._cr.dominant, descs, now=rnd, lam=self.lam)
        sel = farthest_point_select(descs, self.cap, force_first=self._target.representative_idx)
        self.tele.event(rnd, "round_setup", {
            "n_failures": len(descs), "cluster_method": self._cr.method,
            "n_clusters": len(self._cr.clusters), "target_label": self._target.label,
            "clusters": [{"label": c.label, "size": c.size,
                          "members": [descs[i].episode_id for i in c.member_idxs],
                          "centroid_rc": [round(v, 2) for v in c.centroid_xyz[:2]],
                          "is_target": c is self._target} for c in self._cr.clusters],
            "analyze_ep_ids": [descs[i].episode_id for i in sel],
            "descriptors": [d.digest() for d in descs]})

    # ── round diagnostics ─────────────────────────────────────────────────────
    def k_star(self) -> int:
        return len(self._cr.clusters) if self._cr else 0

    def cluster_method(self):
        return self._cr.method if self._cr else None

    def target_label(self):
        return self._target.label if self._target is not None else None

    def target_descs(self) -> List[Any]:
        if self._target is None:
            return []
        return [self._descs[i] for i in self._target.member_idxs[:self.cap]]

    def _by_ep(self, eid: int):
        for d in self._descs:
            if d.episode_id == int(eid):
                return d
        return None

    def _nearest_untried(self):
        order = sorted(self._target.member_idxs, key=lambda i: self._descs[i].peak_loss, reverse=True)
        for i in order:
            if self._descs[i].episode_id not in self._tried:
                return self._descs[i]
        return self._descs[self._target.representative_idx] if self._descs else None

    def _spread(self) -> float:
        pts = np.asarray([self._descs[i].agent_cell for i in self._target.member_idxs], dtype=float)
        if len(pts) < 2:
            return 0.0
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        return float(d.max())

    def _select_spec(self, d, choice="select", **prov) -> GWLayoutSpec:
        self._tried.add(d.episode_id)
        return GWLayoutSpec(mode="select", choice=choice, desc=d,
                            cited_ids=[d.episode_id], provenance=prov)

    def _bridge_spec(self, cited, choice="bridge") -> GWLayoutSpec:
        return GWLayoutSpec(mode="bridge", choice=choice, cited=cited,
                            cited_ids=[c.episode_id for c in cited],
                            provenance={"n_cited": len(cited)})

    def decide(self, label: Optional[str], attempt: int) -> GWLayoutSpec:
        if self.allocation == "random" and attempt == 0 and self._descs:
            d = self._descs[int(self._rng.integers(len(self._descs)))]
            return self._select_spec(d, "random_alloc")
        if attempt >= 1 or not self._descs:
            d = self._nearest_untried()
            if d is None and self.bridge_supported:
                return self._bridge_spec([self._descs[i] for i in self._target.member_idxs[:2]],
                                         "escalated_bridge")
            return self._select_spec(d or self._descs[self._target.representative_idx],
                                     "escalated_select", attempt=attempt)

        choice, ids = parse_choice(label or "")
        if choice is None:                                   # geometric fallback
            if not self.bridge_supported or self._spread() <= self.snap_eps or self._target.size < 2:
                return self._select_spec(self._descs[self._target.representative_idx], "geometric_select")
            idxs = self._target.member_idxs
            pts = np.asarray([self._descs[i].agent_cell for i in idxs], dtype=float)
            D = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
            a, b = np.unravel_index(int(D.argmax()), D.shape)
            return self._bridge_spec([self._descs[idxs[a]], self._descs[idxs[b]]], "geometric_bridge")

        if choice == "select" or not self.bridge_supported:
            d = self._by_ep(ids[0]) if ids else None
            if d is None or d.episode_id in self._tried:
                d = self._nearest_untried() or self._descs[self._target.representative_idx]
            return self._select_spec(d, "select")

        cited = [d for d in (self._by_ep(i) for i in ids) if d is not None]
        if not cited:
            cited = [self._descs[i] for i in self._target.member_idxs[:2]]
        return self._bridge_spec(cited, "bridge")

    def collect(self, spec: GWLayoutSpec):
        if spec.mode == "select":
            return collect_select_gw(spec.desc)
        return collect_bridge_gw(spec.cited)

    def note_collect(self, spec: GWLayoutSpec, ok: bool, meta: Dict[str, Any]) -> None:
        self.tele.event(self._rnd, "collect", {"spec": spec.digest(), "success": ok, **meta})
        if ok and self._target is not None:
            self.memory.append(self._rnd, self._target.centroid_xyz, self._target.centroid_theta)
            self.tele.summary_row({"round": self._rnd, "mode": spec.mode,
                                   "choice": spec.choice, "cited": spec.cited_ids, **meta})
