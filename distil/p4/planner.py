"""RobosuiteHybridPlanner — the V3-hybrid decision engine for robosuite.

Per round: cluster the screening failures into modes (reusable compression core),
pick a dominant target cluster with cross-round memory rotation, then decide how
to spend ONE demo — SELECT (on-policy correct one real failure) or BRIDGE
(prescribe a middle-ground scene covering 2-3). The choice is the LLM's (parsed
from its 'SELECT ep#' / 'BRIDGE ep#,ep#' label); with no LLM a GEOMETRIC fallback
is used (tight cluster -> SELECT the representative; spread -> BRIDGE the extremes),
so the loop is testable before the LLM is wired.

Mirrors the ManiSkill StackCubeHybridPlanner; reuses clustering/diversity/memory/
telemetry/parse verbatim, and dispatches to the robosuite collect primitives.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import cluster_failures, farthest_point_select, CentroidMemory, Telemetry, parse_choice
from .bounds import clamp_obj_xy
from .clustering import Cluster, ClusterResult
from .collect import collect_bridge, collect_select
from .descriptor import RSFailureDescriptor


def _single_cluster(descs) -> ClusterResult:
    """clustering=off: one all-in-one group; rep = highest peak-loss (no modes)."""
    idxs = list(range(len(descs)))
    cp = [d.centroid_pos() for d in descs]
    cx = float(np.mean([p[0] for p in cp])); cy = float(np.mean([p[1] for p in cp]))
    cz = float(np.mean([p[2] for p in cp]))
    ct = math.atan2(float(np.mean([math.sin(d.centroid_theta()) for d in descs])),
                    float(np.mean([math.cos(d.centroid_theta()) for d in descs])))
    rep = max(idxs, key=lambda i: descs[i].peak_loss)
    c = Cluster(label=0, member_idxs=idxs, centroid_xyz=[cx, cy, cz], centroid_theta=ct,
                mean_peak_loss=float(np.mean([d.peak_loss for d in descs])),
                size=len(idxs), representative_idx=rep)
    return ClusterResult(clusters=[c], labels=[0] * len(descs), method="clustering_off", dominant=c)


@dataclass
class LayoutSpec:
    mode: str                      # "select" | "bridge"
    choice: str                    # provenance tag: select | bridge | escalated_select | geometric_*
    cand: Optional[Dict[str, Any]] = None   # SELECT: {seed, exec_actions, t_star}
    bridge_xy: Optional[List[float]] = None  # BRIDGE: middle-ground (x, y)
    bridge_seed: Optional[int] = None
    cited: List[int] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    bridge_z: Optional[float] = None         # BRIDGE full pose (Door): z ...
    bridge_quat: Optional[List[float]] = None  # ... and orientation [w,x,y,z]

    def digest(self) -> Dict[str, Any]:
        return {"mode": self.mode, "choice": self.choice, "cited": self.cited,
                "cand_seed": (self.cand or {}).get("seed"),
                "bridge_xy": [round(v, 4) for v in self.bridge_xy] if self.bridge_xy else None,
                "bridge_z": self.bridge_z, "bridge_quat": self.bridge_quat,
                "bridge_seed": self.bridge_seed, "provenance": self.provenance}


class RobosuiteHybridPlanner:
    def __init__(self, work_dir: str, task: str, cfg: Dict[str, Any]):
        self.task = task
        self.cfg = dict(cfg or {})
        # Wipe randomizes a dirt-marker PATH (no single object pose) -> SELECT-only.
        # bridge=off ablation (05_..md) forces SELECT-only on Lift/Door too.
        self.bridge_supported = (task in ("Lift", "Door")) and bool(self.cfg.get("bridge", True))
        self.max_k = int(self.cfg.get("max_clusters", 4))
        self.cap = int(self.cfg.get("analyze_cap", 4))
        self.snap_eps = float(self.cfg.get("snap_eps", 0.02))
        self.bridge_seed_base = int(self.cfg.get("bridge_seed_base", 8_000_000))
        self.tele = Telemetry(work_dir)
        self.memory = CentroidMemory(str(Path(work_dir) / "telemetry" / "centroid_memory.json"),
                                     gamma=float(self.cfg.get("memory_gamma", 0.6)),
                                     sigma=float(self.cfg.get("memory_sigma", 0.06)))
        # Eq 9 lambda: 1.0 = full DISTIL; 0.0 = cluster-memory OFF (Tier-1 knockout,
        # 05_..md). Threaded into memory.select_target so `memory.lambda=0` is one flag.
        self.lam = float(self.cfg.get("memory_lambda", 1.0))
        # Tier-1 allocation knockouts (05_..md): 'random' => correct a uniformly random
        # recorded failure each round (Stagger control); clustering 'off' => one all-in-one
        # group, no modes (target highest-peak-loss).
        self.allocation = self.cfg.get("allocation", "distil")
        self.clustering = self.cfg.get("clustering", "silhouette")
        import numpy as _np
        self._rng = _np.random.default_rng(int(self.cfg.get("_seed", 0)))
        # per-round state
        self._rnd = -1
        self._descs: List[RSFailureDescriptor] = []
        self._cr = None
        self._target = None
        self._tried_seeds: set = set()
        self._bridge_count = 0

    # ── per round ────────────────────────────────────────────────────────────
    def set_round(self, rnd: int, descs: List[RSFailureDescriptor]) -> None:
        self._rnd = rnd
        self._descs = descs
        self._tried_seeds = set()
        if self.clustering == "off":
            # no mode structure: one all-in-one cluster, rep = highest peak-loss.
            self._cr = _single_cluster(descs)
            self._target = self._cr.dominant
        else:
            self._cr = cluster_failures(descs, max_k=self.max_k)
            self._target, mem_scores = self.memory.select_target(
                self._cr.clusters, self._cr.dominant, descs, now=rnd, lam=self.lam)
        sel = farthest_point_select(descs, self.cap,
                                    force_first=self._target.representative_idx)
        self.tele.event(rnd, "round_setup", {
            "n_failures": len(descs), "cluster_method": self._cr.method,
            "n_clusters": len(self._cr.clusters),
            "clusters": [{"label": c.label, "size": c.size,
                          "members": [descs[i].episode_id for i in c.member_idxs],
                          "centroid_xy": [round(v, 4) for v in c.centroid_xyz[:2]],
                          "is_target": c is self._target} for c in self._cr.clusters],
            "target_label": self._target.label,
            "analyze_ep_ids": [descs[i].episode_id for i in sel],
            "descriptors": [d.digest() for d in descs]})

    def target_descs(self, cap: int = None) -> List[RSFailureDescriptor]:
        """Target-cluster member descriptors the multi-stage LLM analyses (VLM +
        reasoning per failure). Capped by analyze_cap to bound LLM cost."""
        if self._target is None:
            return []
        cap = int(cap or self.cap)
        return [self._descs[i] for i in self._target.member_idxs[:cap]]

    # ── round diagnostics (for 09_..md telemetry) ────────────────────────────
    def k_star(self) -> int:
        return len(self._cr.clusters) if self._cr else 0

    def cluster_method(self) -> Optional[str]:
        return self._cr.method if self._cr else None

    def target_label(self):
        return self._target.label if self._target is not None else None

    def decision_context(self) -> str:
        """Text block appended to the LLM prompt (the target cluster's members).
        On SELECT-only tasks (Wipe) the BRIDGE option is omitted."""
        t = self._target
        lines = []
        for i in t.member_idxs[:8]:
            d = self._descs[i]
            ox, oy = d.snap["obj_xyz"][0], d.snap["obj_xyz"][1]
            hint = ""
            if "grasp" in d.snap:
                hint = f" grasp={d.snap.get('grasp')}"
            elif "hinge" in d.snap:
                hint = f" hinge={d.snap.get('hinge'):.3f}"
            elif "prop_wiped" in d.snap:
                hint = f" wiped={d.snap.get('prop_wiped'):.2f}"
            lines.append(f"    - ep{d.episode_id}: object_xy=({ox:.3f},{oy:.3f})"
                         f"{hint} progress={d.t_star}/{d.T} peak_loss={d.peak_loss:.4f}")
        members = "\n".join(lines)
        select_line = (
            "  (A) SELECT ep<ID> — one recorded failure represents the mode; re-run "
            "that exact scene and the expert corrects it on-policy from where the "
            "policy diverged. Use when the cluster is tight / one failure dominates.\n")
        bridge_line = (
            "  (B) BRIDGE ep<ID>,ep<ID> — no single failure covers the mode; prescribe "
            "ONE new object placement in the MIDDLE GROUND between 2-3 cited failures; "
            "the expert demonstrates from there. Use when members are spread out.\n")
        out_line = ("  OUTPUT: emit a line beginning 'SELECT ep<ID>'"
                    + (" or 'BRIDGE ep<ID>,ep<ID>'.\n" if self.bridge_supported
                       else ". (This task supports SELECT only.)\n"))
        return (
            f"DEMONSTRATION DECISION (robosuite {self.task}). One demo this round. Choose:\n"
            + select_line + (bridge_line if self.bridge_supported else "")
            + f"  Target failure cluster (label {t.label}, {t.size} member(s)):\n{members}\n"
            + out_line)

    # ── decision ─────────────────────────────────────────────────────────────
    def _by_ep(self, eid: int) -> Optional[RSFailureDescriptor]:
        for d in self._descs:
            if d.episode_id == int(eid):
                return d
        return None

    def _cand(self, d: RSFailureDescriptor) -> Dict[str, Any]:
        return {"seed": d.seed, "exec_actions": d.exec_actions, "t_star": d.t_star}

    def _nearest_untried(self) -> Optional[RSFailureDescriptor]:
        """Nearest-to-centroid untried member of the target cluster (safety floor)."""
        t = self._target
        order = sorted(t.member_idxs, key=lambda i: self._descs[i].peak_loss, reverse=True)
        for i in order:
            if self._descs[i].seed not in self._tried_seeds:
                return self._descs[i]
        return self._descs[t.representative_idx] if self._descs else None

    def _target_spread(self) -> float:
        pts = np.asarray([[self._descs[i].snap["obj_xyz"][0], self._descs[i].snap["obj_xyz"][1]]
                          for i in self._target.member_idxs], dtype=float)
        if len(pts) < 2:
            return 0.0
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        return float(d.max())

    def _select_spec(self, m: RSFailureDescriptor, choice: str = "select",
                     **prov) -> LayoutSpec:
        self._tried_seeds.add(m.seed)
        return LayoutSpec(mode="select", choice=choice, cand=self._cand(m),
                          cited=[m.episode_id], provenance=prov)

    def _bridge_spec(self, cited: List[RSFailureDescriptor]) -> LayoutSpec:
        xy = np.mean([[c.snap["obj_xyz"][0], c.snap["obj_xyz"][1]] for c in cited], axis=0)
        x, y = clamp_obj_xy(self.task, float(xy[0]), float(xy[1]))
        self._bridge_count += 1
        # Door has no free joint: keep the representative failure's real z + quat
        # (a valid door orientation) and only move xy to the middle ground.
        z, quat = None, None
        if self.task == "Door":
            rep = cited[0]
            z = float(rep.snap["obj_xyz"][2])
            quat = list(rep.snap.get("obj_quat", []))
        return LayoutSpec(mode="bridge", choice="bridge", bridge_xy=[x, y],
                          bridge_z=z, bridge_quat=quat,
                          bridge_seed=self.bridge_seed_base + self._rnd * 17 + self._bridge_count,
                          cited=[c.episode_id for c in cited],
                          provenance={"n_cited": len(cited)})

    def decide(self, label: Optional[str], attempt: int) -> LayoutSpec:
        """Map an LLM label (or geometric fallback) to a LayoutSpec. On retry
        (attempt>=1) escalate to SELECT of the nearest untried failure (safety floor)."""
        # Tier-1 random-allocation control (Stagger): SELECT a uniformly random recorded
        # failure each round — no clustering, no memory, no LLM.
        if self.allocation == "random" and attempt == 0 and self._descs:
            m = self._descs[int(self._rng.integers(len(self._descs)))]
            return self._select_spec(m, "random_alloc")
        if attempt >= 1 or not self._descs:
            m = self._nearest_untried()
            if m is None:                                   # nothing untried left
                if self.bridge_supported:
                    return self._bridge_spec([self._descs[i] for i in self._target.member_idxs[:2]])
                m = self._descs[self._target.representative_idx]
            return self._select_spec(m, "escalated_select", attempt=attempt)

        choice, ids = parse_choice(label or "")
        if choice is None:                                  # geometric fallback
            if (not self.bridge_supported or self._target_spread() <= self.snap_eps
                    or self._target.size < 2):
                return self._select_spec(self._descs[self._target.representative_idx],
                                         "geometric_select")
            idxs = self._target.member_idxs
            pts = [[self._descs[i].snap["obj_xyz"][0], self._descs[i].snap["obj_xyz"][1]] for i in idxs]
            D = np.linalg.norm(np.asarray(pts)[:, None] - np.asarray(pts)[None], axis=-1)
            a, b = np.unravel_index(int(D.argmax()), D.shape)
            spec = self._bridge_spec([self._descs[idxs[a]], self._descs[idxs[b]]])
            spec.choice = "geometric_bridge"
            return spec

        # SELECT (or, on a SELECT-only task, coerce a BRIDGE label into SELECT)
        if choice == "select" or not self.bridge_supported:
            m = self._by_ep(ids[0]) if (choice == "select" and ids) else None
            if m is None or m.seed in self._tried_seeds:
                m = self._nearest_untried() or self._descs[self._target.representative_idx]
            return self._select_spec(m, "select")

        cited = [m for m in (self._by_ep(i) for i in ids) if m is not None]
        if not cited:
            cited = [self._descs[i] for i in self._target.member_idxs[:2]]
        return self._bridge_spec(cited)

    # ── collect one demo for a spec ──────────────────────────────────────────
    def collect(self, spec: LayoutSpec, env, expert, cfg) -> tuple:
        ms = int(cfg["env_horizon"])
        img = int(cfg["image_size"]) if cfg.get("modality") == "image" else None
        if spec.mode == "select":
            demo, ok, t = collect_select(env, expert, seed=spec.cand["seed"],
                                         exec_actions=spec.cand["exec_actions"],
                                         t_star=spec.cand["t_star"], max_steps=ms, image_size=img)
            return demo, ok, {"len": t}
        demo, ok, t, applied = collect_bridge(env, expert, task=self.task,
                                              x=spec.bridge_xy[0], y=spec.bridge_xy[1],
                                              z=spec.bridge_z, quat=spec.bridge_quat,
                                              seed=spec.bridge_seed, max_steps=ms, image_size=img)
        return demo, ok, {"len": t, "applied": applied}

    def note_collect(self, spec: LayoutSpec, ok: bool, meta: Dict[str, Any]) -> None:
        self.tele.event(self._rnd, "collect", {"spec": spec.digest(), "success": ok, **meta})
        if ok and self._target is not None:
            self.memory.append(self._rnd, self._target.centroid_xyz, self._target.centroid_theta)
            self.tele.summary_row({"round": self._rnd, "mode": spec.mode, "choice": spec.choice,
                                   "cited": spec.cited, "len": meta.get("len")})
