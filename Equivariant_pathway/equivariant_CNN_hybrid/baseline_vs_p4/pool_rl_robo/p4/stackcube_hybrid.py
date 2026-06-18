"""StackCube V3 HYBRID — the LLM decides SELECT vs BRIDGE each round.

This is the StackCube port of the PushT V3 hybrid (``p4_subtask``). PushT's hybrid
hooks into the FORK pipeline; StackCube uses the suite-native engine
(``p4/select_arm.py::run_p4_top3_arm``), so this module is injected INLINE there
(gated behind ``p4.subtask.collect: hybrid``), NOT via fork hooks.

Per round, ``run_p4_top3_arm`` already rolls ``n_cand`` candidates and keeps the
FAILURES. This module:

  1. Builds a cube descriptor per failure by REPLAYING the recorded
     ``exec_actions`` to ``t*`` and snapshotting cubeA/cubeB pose + grasp state
     (the select_arm engine has no meta.json — failures are in-memory candidates).
  2. Clusters the failures on a 6-d cube signature, picks the dominant cluster
     (memory-rotated so coverage rotates across rounds).
  3. Builds a DECISION CONTEXT block appended to the cube-prescriber prompt: the
     LLM emits ONE cube layout AND prefixes its rationale with ``SELECT ep<id>``
     or ``BRIDGE ep<a>,ep<b>``.
  4. Maps that choice to a ``CubeLayoutSpec``:
       * SELECT  → on-policy correction of the cited REAL failure
                   (``_correct_onpolicy_from`` — re-roll its seed, expert takes
                   over from the divergence state). The faithful DAgger primitive.
       * BRIDGE  → the LLM's middle-ground cube layout, realised by
                   ``StackCube-Start-v0.set_prescription`` + motion-planner solve
                   (``_collect_prescribed_demo``). The creative compression.
  5. Any in-round retry (the infeasibility loop) ESCALATES to SELECT of the
     nearest untried real failure — a bad bridge can never waste the round.

Reuses the task-agnostic compression core verbatim: clustering (silhouette
k-sweep + ``pick_dominant``), ``CentroidMemory`` recency rotation, and the
``Telemetry`` writer. Cube geometry / prompt text are StackCube-specific here.
With the flag absent the planner is never instantiated and p4_top3 is byte-
identical (pure prescription), preserving the apples-to-apples baseline.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..p4_subtask.clustering import (
    _best_k_clustering, _standardize, pick_dominant, Cluster, ClusterResult,
)
from ..p4_subtask.memory import CentroidMemory
from ..p4_subtask.telemetry import Telemetry


# StackCube table workspace for cube centres — MIRRORS envs/stackcube_start.py
# (set_prescription clamps to these too; kept here for clamping + prompt text).
CUBE_X: tuple = (-0.15, 0.15)
CUBE_Y: tuple = (-0.25, 0.25)
CUBE_Z: float = 0.02
MIN_SEP: float = 0.05


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def yaw_from_quat_wxyz(q) -> float:
    """Z-yaw (radians) from a [w, x, y, z] quaternion."""
    if q is None or len(q) < 4:
        return 0.0
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def parse_choice(label: str):
    """Parse the LLM's explicit decision from free text (the prescriber rationale).
    Returns ("select", [id]) | ("bridge", [ids]) | (None, []). Same protocol as the
    PushT planner's _parse_choice (kept inline to avoid importing the tee-coupled
    planner module)."""
    s = str(label or "")
    m = re.search(r"\bSELECT\s*[: ]\s*(?:ep\s*)?(-?\d+)", s, re.IGNORECASE)
    if m:
        return "select", [int(m.group(1))]
    m = re.search(r"\bBRIDGE\s*[: ]\s*((?:(?:ep\s*)?-?\d+[,\s]*)+)", s, re.IGNORECASE)
    if m:
        ids = [int(x) for x in re.findall(r"-?\d+", m.group(1))]
        return ("bridge", ids[:3]) if ids else (None, [])
    return None, []


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class StackCubeFailureDescriptor:
    """One failed candidate's reconstructed t* state. ``cand`` is the live
    select_arm candidate dict ({seed, exec_actions, t_star, ...}) so a SELECT
    can re-roll + correct it on-policy."""
    episode_id: int            # == screening seed (scene identity for SELECT)
    seed: int
    peak_loss: float
    t_star: int
    T: int
    cubeA_xyz: List[float]     # [x, y, z] of the cube to pick, at t*
    cubeA_zrot: float
    cubeB_xyz: List[float]     # [x, y, z] of the base cube, at t*
    cubeB_zrot: float
    grasp: float               # 1.0 if the gripper is grasping cubeA at t*, else 0
    cand: Dict[str, Any] = field(default_factory=dict, repr=False)

    def feature(self) -> List[float]:
        """6-d clustering signature: where cubeA is, the cubeA→cubeB xy offset
        (the alignment error that distinguishes stacking-failure modes), the grasp
        state, and task progress (t*/T)."""
        progress = self.t_star / max(1, self.T)
        dx = self.cubeA_xyz[0] - self.cubeB_xyz[0]
        dy = self.cubeA_xyz[1] - self.cubeB_xyz[1]
        return [self.cubeA_xyz[0], self.cubeA_xyz[1], dx, dy,
                float(self.grasp), float(progress)]

    def digest(self) -> Dict[str, Any]:
        return {"episode_id": self.episode_id, "seed": self.seed,
                "peak_loss": round(self.peak_loss, 6),
                "t_star": self.t_star, "T": self.T,
                "cubeA_xyz": [round(v, 4) for v in self.cubeA_xyz],
                "cubeA_zrot": round(self.cubeA_zrot, 4),
                "cubeB_xyz": [round(v, 4) for v in self.cubeB_xyz],
                "cubeB_zrot": round(self.cubeB_zrot, 4),
                "grasp": round(float(self.grasp), 3)}


@dataclass
class CubeLayoutSpec:
    """The per-round collection decision."""
    mode: str                              # "onpolicy_correction" | "bridge"
    choice: str = "none"                   # select | bridge | escalated_select | …
    cand: Optional[Dict[str, Any]] = field(default=None, repr=False)   # SELECT
    seed: Optional[int] = None
    episode_id: Optional[int] = None
    layout: Optional[Dict[str, Any]] = None   # BRIDGE cube layout (or None→random)
    cited: List[int] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def digest(self) -> Dict[str, Any]:
        return {"mode": self.mode, "choice": self.choice, "seed": self.seed,
                "episode_id": self.episode_id, "layout": self.layout,
                "cited": self.cited, "provenance": self.provenance}


def cluster_stackcube(descs: List[StackCubeFailureDescriptor], *, max_k: int = 6
                      ) -> ClusterResult:
    """StackCube clustering: reuses the task-agnostic silhouette k-sweep
    (``_best_k_clustering``) + ``pick_dominant`` kernels, but computes cluster
    centroids in CUBE-A xy (for the 2-d coverage memory) — NOT the PushT tee
    bounds baked into ``clustering.cluster_failures``."""
    n = len(descs)
    if n == 0:
        return ClusterResult([], [], "empty", None)
    feats = np.asarray([d.feature() for d in descs], dtype=float)
    if n <= 3:
        labels = np.arange(n, dtype=int)
        method = "singletons"
    else:
        Xs = _standardize(feats)
        kmax = max(2, min(int(max_k), n - 1))
        labels, method = _best_k_clustering(Xs, kmax)

    Xs_rep = _standardize(feats) if n > 1 else feats
    clusters: List[Cluster] = []
    for lab in sorted(set(int(x) for x in labels.tolist())):
        idxs = [i for i in range(n) if int(labels[i]) == lab]
        cx = float(np.mean([descs[i].cubeA_xyz[0] for i in idxs]))
        cy = float(np.mean([descs[i].cubeA_xyz[1] for i in idxs]))
        cs = float(np.mean([math.sin(descs[i].cubeA_zrot) for i in idxs]))
        cc = float(np.mean([math.cos(descs[i].cubeA_zrot) for i in idxs]))
        ctheta = math.atan2(cs, cc)
        mpl = float(np.mean([descs[i].peak_loss for i in idxs]))
        cfeat = np.mean(Xs_rep[idxs], axis=0)
        rep = min(idxs, key=lambda i: float(np.linalg.norm(Xs_rep[i] - cfeat)))
        clusters.append(Cluster(
            label=lab, member_idxs=idxs,
            centroid_xyz=[_clamp(cx, *CUBE_X), _clamp(cy, *CUBE_Y), CUBE_Z],
            centroid_theta=float(ctheta), mean_peak_loss=mpl,
            size=len(idxs), representative_idx=rep))
    dominant = pick_dominant(clusters, descs)
    return ClusterResult(clusters=clusters,
                         labels=[int(x) for x in labels.tolist()],
                         method=method, dominant=dominant)


class StackCubeHybridPlanner:
    """Per-round SELECT/BRIDGE decision for StackCube. Stateful across the
    in-round infeasibility retries (escalates to SELECT) and across rounds (memory
    rotation, tried-seed bookkeeping)."""

    def __init__(self, *, work_dir: str, cfg: Dict[str, Any]):
        self.cfg = dict(cfg or {})
        self.collect_mode = str(self.cfg.get("collect", "hybrid")).lower()
        self.max_k = int(self.cfg.get("max_clusters", 4))
        self.snap_eps = float(self.cfg.get("snap_eps", 0.03))
        self.tele = Telemetry(work_dir)
        self.memory = CentroidMemory(
            str(Path(work_dir) / "telemetry" / "centroid_memory.json"),
            gamma=float(self.cfg.get("memory_gamma", 0.6)),
            sigma=float(self.cfg.get("memory_sigma", 0.05)))
        # per-round state
        self._round: Optional[int] = None
        self._descs: List[StackCubeFailureDescriptor] = []
        self._cr: Optional[ClusterResult] = None
        self._target: Optional[Cluster] = None
        self._tried: set = set()

    # ── called once per round, after the failures are built ──────────────────
    def set_round(self, rnd: int, descs: List[StackCubeFailureDescriptor]) -> None:
        self._round = rnd
        self._descs = list(descs)
        self._tried = set()
        self._cr = None
        self._target = None
        if not descs:
            self.tele.event(rnd, "no_descriptors",
                            {"note": "no cube failure states — hybrid inert this round"})
            return
        self.tele.tic("cluster")
        cr = cluster_stackcube(descs, max_k=self.max_k)
        self._cr = cr
        target, mem_scores = self.memory.select_target(
            cr.clusters, cr.dominant, descs, now=rnd)
        self._target = target
        self.tele.event(rnd, "round_setup", {
            "cluster_method": cr.method,
            "n_failures": len(descs),
            "n_clusters": len(cr.clusters),
            "dominant_label": cr.dominant.label if cr.dominant else None,
            "target_label": target.label if target else None,
            "target_size": target.size if target else None,
            "target_centroid_xy": [round(v, 4) for v in target.centroid_xyz[:2]]
                                   if target else None,
            "memory_scores": mem_scores,
            "descriptors": [d.digest() for d in descs],
            "cluster_secs": self.tele.toc("cluster"),
        })

    # ── the decision-context block appended to the cube-prescriber prompt ─────
    def decision_addendum(self) -> str:
        if self.collect_mode != "hybrid" or not self._descs:
            return ""
        t = self._target
        members = t.member_idxs if t is not None else range(len(self._descs))
        lines = []
        for i in members:
            d = self._descs[i]
            lines.append(
                f"    - ep{d.episode_id}: cubeA=({round(d.cubeA_xyz[0],3)},"
                f"{round(d.cubeA_xyz[1],3)}) cubeB=({round(d.cubeB_xyz[0],3)},"
                f"{round(d.cubeB_xyz[1],3)}) grasped={'yes' if d.grasp>0.5 else 'no'} "
                f"peak_loss={round(d.peak_loss,5)} t*={d.t_star}/{d.T}")
        member_block = "\n".join(lines)
        return (
            "DEMONSTRATION DECISION (code-computed, authoritative — do NOT recompute):\n"
            "  You also DECIDE how the next demonstration is collected. Two options — "
            "choose whichever closes MORE failures per demo:\n"
            "  (A) SELECT — one recorded failure represents this mode: that exact "
            "failed episode is re-run and the expert corrects it on-policy from the "
            "divergence point (a faithful DAgger correction). Use when the cluster is "
            "tight or one failure clearly dominates.\n"
            "  (B) BRIDGE — no single failure covers the mode: place ONE new cube "
            "layout in the MIDDLE GROUND between 2-3 cited failures (e.g. cubeA at "
            "(0.1,0.1) and (-0.1,-0.1) → bridge near (0,0)); the motion-planner expert "
            "demonstrates a clean stack from there, teaching the whole spread at once. "
            "Use when members are geometrically spread but share a root cause.\n"
            "  Target failure cluster: "
            f"label {t.label if t else '-'}, "
            f"{t.size if t else len(self._descs)} member(s)"
            f"{', mean_peak_loss='+str(round(t.mean_peak_loss,5)) if t else ''}.\n"
            f"  Recorded failures (cite these by ep<id>):\n{member_block}\n"
            "  OUTPUT CONTRACT (hard): emit the cube layout as usual, and START the "
            "'rationale' field with EXACTLY ONE of:\n"
            "    'SELECT ep<ID>'  (then set cubeA_xyz/cubeB_xyz to that member's "
            "recorded layout), or\n"
            "    'BRIDGE ep<ID>,ep<ID>[,ep<ID>]'  (then set cubeA_xyz/cubeB_xyz to "
            "your middle-ground layout between the cited members).\n"
            f"  Cube XY bounds: x∈[{CUBE_X[0]},{CUBE_X[1]}], y∈[{CUBE_Y[0]},{CUBE_Y[1]}]"
            f" (min cubeA–cubeB gap {MIN_SEP} m; auto-clamped)."
        )

    # ── helpers ──────────────────────────────────────────────────────────────
    def _member_by_episode_id(self, eid: int) -> Optional[StackCubeFailureDescriptor]:
        for d in self._descs:
            if d.episode_id == int(eid):
                return d
        return None

    @staticmethod
    def _select_feasible(d: StackCubeFailureDescriptor) -> bool:
        """On-policy SELECT can only correct a failure whose cubeA is still
        GRASPABLE at t* (on/near the table). If the policy knocked the cube off the
        table or below the surface, no expert correction exists from that state —
        the motion planner would grind on "screw plan failed" — so the caller must
        BRIDGE (a fresh, solvable layout) instead."""
        x, y, z = d.cubeA_xyz[0], d.cubeA_xyz[1], d.cubeA_xyz[2]
        m = 0.04   # small margin: cubes just past the spawn box are still reachable
        return (CUBE_X[0] - m <= x <= CUBE_X[1] + m
                and CUBE_Y[0] - m <= y <= CUBE_Y[1] + m and z > 0.0)

    def _pick_untried_member(self, pres: Optional[Dict[str, Any]]
                             ) -> Optional[StackCubeFailureDescriptor]:
        """Nearest untried, SELECT-FEASIBLE real failure to the prescribed cubeA
        (else by peak loss): target cluster first, then other clusters (size desc,
        peak-loss desc). Returns None when no graspable untried failure remains —
        the caller then BRIDGEs. Mirrors the PushT safety floor + a StackCube
        feasibility gate (off-table cubes can't be on-policy corrected)."""
        descs, t = self._descs, self._target
        if not descs:
            return None
        ax = ay = None
        if isinstance(pres, dict) and pres.get("cubeA_xyz"):
            ax, ay = float(pres["cubeA_xyz"][0]), float(pres["cubeA_xyz"][1])

        def near(i: int) -> float:
            if ax is None:
                return -descs[i].peak_loss
            return math.hypot(descs[i].cubeA_xyz[0] - ax, descs[i].cubeA_xyz[1] - ay)

        order: List[int] = []
        if t is not None:
            order += sorted(t.member_idxs, key=near)
        if self._cr is not None:
            for c in sorted((c for c in self._cr.clusters if c is not t),
                            key=lambda c: (-c.size, -c.mean_peak_loss)):
                order += sorted(c.member_idxs, key=lambda i: -descs[i].peak_loss)
        for i in order:
            if descs[i].seed not in self._tried and self._select_feasible(descs[i]):
                return descs[i]
        feas = [d for d in descs
                if d.seed not in self._tried and self._select_feasible(d)]
        return max(feas, key=lambda d: d.peak_loss) if feas else None

    @staticmethod
    def _layout_from_pres(pres: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(pres, dict) or not pres.get("cubeA_xyz") or not pres.get("cubeB_xyz"):
            return None
        return {"cubeA_xyz": [float(v) for v in pres["cubeA_xyz"][:3]],
                "cubeB_xyz": [float(v) for v in pres["cubeB_xyz"][:3]],
                "cubeA_zrot": pres.get("cubeA_zrot"),
                "cubeB_zrot": pres.get("cubeB_zrot")}

    def _select_spec(self, member, choice: str) -> CubeLayoutSpec:
        self._tried.add(member.seed)
        return CubeLayoutSpec(
            mode="onpolicy_correction", choice=choice, cand=member.cand,
            seed=int(member.seed), episode_id=int(member.episode_id),
            provenance={"t_star_discovery": member.t_star, "T": member.T,
                        "peak_loss": round(member.peak_loss, 6)})

    # ── the per-attempt decision ─────────────────────────────────────────────
    def decide(self, pres: Optional[Dict[str, Any]], *, attempt: int) -> CubeLayoutSpec:
        """Map the LLM's prescription (+ its SELECT/BRIDGE tag in the rationale) to
        a collection spec. ``attempt`` is the in-round infeasibility-retry index;
        any retry escalates to SELECT (safety floor)."""
        label = pres.get("rationale", "") if isinstance(pres, dict) else ""
        choice, ids = parse_choice(label)

        # Escalation: any retry, or no usable failures → SELECT nearest untried.
        if attempt >= 1 or not self._descs:
            member = self._pick_untried_member(pres)
            if member is None:
                return CubeLayoutSpec(mode="bridge", choice="bridge_fallback",
                                      layout=self._layout_from_pres(pres))
            return self._select_spec(member, "escalated_select")

        # No explicit tag: geometric fallback — prescribed cubeA within snap_eps of
        # a real failure ⇒ SELECT it; otherwise BRIDGE the prescribed layout.
        if choice is None:
            layout = self._layout_from_pres(pres)
            if layout is not None:
                ax, ay = layout["cubeA_xyz"][0], layout["cubeA_xyz"][1]
                nearest = min(self._descs,
                              key=lambda d: math.hypot(d.cubeA_xyz[0] - ax,
                                                       d.cubeA_xyz[1] - ay))
                dist = math.hypot(nearest.cubeA_xyz[0] - ax, nearest.cubeA_xyz[1] - ay)
                if dist <= self.snap_eps:
                    choice, ids = "select", [nearest.episode_id]
                else:
                    choice = "bridge"
            else:
                choice = "select"   # no LLM layout → faithful, budget-safe SELECT

        if choice == "select":
            member = self._member_by_episode_id(ids[0]) if ids else None
            if (member is None or member.seed in self._tried
                    or not self._select_feasible(member)):
                member = self._pick_untried_member(pres)   # nearest graspable untried
            if member is None:
                # no graspable real failure to correct → BRIDGE a solvable layout
                return CubeLayoutSpec(mode="bridge", choice="bridge_no_feasible_select",
                                      layout=self._layout_from_pres(pres))
            return self._select_spec(member, "select")

        # BRIDGE: the LLM's middle-ground layout (set_prescription clamps it).
        layout = self._layout_from_pres(pres)
        if layout is None:
            member = self._pick_untried_member(pres)
            if member is not None:
                return self._select_spec(member, "escalated_select")
        cited = [d.episode_id for d in
                 (self._member_by_episode_id(i) for i in ids) if d is not None]
        return CubeLayoutSpec(mode="bridge", choice="bridge", layout=layout,
                              cited=cited, provenance={"cited_episode_ids": cited})

    # ── outcome bookkeeping + coverage memory ────────────────────────────────
    def note_collect(self, spec: CubeLayoutSpec, result: Dict[str, Any], *,
                     attempt: int) -> None:
        rnd = self._round if self._round is not None else 0
        success = bool(result.get("success"))
        self.tele.event(rnd, "collect_outcome", {
            "mode": spec.mode, "choice": spec.choice, "attempt": attempt,
            "success": success, "episode_length": result.get("episode_length"),
            "select_seed": spec.seed, "select_episode_id": spec.episode_id,
            "bridge_cited": spec.cited, "bridge_layout": spec.layout,
            "applied": result.get("applied")})
        if success:
            # record the covered coverage point so future rounds rotate away
            if spec.mode == "bridge" and spec.layout and spec.layout.get("cubeA_xyz"):
                cx = spec.layout["cubeA_xyz"]
                self.memory.append(rnd, [cx[0], cx[1], CUBE_Z],
                                   float(spec.layout.get("cubeA_zrot") or 0.0))
            elif spec.episode_id is not None:
                d = self._member_by_episode_id(spec.episode_id)
                if d is not None:
                    self.memory.append(rnd, d.cubeA_xyz, d.cubeA_zrot)
            self.tele.summary_row({
                "round": rnd, "mode": spec.mode, "choice": spec.choice,
                "attempt": attempt, "select_episode_id": spec.episode_id,
                "bridge_cited": spec.cited,
                "target_cluster": (self._target.label if self._target else None),
                "target_size": (self._target.size if self._target else None),
                "episode_length": result.get("episode_length")})
