"""PlugCharger V3 HYBRID — the LLM decides SELECT vs BRIDGE each round.

PlugCharger port of the StackCube V3 hybrid (``stackcube_hybrid``). Same suite-
native engine (``p4/select_arm.py::run_p4_top3_arm``); this module is injected
INLINE there (gated behind ``p4.subtask.collect: hybrid``), NOT via fork hooks.

Per round, ``run_p4_top3_arm`` already rolls ``n_cand`` candidates and keeps the
FAILURES. This module:

  1. Builds a charger descriptor per failure by REPLAYING the recorded
     ``exec_actions`` to ``t*`` and snapshotting charger/receptacle pose + grasp
     state (the select_arm engine has no meta.json — failures are in-memory
     candidates).
  2. Clusters the failures on a 6-d charger signature, picks the dominant cluster
     (memory-rotated so coverage rotates across rounds).
  3. Builds a DECISION CONTEXT block appended to the charger-prescriber prompt:
     the LLM emits ONE charger/receptacle layout AND prefixes its rationale with
     ``SELECT ep<id>`` or ``BRIDGE ep<a>,ep<b>``.
  4. Maps that choice to a ``PlugChargerLayoutSpec``:
       * SELECT  → on-policy correction of the cited REAL failure
                   (``_correct_onpolicy_from`` — re-roll its seed, expert takes
                   over from the divergence state). The faithful DAgger primitive.
       * BRIDGE  → the LLM's middle-ground layout, realised by
                   ``PlugCharger-Start-v0.set_prescription`` + motion-planner solve
                   (``_collect_prescribed_demo``). The creative compression.
  5. Any in-round retry (the infeasibility loop) ESCALATES to SELECT of the
     nearest untried real failure — a bad bridge can never waste the round.

Reuses the task-agnostic compression core verbatim: clustering (silhouette
k-sweep + ``pick_dominant``), ``CentroidMemory`` recency rotation, and the
``Telemetry`` writer. Charger geometry / prompt text are PlugCharger-specific.
With the flag absent the planner is never instantiated and p4_top3 is byte-
identical (pure prescription), preserving the apples-to-apples baseline.

GEOMETRY MAPPING (vs StackCube cubeA/cubeB):
  * charger    (dynamic, graspable)  ↔ cubeA  (the thing to pick + move)
  * receptacle (kinematic, target)   ↔ cubeB  (the place to align/insert to)
The 6-d feature mirrors StackCube exactly: [charger_x, charger_y, charger−recep
dx, charger−recep dy, grasp, progress]. The clustering centroid / coverage memory
read the charger XY (the cubeA_xyz analogue). The clamp bounds below are kept in
SYNC with envs/plugcharger_start.py (set_prescription clamps to the same ranges).
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


# PlugCharger spawn workspace — MIRRORS envs/plugcharger_start.py (set_prescription
# clamps to these too; kept here for clamping + prompt text + the feasibility gate).
CHARGER_X: tuple = (-0.10, -0.026)
CHARGER_Y: tuple = (-0.20, 0.20)
CHARGER_Z: float = 0.012
CHARGER_ZROT: tuple = (-math.pi / 3, math.pi / 3)
RECEP_X: tuple = (0.01, 0.10)
RECEP_Y: tuple = (-0.10, 0.10)
RECEP_Z: float = 0.10
RECEP_ZROT: tuple = (math.pi - math.pi / 8, math.pi + math.pi / 8)


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
    StackCube planner (kept inline so the module is self-contained)."""
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
class PlugChargerFailureDescriptor:
    """One failed candidate's reconstructed t* state. ``cand`` is the live
    select_arm candidate dict ({seed, exec_actions, t_star, ...}) so a SELECT
    can re-roll + correct it on-policy. ``charger_xyz`` is the cubeA_xyz analogue
    that the coverage memory / clustering centroid read."""
    episode_id: int            # == screening seed (scene identity for SELECT)
    seed: int
    peak_loss: float
    t_star: int
    T: int
    charger_xyz: List[float]   # [x, y, z] of the charger (graspable), at t*
    charger_zrot: float
    receptacle_xyz: List[float]  # [x, y, z] of the receptacle (target), at t*
    receptacle_zrot: float
    grasp: float               # 1.0 if the gripper is grasping the charger at t*, else 0
    cand: Dict[str, Any] = field(default_factory=dict, repr=False)

    def feature(self) -> List[float]:
        """6-d clustering signature, mirroring StackCube: where the charger is, the
        charger→receptacle xy offset (the alignment error that distinguishes
        insertion-failure modes), the grasp state, and task progress (t*/T).

        NOTE on 6-d vs 7-d: a 7th "insertion yaw error" dim was considered (charger
        yaw vs the receptacle's hole-facing yaw). It is left OUT by default: the xy
        offset already separates the dominant modes (missed-grasp vs misaligned vs
        partial), the receptacle zrot range is narrow (pi +/- pi/8), and keeping 6-d
        matches the agnostic clustering core's standardization scale (one extra
        circular dim of small variance mostly adds noise). The charger zrot is still
        carried on the descriptor (used for the coverage centroid + prompt text), it
        just does not enter the clustering signature."""
        progress = self.t_star / max(1, self.T)
        dx = self.charger_xyz[0] - self.receptacle_xyz[0]
        dy = self.charger_xyz[1] - self.receptacle_xyz[1]
        return [self.charger_xyz[0], self.charger_xyz[1], dx, dy,
                float(self.grasp), float(progress)]

    def digest(self) -> Dict[str, Any]:
        return {"episode_id": self.episode_id, "seed": self.seed,
                "peak_loss": round(self.peak_loss, 6),
                "t_star": self.t_star, "T": self.T,
                "charger_xyz": [round(v, 4) for v in self.charger_xyz],
                "charger_zrot": round(self.charger_zrot, 4),
                "receptacle_xyz": [round(v, 4) for v in self.receptacle_xyz],
                "receptacle_zrot": round(self.receptacle_zrot, 4),
                "grasp": round(float(self.grasp), 3)}


@dataclass
class PlugChargerLayoutSpec:
    """The per-round collection decision."""
    mode: str                              # "onpolicy_correction" | "bridge"
    choice: str = "none"                   # select | bridge | escalated_select | …
    cand: Optional[Dict[str, Any]] = field(default=None, repr=False)   # SELECT
    seed: Optional[int] = None
    episode_id: Optional[int] = None
    layout: Optional[Dict[str, Any]] = None   # BRIDGE layout (or None→random)
    cited: List[int] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def digest(self) -> Dict[str, Any]:
        return {"mode": self.mode, "choice": self.choice, "seed": self.seed,
                "episode_id": self.episode_id, "layout": self.layout,
                "cited": self.cited, "provenance": self.provenance}


def cluster_plugcharger(descs: List[PlugChargerFailureDescriptor], *, max_k: int = 6
                        ) -> ClusterResult:
    """PlugCharger clustering: reuses the task-agnostic silhouette k-sweep
    (``_best_k_clustering``) + ``pick_dominant`` kernels, but computes cluster
    centroids in CHARGER xy (for the 2-d coverage memory) — the analogue of
    StackCube's cubeA-xy centroid."""
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
        cx = float(np.mean([descs[i].charger_xyz[0] for i in idxs]))
        cy = float(np.mean([descs[i].charger_xyz[1] for i in idxs]))
        cs = float(np.mean([math.sin(descs[i].charger_zrot) for i in idxs]))
        cc = float(np.mean([math.cos(descs[i].charger_zrot) for i in idxs]))
        ctheta = math.atan2(cs, cc)
        mpl = float(np.mean([descs[i].peak_loss for i in idxs]))
        cfeat = np.mean(Xs_rep[idxs], axis=0)
        rep = min(idxs, key=lambda i: float(np.linalg.norm(Xs_rep[i] - cfeat)))
        clusters.append(Cluster(
            label=lab, member_idxs=idxs,
            centroid_xyz=[_clamp(cx, *CHARGER_X), _clamp(cy, *CHARGER_Y), CHARGER_Z],
            centroid_theta=float(ctheta), mean_peak_loss=mpl,
            size=len(idxs), representative_idx=rep))
    dominant = pick_dominant(clusters, descs)
    return ClusterResult(clusters=clusters,
                         labels=[int(x) for x in labels.tolist()],
                         method=method, dominant=dominant)


class PlugChargerHybridPlanner:
    """Per-round SELECT/BRIDGE decision for PlugCharger. Stateful across the
    in-round infeasibility retries (escalates to SELECT) and across rounds (memory
    rotation, tried-seed bookkeeping)."""

    def __init__(self, *, work_dir: str, cfg: Dict[str, Any]):
        self.cfg = dict(cfg or {})
        self.collect_mode = str(self.cfg.get("collect", "hybrid")).lower()
        self.max_k = int(self.cfg.get("max_clusters", 4))
        self.snap_eps = float(self.cfg.get("snap_eps", 0.02))
        self.tele = Telemetry(work_dir)
        self.memory = CentroidMemory(
            str(Path(work_dir) / "telemetry" / "centroid_memory.json"),
            gamma=float(self.cfg.get("memory_gamma", 0.6)),
            sigma=float(self.cfg.get("memory_sigma", 0.03)))
        # per-round state
        self._round: Optional[int] = None
        self._descs: List[PlugChargerFailureDescriptor] = []
        self._cr: Optional[ClusterResult] = None
        self._target: Optional[Cluster] = None
        self._tried: set = set()

    # ── called once per round, after the failures are built ──────────────────
    def set_round(self, rnd: int, descs: List[PlugChargerFailureDescriptor]) -> None:
        self._round = rnd
        self._descs = list(descs)
        self._tried = set()
        self._cr = None
        self._target = None
        if not descs:
            self.tele.event(rnd, "no_descriptors",
                            {"note": "no charger failure states — hybrid inert this round"})
            return
        self.tele.tic("cluster")
        cr = cluster_plugcharger(descs, max_k=self.max_k)
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

    # ── the decision-context block appended to the charger-prescriber prompt ──
    def decision_addendum(self) -> str:
        if self.collect_mode != "hybrid" or not self._descs:
            return ""
        t = self._target
        members = t.member_idxs if t is not None else range(len(self._descs))
        lines = []
        for i in members:
            d = self._descs[i]
            lines.append(
                f"    - ep{d.episode_id}: charger=({round(d.charger_xyz[0],3)},"
                f"{round(d.charger_xyz[1],3)}) receptacle=({round(d.receptacle_xyz[0],3)},"
                f"{round(d.receptacle_xyz[1],3)}) grasped={'yes' if d.grasp>0.5 else 'no'} "
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
            "  (B) BRIDGE — no single failure covers the mode: place ONE new "
            "charger+receptacle layout in the MIDDLE GROUND between 2-3 cited "
            "failures (e.g. charger at (-0.05,0.1) and (-0.05,-0.1) → bridge near "
            "(-0.05,0)); the motion-planner expert demonstrates a clean grasp + "
            "insertion from there, teaching the whole spread at once. Use when "
            "members are geometrically spread but share a root cause (e.g. a common "
            "peg-to-hole misalignment).\n"
            "  Target failure cluster: "
            f"label {t.label if t else '-'}, "
            f"{t.size if t else len(self._descs)} member(s)"
            f"{', mean_peak_loss='+str(round(t.mean_peak_loss,5)) if t else ''}.\n"
            f"  Recorded failures (cite these by ep<id>):\n{member_block}\n"
            "  OUTPUT CONTRACT (hard): emit the charger/receptacle layout as usual, "
            "and START the 'rationale' field with EXACTLY ONE of:\n"
            "    'SELECT ep<ID>'  (then set charger_xyz/receptacle_xyz to that "
            "member's recorded layout), or\n"
            "    'BRIDGE ep<ID>,ep<ID>[,ep<ID>]'  (then set charger_xyz/receptacle_xyz "
            "to your middle-ground layout between the cited members).\n"
            f"  Charger XY bounds: x∈[{CHARGER_X[0]},{CHARGER_X[1]}], "
            f"y∈[{CHARGER_Y[0]},{CHARGER_Y[1]}] (z={CHARGER_Z}). "
            f"Receptacle XY bounds: x∈[{RECEP_X[0]},{RECEP_X[1]}], "
            f"y∈[{RECEP_Y[0]},{RECEP_Y[1]}] (z={RECEP_Z}). Out-of-range poses are "
            "auto-clamped to a graspable, insertable layout."
        )

    # ── helpers ──────────────────────────────────────────────────────────────
    def _member_by_episode_id(self, eid: int) -> Optional[PlugChargerFailureDescriptor]:
        for d in self._descs:
            if d.episode_id == int(eid):
                return d
        return None

    @staticmethod
    def _select_feasible(d: PlugChargerFailureDescriptor) -> bool:
        """On-policy SELECT can only correct a failure whose charger is still
        GRASPABLE/INSERTABLE at t* (on/near the table, inside the reachable spawn
        region). If the policy knocked the charger off the table or far out of the
        workspace, no expert correction exists from that state — the motion planner
        would grind on "screw plan failed" — so the caller must BRIDGE (a fresh,
        solvable layout) instead. Mirrors StackCube's on-table check; uses the
        charger spawn box (the region where the planner is known to solve)."""
        x, y, z = d.charger_xyz[0], d.charger_xyz[1], d.charger_xyz[2]
        m = 0.04   # small margin: chargers just past the spawn box are still reachable
        return (CHARGER_X[0] - m <= x <= CHARGER_X[1] + m
                and CHARGER_Y[0] - m <= y <= CHARGER_Y[1] + m and z > 0.0)

    def _pick_untried_member(self, pres: Optional[Dict[str, Any]]
                             ) -> Optional[PlugChargerFailureDescriptor]:
        """Nearest untried, SELECT-FEASIBLE real failure to the prescribed charger
        (else by peak loss): target cluster first, then other clusters (size desc,
        peak-loss desc). Returns None when no graspable untried failure remains —
        the caller then BRIDGEs. Mirrors the StackCube safety floor + feasibility
        gate (off-workspace chargers can't be on-policy corrected)."""
        descs, t = self._descs, self._target
        if not descs:
            return None
        ax = ay = None
        if isinstance(pres, dict) and pres.get("charger_xyz"):
            ax, ay = float(pres["charger_xyz"][0]), float(pres["charger_xyz"][1])

        def near(i: int) -> float:
            if ax is None:
                return -descs[i].peak_loss
            return math.hypot(descs[i].charger_xyz[0] - ax, descs[i].charger_xyz[1] - ay)

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
        if (not isinstance(pres, dict) or not pres.get("charger_xyz")
                or not pres.get("receptacle_xyz")):
            return None
        return {"charger_xyz": [float(v) for v in pres["charger_xyz"][:3]],
                "receptacle_xyz": [float(v) for v in pres["receptacle_xyz"][:3]],
                "charger_zrot": pres.get("charger_zrot"),
                "receptacle_zrot": pres.get("receptacle_zrot")}

    def _select_spec(self, member, choice: str) -> PlugChargerLayoutSpec:
        self._tried.add(member.seed)
        return PlugChargerLayoutSpec(
            mode="onpolicy_correction", choice=choice, cand=member.cand,
            seed=int(member.seed), episode_id=int(member.episode_id),
            provenance={"t_star_discovery": member.t_star, "T": member.T,
                        "peak_loss": round(member.peak_loss, 6)})

    # ── the per-attempt decision ─────────────────────────────────────────────
    def decide(self, pres: Optional[Dict[str, Any]], *, attempt: int) -> PlugChargerLayoutSpec:
        """Map the LLM's prescription (+ its SELECT/BRIDGE tag in the rationale) to
        a collection spec. ``attempt`` is the in-round infeasibility-retry index;
        any retry escalates to SELECT (safety floor)."""
        label = pres.get("rationale", "") if isinstance(pres, dict) else ""
        choice, ids = parse_choice(label)

        # Escalation: any retry, or no usable failures → SELECT nearest untried.
        if attempt >= 1 or not self._descs:
            member = self._pick_untried_member(pres)
            if member is None:
                return PlugChargerLayoutSpec(mode="bridge", choice="bridge_fallback",
                                             layout=self._layout_from_pres(pres))
            return self._select_spec(member, "escalated_select")

        # No explicit tag: geometric fallback — prescribed charger within snap_eps of
        # a real failure ⇒ SELECT it; otherwise BRIDGE the prescribed layout.
        if choice is None:
            layout = self._layout_from_pres(pres)
            if layout is not None:
                ax, ay = layout["charger_xyz"][0], layout["charger_xyz"][1]
                nearest = min(self._descs,
                              key=lambda d: math.hypot(d.charger_xyz[0] - ax,
                                                       d.charger_xyz[1] - ay))
                dist = math.hypot(nearest.charger_xyz[0] - ax,
                                  nearest.charger_xyz[1] - ay)
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
                return PlugChargerLayoutSpec(mode="bridge",
                                             choice="bridge_no_feasible_select",
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
        return PlugChargerLayoutSpec(mode="bridge", choice="bridge", layout=layout,
                                     cited=cited, provenance={"cited_episode_ids": cited})

    # ── outcome bookkeeping + coverage memory ────────────────────────────────
    def note_collect(self, spec: PlugChargerLayoutSpec, result: Dict[str, Any], *,
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
            if spec.mode == "bridge" and spec.layout and spec.layout.get("charger_xyz"):
                cx = spec.layout["charger_xyz"]
                self.memory.append(rnd, [cx[0], cx[1], CHARGER_Z],
                                   float(spec.layout.get("charger_zrot") or 0.0))
            elif spec.episode_id is not None:
                d = self._member_by_episode_id(spec.episode_id)
                if d is not None:
                    self.memory.append(rnd, d.charger_xyz, d.charger_zrot)
            self.tele.summary_row({
                "round": rnd, "mode": spec.mode, "choice": spec.choice,
                "attempt": attempt, "select_episode_id": spec.episode_id,
                "bridge_cited": spec.cited,
                "target_cluster": (self._target.label if self._target else None),
                "target_size": (self._target.size if self._target else None),
                "episode_length": result.get("episode_length")})


# ═════════════════════════════════════════════════════════════════════════════
# Charger prescriber + prompt text (PlugCharger analogue of CubePrescriptionClass /
# cube_prescription_prompt). Self-contained here so select_arm only needs an env-id
# dispatch to swap in the right prescriber; the cube prescriber stays untouched.
# ═════════════════════════════════════════════════════════════════════════════
from .prescription import PrescriptionEmptyError, _valid_pose  # noqa: E402
from .runner import extract_json  # noqa: E402


CHARGER_OUTPUT_REQUIREMENT = (
    "OUTPUT REQUIREMENT (strict, non-negotiable):\n"
    "- A failure IS present, so you MUST prescribe a CONCRETE corrective scene: the "
    "world XY positions of the CHARGER (the movable part to grasp) and the "
    "RECEPTACLE (the fixed wall socket to insert into). Provide charger_xyz ([x,y,z]) "
    "and receptacle_xyz ([x,y,z]); the z heights are pinned automatically (charger "
    f"~{CHARGER_Z} on the table, receptacle ~{RECEP_Z}). Optionally charger_zrot / "
    "receptacle_zrot (z-rotation in radians; omit/null = random within range).\n"
    "- DO NOT emit an empty object, null positions, or refuse. An empty/invalid "
    "prescription is treated as ZERO prescriptions and collects ZERO demos this round "
    "— the worst possible outcome.\n"
    f"- Keep the charger inside x∈[{CHARGER_X[0]},{CHARGER_X[1]}], "
    f"y∈[{CHARGER_Y[0]},{CHARGER_Y[1]}] (zrot∈[{round(CHARGER_ZROT[0],3)},"
    f"{round(CHARGER_ZROT[1],3)}]) and the receptacle inside "
    f"x∈[{RECEP_X[0]},{RECEP_X[1]}], y∈[{RECEP_Y[0]},{RECEP_Y[1]}] "
    f"(zrot∈[{round(RECEP_ZROT[0],3)},{round(RECEP_ZROT[1],3)}]); out-of-range poses "
    "are auto-clamped, and a charger/receptacle that can't be grasped or inserted "
    "wastes the round.\n"
    "- CHOOSE the layout to TEACH the missing behaviour the analysis identified: e.g. "
    "for missed-grasp failures, place the charger in the open with a clean approach so "
    "the expert demonstrates a secure grasp first; for misaligned/partial insertions, "
    "place the charger and receptacle so only a small re-orientation is needed, "
    "rehearsing precise peg-to-hole alignment; for collisions, leave a clear approach "
    "corridor. When unsure, copy a real failure episode's layout and adjust minimally.")

CHARGER_PRESCRIPTION_SYSTEM = (
    "You are a demonstration coach for a charger-insertion robot. You prescribe the "
    "charger + receptacle layout of the NEXT corrective episode so the motion-planner "
    "expert can demonstrate the missing behaviour (grasp the charger, align its pegs, "
    "insert to full depth). Apply every OUTPUT REQUIREMENT. Output strict JSON, no "
    "prose, no code fences.")


def charger_prescription_prompt(task_description: str, kag_text: str,
                                analysis_json: str, failures_summary: str) -> str:
    return (
        f"TASK: {task_description}\n\n"
        f"{kag_text}\n\n"
        f"ROOT-CAUSE ANALYSIS (authoritative):\n{analysis_json}\n\n"
        f"TOP FAILURE EPISODES (compress these into ONE corrective layout):\n"
        f"{failures_summary}\n\n"
        f"{CHARGER_OUTPUT_REQUIREMENT}\n\n"
        'Output ONLY this JSON:\n'
        '{"charger_xyz": [x,y,z], "receptacle_xyz": [x,y,z], '
        '"charger_zrot": <float|null>, "receptacle_zrot": <float|null>, '
        '"n_demos": 1, '
        '"rationale": "<one sentence tying this layout to the failure>"}')


def _charger_validate(obj: Dict) -> Dict:
    """Raise PrescriptionEmptyError unless obj is a concrete charger layout
    (charger_xyz + receptacle_xyz as 3-floats; z-rotations optional)."""
    if not isinstance(obj, dict) or not obj:
        raise PrescriptionEmptyError("charger prescription empty / not an object")
    a, b = obj.get("charger_xyz"), obj.get("receptacle_xyz")
    if not _valid_pose(a, 3) or not _valid_pose(b, 3):
        raise PrescriptionEmptyError(
            f"charger/receptacle positions missing/invalid (charger={a}, receptacle={b})")

    def _optf(v):
        return float(v) if isinstance(v, (int, float)) else None

    return {"charger_xyz": [float(x) for x in a],
            "receptacle_xyz": [float(x) for x in b],
            "charger_zrot": _optf(obj.get("charger_zrot")),
            "receptacle_zrot": _optf(obj.get("receptacle_zrot")),
            "n_demos": int(obj.get("n_demos", 1) or 1),
            "rationale": str(obj.get("rationale", ""))[:300]}


class ChargerPrescriptionClass:
    """Charger-layout prescription (PlugCharger): the LLM aggregates the top-K
    failure analyses into ONE corrective scene = where to place the charger
    (movable) and the receptacle (target). Same empty-prescription HARD-FLOOR guard
    + retry as the cube/PushT prescribers. API-identical to ``CubePrescriptionClass``
    (``prescribe(analysis_json, failures, extra_addendum="")``) so select_arm can
    swap it in via an env-id dispatch with no other change."""

    def __init__(self, client, task_description: str, kag_text: str,
                 max_retries: int = 2):
        self.client = client
        self.task = task_description
        self.kag = kag_text
        self.max_retries = int(max_retries)

    def prescribe(self, analysis_json: str, failures: List[dict],
                  extra_addendum: str = "") -> Dict:
        from . import prompts as P
        if self.client is None:
            raise PrescriptionEmptyError("LLM client unavailable")
        base = charger_prescription_prompt(self.task, self.kag, analysis_json,
                                           P.failures_summary(failures))
        if extra_addendum:
            base = base + "\n\n" + extra_addendum
        last_err = None
        for attempt in range(self.max_retries + 1):
            prompt = base if attempt == 0 else (
                base + "\n\nYOU MUST PRESCRIBE a concrete, non-empty charger layout "
                "now. Your previous output was empty or invalid; copy a real failure's "
                "charger/receptacle positions and adjust minimally toward a graspable, "
                "insertable layout rather than emitting nothing.")
            try:
                txt = self.client.reason(prompt, system_prompt=CHARGER_PRESCRIPTION_SYSTEM)
                return _charger_validate(extract_json(txt) or {})
            except PrescriptionEmptyError as exc:
                last_err = exc
                continue
            except Exception as exc:  # transport error
                last_err = exc
                continue
        raise PrescriptionEmptyError(
            f"no valid charger prescription after {self.max_retries + 1} attempts: {last_err}")
