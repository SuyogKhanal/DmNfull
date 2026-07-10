"""Per-failure descriptor for the robosuite P4-LLM hybrid.

A failure is a screening episode where the current policy failed. To cluster
failures into modes and to prescribe/correct them, we snapshot the PRIVILEGED
geometry at the peak-uncertainty step t* (reached by replaying the failure's
recorded actions to t*), and turn it into a small standardized feature vector.

This mirrors the ManiSkill StackCubeFailureDescriptor (live-env snapshot, NOT a
meta.json parse) but reads robosuite state. The dataclass satisfies the clustering
duck-type contract: cluster_feature(), centroid_pos(), centroid_theta(),
peak_loss, episode_id.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ── privileged state snapshot (per task) ─────────────────────────────────────
def snapshot_lift(raw) -> Dict[str, Any]:
    """Ground-truth Lift geometry from the raw robosuite env at the CURRENT state."""
    eef = np.asarray(raw.sim.data.site_xpos[raw.robots[0].eef_site_id["right"]], dtype=float)
    obj = np.asarray(raw.sim.data.body_xpos[raw.cube_body_id], dtype=float)
    quat = np.asarray(raw.sim.data.body_xquat[raw.cube_body_id], dtype=float)  # [w,x,y,z]
    table_h = float(raw.model.mujoco_arena.table_offset[2])
    grasped = bool(raw._check_grasp(gripper=raw.robots[0].gripper, object_geoms=raw.cube))
    return {"obj_xyz": obj.tolist(), "obj_yaw": _yaw_wxyz(quat),
            "eef_xyz": eef.tolist(), "grasp": float(grasped), "table_h": table_h}


def snapshot_door(raw) -> Dict[str, Any]:
    """Ground-truth Door geometry. The randomized 'object' is the door frame
    (root body, placed via body_pos/body_quat — no free joint). Success = hinge
    angle > 0.3 rad (range 0..0.4)."""
    eef = np.asarray(raw.sim.data.site_xpos[raw.robots[0].eef_site_id["right"]], dtype=float)
    handle = np.asarray(raw._handle_xpos, dtype=float)
    did = raw.sim.model.body_name2id(raw.door.root_body)
    dpos = np.asarray(raw.sim.data.body_xpos[did], dtype=float)
    dquat = np.asarray(raw.sim.data.body_xquat[did], dtype=float)   # [w,x,y,z]
    hinge = float(raw.sim.data.qpos[raw.hinge_qpos_addr])
    return {"obj_xyz": dpos.tolist(), "obj_yaw": _yaw_wxyz(dquat), "obj_quat": dquat.tolist(),
            "eef_xyz": eef.tolist(), "handle_xyz": handle.tolist(), "hinge": hinge}


def snapshot_wipe(raw) -> Dict[str, Any]:
    """Ground-truth Wipe geometry. The randomized quantity is a SET of dirt
    markers (a path), so there is no single object pose to prescribe -> Wipe is
    SELECT-only. We snapshot the remaining-dirt centroid + coverage for clustering."""
    eef = np.asarray(raw.sim.data.site_xpos[raw.robots[0].eef_site_id["right"]], dtype=float)
    wiped = set(id(m) for m in raw.wiped_markers)
    pos = []
    for m in raw.model.mujoco_arena.markers:
        if id(m) in wiped:
            continue
        bid = raw.sim.model.body_name2id(m.root_body)
        pos.append(raw.sim.data.body_xpos[bid])
    pts = np.asarray(pos, dtype=float) if pos else np.zeros((0, 3))
    centroid = pts.mean(axis=0) if len(pts) else eef.copy()
    n_total = int(getattr(raw, "num_markers", max(1, len(raw.wiped_markers) + len(pts))))
    prop_wiped = len(raw.wiped_markers) / max(1, n_total)
    return {"obj_xyz": centroid.tolist(), "obj_yaw": 0.0, "eef_xyz": eef.tolist(),
            "prop_wiped": float(prop_wiped), "n_remaining": int(len(pts)), "n_total": n_total}


SNAPSHOT = {"Lift": snapshot_lift, "Door": snapshot_door, "Wipe": snapshot_wipe}


def snapshot_state(task: str, raw) -> Dict[str, Any]:
    return SNAPSHOT[task](raw)


def _yaw_wxyz(q) -> float:
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# ── the descriptor ───────────────────────────────────────────────────────────
@dataclass
class RSFailureDescriptor:
    task: str
    episode_id: int          # screening seed = scene identity (for SELECT re-roll)
    seed: int
    t_star: int
    T: int
    peak_loss: float
    snap: Dict[str, Any]     # snapshot_state(...) at t*
    exec_actions: List[Any]  # recorded policy actions (replay to t* for SELECT)
    visual_embedding: Optional[List[float]] = None  # VLM path (optional)

    # ---- clustering duck-type contract ----
    def feature(self) -> List[float]:
        """Task-specific standardized-friendly signature (z-scored by clustering)."""
        s = self.snap
        prog = self.t_star / max(1, self.T)
        if self.task == "Lift":
            ox, oy, oz = s["obj_xyz"]
            ex, ey, ez = s["eef_xyz"]
            g2o_xy = math.hypot(ex - ox, ey - oy)   # gripper→cube planar distance
            z_gap = ez - oz                          # gripper height above cube
            return [ox, oy, float(prog), g2o_xy, z_gap, s["grasp"]]
        if self.task == "Door":
            dx, dy, _dz = s["obj_xyz"]
            ex, ey = s["eef_xyz"][0], s["eef_xyz"][1]
            e2h = math.hypot(ex - s["handle_xyz"][0], ey - s["handle_xyz"][1])
            hinge_prog = s["hinge"] / 0.4            # door opening fraction
            return [dx, dy, s["obj_yaw"], hinge_prog, e2h, float(prog)]
        if self.task == "Wipe":
            cx, cy, _cz = s["obj_xyz"]               # remaining-dirt centroid
            ex, ey = s["eef_xyz"][0], s["eef_xyz"][1]
            e2c = math.hypot(ex - cx, ey - cy)
            return [cx, cy, s["prop_wiped"], e2c,
                    s["n_remaining"] / max(1, s["n_total"]), float(prog)]
        raise NotImplementedError(f"feature() for task {self.task}")

    def cluster_feature(self) -> List[float]:
        return list(self.visual_embedding) if self.visual_embedding is not None else self.feature()

    def centroid_pos(self) -> List[float]:
        return [float(v) for v in self.snap["obj_xyz"]]

    def centroid_theta(self) -> float:
        return float(self.snap.get("obj_yaw", 0.0))

    def digest(self) -> Dict[str, Any]:
        return {"task": self.task, "episode_id": self.episode_id, "seed": self.seed,
                "t_star": self.t_star, "T": self.T, "peak_loss": round(self.peak_loss, 6),
                "obj_xyz": [round(v, 4) for v in self.snap["obj_xyz"]],
                "obj_yaw": round(self.snap.get("obj_yaw", 0.0), 4),
                "eef_xyz": [round(v, 4) for v in self.snap["eef_xyz"]],
                "grasp": self.snap.get("grasp"), "feature": [round(v, 4) for v in self.feature()]}
