"""BRIDGE scene-prescription for robosuite: place the primary object at a
prescribed pose AFTER env.reset(), so the expert can demonstrate from a
middle-ground configuration covering several failures.

robosuite has no direct "place object at (x,y)" API; the reliable per-episode
mechanism is to set the object's free-joint qpos post-reset and forward the sim
(validated: places the Lift cube exactly, expert then solves it). Task-dispatched;
Door/Wipe added in P3g.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

from .bounds import clamp_obj_xy, obj_z

# Primary-object free joint per task (discovered from the model). Fallback logic
# below finds a joint containing the object name if this map lacks the task.
OBJ_JOINT = {"Lift": "cube_joint0"}
OBJ_TOKEN = {"Lift": "cube"}


def _obj_joint(task: str, raw) -> str:
    name = OBJ_JOINT.get(task)
    if name is not None:
        return name
    tok = OBJ_TOKEN.get(task, "")
    for j in range(raw.sim.model.njnt):
        n = raw.sim.model.joint_id2name(j)
        if n and tok in n.lower():
            return n
    raise RuntimeError(f"could not find primary-object free joint for task {task}")


def _yaw_quat_wxyz(yaw: float):
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def set_object_pose(raw, task: str, x: float, y: float, yaw: float = 0.0,
                    z: Optional[float] = None, quat=None) -> Dict[str, float]:
    """Place the task's primary object at a prescribed pose (post-reset), then
    forward the sim. Returns the applied pose.

    Lift uses the cube's FREE JOINT qpos. Door has no free joint (the frame is a
    static body placed by the sampler via model.body_pos/body_quat in
    _reset_internal), so BRIDGE mirrors that: set the frame body's model pose +
    forward. For Door a full quat (the representative failure's real orientation)
    is passed so the door keeps a valid orientation; xy is the middle ground."""
    x, y = clamp_obj_xy(task, x, y)
    z = obj_z(task) if z is None else float(z)
    if task == "Door":
        did = raw.sim.model.body_name2id(raw.door.root_body)
        q = list(quat) if quat is not None else _yaw_quat_wxyz(yaw)
        raw.sim.model.body_pos[did] = [x, y, z]
        raw.sim.model.body_quat[did] = q
        raw.sim.forward()
        return {"x": x, "y": y, "z": z, "quat": q}
    qw, qz = math.cos(yaw / 2.0), math.sin(yaw / 2.0)     # yaw about +z
    raw.sim.data.set_joint_qpos(_obj_joint(task, raw), [x, y, z, qw, 0.0, 0.0, qz])
    raw.sim.forward()
    return {"x": x, "y": y, "z": z, "yaw": yaw}
