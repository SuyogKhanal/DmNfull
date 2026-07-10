"""Per-task workspace bounds for P4-LLM BRIDGE scene-prescription (robosuite UR5e).

BRIDGE prescribes a middle-ground object placement; it must stay inside the task's
valid placement range (the same range robosuite's UniformRandomSampler draws from),
so the scripted expert can still solve it and the demo is on-distribution. Ranges
are the empirically-confirmed sampler ranges (see the Lift note); Door/Wipe filled
when those tasks are wired.
"""
from __future__ import annotations

from typing import Dict, Tuple

# obj xy = the primary randomized object (Lift cube / Door handle-frame / Wipe n/a).
# z is the resting height (informational); yaw range (0,0) => no yaw randomization.
TASK_BOUNDS: Dict[str, Dict] = {
    # Lift: UniformRandomSampler x_range=[-0.03,0.03] y_range=[-0.03,0.03], rot=None,
    # table_offset z=0.8, cube rests at ~0.831. Confirmed empirically over 12 seeds.
    "Lift": {"obj_x": (-0.03, 0.03), "obj_y": (-0.03, 0.03), "obj_z": 0.831, "yaw": (0.0, 0.0)},
    # Door: the randomized "object" is the door FRAME root body (placed via
    # body_pos/body_quat, no free joint). ABSOLUTE door body_xpos range measured
    # over seeds (padded): x~[-0.135,-0.108], y~[-0.366,-0.340], z=1.10,
    # yaw~[-1.82,-1.57]. BRIDGE keeps the representative's z+quat and shifts xy.
    "Door": {"obj_x": (-0.135, -0.108), "obj_y": (-0.366, -0.340), "obj_z": 1.10, "yaw": (-1.82, -1.57)},
    # Wipe: the randomized quantity is a 100-marker dirt PATH (arena-driven) — no
    # single object pose exists, so BRIDGE is infeasible and Wipe is SELECT-only
    # (no bounds needed).
}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def clamp_obj_xy(task: str, x: float, y: float) -> Tuple[float, float]:
    """Clamp a prescribed object (x,y) to the task's valid placement range."""
    b = TASK_BOUNDS[task]
    return _clamp(x, *b["obj_x"]), _clamp(y, *b["obj_y"])


def obj_z(task: str) -> float:
    return float(TASK_BOUNDS[task]["obj_z"])


def as_dict(task: str) -> Dict:
    return dict(TASK_BOUNDS.get(task, {}))
