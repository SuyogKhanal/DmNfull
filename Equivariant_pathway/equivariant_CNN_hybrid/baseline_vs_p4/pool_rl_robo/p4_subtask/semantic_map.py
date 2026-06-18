"""KAG-grounded semantic direction helpers (prompt-side, Fix-3 spirit).

PushT already prescribes a SCALAR `tee_zrot` (no quaternions), so the literal
Fix 3 ("stop prescribing quaternions") is moot. We keep its spirit: give the LLM
grounded SEMANTIC directions ("toward the goal", "rotate clockwise") with their
numeric meaning, so its absolute-pose output is geometrically anchored rather
than hallucinated. These are used to build the prompt context; the planner still
clamps the LLM's numeric output to a bounded perturbation around the anchor.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

# Goal T pose (PushTEnv2): centre of mass + z-rotation (radians).
GOAL_XY: Tuple[float, float] = (-0.156, -0.1)
GOAL_ZROT: float = (5.0 / 3.0) * math.pi    # PushT-v1 goal orientation


def direction_to_goal(tee_xyz) -> Dict[str, object]:
    """Unit vector + bearing from a tee position toward the goal."""
    dx = GOAL_XY[0] - float(tee_xyz[0])
    dy = GOAL_XY[1] - float(tee_xyz[1])
    n = math.hypot(dx, dy) or 1.0
    return {"unit": [round(dx / n, 4), round(dy / n, 4)],
            "distance": round(n, 4),
            "bearing_rad": round(math.atan2(dy, dx), 4)}


def semantic_lines(anchor_tee_xyz, centroid_xyz) -> List[str]:
    """Human-readable grounded hints for the prompt context."""
    g = direction_to_goal(anchor_tee_xyz)
    cdx = centroid_xyz[0] - anchor_tee_xyz[0]
    cdy = centroid_xyz[1] - anchor_tee_xyz[1]
    return [
        f"Goal T is at xy=({GOAL_XY[0]}, {GOAL_XY[1]}), zrot={round(GOAL_ZROT,3)} rad.",
        f"From the anchor tee, 'toward_goal' = unit {g['unit']} (distance {g['distance']} m).",
        f"The dominant cluster centroid is offset ({round(cdx,3)}, {round(cdy,3)}) m "
        f"from the anchor tee — prescribe a coverage point in that direction.",
    ]
