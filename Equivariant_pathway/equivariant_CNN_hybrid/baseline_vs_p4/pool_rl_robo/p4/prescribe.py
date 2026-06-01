"""P4 PRESCRIPTION pass (text LLM). Given the top-K failure analyses + the env
KAG + the workspace bounds, the planner LLM COMPRESSES the failures into ONE
environment configuration (a goal, plus an object placement for
FetchPickAndPlace) that a known-good expert can demonstrate — cutting the extra
demos the IIL baselines would spend. Returns strict JSON; the config is then
loaded into the sim and strictly checked (loadable + expert-solvable)."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from . import kag as kag_mod


def _fmt(v) -> str:
    return "[" + ", ".join(f"{float(x):.3f}" for x in np.asarray(v).ravel()) + "]"


def prescribe_config(env_id: str, analyses: List[str], bounds: Dict, client,
                     resolved_env_id: str = None) -> Dict:
    is_pick = env_id.startswith("FetchPickAndPlace")
    schema = ('{"goal": [x, y, z]' + (', "object": [x, y]' if is_pick else '')
              + ', "rationale": "<short>"}')
    obj_line = (f"\n- object MUST be within obj_low={_fmt(bounds['obj_low'])} "
                f"obj_high={_fmt(bounds['obj_high'])} (the box's table x,y)") if is_pick else ""
    system = (
        "You are P4, a demonstration planner. Given analyses of the top-3 failures "
        "of a behavior-cloning policy on a Fetch robot task, prescribe EXACTLY ONE "
        "environment configuration that a known-good EXPERT can then demonstrate, "
        "COMPRESSING the 3 failures into ONE corrective demonstration (so we spend "
        "1 expert demo instead of 3). The configuration must be inside the workspace "
        "bounds so it is loadable and solvable. Reply with ONLY JSON: " + schema
    )
    kag_text = kag_mod.load_kag(env_id, resolved_env_id)
    user = (
        (f"ENVIRONMENT KNOWLEDGE (KAG):\n{kag_text.strip()}\n\n" if kag_text.strip() else "")
        + "WORKSPACE BOUNDS (your config MUST lie inside these):\n"
        + f"- goal in goal_low={_fmt(bounds['goal_low'])} .. goal_high={_fmt(bounds['goal_high'])}"
        + obj_line + "\n\n"
        + "TOP-3 FAILURE ANALYSES:\n"
        + "\n".join(f"  [{i + 1}] {a}" for i, a in enumerate(analyses)) + "\n\n"
        + "Prescribe ONE configuration that targets the COMMON failure mode across "
        + "these (e.g. a goal" + (" + object placement" if is_pick else "")
        + " in the region the policy keeps failing). Return JSON " + schema + "."
    )
    obj = client.chat_json(system, user, max_tokens=1024)
    out = {"goal": obj.get("goal"), "object": (obj.get("object") if is_pick else None),
           "rationale": str(obj.get("rationale", ""))[:300]}
    return out
