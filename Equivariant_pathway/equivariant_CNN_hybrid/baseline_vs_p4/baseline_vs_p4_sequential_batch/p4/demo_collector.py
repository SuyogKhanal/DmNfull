"""Corridor-aware demo collector for the P4 variants.

Wraps the demo-recording logic from
``Equivariant_pathway.collect_demos`` (specifically ``_record_one`` +
``_build_forced_env``) but builds the A\* expert on a corridor-blocked
grid when the prescribed layout has a ``steps`` field.

Public entry point::

    collect(rec_path, demo_dir, seed, corridor_blocking=True) -> CollectResult

``CollectResult`` is a small dataclass capturing how many demos we kept,
how many were infeasible, and the per-layout outcomes so the round can
populate the compression_log + prescription_overlap.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from Equivariant_pathway.collect_demos import (  # noqa: E402
    _build_forced_env, _record_one,
)
from Equivariant_pathway.expert import AStarExpert, build_grid  # noqa: E402

from ..corridor.blocker import (
    build_constrained_expert,
    parse_steps_string,
)


@dataclass
class CollectedDemo:
    layout_id: str
    path: Optional[Path]
    feasible: bool
    corridor_used: bool
    path_length: int
    rationale: str = ""
    steps_str: str = ""
    start_pos: Optional[List[int]] = None
    goal_pos: Optional[List[int]] = None
    fire_positions: Optional[List[List[int]]] = None


@dataclass
class CollectResult:
    saved: List[CollectedDemo] = field(default_factory=list)
    infeasible: List[CollectedDemo] = field(default_factory=list)

    @property
    def n_saved(self) -> int:
        return len(self.saved)

    @property
    def n_infeasible(self) -> int:
        return len(self.infeasible)


def _layout_id(layout: Dict, idx: int) -> str:
    if "parent_demo_id" in layout:
        return (
            f"demo{layout.get('parent_demo_id','?')}"
            f"_layout{layout.get('layout_index','?')}"
            f"_rep{layout.get('repetition','1')}"
        )
    return f"prescribed_{idx + 1}"


def _make_expert(layout: Dict, env, corridor_blocking: bool):
    """Return (expert, used_corridor, feasible, path_len, corridor_str).

    The expert is built on either the corridor-blocked grid (if the
    prescription has ``steps`` and corridor_blocking is True), or the
    plain unblocked grid otherwise.
    """
    grid_size = int(env.grid.shape[0])
    fires = [tuple(p) for p in env.fire_positions]
    start = tuple(env.agent_pos)
    goal = tuple(env.goal_pos)

    steps_str = str(layout.get("steps") or "")
    if corridor_blocking and steps_str:
        steps = parse_steps_string(steps_str)
        if steps:
            _grid, expert, feasible, dist = build_constrained_expert(
                grid_size=grid_size,
                start_pos=start,
                goal_pos=goal,
                fire_positions=fires,
                corridor_steps=steps,
            )
            return expert, True, feasible, dist, steps_str

    # No corridor (or corridor disabled): use the unmasked expert exactly
    # like Equivariant_pathway.collect_demos does.
    grid = build_grid(grid_size=grid_size, fire_positions=fires, goal_pos=goal)
    expert = AStarExpert(grid, goal)
    path_len = expert.shortest_path_length(start)
    return expert, False, (path_len > 0), int(path_len), steps_str


def collect(
    rec_path: Path,
    demo_dir: Path,
    seed: int,
    corridor_blocking: bool = True,
) -> CollectResult:
    """Materialize prescribed layouts into demos.

    Reads ``rec_path`` (a recommended_layouts.json with the upstream
    schema) and writes demos under ``demo_dir`` in the same JSON schema
    as ``Equivariant_pathway.collect_demos``. When ``corridor_blocking``
    is True and the layout has a non-empty ``steps`` field, we force A\*
    down the prescribed corridor by walling off every other FREE cell.
    """
    with open(rec_path, "r") as f:
        spec = json.load(f)
    layouts: List[Dict] = (spec.get("layouts") if isinstance(spec, dict) else spec) or []

    demo_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    out = CollectResult()

    for idx, layout in enumerate(layouts):
        lid = _layout_id(layout, idx)
        ep_seed = seed + idx
        try:
            env = _build_forced_env(layout, seed=ep_seed)
        except Exception as exc:  # noqa: BLE001
            print(f"[p4-collect] {lid}: build_env failed ({exc!r}); skip", flush=True)
            out.infeasible.append(CollectedDemo(
                layout_id=lid, path=None, feasible=False,
                corridor_used=False, path_length=-1,
                rationale=str(layout.get("rationale", "")),
                steps_str=str(layout.get("steps", "")),
                start_pos=list(layout.get("start_pos") or []),
                goal_pos=list(layout.get("goal_pos") or []),
                fire_positions=[list(p) for p in (layout.get("fire_positions") or [])],
            ))
            continue

        expert, used_corridor, feasible, dist, steps_str = _make_expert(
            layout, env, corridor_blocking,
        )

        info = CollectedDemo(
            layout_id=lid,
            path=None,
            feasible=feasible,
            corridor_used=used_corridor,
            path_length=int(dist),
            rationale=str(layout.get("rationale", "")),
            steps_str=steps_str,
            start_pos=list(layout.get("start_pos") or []),
            goal_pos=list(layout.get("goal_pos") or []),
            fire_positions=[list(p) for p in (layout.get("fire_positions") or [])],
        )

        if not feasible or expert is None:
            print(f"[p4-collect] {lid}: corridor infeasible "
                  f"(used_corridor={used_corridor} dist={dist}); skip",
                  flush=True)
            env.close()
            out.infeasible.append(info)
            continue

        # When using corridor blocking, override the env's grid with the
        # masked one so the rollout sees the blocked walls. The expert
        # already operates on the masked grid; we mirror it into the env
        # so the policy's observations / images reflect the corridor.
        if used_corridor:
            env.grid = expert.grid

        path = _record_one(env, expert, rng, lid, demo_dir)
        env.close()
        info.path = path
        if path is not None:
            out.saved.append(info)
        else:
            out.infeasible.append(info)

    return out


def write_summary(result: CollectResult, out_path: Path) -> None:
    """Persist a small JSON summary of the collection outcome."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_saved": result.n_saved,
        "n_infeasible": result.n_infeasible,
        "saved": [
            {
                "layout_id": d.layout_id,
                "path": str(d.path) if d.path else None,
                "corridor_used": d.corridor_used,
                "path_length": d.path_length,
                "steps_str": d.steps_str,
                "start_pos": d.start_pos,
                "goal_pos": d.goal_pos,
                "fire_positions": d.fire_positions,
            }
            for d in result.saved
        ],
        "infeasible": [
            {
                "layout_id": d.layout_id,
                "corridor_used": d.corridor_used,
                "path_length": d.path_length,
                "steps_str": d.steps_str,
                "rationale": d.rationale,
                "start_pos": d.start_pos,
                "goal_pos": d.goal_pos,
                "fire_positions": d.fire_positions,
            }
            for d in result.infeasible
        ],
        "timestamp": int(time.time() * 1000),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
