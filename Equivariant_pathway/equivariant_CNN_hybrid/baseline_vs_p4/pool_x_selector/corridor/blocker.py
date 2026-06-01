"""Corridor blocking — convert an LLM-prescribed pathway into a constrained
A\* expert by walling off every FREE cell that is not on the path.

The flow for a P4 round (sequential or batch) is:

    1. LLM prescribes a layout with ``steps: (r,c)->(r,c)->...``.
    2. We extract those steps together with ``start_pos``, ``goal_pos``,
       and ``fire_positions``.
    3. ``build_constrained_grid`` constructs the unconstrained grid as
       ``Equivariant_pathway.expert.build_grid`` would, then overwrites
       every TILE_FREE cell *outside* the corridor with TILE_WALL.
    4. ``feasibility_check`` confirms the A\* distance map still reaches
       the start from the goal; if not, this prescription is dropped and
       a ``corridor_infeasible`` event is logged.
    5. The constrained grid is handed to ``AStarExpert`` so the BFS demo
       collector is forced down exactly that path.

The module has a ``__main__`` self-test exercising the three core
guarantees so it can be smoke-tested independently of the rest of the
suite.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Equivariant_pathway.expert import (  # noqa: E402
    AStarExpert,
    build_grid,
    compute_distance_map,
)
from envs.maze_env import TILE_FIRE, TILE_FREE, TILE_GOAL, TILE_WALL  # noqa: E402


Cell = Tuple[int, int]


# Matches ``(row,col)`` allowing arbitrary whitespace. Coordinates may be
# negative in the LLM's output (we'll reject those during validation).
_STEP_RE = re.compile(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")


def parse_steps_string(s: str) -> List[Cell]:
    """Parse ``(r,c)->(r,c)->...`` into an ordered list of cells.

    Tolerates whitespace, alternative arrows ("->", "→", " -> "), and
    stray text around the matches. Empty input yields an empty list.
    """
    if not s:
        return []
    return [(int(r), int(c)) for r, c in _STEP_RE.findall(s)]


def is_inside_grid(cell: Cell, grid_size: int) -> bool:
    r, c = cell
    return 0 <= r < grid_size and 0 <= c < grid_size


def build_constrained_grid(
    grid_size: int,
    start_pos: Cell,
    goal_pos: Cell,
    fire_positions: Sequence[Cell],
    corridor_steps: Iterable[Cell],
) -> Tuple[np.ndarray, Set[Cell]]:
    """Construct a maze grid in which everything outside the prescribed
    corridor is walled off.

    Returns ``(grid, corridor_cells)``. ``corridor_cells`` always includes
    ``start_pos`` and ``goal_pos`` so the start/goal are guaranteed free
    even if the LLM forgot to list them.

    A fire on a corridor cell is *kept* — the prescription is honored, but
    the BFS expert will then be unable to traverse it, and the
    ``feasibility_check`` below will report infeasibility downstream.
    """
    grid = build_grid(grid_size, fire_positions, goal_pos)

    corridor: Set[Cell] = {tuple(c) for c in corridor_steps if is_inside_grid(tuple(c), grid_size)}
    corridor.add(tuple(start_pos))
    corridor.add(tuple(goal_pos))

    # Overwrite every FREE cell outside the corridor with WALL. Leave
    # WALL/FIRE/GOAL tiles alone — turning a fire on the corridor into a
    # wall would silently fix the LLM's mistake.
    for r in range(grid_size):
        for c in range(grid_size):
            if (r, c) in corridor:
                continue
            if grid[r, c] == TILE_FREE:
                grid[r, c] = TILE_WALL
    return grid, corridor


def feasibility_check(
    grid: np.ndarray, start_pos: Cell, goal_pos: Cell,
) -> Tuple[bool, int]:
    """Return ``(feasible, distance_from_start_to_goal)``.

    ``feasible`` is True iff A\* on the constrained grid reaches the goal
    from the start with finite distance. The distance returned is -1 when
    infeasible so it serializes nicely into JSON.
    """
    dist_map = compute_distance_map(grid, goal_pos)
    sr, sc = int(start_pos[0]), int(start_pos[1])
    if not (0 <= sr < grid.shape[0] and 0 <= sc < grid.shape[1]):
        return False, -1
    d = int(dist_map[sr, sc])
    INF = 10 ** 9
    if d >= INF:
        return False, -1
    return True, d


def build_constrained_expert(
    grid_size: int,
    start_pos: Cell,
    goal_pos: Cell,
    fire_positions: Sequence[Cell],
    corridor_steps: Iterable[Cell],
) -> Tuple[np.ndarray, Optional[AStarExpert], bool, int]:
    """One-shot constructor used by the P4 demo collectors.

    Returns ``(grid, expert_or_None, feasible, path_length)``. When the
    corridor is infeasible we return ``(grid, None, False, -1)`` so the
    caller can log the event and skip the prescription without crashing.
    """
    grid, _corridor = build_constrained_grid(
        grid_size=grid_size,
        start_pos=start_pos,
        goal_pos=goal_pos,
        fire_positions=fire_positions,
        corridor_steps=corridor_steps,
    )
    feasible, dist = feasibility_check(grid, start_pos, goal_pos)
    if not feasible:
        return grid, None, False, dist
    expert = AStarExpert(grid=grid, goal_pos=tuple(goal_pos))
    return grid, expert, True, dist


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _selftest() -> None:
    # ---- step parser
    steps = parse_steps_string("(0,0)->(0,1)->(1,1)")
    assert steps == [(0, 0), (0, 1), (1, 1)], f"unexpected: {steps}"

    spaced = parse_steps_string("  ( 2 , 3 ) -> ( 2 , 4 ) ")
    assert spaced == [(2, 3), (2, 4)], f"unexpected: {spaced}"

    assert parse_steps_string("") == []
    assert parse_steps_string("no steps here") == []

    # ---- happy path: a 5×5 grid, prescribed L-shaped corridor
    grid, corridor = build_constrained_grid(
        grid_size=5,
        start_pos=(0, 0),
        goal_pos=(2, 2),
        fire_positions=[(0, 2), (3, 3)],
        corridor_steps=[(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],
    )
    expected_corridor = {(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)}
    assert expected_corridor.issubset(corridor), f"corridor missing cells: {expected_corridor - corridor}"

    # Every non-corridor FREE cell is now WALL; corridor cells stay FREE
    # (except the goal which is TILE_GOAL); fires stay fires.
    for r in range(5):
        for c in range(5):
            tile = int(grid[r, c])
            if (r, c) == (2, 2):
                assert tile == TILE_GOAL, f"goal tile wrong: {tile}"
            elif (r, c) in expected_corridor:
                assert tile in (TILE_FREE, TILE_GOAL), f"corridor cell {r,c} got tile {tile}"
            elif (r, c) in {(0, 2), (3, 3)}:
                assert tile == TILE_FIRE, f"fire {r,c} got tile {tile}"
            else:
                assert tile == TILE_WALL, f"non-corridor cell {r,c} should be WALL got {tile}"

    feasible, dist = feasibility_check(grid, (0, 0), (2, 2))
    assert feasible, "expected corridor to be feasible"
    assert dist == 4, f"expected path length 4 (corridor of 5 cells), got {dist}"

    # ---- infeasible corridor: a disconnected stub that doesn't reach goal
    bad_grid, _ = build_constrained_grid(
        grid_size=5,
        start_pos=(0, 0),
        goal_pos=(4, 4),
        fire_positions=[],
        corridor_steps=[(0, 0), (0, 1)],
    )
    feasible_bad, dist_bad = feasibility_check(bad_grid, (0, 0), (4, 4))
    assert not feasible_bad, "expected disconnected corridor to be infeasible"
    assert dist_bad == -1, f"infeasible distance must be -1, got {dist_bad}"

    # ---- one-shot expert builder
    g2, expert, feas, d = build_constrained_expert(
        grid_size=5,
        start_pos=(0, 0),
        goal_pos=(2, 2),
        fire_positions=[(0, 2), (3, 3)],
        corridor_steps=[(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],
    )
    assert feas and expert is not None and d == 4
    mask = expert.optimal_actions((0, 0))
    assert mask.sum() >= 1, f"expert should pick at least one action at start, got mask={mask}"

    # ---- expert says None when infeasible
    _, expert_none, feas_none, _ = build_constrained_expert(
        grid_size=5, start_pos=(0, 0), goal_pos=(4, 4),
        fire_positions=[], corridor_steps=[(0, 0), (0, 1)],
    )
    assert expert_none is None and not feas_none

    print("[corridor.blocker] self-test PASSED")


if __name__ == "__main__":
    _selftest()
