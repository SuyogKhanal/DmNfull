"""DISTIL GridWorld 5x5 — vendored, self-contained subpackage.

The p4m-equivariant classifier (escnn) on a 5-channel semantic grid, a BFS multi-
label expert, and the full DISTIL loop (screen -> geometric silhouette-k cluster ->
Eq-9 memory rotation -> LLM SELECT/BRIDGE + confidence -> one BFS demo/round). Reuses
the task-agnostic distil/p4 compression core + the OpenRouter LLM client. Runs on CPU
(no mujoco, no GPU render — VLM frames are numpy bird-eye rasters).
"""
from .env import make_maze, bfs_path
from .layouts import sample_layout, sample_layouts
from .loop import run_distil_gridworld, bootstrap_demos
from .policy import GridWorldPolicy

__all__ = ["make_maze", "bfs_path", "sample_layout", "sample_layouts",
           "run_distil_gridworld", "bootstrap_demos", "GridWorldPolicy"]
