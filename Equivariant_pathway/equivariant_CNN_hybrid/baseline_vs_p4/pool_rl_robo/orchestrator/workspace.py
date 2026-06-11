"""Per-(env, run) path resolver (continuous-control analogue of
pool_x_selector's ``orchestrator/workspace.py``). Here the run grid is indexed
by ENVIRONMENT (one SLURM array task per env) rather than by run number, so
results live under ``results/{ENV}/run_{id}/{method}/``."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

from ..envs.env_setup import RESULTS_ROOT, SUITE_ROOT

# LOCKSTEP method set (7): p4_top3 (ours) + diff_dagger + the 5 IIL baselines.
# Keep in sync with config.yaml::methods, run_one.py::METHOD_SPEC,
# aggregation/aggregate.py, and nb_plot.py.
METHOD_DIR_NAMES = (
    "p4_top3",
    "diff_dagger",
    "safe_dagger", "dropout_dagger", "ensemble_dagger",
    "thrifty_dagger", "stagger",
    "p4_select",   # industry variant (P4-LLM selection on top of SafeDAgger)
)


@dataclass(frozen=True)
class RunWorkspace:
    env_id: str
    run_id: int
    root: Path                # results/{ENV}/run_{id}/
    shared: Path              # results/{ENV}/run_{id}/shared/

    def method_root(self, method: str) -> Path:
        return self.root / method

    def method_dirs(self, method: str) -> Dict[str, Path]:
        m = self.method_root(method)
        return {"root": m, "demos": m / "demos", "checkpoints": m / "checkpoints",
                "results": m / "results"}


def workspace_for(env_id: str, run_id: int = 0) -> RunWorkspace:
    root = RESULTS_ROOT / str(env_id) / f"run_{int(run_id)}"
    ws = RunWorkspace(env_id=str(env_id), run_id=int(run_id), root=root,
                      shared=root / "shared")
    for d in (ws.root, ws.shared, *(ws.method_root(m) for m in METHOD_DIR_NAMES)):
        d.mkdir(parents=True, exist_ok=True)
    return ws


def aggregate_dir() -> Path:
    p = RESULTS_ROOT / "aggregate"
    p.mkdir(parents=True, exist_ok=True)
    return p


def logs_dir() -> Path:
    p = SUITE_ROOT / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def all_envs() -> Iterable[str]:
    if not RESULTS_ROOT.exists():
        return []
    return sorted(d.name for d in RESULTS_ROOT.glob("*")
                  if d.is_dir() and d.name != "aggregate")
