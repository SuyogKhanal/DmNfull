"""Per-run path resolver. One stop for "where does run N's foo live"."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

from ..layouts.layout_setup import RESULTS_ROOT, SUITE_ROOT


METHOD_DIR_NAMES = (
    "baseline_highloss_fixed", "baseline_random_fixed",
    "baseline_highloss_rotate", "baseline_random_rotate",
    "p4_top3_fixed", "p4_top3_rotate",
    "p4_all_fixed", "p4_all_rotate",
    # IIL baselines (rotated-only) added for the P4-LLM comparison. They write
    # into the same results/run_N/<method>/ tree and read P4's shared pools.
    "safe_dagger", "dropout_dagger", "ensemble_dagger",
    "thrifty_dagger", "stagger",
)


@dataclass(frozen=True)
class RunWorkspace:
    run_id: int
    root: Path                # results/run_{id}/
    shared: Path              # results/run_{id}/shared/
    init_demos: Path          # shared/init_demos/
    init_checkpoints: Path    # shared/init_checkpoints/
    correction_yaml: Path     # shared/correction_layouts.yaml

    def method_root(self, method: str) -> Path:
        return self.root / method

    def method_dirs(self, method: str) -> Dict[str, Path]:
        m = self.method_root(method)
        return {
            "root": m,
            "demos": m / "demos",
            "checkpoints": m / "checkpoints",
            "results": m / "results",
        }


def workspace_for(run_id: int) -> RunWorkspace:
    """Return a frozen workspace for ``run_id`` and ensure all dirs exist."""
    root = RESULTS_ROOT / f"run_{int(run_id)}"
    shared = root / "shared"
    ws = RunWorkspace(
        run_id=int(run_id),
        root=root,
        shared=shared,
        init_demos=shared / "init_demos",
        init_checkpoints=shared / "init_checkpoints",
        correction_yaml=shared / "correction_layouts.yaml",
    )
    for d in (
        ws.root, ws.shared, ws.init_demos, ws.init_checkpoints,
        *(ws.method_root(m) for m in METHOD_DIR_NAMES),
    ):
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


def all_run_ids() -> Iterable[int]:
    """Run IDs that have a results/run_{id}/ directory on disk."""
    if not RESULTS_ROOT.exists():
        return []
    out = []
    for d in RESULTS_ROOT.glob("run_*"):
        if d.is_dir():
            try:
                out.append(int(d.name.split("_")[-1]))
            except ValueError:
                continue
    return sorted(out)
