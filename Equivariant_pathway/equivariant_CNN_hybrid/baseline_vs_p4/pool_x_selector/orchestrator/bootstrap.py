"""One-time and per-run bootstrap of shared assets.

There are two layers of "shared":

1. Upstream-shared, repo-wide (read-only here):
     baseline_vs_p4/shared/{training_layouts.yaml, heldout_layouts.yaml,
                            init_demos/, init_checkpoints/}
   These are seed-deterministic, byte-identical across the whole repo,
   and shared with the upstream notebooks/swatch.sh. We *reuse* them.

2. Per-run, this-suite-only (writable):
     results/run_{id}/shared/{correction_layouts.yaml, init_demos/,
                              init_checkpoints/}
   The init_demos / init_checkpoints under per-run shared are copies of
   the upstream-shared cache so each run's method gets an isolated
   initial state and can corrupt its own copy without affecting peers.

``ensure_upstream_bootstrap`` triggers the upstream cache build only if
the artifact is missing on disk (idempotent). It calls
``Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.run_one
--bootstrap_only`` as a subprocess so we don't duplicate the upstream
demo collector + initial trainer.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from ..layouts.layout_setup import (
    REPO_ROOT,
    UPSTREAM_SHARED_DIR,
    ensure_shared_layouts,
)


def _info(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [bootstrap] {msg}", flush=True)


def ensure_upstream_bootstrap(
    seed: int, initial_epochs: int, initial_demos: int, heldout_n: int,
) -> None:
    """Make sure the upstream-shared assets exist.

    Idempotent: if every artifact is already on disk, this is a no-op.
    Otherwise we call upstream's ``run_one --bootstrap_only`` which is the
    canonical way to materialize them, then sanity-check we got what we
    needed.
    """
    shared = ensure_shared_layouts(initial_demos=initial_demos, heldout_n=heldout_n)
    upstream_demos_dir = UPSTREAM_SHARED_DIR / "init_demos"
    upstream_ckpt_dir = UPSTREAM_SHARED_DIR / "init_checkpoints"

    have_demos = (
        upstream_demos_dir.exists()
        and sum(1 for _ in upstream_demos_dir.glob("*.json")) >= initial_demos
    )
    have_ckpt = (upstream_ckpt_dir / "best_hybrid_policy.pth").exists()
    if have_demos and have_ckpt:
        _info(
            f"upstream cache OK "
            f"(demos={upstream_demos_dir}, ckpt={upstream_ckpt_dir})"
        )
        return

    _info("upstream cache incomplete — invoking upstream bootstrap_only")
    cmd = [
        sys.executable, "-u", "-m",
        "Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.run_one",
        "--run_index", "0",
        "--bootstrap_only",
        "--initial_demos", str(int(initial_demos)),
        "--heldout_n", str(int(heldout_n)),
        "--initial_epochs", str(int(initial_epochs)),
        "--seed", str(int(seed)),
    ]
    _info("$ " + " ".join(cmd))
    rc = subprocess.call(cmd, cwd=str(REPO_ROOT))
    if rc != 0:
        raise RuntimeError(f"upstream bootstrap_only failed rc={rc}")

    # Re-verify.
    if not (upstream_ckpt_dir / "best_hybrid_policy.pth").exists():
        raise RuntimeError(
            f"upstream bootstrap finished but {upstream_ckpt_dir/'best_hybrid_policy.pth'} "
            "is still missing"
        )
    _info("upstream cache populated")


def mirror_upstream_into_run(
    upstream_demos_dir: Path, upstream_ckpt_dir: Path,
    per_run_demos_dir: Path, per_run_ckpt_dir: Path,
) -> None:
    """Copy upstream-shared init demos + init checkpoints into the per-run
    workspace so the run owns its own mutable copy.
    """
    per_run_demos_dir.mkdir(parents=True, exist_ok=True)
    per_run_ckpt_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(upstream_demos_dir.glob("*.json")):
        dst = per_run_demos_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
    for name in ("best_hybrid_policy.pth", "last_hybrid_policy.pth"):
        src = upstream_ckpt_dir / name
        if src.exists():
            shutil.copy2(src, per_run_ckpt_dir / name)
