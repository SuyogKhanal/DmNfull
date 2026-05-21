"""Per-run layout sourcing for the baseline_vs_p4_sequential_batch suite.

Adapted from
``Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/layout_setup.py``.

Two changes from upstream:

1. Outputs live under this suite's ``results/run_{id}/shared/`` so the
   suite never touches upstream's ``runs/`` tree.
2. ``correction_n`` defaults are driven by config (50 in this suite vs 40
   upstream); the function accepts it as a parameter so the orchestrator
   can pass it through.

We *reuse* upstream's training_layouts.yaml + heldout_layouts.yaml verbatim
(they are stable, seed-deterministic, and shared across the whole repo).
If those files do not exist, we materialize them via the same seed scheme
upstream uses so the initial-20 BFS demos + initial checkpoint stay
byte-identical to upstream's `shared/init_*/` cache.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import yaml

# Resolve the repo root so we can import the upstream shared utilities.
# This file lives at: …/baseline_vs_p4/baseline_vs_p4_sequential_batch/layouts/
# Parents:           [4] = baseline_vs_p4_sequential_batch, [5] = baseline_vs_p4,
#                    [6] = equivariant_CNN_hybrid, [7] = Equivariant_pathway,
#                    [8] = repo root.
REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Equivariant_pathway.layout_sampler import (  # noqa: E402
    _load_blocked_signatures,
    _signature,
    sample_layouts,
    write_yaml,
)

# Upstream shared dir — read-only source for training_layouts.yaml +
# heldout_layouts.yaml. We never write here.
UPSTREAM_COMPARE_ROOT = (
    REPO_ROOT / "Equivariant_pathway" / "equivariant_CNN_hybrid" / "baseline_vs_p4"
)
UPSTREAM_SHARED_DIR = UPSTREAM_COMPARE_ROOT / "shared"

# This suite's own root: …/baseline_vs_p4/baseline_vs_p4_sequential_batch/
SUITE_ROOT = UPSTREAM_COMPARE_ROOT / "baseline_vs_p4_sequential_batch"
RESULTS_ROOT = SUITE_ROOT / "results"

# Seeds for the GLOBAL layout sets — kept identical to upstream so that the
# initial-20 demos + initial checkpoint cached at
# ``shared/init_demos/`` and ``shared/init_checkpoints/`` remain valid.
GLOBAL_TRAIN_SEED = 11
GLOBAL_HELDOUT_SEED = 22

# Per-run correction-pool seed base. Kept identical to upstream so the
# in-run contamination guard semantics are unchanged.
CORRECTION_SEED_BASE = 91_000_000


def _write_training_layouts(layouts: List[Dict], path: Path) -> None:
    payload = {
        "img_size": 80, "grid_size": 5, "cell_px": 16,
        "num_fires": 3, "min_manhattan": 4, "n_repetitions": 1,
        "training_layouts": [
            {**L, "n_repetitions": int(L.get("n_repetitions", 1) or 1)}
            for L in layouts
        ],
    }
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def ensure_shared_layouts(initial_demos: int, heldout_n: int) -> Dict[str, Path]:
    """Return paths to training_layouts.yaml and heldout_layouts.yaml.

    These live in *upstream's* shared dir. They are seed-deterministic and
    shared across the entire repo; we do not duplicate them.

    If they are missing (fresh checkout), we materialize them in-place
    using the upstream seed scheme so the byte content matches upstream's.
    """
    UPSTREAM_SHARED_DIR.mkdir(parents=True, exist_ok=True)
    train_yaml = UPSTREAM_SHARED_DIR / "training_layouts.yaml"
    heldout_yaml = UPSTREAM_SHARED_DIR / "heldout_layouts.yaml"

    if not heldout_yaml.exists():
        heldout = sample_layouts(
            n=heldout_n, grid_size=5, num_fires=3, min_manhattan=4,
            seed=GLOBAL_HELDOUT_SEED, blocked_signatures=set(),
        )
        write_yaml(heldout, heldout_yaml, grid_size=5)

    if not train_yaml.exists():
        blocked = _load_blocked_signatures([str(heldout_yaml)])
        train = sample_layouts(
            n=initial_demos, grid_size=5, num_fires=3, min_manhattan=4,
            seed=GLOBAL_TRAIN_SEED, blocked_signatures=blocked,
        )
        _write_training_layouts(train, train_yaml)

    return {"train": train_yaml, "heldout": heldout_yaml}


def _signatures_from_yaml(path: Path) -> Set[Tuple]:
    if not path.exists():
        return set()
    with open(path, "r") as f:
        spec = yaml.safe_load(f) or {}
    out: Set[Tuple] = set()
    for key in ("test_layouts", "training_layouts", "heldout_test_layouts", "layouts"):
        for L in spec.get(key, []) or []:
            out.add(_signature(L["start_pos"], L["goal_pos"], L["fire_positions"]))
    return out


def assert_no_contamination(
    correction_yaml: Path, train_yaml: Path, heldout_yaml: Path,
) -> Dict[str, int]:
    """Verify correction ⊥ training ⊥ heldout. Raises SystemExit on leak.

    Verbatim semantics from upstream — same signature scheme, same error
    surface so existing log readers keep working.
    """
    corr_sigs = _signatures_from_yaml(correction_yaml)
    train_sigs = _signatures_from_yaml(train_yaml)
    heldout_sigs = _signatures_from_yaml(heldout_yaml)

    leak_train = corr_sigs & train_sigs
    leak_heldout = corr_sigs & heldout_sigs
    if leak_train or leak_heldout:
        lines = ["[layout-setup] CONTAMINATION DETECTED in correction pool:"]
        for sig in sorted(leak_train):
            lines.append(f"  * shares (start,goal,fires) with TRAINING set: {sig}")
        for sig in sorted(leak_heldout):
            lines.append(f"  * shares (start,goal,fires) with HELDOUT  set: {sig}")
        raise SystemExit("\n".join(lines))

    return {
        "n_correction": len(corr_sigs),
        "n_train": len(train_sigs),
        "n_heldout": len(heldout_sigs),
        "overlap_train": 0,
        "overlap_heldout": 0,
    }


def ensure_correction_layouts_for_run(
    run_id: int,
    correction_n: int,
    train_yaml: Path,
    heldout_yaml: Path,
    out_dir: Path,
) -> Path:
    """Sample correction layouts for a single run and write
    ``out_dir/correction_layouts.yaml``.

    Per-run seed = CORRECTION_SEED_BASE + run_id so each of the N runs
    produces a deterministically distinct pool.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "correction_layouts.yaml"
    if out_path.exists():
        report = assert_no_contamination(out_path, train_yaml, heldout_yaml)
        _save_report(out_dir, run_id, report)
        return out_path

    blocked = _load_blocked_signatures([str(train_yaml), str(heldout_yaml)])
    layouts = sample_layouts(
        n=correction_n, grid_size=5, num_fires=3, min_manhattan=4,
        seed=CORRECTION_SEED_BASE + int(run_id),
        blocked_signatures=blocked,
    )
    for i, L in enumerate(layouts):
        L["name"] = f"corr_r{int(run_id):02d}_{i + 1:03d}"
    write_yaml(layouts, out_path, grid_size=5)

    report = assert_no_contamination(out_path, train_yaml, heldout_yaml)
    _save_report(out_dir, run_id, report)
    return out_path


def _save_report(out_dir: Path, run_id: int, report: Dict[str, int]) -> None:
    out = {
        "run_id": int(run_id),
        "correction_seed": CORRECTION_SEED_BASE + int(run_id),
        "global_train_seed": GLOBAL_TRAIN_SEED,
        "global_heldout_seed": GLOBAL_HELDOUT_SEED,
        **report,
    }
    with open(out_dir / "layout_setup_report.json", "w") as f:
        json.dump(out, f, indent=2, default=str)


def run_dir_for(run_id: int) -> Path:
    return RESULTS_ROOT / f"run_{int(run_id)}"


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--run_id", type=int, required=True)
    p.add_argument("--initial_demos", type=int, default=20)
    p.add_argument("--heldout_n", type=int, default=200)
    p.add_argument("--correction_n", type=int, default=50)
    args = p.parse_args()

    shared = ensure_shared_layouts(args.initial_demos, args.heldout_n)
    out_dir = run_dir_for(args.run_id) / "shared"
    corr = ensure_correction_layouts_for_run(
        args.run_id, args.correction_n,
        train_yaml=shared["train"], heldout_yaml=shared["heldout"],
        out_dir=out_dir,
    )
    print(f"[layout-setup] OK run_id={args.run_id}")
    print(f"  train   -> {shared['train']}")
    print(f"  heldout -> {shared['heldout']}")
    print(f"  corr    -> {corr}")
