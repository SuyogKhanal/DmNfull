"""Shared-layout setup for the baseline-vs-P4 comparison.

One repetition of the comparison needs three layout sets:

* ``training_layouts.yaml``    — the initial 20 BFS layouts used to
  collect ``initial_demos``. Constant across all repetitions so the
  initial model is identical for every run.
* ``heldout_layouts.yaml``     — the 200 held-out layouts used by the
  per-round held-out eval. Constant across all repetitions for the
  same reason — and explicitly NEVER allowed to leak into the
  correction pool.
* ``correction_layouts.yaml``  — the per-run correction pool of size
  ``correction_n``. Seeded by the run index so the composition is
  different in every repetition, with a hard guard that every signature
  ``(start, goal, sorted(fires))`` is disjoint from BOTH the training
  and the heldout sets.

The training and heldout layouts are written ONCE to
``baseline_vs_p4/shared/``; correction layouts are written under
``baseline_vs_p4/runs/run_NN/shared/`` so concurrent runs never touch
each other's files. Within a single run, baseline_only and p4_only both
read from that same ``run_NN/shared/`` so they see byte-identical
correction pools.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Equivariant_pathway.layout_sampler import (
    _load_blocked_signatures,
    _signature,
    sample_layouts,
    write_yaml,
)

HYBRID_ROOT = REPO_ROOT / "Equivariant_pathway" / "equivariant_CNN_hybrid"
COMPARE_ROOT = HYBRID_ROOT / "baseline_vs_p4"
SHARED_DIR = COMPARE_ROOT / "shared"
RUNS_DIR = COMPARE_ROOT / "runs"

# Seeds for the GLOBAL layout sets (constant across runs).
GLOBAL_TRAIN_SEED   = 11
GLOBAL_HELDOUT_SEED = 22

# Offset added to the run index when seeding the correction pool. Keeps
# the correction RNG state well separated from the train / heldout
# seeds so a future change to either does not silently overlap with an
# existing correction sample.
CORRECTION_SEED_BASE = 91_000_000


def _shared_paths():
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "train":   SHARED_DIR / "training_layouts.yaml",
        "heldout": SHARED_DIR / "heldout_layouts.yaml",
    }


def _write_training_layouts(layouts: List[Dict], path: Path):
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
    """Idempotently create the training_layouts + heldout_layouts files
    that all 10 (or N) repetitions share.

    Returns the resolved paths in a dict {"train": ..., "heldout": ...}.
    """
    paths = _shared_paths()

    if not paths["heldout"].exists():
        heldout = sample_layouts(
            n=heldout_n, grid_size=5, num_fires=3, min_manhattan=4,
            seed=GLOBAL_HELDOUT_SEED, blocked_signatures=set(),
        )
        write_yaml(heldout, paths["heldout"], grid_size=5)

    if not paths["train"].exists():
        blocked = _load_blocked_signatures([str(paths["heldout"])])
        train = sample_layouts(
            n=initial_demos, grid_size=5, num_fires=3, min_manhattan=4,
            seed=GLOBAL_TRAIN_SEED, blocked_signatures=blocked,
        )
        _write_training_layouts(train, paths["train"])

    return paths


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


def assert_no_contamination(correction_yaml: Path, train_yaml: Path,
                            heldout_yaml: Path) -> Dict[str, int]:
    """Verify the correction pool shares ZERO signatures with the
    training or heldout sets. Raises SystemExit on any leak so the run
    cannot proceed with a contaminated pool.

    Returns a small diagnostic dict for logging.
    """
    corr_sigs    = _signatures_from_yaml(correction_yaml)
    train_sigs   = _signatures_from_yaml(train_yaml)
    heldout_sigs = _signatures_from_yaml(heldout_yaml)

    leak_train   = corr_sigs & train_sigs
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
        "n_train":      len(train_sigs),
        "n_heldout":    len(heldout_sigs),
        "overlap_train":   0,
        "overlap_heldout": 0,
    }


def ensure_correction_layouts_for_run(
    run_index: int, correction_n: int,
    train_yaml: Path, heldout_yaml: Path,
    out_dir: Path,
) -> Path:
    """Sample correction layouts for repetition ``run_index`` and write
    them to ``out_dir/correction_layouts.yaml``.

    The blocked-signature set is union(train, heldout) so the
    contamination check downstream is purely a safety net — the
    sampler already refuses to emit a conflicting signature.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "correction_layouts.yaml"
    if out_path.exists():
        # Re-verify the cached pool to guard against an old contaminated file.
        report = assert_no_contamination(out_path, train_yaml, heldout_yaml)
        _save_report(out_dir, run_index, report)
        return out_path

    blocked = _load_blocked_signatures([str(train_yaml), str(heldout_yaml)])
    layouts = sample_layouts(
        n=correction_n, grid_size=5, num_fires=3, min_manhattan=4,
        seed=CORRECTION_SEED_BASE + int(run_index),
        blocked_signatures=blocked,
    )
    # Rename so they're clearly per-run + per-index correction layouts.
    for i, L in enumerate(layouts):
        L["name"] = f"corr_r{run_index:02d}_{i+1:03d}"
    write_yaml(layouts, out_path, grid_size=5)

    report = assert_no_contamination(out_path, train_yaml, heldout_yaml)
    _save_report(out_dir, run_index, report)
    return out_path


def _save_report(out_dir: Path, run_index: int, report: Dict[str, int]):
    out = {
        "run_index": int(run_index),
        "correction_seed": CORRECTION_SEED_BASE + int(run_index),
        "global_train_seed":   GLOBAL_TRAIN_SEED,
        "global_heldout_seed": GLOBAL_HELDOUT_SEED,
        **report,
    }
    with open(out_dir / "layout_setup_report.json", "w") as f:
        json.dump(out, f, indent=2, default=str)


def run_layout_dir(run_index: int) -> Path:
    return RUNS_DIR / f"run_{int(run_index):02d}"


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--run_index", type=int, required=True)
    p.add_argument("--initial_demos", type=int, default=20)
    p.add_argument("--heldout_n", type=int, default=200)
    p.add_argument("--correction_n", type=int, default=40)
    args = p.parse_args()

    shared = ensure_shared_layouts(args.initial_demos, args.heldout_n)
    out_dir = run_layout_dir(args.run_index) / "shared"
    corr = ensure_correction_layouts_for_run(
        args.run_index, args.correction_n, shared["train"], shared["heldout"], out_dir,
    )
    print(f"[layout-setup] OK run={args.run_index}")
    print(f"  train   -> {shared['train']}")
    print(f"  heldout -> {shared['heldout']}")
    print(f"  corr    -> {corr}")
