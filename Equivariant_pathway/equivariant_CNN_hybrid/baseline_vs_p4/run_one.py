"""Drive a single repetition of the baseline-vs-P4 budget comparison.

Phases per repetition:
  1. Layout setup: shared training + heldout (idempotent); per-run
     correction pool sampled with contamination guard.
  2. Bootstrap: collect the initial 20 BFS demos and train the initial
     EquivariantCNNHybridPolicy. Both are CACHED on disk so concurrent
     repetitions all share the same initial checkpoint without redoing
     the work — but they each run in their own ``runs/run_NN/shared/``
     so the cache lookup never races.
  3. Two budget loops run sequentially against an isolated copy of the
     initial state:
       - baseline_only (top-K by loss with tiebreakers)
       - p4_only       (LLM-prescribed with dynamic budget addendum)
     Both write a standard ``learning_curve.json`` so the aggregate
     chart can read them with a single loader.

CLI:
  python -m Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.run_one \
      --run_index 0 --budget 15 --target_sr 0.90
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4 import layout_setup
from Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.baseline_budget import (
    run_baseline_budget,
)
from Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.p4_budget import (
    run_p4_budget,
)

CONFIG_PATH = Path(__file__).resolve().parent / "config.yml"


def _info(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [run-one] {msg}", flush=True)


def _section(title: str, char: str = "="):
    print(f"\n{char*80}\n{title}\n{char*80}", flush=True)


def _load_config() -> Dict:
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f) or {}


def _to_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in ("1", "true", "yes", "on")


GLOBAL_SHARED_DEMO_DIR = layout_setup.SHARED_DIR / "init_demos"
GLOBAL_SHARED_CKPT_DIR = layout_setup.SHARED_DIR / "init_checkpoints"


def _bootstrap_initial(seed: int, initial_epochs: int, initial_demos: int,
                       train_yaml: Path, shared_demo_dir: Path,
                       shared_ckpt_dir: Path,
                       prefer_global: bool = True) -> None:
    """Collect the initial 20 BFS demos (if missing) and train the
    initial policy (if missing).

    The initial-20 BFS demos + the initial policy depend only on the
    constant training_layouts.yaml + seed, so the byte-output is
    identical across runs. We therefore cache them under
    ``baseline_vs_p4/shared/init_*/`` and copy into each run's
    per-run ``shared/`` dir. ``swatch.sh`` pre-seeds the global cache
    before launching the parallel children so there is no write race.
    If the global cache is missing, this function falls back to
    materialising it itself (safe for serial / single-run usage).
    """
    shared_demo_dir.mkdir(parents=True, exist_ok=True)
    shared_ckpt_dir.mkdir(parents=True, exist_ok=True)

    if prefer_global:
        GLOBAL_SHARED_DEMO_DIR.mkdir(parents=True, exist_ok=True)
        GLOBAL_SHARED_CKPT_DIR.mkdir(parents=True, exist_ok=True)
        n_global = sum(1 for _ in GLOBAL_SHARED_DEMO_DIR.glob("*.json"))
        best_global = GLOBAL_SHARED_CKPT_DIR / "best_hybrid_policy.pth"
        if n_global < initial_demos or not best_global.exists():
            _populate_global_bootstrap(
                seed=seed, initial_epochs=initial_epochs,
                initial_demos=initial_demos, train_yaml=train_yaml,
            )
        # Mirror the global cache into the per-run shared dir.
        for src in sorted(GLOBAL_SHARED_DEMO_DIR.glob("*.json")):
            dst = shared_demo_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
        for name in ("best_hybrid_policy.pth", "last_hybrid_policy.pth"):
            src = GLOBAL_SHARED_CKPT_DIR / name
            if src.exists():
                shutil.copy2(src, shared_ckpt_dir / name)
        return

    n_now = sum(1 for _ in shared_demo_dir.glob("*.json"))
    if n_now < initial_demos:
        cmd = [
            sys.executable, "-u", "-m", "Equivariant_pathway.collect_demos",
            "--layouts", str(train_yaml),
            "--demo_dir", str(shared_demo_dir),
            "--num_demos", str(initial_demos),
            "--seed", str(seed),
        ]
        _info("$ " + " ".join(cmd))
        rc = subprocess.call(cmd, cwd=str(REPO_ROOT))
        if rc != 0:
            raise RuntimeError(f"initial BFS collection failed rc={rc}")

    best = shared_ckpt_dir / "best_hybrid_policy.pth"
    if not best.exists():
        cmd = [
            sys.executable, "-u", "-m",
            "Equivariant_pathway.equivariant_CNN_hybrid.train",
            "--demo_dir", str(shared_demo_dir),
            "--checkpoint_dir", str(shared_ckpt_dir),
            "--epochs", str(initial_epochs),
            "--seed", str(seed),
            "--max_demos", str(initial_demos),
        ]
        _info("$ " + " ".join(cmd))
        rc = subprocess.call(cmd, cwd=str(REPO_ROOT))
        if rc != 0:
            raise RuntimeError(f"initial training failed rc={rc}")


def _populate_global_bootstrap(seed: int, initial_epochs: int,
                               initial_demos: int, train_yaml: Path) -> None:
    """Materialise the GLOBAL initial-20 demos + initial checkpoint in
    ``baseline_vs_p4/shared/init_*/``. Safe to call repeatedly; both
    steps are skipped when their artefact is already on disk.
    """
    GLOBAL_SHARED_DEMO_DIR.mkdir(parents=True, exist_ok=True)
    GLOBAL_SHARED_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    n_now = sum(1 for _ in GLOBAL_SHARED_DEMO_DIR.glob("*.json"))
    if n_now < initial_demos:
        cmd = [
            sys.executable, "-u", "-m", "Equivariant_pathway.collect_demos",
            "--layouts", str(train_yaml),
            "--demo_dir", str(GLOBAL_SHARED_DEMO_DIR),
            "--num_demos", str(initial_demos),
            "--seed", str(seed),
        ]
        _info("$ " + " ".join(cmd))
        rc = subprocess.call(cmd, cwd=str(REPO_ROOT))
        if rc != 0:
            raise RuntimeError(f"global initial BFS collection failed rc={rc}")
    if not (GLOBAL_SHARED_CKPT_DIR / "best_hybrid_policy.pth").exists():
        cmd = [
            sys.executable, "-u", "-m",
            "Equivariant_pathway.equivariant_CNN_hybrid.train",
            "--demo_dir", str(GLOBAL_SHARED_DEMO_DIR),
            "--checkpoint_dir", str(GLOBAL_SHARED_CKPT_DIR),
            "--epochs", str(initial_epochs),
            "--seed", str(seed),
            "--max_demos", str(initial_demos),
        ]
        _info("$ " + " ".join(cmd))
        rc = subprocess.call(cmd, cwd=str(REPO_ROOT))
        if rc != 0:
            raise RuntimeError(f"global initial training failed rc={rc}")


def main():
    cfg = _load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--run_index", type=int, required=True)
    p.add_argument("--budget", type=int, default=int(cfg.get("budget", 15)))
    p.add_argument("--target_sr", type=float, default=float(cfg.get("target_sr", 0.90)))
    p.add_argument("--initial_demos", type=int, default=int(cfg.get("initial_demos", 20)))
    p.add_argument("--heldout_n", type=int, default=int(cfg.get("heldout_n", 200)))
    p.add_argument("--correction_n", type=int, default=int(cfg.get("correction_n", 40)))
    p.add_argument("--initial_epochs", type=int, default=int(cfg.get("initial_epochs", 500)))
    p.add_argument("--round_epochs", type=int, default=int(cfg.get("round_epochs", 500)))
    p.add_argument("--baseline_round_epochs", type=int,
                   default=int(cfg.get("baseline_round_epochs", 100)))
    p.add_argument("--max_rounds", type=int, default=int(cfg.get("max_rounds", 50)))
    p.add_argument("--max_steps", type=int, default=int(cfg.get("max_steps", 60)))
    p.add_argument("--seed", type=int, default=int(cfg.get("seed", 0)))
    p.add_argument("--train_from_scratch", type=_to_bool,
                   default=_to_bool(cfg.get("train_from_scratch", True)))
    p.add_argument("--methods", type=str, default="baseline,p4",
                   help="Comma-separated subset of {baseline,p4} to run.")
    p.add_argument("--force_restart", action="store_true",
                   help="Delete this run's baseline_only/ and p4_only/ outputs "
                        "before starting. Does NOT delete the shared bootstrap.")
    p.add_argument("--bootstrap_only", action="store_true",
                   help="Pre-create global shared layouts and the global initial "
                        "demos + initial checkpoint, then exit. Used by swatch.sh "
                        "to eliminate write races between parallel children.")
    args = p.parse_args()

    if args.bootstrap_only:
        shared = layout_setup.ensure_shared_layouts(
            initial_demos=args.initial_demos, heldout_n=args.heldout_n,
        )
        _populate_global_bootstrap(
            seed=args.seed, initial_epochs=args.initial_epochs,
            initial_demos=args.initial_demos, train_yaml=shared["train"],
        )
        _info(f"bootstrap_only done: {GLOBAL_SHARED_DEMO_DIR} / {GLOBAL_SHARED_CKPT_DIR}")
        return

    methods = {m.strip() for m in args.methods.split(",") if m.strip()}
    run_dir = layout_setup.run_layout_dir(args.run_index)
    run_dir.mkdir(parents=True, exist_ok=True)
    shared_run_dir = run_dir / "shared"
    shared_run_dir.mkdir(parents=True, exist_ok=True)

    _section(f"RUN {args.run_index}  (budget={args.budget}, target={args.target_sr:.2f})")

    # --- Phase 1: layouts -------------------------------------------------
    _section("PHASE 1: LAYOUTS", char="-")
    shared = layout_setup.ensure_shared_layouts(
        initial_demos=args.initial_demos, heldout_n=args.heldout_n,
    )
    correction_yaml = layout_setup.ensure_correction_layouts_for_run(
        args.run_index, args.correction_n,
        train_yaml=shared["train"], heldout_yaml=shared["heldout"],
        out_dir=shared_run_dir,
    )
    _info(f"train   -> {shared['train']}")
    _info(f"heldout -> {shared['heldout']}")
    _info(f"corr    -> {correction_yaml}")

    # --- Phase 2: bootstrap ------------------------------------------------
    _section("PHASE 2: BOOTSTRAP (initial demos + initial model)", char="-")
    shared_demo_dir = shared_run_dir / "init_demos"
    shared_ckpt_dir = shared_run_dir / "init_checkpoints"
    _bootstrap_initial(
        seed=args.seed,
        initial_epochs=args.initial_epochs,
        initial_demos=args.initial_demos,
        train_yaml=shared["train"],
        shared_demo_dir=shared_demo_dir,
        shared_ckpt_dir=shared_ckpt_dir,
    )

    bo_root = run_dir / "baseline_only"
    p4_root = run_dir / "p4_only"
    if args.force_restart:
        for d in (bo_root, p4_root):
            if d.exists():
                shutil.rmtree(d)
    bo_root.mkdir(parents=True, exist_ok=True)
    p4_root.mkdir(parents=True, exist_ok=True)

    summary = {"run_index": args.run_index, "budget": args.budget,
               "target_sr": args.target_sr,
               "correction_yaml": str(correction_yaml),
               "heldout_yaml": str(shared["heldout"]),
               "training_yaml": str(shared["train"])}

    # --- Phase 3a: baseline -----------------------------------------------
    if "baseline" in methods:
        _section("PHASE 3a: BASELINE-BUDGET", char="-")
        result_bo = run_baseline_budget(
            run_index=args.run_index,
            bo_root=bo_root,
            shared_demo_dir=shared_demo_dir,
            shared_ckpt_dir=shared_ckpt_dir,
            correction_yaml=correction_yaml,
            heldout_yaml=shared["heldout"],
            budget=args.budget,
            target_sr=args.target_sr,
            round_epochs=args.baseline_round_epochs,
            max_rounds=args.max_rounds,
            max_steps=args.max_steps,
            seed=args.seed,
            train_from_scratch=args.train_from_scratch,
        )
        summary["baseline_stopped_reason"] = result_bo.get("stopped_reason")

    # --- Phase 3b: p4 ------------------------------------------------------
    if "p4" in methods:
        _section("PHASE 3b: P4-BUDGET", char="-")
        result_p4 = run_p4_budget(
            run_index=args.run_index,
            p4_root=p4_root,
            shared_demo_dir=shared_demo_dir,
            shared_ckpt_dir=shared_ckpt_dir,
            correction_yaml=correction_yaml,
            heldout_yaml=shared["heldout"],
            budget=args.budget,
            target_sr=args.target_sr,
            round_epochs=args.round_epochs,
            max_rounds=args.max_rounds,
            max_steps=args.max_steps,
            seed=args.seed,
            train_from_scratch=args.train_from_scratch,
        )
        summary["p4_stopped_reason"] = result_p4.get("stopped_reason")

    import json
    with open(run_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    _section(f"RUN {args.run_index} DONE")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
