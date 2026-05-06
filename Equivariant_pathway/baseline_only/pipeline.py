"""Minimal baseline-DAgger-only equivariant policy pipeline.

End-to-end flow (single invocation = full cycle)
================================================
1. Generate 20 unique random training layouts and 50 unique random
   heldout layouts. The heldout layouts are sampled FIRST and the
   training layouts are sampled with the heldout signatures as a
   blocked set, so the two sets are guaranteed disjoint by signature
   (start_pos, goal_pos, sorted fire_positions). The training layouts
   YAML drives initial demo collection; the heldout layouts YAML drives
   every rollout-and-correct pass.

2. Collect 20 BFS-expert demonstrations on the training layouts (one
   demo per layout), saved under <baseline_only>/demos/.

3. Train the equivariant policy on those 20 demos for 500 epochs.
   Best/last checkpoints land in <baseline_only>/checkpoints/.

4. Until heldout success rate reaches 0.90:
     a. Run a baseline DAgger correction pass over the 50 heldout
        layouts (n_episodes = 50, no round-robin since the layout pool
        already has 50). When the policy fails an episode, the BFS
        expert resumes from the deviation step and finishes the
        trajectory; the corrective demo is written into the demos/
        folder. The pass triggers a retrain every 10 corrective demos
        accumulated GLOBALLY (across passes). If a pass ends with at
        least one correction but fewer than 10 fresh ones (i.e. no
        retrain fired this pass), we force a single retrain on the
        accumulated demo set so progress is never silently dropped.
     b. Re-evaluate on the 50 heldout layouts (no corrections) and
        record (round, cumulative_demos, success_rate).

5. Persist a learning_curve.json + a chart of success_rate vs
   cumulative_demos under <baseline_only>/results/.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PATHWAY_ROOT = REPO_ROOT / "Equivariant_pathway"
ROOT = PATHWAY_ROOT / "baseline_only"
DEMO_DIR = ROOT / "demos"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
TRAIN_YAML = ROOT / "training_layouts.yaml"
HELDOUT_YAML = ROOT / "heldout_layouts.yaml"

TARGET_SR = 0.90
DEFAULT_TRAIN_EVERY_N = 10
DEFAULT_INITIAL_DEMOS = 20
DEFAULT_HELDOUT_N = 50
DEFAULT_INITIAL_EPOCHS = 500
DEFAULT_ROUND_EPOCHS = 100


def _info(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [bo] {msg}", flush=True)


def _section(title: str, char: str = "=") -> None:
    print(f"\n{char * 80}\n{title}\n{char * 80}", flush=True)


def _run(cmd: List[str]) -> int:
    _info("$ " + " ".join(cmd))
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def _count_demos() -> int:
    if not DEMO_DIR.exists():
        return 0
    return sum(1 for _ in DEMO_DIR.rglob("*.json"))


def _ensure_layouts(seed: int, n_train: int, n_heldout: int) -> None:
    """Sample heldout layouts FIRST, then sample training layouts with the
    heldout signatures blocked. Guarantees signature-disjointness between
    training and heldout."""
    from Equivariant_pathway.layout_sampler import (
        sample_layouts, write_yaml, _load_blocked_signatures,
    )

    if not HELDOUT_YAML.exists():
        h_layouts = sample_layouts(
            n=n_heldout, grid_size=5, num_fires=3,
            min_manhattan=4, seed=seed + 7919, blocked_signatures=set(),
        )
        write_yaml(h_layouts, HELDOUT_YAML, grid_size=5)
        _info(f"wrote {len(h_layouts)} heldout layouts -> {HELDOUT_YAML}")
    else:
        _info(f"using existing {HELDOUT_YAML}")

    if not TRAIN_YAML.exists():
        blocked = _load_blocked_signatures([str(HELDOUT_YAML)])
        t_layouts = sample_layouts(
            n=n_train, grid_size=5, num_fires=3,
            min_manhattan=4, seed=seed, blocked_signatures=blocked,
        )
        for L in t_layouts:
            L["n_repetitions"] = 1
        payload = {
            "img_size": 80, "grid_size": 5, "cell_px": 16,
            "num_fires": 3, "min_manhattan": 4, "n_repetitions": 1,
            "training_layouts": t_layouts,
        }
        with open(TRAIN_YAML, "w") as f:
            yaml.safe_dump(payload, f, sort_keys=False)
        _info(f"wrote {len(t_layouts)} training layouts -> {TRAIN_YAML} "
              f"(disjoint from heldout by signature)")
    else:
        _info(f"using existing {TRAIN_YAML}")


def _initial_collect(seed: int, n_demos: int) -> None:
    rc = _run([
        sys.executable, "-u", "-m", "Equivariant_pathway.collect_demos",
        "--layouts", str(TRAIN_YAML),
        "--demo_dir", str(DEMO_DIR),
        "--num_demos", str(n_demos),
        "--seed", str(seed),
    ])
    if rc != 0:
        raise RuntimeError(f"initial demo collection failed (rc={rc})")


def _initial_train(seed: int, epochs: int, max_demos: int) -> None:
    rc = _run([
        sys.executable, "-u", "-m", "Equivariant_pathway.train",
        "--demo_dir", str(DEMO_DIR),
        "--checkpoint_dir", str(CKPT_DIR),
        "--epochs", str(epochs),
        "--seed", str(seed),
        "--max_demos", str(max_demos),
    ])
    if rc != 0:
        raise RuntimeError(f"initial training failed (rc={rc})")


def _force_retrain(seed: int, epochs: int) -> None:
    rc = _run([
        sys.executable, "-u", "-m", "Equivariant_pathway.train",
        "--demo_dir", str(DEMO_DIR),
        "--checkpoint_dir", str(CKPT_DIR),
        "--epochs", str(epochs),
        "--seed", str(seed),
        "--resume",
    ])
    if rc != 0:
        raise RuntimeError(f"forced retrain failed (rc={rc})")


def _eval_heldout(seed: int, tag: str) -> Dict:
    """Pure rollout (no corrections) on the 50 heldout layouts. Returns
    {n_episodes, n_successes, success_rate, out_dir}."""
    out_dir = RESULTS_DIR / f"eval_{tag}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rc = _run([
        sys.executable, "-u", "-m", "Equivariant_pathway.rollout_test",
        "--checkpoint", str(CKPT_DIR / "best_eq_policy.pth"),
        "--layouts", str(HELDOUT_YAML),
        "--out_dir", str(out_dir),
        "--seed", str(seed),
    ])
    if rc != 0:
        raise RuntimeError(f"heldout eval failed (rc={rc})")
    full = json.load(open(out_dir / "full_output.json"))
    md = full.get("metadata", {}) or {}
    n = int(md.get("n_episodes", 0) or 0)
    ns = int(md.get("n_successes", 0) or 0)
    return {
        "n_episodes": n,
        "n_successes": ns,
        "success_rate": ns / n if n else 0.0,
        "out_dir": str(out_dir),
    }


def main():
    p = argparse.ArgumentParser(description="Baseline-DAgger-only equivariant pipeline.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--initial_demos", type=int, default=DEFAULT_INITIAL_DEMOS)
    p.add_argument("--heldout_n", type=int, default=DEFAULT_HELDOUT_N)
    p.add_argument("--initial_epochs", type=int, default=DEFAULT_INITIAL_EPOCHS)
    p.add_argument("--round_epochs", type=int, default=DEFAULT_ROUND_EPOCHS)
    p.add_argument("--train_every_n", type=int, default=DEFAULT_TRAIN_EVERY_N,
                   help="Retrain after every N corrective demos collected GLOBALLY. "
                        "If a rollout pass ends with corrections >0 but no train fired, "
                        "one retrain is forced on the accumulated demo set.")
    p.add_argument("--max_rounds", type=int, default=50,
                   help="Hard guard against infinite loops; the loop normally exits "
                        "when heldout success rate >= 0.90.")
    p.add_argument("--max_steps", type=int, default=60,
                   help="Per-episode step cap during rollout.")
    p.add_argument("--force_restart", action="store_true",
                   help="Wipe demos/, checkpoints/, results/, and the layout YAMLs "
                        "before running so every submission starts from scratch.")
    args = p.parse_args()

    _section("BASELINE-DAGGER-ONLY EQUIVARIANT PIPELINE")
    _info(f"baseline_only root: {ROOT}")
    _info(f"seed={args.seed}  initial_demos={args.initial_demos}  "
          f"heldout_n={args.heldout_n}  initial_epochs={args.initial_epochs}  "
          f"round_epochs={args.round_epochs}  train_every_n={args.train_every_n}")

    if args.force_restart:
        _info("--force_restart: wiping demos/, checkpoints/, results/, layout YAMLs.")
        for d in (DEMO_DIR, CKPT_DIR, RESULTS_DIR):
            if d.exists():
                shutil.rmtree(d)
        for f in (TRAIN_YAML, HELDOUT_YAML):
            if f.exists():
                f.unlink()
    for d in (DEMO_DIR, CKPT_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    _section("PHASE 1: LAYOUTS", char="-")
    _ensure_layouts(args.seed, args.initial_demos, args.heldout_n)

    _section("PHASE 2: INITIAL DEMO COLLECTION", char="-")
    if _count_demos() < args.initial_demos:
        _initial_collect(args.seed, args.initial_demos)
    else:
        _info(f"demos/ already has {_count_demos()} demos; skipping initial collection")
    _info(f"final initial demo count: {_count_demos()}")

    _section("PHASE 3: INITIAL TRAINING", char="-")
    if not (CKPT_DIR / "best_eq_policy.pth").exists():
        _initial_train(args.seed, args.initial_epochs, args.initial_demos)
    else:
        _info(f"best_eq_policy.pth already exists at {CKPT_DIR}; skipping initial training")

    _section("PHASE 4: HELDOUT EVAL ON INITIAL MODEL", char="-")
    init_metrics = _eval_heldout(args.seed, tag="round_0")
    n_demos = _count_demos()
    _info(f"INITIAL: cum_demos={n_demos}  heldout_sr={init_metrics['success_rate']:.3f}")
    history: List[Dict] = [{
        "round": 0,
        "cum_demos": n_demos,
        "heldout_sr": init_metrics["success_rate"],
        "heldout_n_successes": init_metrics["n_successes"],
        "heldout_n_episodes": init_metrics["n_episodes"],
        "n_corrections_in_pass": 0,
        "n_train_calls_in_pass": 0,
    }]
    _persist_curve(history)

    if init_metrics["success_rate"] >= TARGET_SR:
        _info(f"target {TARGET_SR} already reached at round 0; done.")
        _make_chart()
        return

    _section("PHASE 5: BASELINE DAGGER LOOP", char="-")
    from Equivariant_pathway.baseline_dagger import run_dagger_correction_pass

    for rnd in range(1, args.max_rounds + 1):
        _section(f"ROUND {rnd}", char="-")
        pass_dir = RESULTS_DIR / f"round_{rnd:03d}_pass"
        summary = run_dagger_correction_pass(
            layouts_yaml=HELDOUT_YAML,
            checkpoint=CKPT_DIR / "best_eq_policy.pth",
            demo_dir=DEMO_DIR,
            out_dir=pass_dir,
            seed=args.seed + rnd * 1000,
            max_steps=args.max_steps,
            train_every_n=args.train_every_n,
            train_demo_paths=None,
            epochs=args.round_epochs,
            n_episodes=args.heldout_n,
        )
        n_corr = int(summary.get("n_corrected_in_pass", 0) or 0)
        n_train = int(summary.get("n_train_calls", 0) or 0)
        _info(f"pass: corrections={n_corr}  train_calls={n_train}  "
              f"pass_success_rate={summary.get('success_rate', 0.0):.3f}")

        # Force one retrain if the pass produced corrections but didn't
        # cross a multiple of train_every_n. Without this, fewer than 10
        # corrections in a pass would silently bypass training.
        if n_corr > 0 and n_train == 0:
            _info(f"pass collected {n_corr} corrections (< {args.train_every_n}); "
                  f"forcing one retrain on the accumulated demo set.")
            _force_retrain(args.seed, args.round_epochs)

        post_metrics = _eval_heldout(args.seed + rnd, tag=f"round_{rnd:03d}")
        n_demos = _count_demos()
        history.append({
            "round": rnd,
            "cum_demos": n_demos,
            "heldout_sr": post_metrics["success_rate"],
            "heldout_n_successes": post_metrics["n_successes"],
            "heldout_n_episodes": post_metrics["n_episodes"],
            "n_corrections_in_pass": n_corr,
            "n_train_calls_in_pass": n_train,
        })
        _persist_curve(history)
        _info(f"round {rnd}: cum_demos={n_demos}  "
              f"heldout_sr={post_metrics['success_rate']:.3f}")

        if post_metrics["success_rate"] >= TARGET_SR:
            _info(f"target {TARGET_SR} reached at round {rnd}; stopping.")
            break
        if n_corr == 0:
            _info(f"no corrections collected this pass; loop has stalled, stopping.")
            break

    _section("PHASE 6: CHART", char="-")
    _make_chart()
    _section("DONE")


def _persist_curve(history: List[Dict]) -> None:
    out = {
        "target_sr": TARGET_SR,
        "demo_dir": str(DEMO_DIR),
        "checkpoint_dir": str(CKPT_DIR),
        "heldout_yaml": str(HELDOUT_YAML),
        "training_yaml": str(TRAIN_YAML),
        "history": history,
    }
    with open(RESULTS_DIR / "learning_curve.json", "w") as f:
        json.dump(out, f, indent=2, default=str)


def _make_chart() -> None:
    rc = _run([sys.executable, "-u", "-m", "Equivariant_pathway.baseline_only.chart"])
    if rc != 0:
        _info(f"chart generation exited rc={rc}")


if __name__ == "__main__":
    main()
