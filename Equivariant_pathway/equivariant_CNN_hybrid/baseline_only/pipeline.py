"""Hybrid baseline_only pipeline (EquivariantCNNHybridPolicy + DAgger).

Mirrors Equivariant_pathway/baseline_only/pipeline.py but with all
subprocess invocations swapped to the hybrid modules and the
checkpoint name set to best_hybrid_policy.pth. Re-uses
Equivariant_pathway.collect_demos and .layout_sampler since both are
policy-agnostic and the hybrid lives INSIDE Equivariant_pathway/ so
the no-cross-dep rule does not apply here.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HYBRID_ROOT = REPO_ROOT / "Equivariant_pathway" / "equivariant_CNN_hybrid"

_ROOT_OVERRIDE = os.environ.get("BASELINE_ONLY_ROOT")
ROOT = Path(_ROOT_OVERRIDE).resolve() if _ROOT_OVERRIDE else HYBRID_ROOT / "baseline_only"
DEMO_DIR = ROOT / "demos"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
TRAIN_YAML = ROOT / "training_layouts.yaml"
HELDOUT_YAML = ROOT / "heldout_layouts.yaml"
CORRECTION_YAML = ROOT / "correction_layouts.yaml"

TARGET_SR = 0.90
DEFAULTS = dict(initial_demos=20, heldout_n=50, correction_n=50,
                initial_epochs=500, round_epochs=100, train_every_n=10)


def _info(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [bo-hybrid] {msg}", flush=True)


def _section(title, char="="):
    print(f"\n{char*80}\n{title}\n{char*80}", flush=True)


def _run(cmd):
    _info("$ " + " ".join(cmd))
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def _count_demos():
    return sum(1 for _ in DEMO_DIR.rglob("*.json")) if DEMO_DIR.exists() else 0


def _ensure_layouts(seed, n_train, n_heldout, n_correction):
    from Equivariant_pathway.layout_sampler import (
        sample_layouts, write_yaml, _load_blocked_signatures,
    )
    if not HELDOUT_YAML.exists():
        h = sample_layouts(n=n_heldout, grid_size=5, num_fires=3,
                           min_manhattan=4, seed=seed + 7919, blocked_signatures=set())
        write_yaml(h, HELDOUT_YAML, grid_size=5)
    if not TRAIN_YAML.exists():
        blocked = _load_blocked_signatures([str(HELDOUT_YAML)])
        t = sample_layouts(n=n_train, grid_size=5, num_fires=3,
                           min_manhattan=4, seed=seed, blocked_signatures=blocked)
        for L in t:
            L["n_repetitions"] = 1
        payload = {"img_size": 80, "grid_size": 5, "cell_px": 16,
                   "num_fires": 3, "min_manhattan": 4, "n_repetitions": 1,
                   "training_layouts": t}
        with open(TRAIN_YAML, "w") as f:
            yaml.safe_dump(payload, f, sort_keys=False)
    if not CORRECTION_YAML.exists():
        blocked = _load_blocked_signatures([str(HELDOUT_YAML), str(TRAIN_YAML)])
        c = sample_layouts(n=n_correction, grid_size=5, num_fires=3,
                           min_manhattan=4, seed=seed + 31337, blocked_signatures=blocked)
        write_yaml(c, CORRECTION_YAML, grid_size=5)


def _initial_collect(seed, n_demos):
    rc = _run([
        sys.executable, "-u", "-m", "Equivariant_pathway.collect_demos",
        "--layouts", str(TRAIN_YAML),
        "--demo_dir", str(DEMO_DIR),
        "--num_demos", str(n_demos),
        "--seed", str(seed),
    ])
    if rc != 0:
        raise RuntimeError(f"initial demo collection failed (rc={rc})")


def _initial_train(seed, epochs, max_demos):
    rc = _run([
        sys.executable, "-u", "-m", "Equivariant_pathway.equivariant_CNN_hybrid.train",
        "--demo_dir", str(DEMO_DIR),
        "--checkpoint_dir", str(CKPT_DIR),
        "--epochs", str(epochs),
        "--seed", str(seed),
        "--max_demos", str(max_demos),
    ])
    if rc != 0:
        raise RuntimeError(f"initial training failed (rc={rc})")


def _force_retrain(seed, epochs, train_from_scratch):
    cmd = [
        sys.executable, "-u", "-m", "Equivariant_pathway.equivariant_CNN_hybrid.train",
        "--demo_dir", str(DEMO_DIR),
        "--checkpoint_dir", str(CKPT_DIR),
        "--epochs", str(epochs),
        "--seed", str(seed),
    ]
    if not train_from_scratch:
        cmd.append("--resume")
    rc = _run(cmd)
    if rc != 0:
        raise RuntimeError(f"forced retrain failed (rc={rc})")


def _eval_heldout(seed, tag):
    out_dir = RESULTS_DIR / f"eval_{tag}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rc = _run([
        sys.executable, "-u", "-m", "Equivariant_pathway.equivariant_CNN_hybrid.rollout_test",
        "--checkpoint", str(CKPT_DIR / "best_hybrid_policy.pth"),
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
    return {"n_episodes": n, "n_successes": ns,
            "success_rate": ns / n if n else 0.0, "out_dir": str(out_dir)}


CONFIG_PATH = Path(__file__).resolve().parent / "config.yml"


def _load_config():
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f) or {}


def _to_bool(x):
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in ("1", "true", "yes", "on")


def main():
    global TARGET_SR
    cfg = _load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=int(cfg.get("seed", 0)))
    p.add_argument("--initial_demos", type=int, default=int(cfg.get("initial_demos", DEFAULTS["initial_demos"])))
    p.add_argument("--heldout_n", type=int, default=int(cfg.get("heldout_n", DEFAULTS["heldout_n"])))
    p.add_argument("--correction_n", type=int, default=int(cfg.get("correction_n", DEFAULTS["correction_n"])))
    p.add_argument("--initial_epochs", type=int, default=int(cfg.get("initial_epochs", DEFAULTS["initial_epochs"])))
    p.add_argument("--round_epochs", type=int, default=int(cfg.get("round_epochs", DEFAULTS["round_epochs"])))
    p.add_argument("--train_every_n", type=int, default=int(cfg.get("train_every_n", DEFAULTS["train_every_n"])))
    p.add_argument("--max_rounds", type=int, default=int(cfg.get("max_rounds", 50)))
    p.add_argument("--max_steps", type=int, default=int(cfg.get("max_steps", 60)))
    p.add_argument("--target_sr", type=float, default=float(cfg.get("target_sr", TARGET_SR)))
    p.add_argument("--train_from_scratch", type=_to_bool, default=_to_bool(cfg.get("train_from_scratch", True)))
    p.add_argument("--force_restart", action="store_true")
    args = p.parse_args()
    TARGET_SR = args.target_sr

    _section("HYBRID BASELINE-DAGGER PIPELINE")
    _info(f"baseline_only root: {ROOT}")

    if args.force_restart:
        for d in (DEMO_DIR, CKPT_DIR, RESULTS_DIR):
            if d.exists():
                shutil.rmtree(d)
        for f in (TRAIN_YAML, HELDOUT_YAML, CORRECTION_YAML):
            if f.exists():
                f.unlink()
    for d in (DEMO_DIR, CKPT_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    _section("PHASE 1: LAYOUTS", char="-")
    _ensure_layouts(args.seed, args.initial_demos, args.heldout_n, args.correction_n)

    _section("PHASE 2: INITIAL DEMO COLLECTION", char="-")
    if _count_demos() < args.initial_demos:
        _initial_collect(args.seed, args.initial_demos)
    _info(f"final initial demo count: {_count_demos()}")

    _section("PHASE 3: INITIAL TRAINING", char="-")
    if not (CKPT_DIR / "best_hybrid_policy.pth").exists():
        _initial_train(args.seed, args.initial_epochs, args.initial_demos)

    initial_best = CKPT_DIR / "initial_best_hybrid_policy.pth"
    initial_last = CKPT_DIR / "initial_last_hybrid_policy.pth"
    if not initial_best.exists() and (CKPT_DIR / "best_hybrid_policy.pth").exists():
        shutil.copy2(CKPT_DIR / "best_hybrid_policy.pth", initial_best)
    if not initial_last.exists() and (CKPT_DIR / "last_hybrid_policy.pth").exists():
        shutil.copy2(CKPT_DIR / "last_hybrid_policy.pth", initial_last)

    _section("PHASE 4: HELDOUT EVAL ON INITIAL MODEL", char="-")
    init_metrics = _eval_heldout(args.seed, tag="round_0")
    n_demos = _count_demos()
    history: List[Dict] = [{
        "round": 0, "cum_demos": n_demos,
        "heldout_sr": init_metrics["success_rate"],
        "heldout_n_successes": init_metrics["n_successes"],
        "heldout_n_episodes": init_metrics["n_episodes"],
        "n_corrections_in_pass": 0, "n_train_calls_in_pass": 0,
    }]
    _persist_curve(history)
    if init_metrics["success_rate"] >= TARGET_SR:
        _make_chart(); return

    _section("PHASE 5: BASELINE DAGGER LOOP", char="-")
    from Equivariant_pathway.equivariant_CNN_hybrid.baseline_dagger import run_dagger_correction_pass

    for rnd in range(1, args.max_rounds + 1):
        _section(f"ROUND {rnd}", char="-")
        pass_dir = RESULTS_DIR / f"round_{rnd:03d}_pass"
        summary = run_dagger_correction_pass(
            layouts_yaml=CORRECTION_YAML,
            checkpoint=CKPT_DIR / "best_hybrid_policy.pth",
            demo_dir=DEMO_DIR, out_dir=pass_dir,
            seed=args.seed + rnd * 1000,
            max_steps=args.max_steps,
            train_every_n=args.train_every_n,
            train_demo_paths=None,
            epochs=args.round_epochs,
            n_episodes=args.correction_n,
            train_from_scratch=args.train_from_scratch,
        )
        n_corr = int(summary.get("n_corrected_in_pass", 0) or 0)
        n_train = int(summary.get("n_train_calls", 0) or 0)
        if n_corr > 0 and n_train == 0:
            _force_retrain(args.seed, args.round_epochs, args.train_from_scratch)
        post_metrics = _eval_heldout(args.seed + rnd, tag=f"round_{rnd:03d}")
        n_demos = _count_demos()
        history.append({
            "round": rnd, "cum_demos": n_demos,
            "heldout_sr": post_metrics["success_rate"],
            "heldout_n_successes": post_metrics["n_successes"],
            "heldout_n_episodes": post_metrics["n_episodes"],
            "n_corrections_in_pass": n_corr,
            "n_train_calls_in_pass": n_train,
        })
        _persist_curve(history)
        _info(f"round {rnd}: cum_demos={n_demos} sr={post_metrics['success_rate']:.3f}")
        if post_metrics["success_rate"] >= TARGET_SR:
            break
        if n_corr == 0:
            break

    _section("PHASE 6: CHART", char="-")
    _make_chart()
    _section("DONE")


def _persist_curve(history):
    out = {
        "target_sr": TARGET_SR,
        "demo_dir": str(DEMO_DIR), "checkpoint_dir": str(CKPT_DIR),
        "heldout_yaml": str(HELDOUT_YAML),
        "correction_yaml": str(CORRECTION_YAML),
        "training_yaml": str(TRAIN_YAML),
        "history": history,
    }
    with open(RESULTS_DIR / "learning_curve.json", "w") as f:
        json.dump(out, f, indent=2, default=str)


def _make_chart():
    rc = _run([sys.executable, "-u", "-m",
               "Equivariant_pathway.equivariant_CNN_hybrid.baseline_only.chart"])
    if rc != 0:
        _info(f"chart rc={rc}")


if __name__ == "__main__":
    main()
