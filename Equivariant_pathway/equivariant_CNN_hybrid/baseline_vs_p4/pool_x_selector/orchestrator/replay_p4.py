"""Replay P4-top3-rotated's ALREADY-PRESCRIBED demonstrations at a different
per-round fine-tune budget (default 90 epochs) — no LLM, no rollout, no
re-prescription.

P4 trained at 500 epochs/round; the 5 IIL baselines trained at 90. To compare
apples-to-apples on the *training budget*, this re-runs P4's exact demo
sequence (the corrective demos it already collected, in order) but fine-tunes
``--epochs`` (default 90) per round, warm-starting from the SAME initial
bootstrap checkpoint P4 started from. The held-out eval (full set) and curve
schema match the rest of the suite, so the resulting ``<dest_method>`` curve
drops straight into the comparison.

For run R it reads ``results/run_R/<source_method>/demos/`` (20 init demos at
top level + ``round_NNN/`` corrective demos) and writes a fresh learning curve
to ``results/run_R/<dest_method>/``. The original P4 (@500) results are left
untouched.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4 import (  # noqa: E402
    baseline_budget as _bb,
)

from ..layouts.layout_setup import RESULTS_ROOT, ensure_shared_layouts
from ..logging_ext.training_log import TrainingLog

SUITE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = SUITE_ROOT / "config_baselines.yaml"


def _info(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [replay_p4] {msg}", flush=True)


def _finetune(cumulative_demo_dir: Path, new_demo_dir: Optional[Path], ckpt_dir: Path,
              seed: int, epochs: int, lr: float, batch_size: int, weight_decay: float,
              replay_mix: float, replay_mix_floor: float, head_dropout: float = 0.0) -> int:
    cmd = [
        sys.executable, "-u", "-m",
        "Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4."
        "pool_x_selector.trainer.finetune_replay",
        "--demo_dir", str(cumulative_demo_dir),
        "--checkpoint_dir", str(ckpt_dir),
        "--epochs", str(int(epochs)),
        "--batch_size", str(int(batch_size)),
        "--lr", str(float(lr)),
        "--weight_decay", str(float(weight_decay)),
        "--replay_mix", str(float(replay_mix)),
        "--replay_mix_floor", str(float(replay_mix_floor)),
        "--head_dropout", str(float(head_dropout)),
        "--seed", str(int(seed)),
        "--resume",
    ]
    if new_demo_dir is not None and new_demo_dir.exists():
        cmd += ["--new_demos_dir", str(new_demo_dir)]
    _info("$ " + " ".join(cmd))
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def _persist_curve(method: str, results_dir: Path, history: List[Dict], budget: int,
                   target_sr: float, demo_dir: Path, ckpt_dir: Path,
                   heldout_yaml: Path, run_id: int) -> None:
    out = {
        "method": method, "run_id": int(run_id), "budget": int(budget),
        "target_sr": float(target_sr), "demo_dir": str(demo_dir),
        "checkpoint_dir": str(ckpt_dir), "heldout_yaml": str(heldout_yaml),
        "source": "replay_p4_demos", "history": history,
    }
    with open(results_dir / "learning_curve.json", "w") as f:
        json.dump(out, f, indent=2, default=str)


def replay(run_id: int, epochs: int, source_method: str, dest_method: str,
           lr: float, batch_size: int, weight_decay: float, replay_mix: float,
           replay_mix_floor: float, seed_base: int, max_steps: int,
           target_sr: float, out_root: Optional[str] = None,
           head_dropout: float = 0.0) -> Dict:
    import torch  # noqa: F401  (ensures torch/env init like the rest of the suite)

    # READ P4's demos + the init checkpoint from the main results tree; WRITE
    # the replay outputs to out_root (sweep dir) when given, else in place.
    read_root = RESULTS_ROOT / f"run_{int(run_id)}"
    write_root = (Path(out_root) if out_root else RESULTS_ROOT) / f"run_{int(run_id)}"
    src = read_root / source_method
    dst = write_root / dest_method
    if not (src / "demos").exists():
        raise SystemExit(f"no P4 demos at {src/'demos'}")

    demo_dir = dst / "demos"
    ckpt_dir = dst / "checkpoints"
    results_dir = dst / "results"
    for d in (demo_dir, ckpt_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Seed the cumulative pool with the 20 init demos (top-level in P4's demos/).
    for f in sorted((src / "demos").glob("*.json")):
        dstf = demo_dir / f.name
        if not dstf.exists():
            shutil.copy2(f, dstf)

    # Warm-start from the run's INITIAL bootstrap checkpoint (the same start P4
    # used) — NOT P4's 500-epoch-trained checkpoint. We're retraining from
    # scratch on P4's demo sequence, only the per-round epoch count differs.
    init_ck = read_root / "shared" / "init_checkpoints"
    for n in ("best_hybrid_policy.pth", "last_hybrid_policy.pth"):
        s = init_ck / n
        if s.exists():
            shutil.copy2(s, ckpt_dir / n)
    if not (ckpt_dir / "best_hybrid_policy.pth").exists():
        raise SystemExit(f"missing init checkpoint at {init_ck}")

    shared = ensure_shared_layouts(initial_demos=20, heldout_n=200)
    heldout = shared["heldout"]
    seed = int(seed_base) + int(run_id)

    # P4's collected demo sequence: round_NNN dirs that actually have a demo.
    src_rounds = sorted((src / "demos").glob("round_*"),
                        key=lambda p: int(p.name.split("_")[-1]))
    budget = sum(len(list(rd.glob("*.json"))) for rd in src_rounds)

    # Round 0 — eval the initial policy on the full held-out set.
    initial = _bb._eval_heldout(ckpt_dir, heldout, results_dir, tag="round_000",
                                seed=seed, max_steps=max_steps)
    history: List[Dict] = [{
        "round": 0, "cum_demos": _bb._count_demos(demo_dir), "extra_demos": 0,
        "budget_remaining": budget, "heldout_sr": initial["success_rate"],
        "heldout_n_successes": initial["n_successes"],
        "heldout_n_episodes": initial["n_episodes"],
        "correction_sr": None, "n_new_demos": 0, "correction_yaml": None,
    }]
    _persist_curve(dest_method, results_dir, history, budget, target_sr,
                   demo_dir, ckpt_dir, heldout, run_id)
    tlog = TrainingLog(results_dir / "training_log.csv")
    tlog.write_row(round_idx=0, demos_added_total=0, training_rounds_total=0,
                   heldout_sr=initial["success_rate"], finetune_epochs=0,
                   lr=lr, replay_mix=replay_mix)
    _info(f"run {run_id}: init heldout_sr={initial['success_rate']:.3f}  "
          f"replaying {len(src_rounds)} P4 rounds ({budget} demos) at {epochs} epochs")

    extras = 0
    training_rounds = 0
    for rd in src_rounds:
        rnd = int(rd.name.split("_")[-1])
        demos = sorted(rd.glob("*.json"))
        if not demos:
            continue  # empty P4 round (no demo collected) — nothing to train on
        # Copy this round's prescribed demo(s) into the cumulative pool.
        round_demo_dir = demo_dir / f"round_{rnd:03d}"
        round_demo_dir.mkdir(parents=True, exist_ok=True)
        for f in demos:
            d = round_demo_dir / f.name
            if not d.exists():
                shutil.copy2(f, d)
        # Fine-tune at the requested epoch budget (replay over the cumulative pool).
        rc = _finetune(demo_dir, round_demo_dir, ckpt_dir, seed, epochs, lr,
                       batch_size, weight_decay, replay_mix, replay_mix_floor,
                       head_dropout=head_dropout)
        if rc != 0:
            _info(f"run {run_id} round {rnd}: finetune rc={rc} (continuing)")
        training_rounds += 1
        extras += len(demos)

        # Carry over P4's per-round prescribed_loss.json (info-gain parity).
        src_pl = src / "results" / f"round_{rnd:03d}" / "prescribed_loss.json"
        if src_pl.exists():
            out_rd = results_dir / f"round_{rnd:03d}"
            out_rd.mkdir(parents=True, exist_ok=True)
            try:
                obj = json.load(open(src_pl))
                obj["method"] = dest_method
                json.dump(obj, open(out_rd / "prescribed_loss.json", "w"),
                          indent=2, default=str)
            except (OSError, json.JSONDecodeError):
                shutil.copy2(src_pl, out_rd / "prescribed_loss.json")

        post = _bb._eval_heldout(ckpt_dir, heldout, results_dir,
                                 tag=f"round_{rnd:03d}", seed=seed + rnd,
                                 max_steps=max_steps)
        history.append({
            "round": rnd, "cum_demos": _bb._count_demos(demo_dir),
            "extra_demos": extras, "budget_remaining": budget - extras,
            "heldout_sr": post["success_rate"],
            "heldout_n_successes": post["n_successes"],
            "heldout_n_episodes": post["n_episodes"],
            "correction_sr": None, "n_new_demos": len(demos),
            "correction_yaml": None,
        })
        _persist_curve(dest_method, results_dir, history, budget, target_sr,
                       demo_dir, ckpt_dir, heldout, run_id)
        tlog.write_row(round_idx=rnd, demos_added_total=extras,
                       training_rounds_total=training_rounds,
                       heldout_sr=post["success_rate"], finetune_epochs=epochs,
                       lr=lr, replay_mix=replay_mix)
        _info(f"run {run_id} round {rnd}: extras={extras} heldout_sr={post['success_rate']:.3f}")

    return {"history": history, "final_sr": history[-1]["heldout_sr"],
            "n_demos": extras}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", type=int, required=True)
    ap.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--epochs", type=int, default=90,
                    help="Per-round fine-tune epochs for the replay (default 90).")
    ap.add_argument("--source_method", type=str, default="p4_top3_rotate")
    ap.add_argument("--dest_method", type=str, default="p4_top3_rotate_e90")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--weight_decay", type=float, default=None)
    ap.add_argument("--replay_mix", type=float, default=None)
    ap.add_argument("--replay_mix_floor", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--target_sr", type=float, default=None)
    ap.add_argument("--head_dropout", type=float, default=0.0,
                    help="Fusion-head dropout for training (warm-start safe).")
    ap.add_argument("--out_root", type=str, default=None,
                    help="Write outputs under <out_root>/run_<id>/<dest_method>/ "
                         "instead of the main results/ tree. Reads (P4 demos + init "
                         "checkpoint) always come from the main results/ tree.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    tr = cfg.get("trainer", {}) or {}
    res = replay(
        run_id=args.run_id, epochs=int(args.epochs),
        source_method=args.source_method, dest_method=args.dest_method,
        lr=float(args.lr if args.lr is not None else tr.get("lr", 5e-5)),
        batch_size=int(args.batch_size if args.batch_size is not None else tr.get("batch_size", 64)),
        weight_decay=float(args.weight_decay if args.weight_decay is not None else tr.get("weight_decay", 1e-4)),
        replay_mix=float(args.replay_mix if args.replay_mix is not None else tr.get("replay_mix", 0.5)),
        replay_mix_floor=float(args.replay_mix_floor if args.replay_mix_floor is not None else tr.get("replay_mix_floor", 0.2)),
        seed_base=int(args.seed if args.seed is not None else cfg.get("seed", 0)),
        max_steps=int(args.max_steps if args.max_steps is not None else cfg.get("max_steps", 60)),
        target_sr=float(args.target_sr if args.target_sr is not None else cfg.get("target_sr", 0.90)),
        out_root=args.out_root,
        head_dropout=float(args.head_dropout),
    )
    _info(f"run {args.run_id} DONE: final_sr={res['final_sr']} n_demos={res['n_demos']}")


if __name__ == "__main__":
    main()
