"""Headless one-shot runner for ONE round of an active-loop profile.

This is the slurm-friendly counterpart to scripts/active_loop.py:

  - active_loop.py is interactive: it loops train -> eval -> launches pygame ->
    waits for the human -> retrains, all in one process.
  - run_round.py is one-shot: it runs ONE round (optionally restoring the
    baseline checkpoint, training, then running the eval pipeline) and exits.
    Demo collection is NOT launched. The recommended_layouts.json the LLM
    produced is saved so you can record the demos manually with
    play_maze.py --layouts-from <path>, then resubmit the slurm job for the
    next round.

Why this matters
----------------
Slurm submits the job, run_round.py finishes, slurm sends mail. You can
close your laptop. When mail arrives, you ssh in (or sync the layouts file
locally), run play_maze with --layouts-from to record the prescribed demos,
then resubmit the slurm script. The next invocation auto-detects the next
round number for the same loop.

Loop / round resolution
-----------------------
  --loop-id <id>   (optional) — explicit loop directory under
                                results/active_loop/<profile>/<loop-id>/.
                                When omitted, run_round PICKS THE MOST RECENT
                                existing loop dir for that profile if one
                                exists, otherwise creates a new one.

  --round <N>      (optional) — explicit round number under that loop.
                                When omitted, run_round picks N = (max
                                existing round_<N>/ in the loop) + 1, or 1
                                if there are none.

  --new-loop                    Force creation of a fresh loop_<timestamp>/
                                directory even when prior loops exist.

Typical workflow
----------------
  # First-ever round for profile p3:
  PROFILE=p3 sbatch scripts/slurm/round.sh
  #   -> run_round creates results/active_loop/p3/loop_<ts>/round_1/
  #   -> trains on baseline, runs pipeline, saves recommended_layouts.json
  #   -> emails you when slurm finishes

  # Record the demos pygame-side (locally, off-cluster is fine):
  python scripts/play_maze.py \
    --layouts-from results/active_loop/p3/loop_<ts>/round_1/recommended_layouts.json

  # Submit round 2 for the same loop:
  PROFILE=p3 LOOP_ID=loop_<ts> sbatch scripts/slurm/round.sh
  #   -> run_round detects round_1 exists, creates round_2/
  #   -> trains on baseline + ALL demos this loop has collected so far
  #   -> ...

CLI
---
  python scripts/run_round.py --profile p3
  python scripts/run_round.py --profile p3 --loop-id loop_20260501_120000
  python scripts/run_round.py --profile p3 --new-loop --skip-train
  python scripts/run_round.py --profile p3 --rounds-to-target-check
"""
import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

# Reuse the helpers active_loop.py already exposes — the heavy lifting (train,
# eval, metric computation, log appending, layout flattening) lives there.
from scripts.active_loop import (
    PROFILE_ALIASES,
    ACTIVE_LOOP_ROOT,
    BASELINE_CKPT_DEFAULT,
    _info,
    _section,
    append_loop_log,
    build_demo_paths,
    compute_metrics,
    count_baseline_demos,
    count_demos_in,
    parse_train_loss,
    print_prescriptions,
    profile_loop_demos_dir,
    resolve_profile,
    restore_baseline_checkpoint,
    run_eval,
    run_train,
    save_loop_graph,
    save_round_artefacts,
    _collect_recommended_layouts,
)


def _existing_loop_ids(profile_root: Path) -> List[str]:
    if not profile_root.exists():
        return []
    return sorted(
        p.name for p in profile_root.iterdir()
        if p.is_dir() and p.name.startswith("loop_")
    )


def _latest_loop_id(profile_root: Path) -> Optional[str]:
    ids = _existing_loop_ids(profile_root)
    return ids[-1] if ids else None


def _next_round_number(loop_root: Path) -> int:
    """Return one more than the highest existing round_<N>/ under loop_root,
    or 1 if there are none."""
    if not loop_root.exists():
        return 1
    nums = []
    for p in loop_root.iterdir():
        if p.is_dir() and p.name.startswith("round_"):
            try:
                nums.append(int(p.name.split("_", 1)[1]))
            except ValueError:
                continue
    return (max(nums) + 1) if nums else 1


def _load_prev_metrics(profile_log_path: Path, loop_id: str) -> Tuple[float, Optional[Dict]]:
    """Pull cumulative_regret + last metrics from prior rounds in this loop so
    the round we run sees the correct rolling state."""
    if not profile_log_path.exists():
        return 0.0, None
    try:
        with open(profile_log_path, "r") as f:
            history = json.load(f) or []
    except Exception:
        return 0.0, None
    relevant = [r for r in history if r.get("loop_id") == loop_id]
    if not relevant:
        return 0.0, None
    last = relevant[-1]
    cr = float(last.get("cumulative_regret", 0.0) or 0.0)
    return cr, last


def _save_recommended_layouts(round_dir: Path, prescriptions: List[Dict],
                              profile_name: str, loop_id: str, rnd: int,
                              demo_dir_for_round: Path) -> Path:
    """Same artefact active_loop writes — but we save it here (no pygame launch)."""
    layouts = _collect_recommended_layouts(prescriptions)
    out = {
        "round":            rnd,
        "profile":          profile_name,
        "loop_id":          loop_id,
        "n_layouts":        len(layouts),
        "demo_dir":         str(demo_dir_for_round),
        "layouts":          layouts,
    }
    path = round_dir / "recommended_layouts.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    return path


def parse_args():
    p = argparse.ArgumentParser(description="Run ONE round of one ablation profile and exit.")
    p.add_argument("--profile", type=str, required=True,
                   help="Short alias (p1..p6) or full profile name.")
    p.add_argument("--config", type=str, default="configs/experiment_config.yaml")
    p.add_argument("--loop-id", type=str, default=None, dest="loop_id",
                   help="Explicit loop directory name. If omitted, the most recent existing "
                        "loop for this profile is reused; if there are none, a fresh "
                        "loop_<timestamp> is created. Pass --new-loop to force a fresh one.")
    p.add_argument("--round", type=int, default=None, dest="round_number",
                   help="Explicit round number. Default: next unused round in the loop.")
    p.add_argument("--new-loop", action="store_true", dest="new_loop",
                   help="Always create a fresh loop_<timestamp>/ directory.")
    p.add_argument("--n_episodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--target-sr", type=float, default=0.90, dest="target_sr",
                   help="Recorded in the round log so downstream tooling can decide if "
                        "the target was hit (run_round itself never stops a loop).")
    p.add_argument("--baseline-checkpoint", type=str, default=str(BASELINE_CKPT_DEFAULT),
                   dest="baseline_checkpoint",
                   help="Baseline checkpoint copied to checkpoints/best_model.pth on round 1 "
                        "of a NEW loop (not on subsequent rounds — those resume from the "
                        "round's own training output).")
    p.add_argument("--skip-baseline-restore", action="store_true",
                   dest="skip_baseline_restore",
                   help="Do not copy the baseline checkpoint even on round 1 of a new loop.")
    p.add_argument("--skip-train", action="store_true", dest="skip_train",
                   help="Skip the training step (use whatever is in checkpoints/ as-is). "
                        "Useful for re-running the LLM pipeline on an unchanged policy.")
    return p.parse_args()


def main():
    args = parse_args()
    profile_name, profile_yaml = resolve_profile(args.profile)
    profile_root = ACTIVE_LOOP_ROOT / profile_name
    profile_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Resolve loop_id.                                                    #
    # ------------------------------------------------------------------ #
    if args.new_loop or args.loop_id is None and _latest_loop_id(profile_root) is None:
        loop_id = f"loop_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        is_new_loop = True
    elif args.loop_id is None:
        loop_id = _latest_loop_id(profile_root)
        is_new_loop = False
    else:
        loop_id = args.loop_id
        is_new_loop = not (profile_root / loop_id).exists()

    loop_root = profile_root / loop_id
    loop_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Resolve round number.                                               #
    # ------------------------------------------------------------------ #
    if args.round_number is not None:
        rnd = int(args.round_number)
    else:
        rnd = _next_round_number(loop_root)
    round_dir = loop_root / f"round_{rnd}"
    round_dir.mkdir(parents=True, exist_ok=True)

    profile_demo_dir       = profile_loop_demos_dir(profile_name, loop_id)
    profile_demo_dir.mkdir(parents=True, exist_ok=True)
    demo_dir_for_round     = profile_demo_dir / f"round_{rnd}"
    demo_paths_arg         = build_demo_paths(profile_name, loop_id)

    profile_log_path       = profile_root / "loop_log.json"
    learning_curve_path    = profile_root / "learning_curve.png"

    cumulative_regret, _last = _load_prev_metrics(profile_log_path, loop_id)

    _section(f"RUN_ROUND — profile={profile_name}", char="=")
    _info(f"loop_id        : {loop_id}  (new_loop={is_new_loop})")
    _info(f"round          : {rnd}")
    _info(f"round_dir      : {round_dir}")
    _info(f"profile demos  : {profile_demo_dir}  (this loop)")
    _info(f"demo globs     : {demo_paths_arg}")
    _info(f"baseline demos : {count_baseline_demos()}")
    _info(f"prev cum_regret: {cumulative_regret:.3f}")

    # ------------------------------------------------------------------ #
    # Baseline checkpoint restore — only on round 1 of a NEW loop, so the #
    # rest of the loop accumulates training on top of itself.             #
    # ------------------------------------------------------------------ #
    if rnd == 1 and is_new_loop and not args.skip_baseline_restore:
        restore_baseline_checkpoint(Path(args.baseline_checkpoint))
    else:
        _info(f"Baseline restore skipped (rnd={rnd}, new_loop={is_new_loop}, "
              f"--skip-baseline-restore={args.skip_baseline_restore}).")

    # Manifest — written once per loop, but it's safe to overwrite the existing
    # one when a loop's earliest round is rerun. Keep the original timestamp so
    # downstream tooling can tell when the loop was first opened.
    manifest_path = loop_root / "loop_manifest.json"
    if not manifest_path.exists():
        with open(manifest_path, "w") as f:
            json.dump({
                "loop_id":       loop_id,
                "profile":       profile_name,
                "profile_yaml":  str(profile_yaml),
                "first_round_at": datetime.now().isoformat(),
                "demo_paths_for_training": demo_paths_arg,
                "profile_demo_dir":        str(profile_demo_dir),
            }, f, indent=2, default=str)

    # ------------------------------------------------------------------ #
    # TRAIN.                                                              #
    # ------------------------------------------------------------------ #
    round_t0 = time.time()
    if args.skip_train:
        _info("Skipping training (--skip-train).")
        train_loss = float("nan")
    else:
        # Resume from the existing best_model.pth in checkpoints/ — that is
        # either the just-restored baseline (round 1 / new loop) or the policy
        # the previous round left behind (rounds 2+).
        train_stdout = run_train(resume=True, demo_paths=demo_paths_arg)
        train_loss = parse_train_loss(train_stdout)
        _info(f"parsed train_loss = {train_loss}")

    # ------------------------------------------------------------------ #
    # EVAL — full pipeline (Phase A + B + C).                             #
    # ------------------------------------------------------------------ #
    full_output_path = run_eval(
        profile_yaml=profile_yaml,
        base_config=args.config,
        round_dir=round_dir,
        n_episodes=args.n_episodes,
        seed=args.seed,
    )

    metrics = compute_metrics(
        full_output_path, train_loss, cumulative_regret,
        round_start_demos=count_demos_in(profile_demo_dir),
        profile_demo_dir=profile_demo_dir,
    )
    save_round_artefacts(round_dir, metrics, full_output_path)
    layouts_path = _save_recommended_layouts(
        round_dir, metrics.get("prescriptions", []),
        profile_name, loop_id, rnd, demo_dir_for_round,
    )
    print_prescriptions(rnd, metrics.get("prescriptions", []))

    record = {
        "round":     rnd,
        "loop_id":   loop_id,
        "profile":   profile_name,
        "target_sr": args.target_sr,
        "round_dir": str(round_dir),
        **metrics,
    }
    append_loop_log(profile_log_path, record)
    save_loop_graph(profile_log_path, learning_curve_path, profile_name)

    target_hit = metrics["success_rate"] >= args.target_sr
    elapsed = time.time() - round_t0

    _section(f"ROUND {rnd} DONE — SR={metrics['success_rate']:.2f}", char="*")
    _info(f"target_sr      : {args.target_sr:.2f}  (hit={target_hit})")
    _info(f"elapsed        : {elapsed/60:.1f}m")
    _info(f"round_dir      : {round_dir}")
    _info(f"layouts file   : {layouts_path}")
    _info(f"profile log    : {profile_log_path}")
    _info(f"learning curve : {learning_curve_path}")
    _info(f"profile demos  : {metrics['total_profile_demos']} (+{metrics['demos_added']} this round)")

    if target_hit:
        _info("TARGET HIT — no further rounds needed for this loop.")
    else:
        n_layouts_to_record = len(metrics.get("prescriptions", []) and
                                  [l for p in metrics["prescriptions"]
                                     for l in (p.get("recommended_layouts") or [])])
        _info(
            f"NEXT STEP — record demos with:\n"
            f"  python scripts/play_maze.py --layouts-from {layouts_path}\n"
            f"  (will save into {demo_dir_for_round})"
        )
        _info(
            f"THEN submit next round with:\n"
            f"  PROFILE={args.profile} LOOP_ID={loop_id} sbatch scripts/slurm/round.sh"
        )

    # Exit code: 0 always (slurm mail fires regardless). The user reads the
    # log to know whether to record demos or stop.
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
