"""Active loop — single entry point for ONE ablation profile.

Why this script exists
======================
The end-goal of the project is to compare profiles p1..p6 on a fair footing.
"Fair footing" means that every profile starts from the SAME initial policy
weights and the SAME baseline demo set, then the active loop measures how many
extra demos each profile needs to reach a target success rate. Mixing demos
across profiles, or letting a later profile inherit an earlier profile's
fine-tuned checkpoint, makes the comparison meaningless. This script enforces
both isolations.

Per-round flow
==============
  0.  (round 1 only) Restore baseline checkpoint into checkpoints/best_model.pth
      so this loop starts from the same weights every other profile started from.
  1.  Train the diffusion policy on (baseline demos UNION this profile's
      already-collected active-loop demos). Always trains on the FULL aggregated
      dataset to avoid catastrophic forgetting — never just the latest batch.
  2.  Run rollout + the chosen profile's pipeline (Phase A + B + C). The
      aggregator emits demonstration_prescriptions[].recommended_layouts —
      explicit (start_pos, goal_pos, fire_positions) tuples grounded in the
      KAG corridor vocabulary.
  3.  For every recommended layout, launch play_maze with --start/--goal/
      --fires/--demo_dir/--layout_id/--single_episode so the human records
      n_repetitions demos on that exact layout. Demos are saved under
      demos/active_loop/<profile>/<loop_id>/round_<N>/. Baseline demos are
      never touched.
  4.  Loop until success_rate >= --target-sr OR --rounds reached.

Per-profile artefacts
=====================
  demos/active_loop/<profile>/<loop_id>/
    round_1/  *.json        # demos collected on layouts the LLM recommended
    round_2/  *.json
    ...

  results/active_loop/<profile>/
    loop_log.json              # one entry per round (cumulative)
    learning_curve.png
    <loop_id>/
      loop_manifest.json       # CLI args, profile YAML, baseline checkpoint
      loop_summary.json        # rounds_to_target, demos_added_in_loop
      round_<N>/
        full_output.json
        prescriptions.json
        recommended_layouts.json   # what the LLM asked for, post-validation
        metrics.json
        config_used.yaml
        rag_bank/                  # owner-id-isolated per pipeline/rag_bank.py

Usage
=====
  # First time setup (once): pin the checkpoint that represents the baseline
  # policy trained on demos/*.json only.
  cp checkpoints/best_model.pth checkpoints/baseline.pth

  # Run a profile end-to-end:
  python scripts/active_loop.py --profile p1 --baseline-checkpoint checkpoints/baseline.pth
  python scripts/active_loop.py --profile p3 --rounds 15 --target-sr 0.9
  python scripts/active_loop.py --profile p6 --n_episodes 10 --seed 42 --no-pygame
"""
import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional, Tuple

from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS   = REPO_ROOT / "scripts"
RESULTS   = REPO_ROOT / "results"
RUNS_DIR  = RESULTS / "runs"
DEMOS_DIR = REPO_ROOT / "demos"
ACTIVE_LOOP_DEMOS_ROOT = DEMOS_DIR / "active_loop"     # per-profile collected demos
ACTIVE_LOOP_ROOT       = RESULTS / "active_loop"       # per-profile artefacts / logs
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
BASELINE_CKPT_DEFAULT = CHECKPOINT_DIR / "baseline.pth"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Logging helpers.
# ---------------------------------------------------------------------------
def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _section(title: str, char: str = "=", width: int = 80):
    bar = char * width
    print(f"\n{bar}\n[{_ts()}] [active_loop] {title}\n{bar}")


def _info(msg: str):
    print(f"[{_ts()}] [active_loop] {msg}")


# ---------------------------------------------------------------------------
# Profile resolution.
# ---------------------------------------------------------------------------
PROFILE_ALIASES = {
    "p1": "p1_vlm_plain_llm",
    "p2": "p2_vlm_reasoning_plain_llm",
    "p3": "p3_vlm_reasoning_cross_plain_llm",
    "p4": "p4_vlm_reasoning_kag_cross_plain_llm",
    "p5": "p5_vlm_reasoning_kag_rag_cross_plain_llm",
    "p6": "p6_vlm_reasoning_kag_rag_tkf_cross_plain_llm",
}


def resolve_profile(name: str) -> Tuple[str, Path]:
    """Return (profile_name, profile_yaml_path) for a short alias or full name."""
    key = name.strip().lower()
    full = PROFILE_ALIASES.get(key, name)
    yaml_path = REPO_ROOT / "configs" / "ablation_profiles" / f"{full}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Profile YAML not found for '{name}': tried {yaml_path}.\n"
            f"Available short aliases: {sorted(PROFILE_ALIASES.keys())}"
        )
    return full, yaml_path


def parse_args():
    p = argparse.ArgumentParser(description="Active loop for ONE ablation profile.")
    p.add_argument("--profile", type=str, required=True,
                   help="Short alias (p1..p6) or full profile name.")
    p.add_argument("--config",     type=str,   default="configs/experiment_config.yaml")
    p.add_argument("--rounds",     type=int,   default=20)
    p.add_argument("--target-sr",  type=float, default=0.90, dest="target_sr")
    p.add_argument("--n_episodes", type=int,   default=None)
    p.add_argument("--seed",       type=int,   default=None)
    p.add_argument("--baseline-checkpoint", type=str, default=str(BASELINE_CKPT_DEFAULT),
                   dest="baseline_checkpoint",
                   help="Path to the pinned baseline checkpoint to copy into "
                        "checkpoints/best_model.pth at loop start so every profile "
                        "begins from identical weights. Default: checkpoints/baseline.pth.")
    p.add_argument("--skip-baseline-restore", action="store_true",
                   dest="skip_baseline_restore",
                   help="Don't copy the baseline checkpoint at loop start (use whatever "
                        "is currently in checkpoints/).")
    p.add_argument("--skip-train-first-round", action="store_true",
                   help="Don't retrain before round 1 (use the freshly restored baseline).")
    p.add_argument("--no-pygame", action="store_true",
                   help="Skip launching pygame for layout-driven demo collection (CI / "
                        "dry-run mode). Recommended layouts are still written to "
                        "recommended_layouts.json so you can record them later.")
    p.add_argument("--no-demo-prompt", action="store_true",
                   help="Skip the 'press ENTER after recording' prompt between rounds.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Subprocess helpers.
# ---------------------------------------------------------------------------
def _stream_subprocess(cmd: List[str]) -> Tuple[int, str]:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(REPO_ROOT),
    )
    captured: List[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        captured.append(line)
    rc = proc.wait()
    return rc, "".join(captured)


def run_train(resume: bool, demo_paths: Optional[str] = None) -> str:
    """Train the diffusion policy.

    demo_paths: comma-separated globs forwarded to train_diffusion as
    --demo_paths. The active loop builds this so the trainer always sees
    (baseline demos UNION this profile's accumulated active-loop demos),
    which is what defends against catastrophic forgetting between rounds.
    """
    cmd = [sys.executable, "-u", str(SCRIPTS / "train_diffusion.py")]  # -u: unbuffered so tqdm streams live
    if resume:
        cmd.append("--resume")
    if demo_paths:
        cmd += ["--demo_paths", demo_paths]
    _section(f"TRAIN — resume={resume}", char="-")
    _info(f"command: {' '.join(cmd)}")
    if demo_paths:
        _info(f"demo globs: {demo_paths}")
    t0 = time.time()
    rc, out = _stream_subprocess(cmd)
    if rc != 0:
        raise RuntimeError(f"train_diffusion.py exited with code {rc}")
    _info(f"TRAIN done in {(time.time()-t0)/60:.1f}m")
    return out


def parse_train_loss(stdout: str) -> float:
    pattern = re.compile(r"Loss:\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)")
    matches = pattern.findall(stdout or "")
    if not matches:
        return float("nan")
    try:
        return float(matches[-1])
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Pipeline invocation — in-process so we can pin the run dir to the
# profile's loop directory and inject the per-loop RAG bank path.
# ---------------------------------------------------------------------------
def run_eval(
    profile_yaml: Path,
    base_config: str,
    round_dir: Path,
    n_episodes: Optional[int],
    seed: Optional[int],
) -> Path:
    """Run the full pipeline (Phase A + B + C) into round_dir.

    The RAG bank for this round is forced to round_dir/rag_bank so per-loop
    isolation matches the per-profile isolation already enforced by the
    ablation suite.
    """
    from pipeline.pipeline_runner import load_config, run_pipeline

    cfg = load_config(str(base_config), str(profile_yaml))

    if n_episodes is not None:
        cfg.setdefault("rollout", {})["n_episodes"] = int(n_episodes)
    if seed is not None:
        cfg.setdefault("rollout", {})["seed"] = int(seed)

    # Per-round isolated RAG bank — fresh every round so a poor early
    # prescription cannot keep contaminating later rounds within the same loop.
    rag_path = round_dir / "rag_bank"
    if rag_path.exists():
        shutil.rmtree(rag_path)
    rag_path.mkdir(parents=True, exist_ok=True)
    cfg.setdefault("rag", {})["bank_path"] = str(rag_path)

    # Pin the pipeline output directory so we don't have to scrape RUNS_DIR.
    round_dir.mkdir(parents=True, exist_ok=True)
    _section(f"EVAL — in-process pipeline → {round_dir.name}", char="-")
    _info(f"profile yaml : {profile_yaml}")
    _info(f"base config  : {base_config}")
    _info(f"n_episodes   : {cfg.get('rollout',{}).get('n_episodes')}  seed={cfg.get('rollout',{}).get('seed')}")
    _info(f"pipeline flags: {cfg.get('pipeline',{})}")
    _info(f"rag bank dir : {rag_path}")
    started_at = time.time()
    run_pipeline(cfg, run_dir=round_dir)
    full_output_path = round_dir / "full_output.json"
    if not full_output_path.exists():
        raise RuntimeError(
            f"Pipeline finished but {full_output_path} was not written "
            f"(elapsed={time.time()-started_at:.1f}s)."
        )
    _info(f"EVAL done in {time.time()-started_at:.1f}s → {full_output_path}")
    return full_output_path


# ---------------------------------------------------------------------------
# Demo accounting + per-profile demo paths.
# ---------------------------------------------------------------------------
def count_baseline_demos() -> int:
    """Count demos at demos/*.json (top level only — does NOT recurse into
    demos/active_loop)."""
    if not DEMOS_DIR.exists():
        return 0
    return len(list(DEMOS_DIR.glob("*.json")))


def count_demos_in(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob("*.json"))


def profile_loop_demos_dir(profile_name: str, loop_id: str) -> Path:
    """Where this profile/loop should write its collected demos."""
    return ACTIVE_LOOP_DEMOS_ROOT / profile_name / loop_id


def build_demo_paths(profile_name: str, loop_id: str) -> str:
    """Comma-separated glob list: baseline + this profile's collected demos.

    The trainer expands these via glob.glob(..., recursive='**' in pattern).
    Baseline is non-recursive (top-level demos/*.json only) so it does NOT
    pick up demos collected by other profiles' active loops. The per-profile
    glob recurses into round_<N>/ subdirs.
    """
    baseline = str(DEMOS_DIR / "*.json")
    profile_glob = str(profile_loop_demos_dir(profile_name, loop_id) / "**" / "*.json")
    return f"{baseline},{profile_glob}"


def restore_baseline_checkpoint(baseline_path: Path):
    """Copy baseline_path into checkpoints/best_model.pth (and best_model_ema.pth
    when there's a sibling EMA file) so each profile's loop starts from the
    same weights. No-op when the baseline file is missing — caller decides
    whether to abort."""
    if not baseline_path.exists():
        print(f"[active_loop] WARNING: baseline checkpoint not found at {baseline_path}; "
              f"using whatever is currently in {CHECKPOINT_DIR}/.")
        return
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    target = CHECKPOINT_DIR / "best_model.pth"
    shutil.copy2(baseline_path, target)
    print(f"[active_loop] Restored baseline checkpoint: {baseline_path} → {target}")
    ema_src = baseline_path.with_name(baseline_path.stem + "_ema.pth")
    if ema_src.exists():
        shutil.copy2(ema_src, CHECKPOINT_DIR / "best_model_ema.pth")
        print(f"[active_loop] Restored baseline EMA checkpoint: {ema_src}")


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------
def _manhattan(a, b) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def compute_metrics(
    full_output_path: Path,
    train_loss: float,
    prev_cumulative_regret: float,
    round_start_demos: int,
    profile_demo_dir: Path,
) -> Dict:
    with open(full_output_path, "r") as f:
        data = json.load(f)

    metadata     = data.get("metadata", {}) or {}
    n_episodes   = int(metadata.get("n_episodes", 0) or 0)
    n_successes  = int(metadata.get("n_successes", 0) or 0)
    n_failures   = int(metadata.get("n_failures", max(n_episodes - n_successes, 0)) or 0)
    run_id       = str(metadata.get("run_id", full_output_path.parent.name))

    success_rate = (n_successes / n_episodes) if n_episodes > 0 else 0.0
    failure_rate = (n_failures  / n_episodes) if n_episodes > 0 else 0.0

    phase_a = data.get("phase_a", {}) or {}
    all_rollouts = phase_a.get("all_rollouts", []) or []
    steps_list   = [int(r.get("total_steps", 0) or 0) for r in all_rollouts]
    reward_list  = [float(r.get("total_reward", 0.0) or 0.0) for r in all_rollouts]
    mean_episode_length = mean(steps_list)  if steps_list  else 0.0
    mean_reward         = mean(reward_list) if reward_list else 0.0
    std_reward          = pstdev(reward_list) if len(reward_list) > 1 else 0.0

    cumulative_regret = prev_cumulative_regret + (1.0 - success_rate)

    per_episode = (data.get("phase_b", {}) or {}).get("per_episode", []) or []
    tkf_results = [ep.get("tkf_result") for ep in per_episode if ep.get("tkf_result") is not None]
    not_found = 0
    for tkf in tkf_results:
        verdict = str((tkf or {}).get("verdict", "")).upper()
        if verdict in ("NOT_FOUND", "PARTIAL"):
            not_found += 1
    demo_coverage_gap = (not_found / len(tkf_results)) if tkf_results else 0.0

    failure_ids = set(phase_a.get("failure_episode_ids", []) or [])
    fire_hits = 0
    for rollout in all_rollouts:
        eid = rollout.get("episode_id")
        if eid not in failure_ids:
            continue
        dyn = rollout.get("dynamic_config", {}) or {}
        fires = dyn.get("fire_positions", []) or []
        final_pos = None
        steps = rollout.get("steps", []) or []
        if steps:
            last_info = (steps[-1] or {}).get("info", {}) or {}
            final_pos = last_info.get("agent_pos")
        if final_pos is None:
            for kf in rollout.get("key_frames", []) or []:
                if kf.get("role") == "end_frame":
                    idx = kf.get("step_idx")
                    if idx is not None and 0 <= int(idx) < len(steps):
                        final_pos = (steps[int(idx)] or {}).get("info", {}).get("agent_pos")
                    break
        if final_pos is None:
            continue
        if any(_manhattan(final_pos, fp) == 1 for fp in fires):
            fire_hits += 1
    fire_proximity_rate = (fire_hits / len(failure_ids)) if failure_ids else 0.0

    # demos_now / demos_added counts THIS PROFILE'S collected demos only —
    # baseline demos are constant across the loop and not part of "added".
    demos_now   = count_demos_in(profile_demo_dir)
    demos_added = demos_now - round_start_demos
    baseline_demos = count_baseline_demos()

    phase_c = data.get("phase_c", {}) or {}
    parsed  = phase_c.get("parsed_prescription", {}) or {}
    prescriptions = parsed.get("demonstration_prescriptions", []) or []

    return {
        "run_id":              run_id,
        "run_dir":             str(full_output_path.parent),
        "train_loss":          float(train_loss) if train_loss == train_loss else float("nan"),
        "n_episodes":          n_episodes,
        "n_successes":         n_successes,
        "n_failures":          n_failures,
        "success_rate":        float(success_rate),
        "failure_rate":        float(failure_rate),
        "mean_episode_length": float(mean_episode_length),
        "mean_reward":         float(mean_reward),
        "std_reward":          float(std_reward),
        "cumulative_regret":   float(cumulative_regret),
        "demo_coverage_gap":   float(demo_coverage_gap),
        "fire_proximity_rate": float(fire_proximity_rate),
        "demos_added":         int(demos_added),
        "total_profile_demos": int(demos_now),
        "baseline_demos":      int(baseline_demos),
        "training_dataset_size": int(baseline_demos + demos_now),
        "prescriptions":       prescriptions,
    }


# ---------------------------------------------------------------------------
# Per-round artefact saving.
# ---------------------------------------------------------------------------
def save_round_artefacts(round_dir: Path, metrics: Dict, full_output_path: Path):
    round_dir.mkdir(parents=True, exist_ok=True)

    # metrics.json — what the loop logged for this round
    with open(round_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # prescriptions.json — extracted aggregator output, easy to inspect
    prescriptions_payload = {
        "run_id":        metrics.get("run_id"),
        "success_rate":  metrics.get("success_rate"),
        "n_failures":    metrics.get("n_failures"),
        "prescriptions": metrics.get("prescriptions", []),
    }
    with open(round_dir / "prescriptions.json", "w") as f:
        json.dump(prescriptions_payload, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Logging.
# ---------------------------------------------------------------------------
def append_loop_log(profile_log_path: Path, record: Dict):
    profile_log_path.parent.mkdir(parents=True, exist_ok=True)
    history: List[Dict] = []
    if profile_log_path.exists():
        try:
            with open(profile_log_path, "r") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    history = loaded
        except Exception:
            history = []
    history.append(record)
    with open(profile_log_path, "w") as f:
        json.dump(history, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Pretty printing.
# ---------------------------------------------------------------------------
def print_table_header():
    header = (
        f"{'Round':>5} | {'SR':>5} | {'FR':>5} | {'AvgLen':>6} | "
        f"{'MeanRew±Std':>14} | {'Loss':>8} | {'CumRegret':>9} | "
        f"{'CovGap':>6} | {'FireProx':>8} | {'DemosAdded':>10} | {'TotalDemos':>10}"
    )
    sep = "-" * len(header)
    print(sep); print(header); print(sep)


def print_table_row(rnd: int, m: Dict):
    loss = m.get("train_loss", float("nan"))
    loss_str = "nan" if loss != loss else f"{loss:.4f}"
    print(
        f"{rnd:>5d} | "
        f"{m['success_rate']:>5.2f} | "
        f"{m['failure_rate']:>5.2f} | "
        f"{m['mean_episode_length']:>6.1f} | "
        f"{m['mean_reward']:>6.2f}±{m['std_reward']:<6.2f} | "
        f"{loss_str:>8} | "
        f"{m['cumulative_regret']:>9.3f} | "
        f"{m['demo_coverage_gap']:>6.2f} | "
        f"{m['fire_proximity_rate']:>8.2f} | "
        f"{m['demos_added']:>10d} | "
        f"{m['total_profile_demos']:>10d}"
    )


def print_prescriptions(rnd: int, prescriptions: List[Dict]):
    if not prescriptions:
        print(f"\n[active_loop] No prescriptions produced this round.")
        return
    print(f"\n[active_loop] Prescriptions for round {rnd}:")
    for i, pres in enumerate(prescriptions, 1):
        demo_id = pres.get("demo_id", i)
        guidance = pres.get("guidance", "(no guidance)")
        region   = pres.get("target_region", pres.get("corridor", "(unspecified)"))
        teaches  = pres.get("what_it_teaches", "")
        reps     = pres.get("n_repetitions", 1)
        print(f"  [{demo_id}] region={region} | repetitions={reps}")
        print(f"       guidance : {guidance}")
        if teaches:
            print(f"       teaches  : {teaches}")


def _collect_recommended_layouts(prescriptions: List[Dict]) -> List[Dict]:
    """Flatten prescription[].recommended_layouts into a single ordered list of
    layouts the user should record. Each entry carries the parent demo_id /
    corridor / guidance fields so the user knows WHY this layout was asked for.
    Layouts are repeated according to layout.n_repetitions so the loop launches
    pygame the right number of times for each spec."""
    flat: List[Dict] = []
    for pres in prescriptions or []:
        demo_id  = pres.get("demo_id", "?")
        corridor = pres.get("corridor", "?")
        guidance = pres.get("guidance", "")
        teaches  = pres.get("what_it_teaches", "")
        layouts  = pres.get("recommended_layouts", []) or []
        for li, layout in enumerate(layouts):
            reps = max(1, int(layout.get("n_repetitions", 1) or 1))
            for k in range(reps):
                flat.append({
                    "parent_demo_id":   demo_id,
                    "corridor":         corridor,
                    "guidance":         guidance,
                    "what_it_teaches":  teaches,
                    "layout_index":     li,
                    "repetition":       k + 1,
                    "n_repetitions":    reps,
                    "start_pos":        list(layout["start_pos"]),
                    "goal_pos":         list(layout["goal_pos"]),
                    "fire_positions":   [list(p) for p in layout["fire_positions"]],
                    "rationale":        layout.get("rationale", ""),
                })
    return flat


def _format_pair(p) -> str:
    return f"{int(p[0])},{int(p[1])}"


def _format_fires(fires) -> str:
    return ";".join(_format_pair(f) for f in fires)


def launch_play_maze_for_layout(
    layout: Dict,
    profile_name: str,
    loop_id: str,
    rnd: int,
    seq_idx: int,
    demo_dir: Path,
    no_pygame: bool,
):
    """Launch play_maze.py with --start/--goal/--fires/--demo_dir/--layout_id.

    Skips the actual launch when no_pygame is True (CI / dry-run); the layout
    spec is still recorded in recommended_layouts.json so the user can record
    it manually later.
    """
    layout_id = (
        f"{profile_name}_{loop_id}_round{rnd}_demo{layout['parent_demo_id']}_"
        f"layout{layout['layout_index']}_rep{layout['repetition']}"
    )
    spec = (
        f"start={layout['start_pos']}  goal={layout['goal_pos']}  "
        f"fires={layout['fire_positions']}  corridor={layout['corridor']}  "
        f"rep={layout['repetition']}/{layout['n_repetitions']}"
    )
    print(f"\n[active_loop] Layout {seq_idx}: {spec}")
    print(f"[active_loop]   guidance: {layout.get('guidance','')}")
    if layout.get("rationale"):
        print(f"[active_loop]   rationale: {layout['rationale']}")

    if no_pygame:
        print(f"[active_loop]   --no-pygame set; skipping play_maze launch for layout_id={layout_id}.")
        return

    cmd = [
        sys.executable, str(SCRIPTS / "play_maze.py"),
        "--demo_dir",   str(demo_dir),
        "--start",      _format_pair(layout["start_pos"]),
        "--goal",       _format_pair(layout["goal_pos"]),
        "--fires",      _format_fires(layout["fire_positions"]),
        "--layout_id",  layout_id,
        "--single_episode",
    ]
    print(f"[active_loop]   launch: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(REPO_ROOT))


def run_demo_collection_for_round(
    prescriptions: List[Dict],
    profile_name: str,
    loop_id: str,
    rnd: int,
    round_dir: Path,
    profile_demo_dir_for_round: Path,
    no_pygame: bool,
    no_prompt: bool,
) -> List[Dict]:
    """Run the layout-driven demo-collection step for one round.

    Returns the flat list of layout specs that were processed (so the round
    artefacts can record exactly what was asked for).
    """
    layouts = _collect_recommended_layouts(prescriptions)
    profile_demo_dir_for_round.mkdir(parents=True, exist_ok=True)

    # Persist what the LLM asked for, BEFORE pygame, so the record exists even
    # if the user aborts or pygame can't launch in this environment.
    with open(round_dir / "recommended_layouts.json", "w") as f:
        json.dump({
            "round":            rnd,
            "profile":          profile_name,
            "loop_id":          loop_id,
            "n_layouts":        len(layouts),
            "demo_dir":         str(profile_demo_dir_for_round),
            "layouts":          layouts,
        }, f, indent=2, default=str)

    if not layouts:
        _info("No recommended layouts this round.")
        if not no_prompt:
            input("→ Press ENTER to continue to the next round.\n")
        return layouts

    _section(f"DEMO COLLECTION — {len(layouts)} layout(s) queued", char="-")
    _info(f"demos will save to: {profile_demo_dir_for_round}")
    _info(f"profile={profile_name}  loop={loop_id}  round={rnd}")
    if no_pygame:
        _info("--no-pygame set; layouts will be recorded to recommended_layouts.json only.")

    pbar = tqdm(layouts, desc=f"layouts r{rnd}", unit="demo", dynamic_ncols=True)
    for i, layout in enumerate(pbar, 1):
        pbar.set_postfix(
            corridor=layout["corridor"],
            start=layout["start_pos"],
            goal=layout["goal_pos"],
            rep=f"{layout['repetition']}/{layout['n_repetitions']}",
        )
        launch_play_maze_for_layout(
            layout, profile_name, loop_id, rnd, i,
            demo_dir=profile_demo_dir_for_round, no_pygame=no_pygame,
        )
        n_collected = count_demos_in(profile_demo_dir_for_round)
        _info(f"after layout {i}/{len(layouts)}: demos in this round dir = {n_collected}")

    if not no_prompt:
        input(
            f"→ Recorded all {len(layouts)} layout(s)? "
            "Press ENTER to retrain on (baseline + accumulated profile demos) and continue.\n"
        )
    return layouts


# ---------------------------------------------------------------------------
# Learning curve.
# ---------------------------------------------------------------------------
def save_loop_graph(profile_log_path: Path, output_path: Path, profile_name: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not profile_log_path.exists():
        print(f"[active_loop] No log at {profile_log_path}; skipping graph.")
        return
    with open(profile_log_path, "r") as f:
        records = json.load(f)
    if not isinstance(records, list) or not records:
        print(f"[active_loop] Log empty; skipping graph.")
        return

    rounds = [int(r.get("round", i + 1)) for i, r in enumerate(records)]
    sr     = [float(r.get("success_rate", 0.0)) for r in records]
    target = next((float(r["target_sr"]) for r in records if "target_sr" in r), None)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rounds, sr, marker="o", linewidth=2, color="#1f77b4", label="Policy success rate")
    if target is not None:
        ax.axhline(y=target, linestyle="--", color="#d62728", alpha=0.7, label=f"Target SR = {target:.2f}")
    ax.set_xlabel("Round")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Active Loop Learning Curve — {profile_name}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    profile_name, profile_yaml = resolve_profile(args.profile)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    loop_id   = f"loop_{timestamp}"

    profile_root = ACTIVE_LOOP_ROOT / profile_name
    loop_root    = profile_root / loop_id
    loop_root.mkdir(parents=True, exist_ok=True)

    profile_log_path = profile_root / "loop_log.json"
    learning_curve   = profile_root / "learning_curve.png"

    profile_demo_dir = profile_loop_demos_dir(profile_name, loop_id)
    profile_demo_dir.mkdir(parents=True, exist_ok=True)
    demo_paths_arg = build_demo_paths(profile_name, loop_id)

    # Restore baseline checkpoint so this profile starts from the same weights
    # every other profile started from. The user is expected to have pinned
    # a baseline.pth themselves once, after the initial baseline training run.
    baseline_path = Path(args.baseline_checkpoint)
    if not args.skip_baseline_restore:
        restore_baseline_checkpoint(baseline_path)
    else:
        print("[active_loop] --skip-baseline-restore set; using current checkpoints/ as-is.")

    baseline_demos = count_baseline_demos()
    manifest = {
        "loop_id":       loop_id,
        "profile":       profile_name,
        "profile_yaml":  str(profile_yaml),
        "timestamp":     datetime.now().isoformat(),
        "cli": {
            "config":             args.config,
            "rounds":             args.rounds,
            "target_sr":          args.target_sr,
            "n_episodes":         args.n_episodes,
            "seed":               args.seed,
            "baseline_checkpoint":     str(baseline_path),
            "skip_baseline_restore":   bool(args.skip_baseline_restore),
            "skip_train_first_round":  bool(args.skip_train_first_round),
            "no_pygame":               bool(args.no_pygame),
            "no_demo_prompt":          bool(args.no_demo_prompt),
        },
        "demo_paths_for_training": demo_paths_arg,
        "profile_demo_dir":        str(profile_demo_dir),
        "baseline_demo_count":     baseline_demos,
    }
    with open(loop_root / "loop_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    _section(f"ACTIVE LOOP START — profile={profile_name}", char="=")
    _info(f"profile          : {profile_name}")
    _info(f"yaml             : {profile_yaml}")
    _info(f"loop dir         : {loop_root}")
    _info(f"profile demos    : {profile_demo_dir}")
    _info(f"training globs   : {demo_paths_arg}")
    _info(f"baseline demos   : {baseline_demos}")
    _info(f"baseline ckpt    : {baseline_path} (restored={not args.skip_baseline_restore})")
    _info(f"target SR / max  : {args.target_sr:.2f} / {args.rounds} rounds")
    _info(f"pygame launches  : {'OFF (--no-pygame)' if args.no_pygame else 'ON'}")
    _info(f"prompts          : {'OFF (--no-demo-prompt)' if args.no_demo_prompt else 'ON'}")

    print_table_header()
    cumulative_regret = 0.0
    rounds_to_target: Optional[int] = None
    last_metrics: Optional[Dict] = None
    rounds_completed = 0
    loop_t0 = time.time()

    rounds_pbar = tqdm(
        range(1, args.rounds + 1),
        desc=f"loop {profile_name}",
        unit="round",
        dynamic_ncols=True,
    )
    for rnd in rounds_pbar:
        rounds_completed = rnd
        round_t0 = time.time()
        _section(f"ROUND {rnd}/{args.rounds} — profile={profile_name}", char="=")
        _info(f"loop_id={loop_id}  loop_elapsed={(time.time()-loop_t0)/60:.1f}m")
        round_start_demos = count_demos_in(profile_demo_dir)
        _info(f"profile demos so far: {round_start_demos}")
        round_dir = loop_root / f"round_{rnd}"
        round_dir.mkdir(parents=True, exist_ok=True)

        if rnd == 1 and args.skip_train_first_round:
            _info("Skipping training before round 1 (using restored baseline weights).")
            train_loss = float("nan")
        else:
            # Always train on the FULL aggregated dataset for this profile —
            # baseline + every demo this profile collected so far. This is the
            # catastrophic-forgetting defense the user explicitly asked for.
            train_stdout = run_train(resume=(rnd > 1), demo_paths=demo_paths_arg)
            train_loss = parse_train_loss(train_stdout)
            _info(f"parsed train_loss = {train_loss}")

        full_output_path = run_eval(
            profile_yaml=profile_yaml,
            base_config=args.config,
            round_dir=round_dir,
            n_episodes=args.n_episodes,
            seed=args.seed,
        )

        metrics = compute_metrics(
            full_output_path, train_loss, cumulative_regret, round_start_demos,
            profile_demo_dir=profile_demo_dir,
        )
        cumulative_regret = metrics["cumulative_regret"]
        last_metrics = metrics

        save_round_artefacts(round_dir, metrics, full_output_path)
        print_table_row(rnd, metrics)
        _info(
            f"round {rnd} took {time.time()-round_t0:.1f}s | "
            f"SR={metrics['success_rate']:.2f} | "
            f"failures={metrics['n_failures']} | "
            f"profile_demos={metrics['total_profile_demos']} (+{metrics['demos_added']})"
        )
        rounds_pbar.set_postfix(
            sr=f"{metrics['success_rate']:.2f}",
            target=f"{args.target_sr:.2f}",
            demos=metrics["total_profile_demos"],
            cum_regret=f"{metrics['cumulative_regret']:.2f}",
        )

        record = {
            "round":     rnd,
            "loop_id":   loop_id,
            "profile":   profile_name,
            "target_sr": args.target_sr,
            "round_dir": str(round_dir),
            **metrics,
        }
        append_loop_log(profile_log_path, record)
        print_prescriptions(rnd, metrics.get("prescriptions", []))

        if metrics["success_rate"] >= args.target_sr:
            rounds_to_target = rnd
            _section(
                f"TARGET REACHED — SR={metrics['success_rate']:.2f} ≥ {args.target_sr:.2f} "
                f"at round {rnd}", char="*",
            )
            break

        # Per-round demo collection: pygame is launched once per recommended
        # layout-repetition, with --start/--goal/--fires forcing the layout.
        round_demo_dir = profile_demo_dir / f"round_{rnd}"
        run_demo_collection_for_round(
            prescriptions=metrics.get("prescriptions", []),
            profile_name=profile_name,
            loop_id=loop_id,
            rnd=rnd,
            round_dir=round_dir,
            profile_demo_dir_for_round=round_demo_dir,
            no_pygame=args.no_pygame,
            no_prompt=args.no_demo_prompt,
        )

    save_loop_graph(profile_log_path, learning_curve, profile_name)

    end_profile_demos = count_demos_in(profile_demo_dir)
    summary = {
        "profile":              profile_name,
        "loop_id":              loop_id,
        "rounds_completed":     rounds_completed,
        "rounds_to_target":     rounds_to_target,
        "target_sr":            args.target_sr,
        "final_success_rate":   last_metrics["success_rate"] if last_metrics else None,
        "baseline_demo_count":  baseline_demos,
        "profile_demos_collected": end_profile_demos,
        "training_dataset_size_final": baseline_demos + end_profile_demos,
        "loop_dir":             str(loop_root),
        "profile_demo_dir":     str(profile_demo_dir),
        "loop_log":             str(profile_log_path),
        "learning_curve":       str(learning_curve),
    }
    with open(loop_root / "loop_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    _section("LOOP SUMMARY", char="=")
    for k, v in summary.items():
        print(f"  {k:<28}: {v}")
    _info(f"total wall time: {(time.time()-loop_t0)/60:.1f}m")
    _info(f"per-round artefacts: {loop_root}")
    _info(f"per-profile log:     {profile_log_path}")
    _info(f"learning curve:      {learning_curve}")


if __name__ == "__main__":
    main()
