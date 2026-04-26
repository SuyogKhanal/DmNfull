import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, pstdev

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS   = REPO_ROOT / "scripts"
RESULTS   = REPO_ROOT / "results"
RUNS_DIR  = RESULTS / "runs"
DEMOS_DIR = REPO_ROOT / "demos"
LOG_FILE  = RESULTS / "active_loop_log.json"


def parse_args():
    p = argparse.ArgumentParser(description="Active loop: train -> eval -> LLM analysis -> demo collection.")
    p.add_argument("--config",     type=str,   default="configs/experiment_config.yaml")
    p.add_argument("--ablation",   type=str,   default=None)
    p.add_argument("--tag",        type=str,   default=None)
    p.add_argument("--rounds",     type=int,   default=10)
    p.add_argument("--target-sr",  type=float, default=0.90, dest="target_sr")
    p.add_argument("--n_episodes", type=int,   default=None)
    p.add_argument("--seed",       type=int,   default=None)
    return p.parse_args()


def _stream_subprocess(cmd):
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(REPO_ROOT),
    )
    captured = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        captured.append(line)
    rc = proc.wait()
    return rc, "".join(captured)


def run_train(resume: bool) -> str:
    cmd = [sys.executable, str(SCRIPTS / "train_diffusion.py")]
    if resume:
        cmd.append("--resume")
    print(f"\n[active_loop] Training: {' '.join(cmd)}")
    rc, out = _stream_subprocess(cmd)
    if rc != 0:
        raise RuntimeError(f"train_diffusion.py exited with code {rc}")
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


def run_eval(config: str, ablation, tag, n_episodes, seed) -> Path:
    cmd = [sys.executable, str(SCRIPTS / "run_pipeline.py"), "--config", config]
    if ablation is not None:
        cmd += ["--ablation", ablation]
    if tag is not None:
        cmd += ["--tag", tag]
    if n_episodes is not None:
        cmd += ["--n_episodes", str(n_episodes)]
    if seed is not None:
        cmd += ["--seed", str(seed)]

    started_at = time.time()
    print(f"\n[active_loop] Eval: {' '.join(cmd)}")
    rc, _out = _stream_subprocess(cmd)
    if rc != 0:
        raise RuntimeError(f"run_pipeline.py exited with code {rc}")

    if not RUNS_DIR.exists():
        raise RuntimeError(f"Runs directory not found: {RUNS_DIR}")

    candidates = []
    for d in RUNS_DIR.iterdir():
        if not d.is_dir():
            continue
        fp = d / "full_output.json"
        if not fp.exists():
            continue
        if fp.stat().st_mtime >= started_at:
            candidates.append((fp.stat().st_mtime, fp))
    if not candidates:
        raise RuntimeError(f"No new full_output.json found under {RUNS_DIR} after {started_at}")
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def count_demos() -> int:
    if not DEMOS_DIR.exists():
        return 0
    return len(list(DEMOS_DIR.glob("*.json")))


def _manhattan(a, b) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def compute_metrics(full_output_path: Path, train_loss: float, prev_cumulative_regret: float, round_start_demos: int) -> dict:
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
            key_frames = rollout.get("key_frames", []) or []
            for kf in key_frames:
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

    demos_now   = count_demos()
    demos_added = demos_now - round_start_demos

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
        "total_demos":         int(demos_now),
        "prescriptions":       prescriptions,
    }


def print_table_header():
    header = (
        f"{'Round':>5} | {'SR':>5} | {'FR':>5} | {'AvgLen':>6} | "
        f"{'MeanRew±Std':>14} | {'Loss':>8} | {'CumRegret':>9} | "
        f"{'CovGap':>6} | {'FireProx':>8} | {'DemosAdded':>10}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)


def print_table_row(rnd: int, m: dict):
    loss = m.get("train_loss", float("nan"))
    loss_str = "nan" if loss != loss else f"{loss:.4f}"
    row = (
        f"{rnd:>5d} | "
        f"{m['success_rate']:>5.2f} | "
        f"{m['failure_rate']:>5.2f} | "
        f"{m['mean_episode_length']:>6.1f} | "
        f"{m['mean_reward']:>6.2f}±{m['std_reward']:<6.2f} | "
        f"{loss_str:>8} | "
        f"{m['cumulative_regret']:>9.3f} | "
        f"{m['demo_coverage_gap']:>6.2f} | "
        f"{m['fire_proximity_rate']:>8.2f} | "
        f"{m['demos_added']:>10d}"
    )
    print(row)


def append_log(record: dict):
    RESULTS.mkdir(parents=True, exist_ok=True)
    history = []
    if LOG_FILE.exists():
        try:
            with open(LOG_FILE, "r") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    history = loaded
        except Exception:
            history = []
    history.append(record)
    with open(LOG_FILE, "w") as f:
        json.dump(history, f, indent=2, default=str)


def announce_run_dir(run_dir: Path):
    print(f"\n[active_loop] Run directory: {run_dir}")
    print(f"[active_loop] Launch `python dashboard/app.py` and load this path in the Gradio dashboard to inspect the run before recording demos.")


def run_demo_collection():
    cmd = [sys.executable, str(SCRIPTS / "play_maze.py")]
    print(f"\n[active_loop] Launching demo collection: {' '.join(cmd)}")
    print("[active_loop] Record corrective demonstrations, then close play_maze.py to return here.")
    subprocess.run(cmd, cwd=str(REPO_ROOT))
    input("→ Press ENTER when done recording demonstrations to continue.\n")


def save_loop_graph(log_path: Path, output_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not log_path.exists():
        print(f"[active_loop] No log file at {log_path}; skipping graph.")
        return

    with open(log_path, "r") as f:
        records = json.load(f)
    if not isinstance(records, list) or not records:
        print(f"[active_loop] Log empty or malformed; skipping graph.")
        return

    rounds = [int(r.get("round", i + 1)) for i, r in enumerate(records)]
    sr     = [float(r.get("success_rate", 0.0)) for r in records]
    target = None
    for r in records:
        if "target_sr" in r:
            target = float(r["target_sr"])
            break

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rounds, sr, marker="o", linewidth=2, color="#1f77b4", label="Policy success rate")
    if target is not None:
        ax.axhline(y=target, linestyle="--", color="#d62728", alpha=0.7, label=f"Target SR = {target:.2f}")
    ax.set_xlabel("Round")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Active Loop Learning Curve")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    cumulative_regret = 0.0
    print_table_header()

    for rnd in range(1, args.rounds + 1):
        print(f"\n{'='*80}\n[active_loop] Round {rnd}/{args.rounds}\n{'='*80}")
        round_start_demos = count_demos()

        train_stdout = run_train(resume=(rnd > 1))
        train_loss = parse_train_loss(train_stdout)

        full_output_path = run_eval(
            config=args.config,
            ablation=args.ablation,
            tag=args.tag,
            n_episodes=args.n_episodes,
            seed=args.seed,
        )
        run_dir = full_output_path.parent

        metrics = compute_metrics(full_output_path, train_loss, cumulative_regret, round_start_demos)
        cumulative_regret = metrics["cumulative_regret"]

        print_table_row(rnd, metrics)

        record = {"round": rnd, "target_sr": args.target_sr, **metrics}
        append_log(record)

        prescriptions = metrics.get("prescriptions", []) or []
        if prescriptions:
            print(f"\n[active_loop] Prescriptions for round {rnd}:")
            for i, pres in enumerate(prescriptions, 1):
                demo_id = pres.get("demo_id", i)
                guidance = pres.get("guidance", "(no guidance)")
                region   = pres.get("target_region", "(unspecified region)")
                teaches  = pres.get("what_it_teaches", "")
                reps     = pres.get("n_repetitions", 1)
                print(f"  [{demo_id}] region={region} | repetitions={reps}")
                print(f"       guidance : {guidance}")
                if teaches:
                    print(f"       teaches  : {teaches}")
        else:
            print(f"\n[active_loop] No prescriptions produced this round.")

        if metrics["success_rate"] >= args.target_sr:
            print(f"\n[active_loop] Target success rate {args.target_sr:.2f} reached at round {rnd} (SR={metrics['success_rate']:.2f}). Stopping.")
            break

        announce_run_dir(run_dir)
        run_demo_collection()

    graph_path = RESULTS / "active_loop_curve.png"
    save_loop_graph(LOG_FILE, graph_path)
    print(f"\n[active_loop] Learning curve saved to: {graph_path}")


if __name__ == "__main__":
    main()