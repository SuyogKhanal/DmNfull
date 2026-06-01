"""Per-job hyperparameter-sweep driver.

Reads the full grid (``grid.json``), takes this job's chunk of configs, and for
every ``(config x run)`` pair launches a ``replay_p4`` worker IN PARALLEL
(bounded concurrency; CPU threads divided across workers). Each worker retrains
P4-LLM's already-prescribed demos (read from the main ``results/`` tree, no LLM)
at that config's hyperparameters and writes the learning curve to
``<out_root>/<config_tag>/run_<R>/<dest_method>/``.

Only P4 is run — the baselines are the fixed production@90 reference, compared
against later by ``sweep_rank.py``.

grid.json schema: a JSON list of config dicts, each:
  {"tag": "...", "epochs": int, "lr": float, "batch_size": int,
   "weight_decay": float, "replay_mix": float, "head_dropout": float}
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

SUITE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]


def _info(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [sweep_hp] {msg}", flush=True)


def _worker_cmd(cfg: Dict, run_id: int, config_yaml: str, out_root: Path,
                source_method: str, dest_method: str) -> List[str]:
    return [
        sys.executable, "-u", "-m",
        "Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4."
        "pool_x_selector.orchestrator.replay_p4",
        "--run_id", str(int(run_id)),
        "--config", str(config_yaml),
        "--epochs", str(int(cfg["epochs"])),
        "--lr", str(float(cfg["lr"])),
        "--batch_size", str(int(cfg.get("batch_size", 64))),
        "--weight_decay", str(float(cfg.get("weight_decay", 1e-4))),
        "--replay_mix", str(float(cfg.get("replay_mix", 0.5))),
        "--head_dropout", str(float(cfg.get("head_dropout", 0.0))),
        "--source_method", source_method,
        "--dest_method", dest_method,
        "--out_root", str(out_root / cfg["tag"]),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=str, required=True, help="Path to grid.json")
    ap.add_argument("--chunk_index", type=int, default=0)
    ap.add_argument("--chunk_size", type=int, default=1)
    ap.add_argument("--runs", type=str, default="1 6", help="Space-sep run ids.")
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--config", type=str, required=True,
                    help="Base config yaml (trainer fallbacks / seed / max_steps).")
    ap.add_argument("--source_method", type=str, default="p4_top3_rotate")
    ap.add_argument("--dest_method", type=str, default="p4_top3_rotate")
    ap.add_argument("--max_concurrency", type=int, default=4)
    args = ap.parse_args()

    grid: List[Dict] = json.load(open(args.grid))
    lo = args.chunk_index * args.chunk_size
    hi = lo + args.chunk_size
    configs = grid[lo:hi]
    runs = [int(x) for x in args.runs.split()]
    out_root = Path(args.out_root)
    log_dir = SUITE_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Build the full (config x run) work list.
    work = [(c, r) for c in configs for r in runs]
    if not work:
        _info(f"chunk {args.chunk_index}: no configs in [{lo}:{hi}] of {len(grid)} — nothing to do")
        return

    ncpus = int(os.environ.get("SLURM_CPUS_PER_TASK")
                or os.environ.get("SLURM_CPUS_ON_NODE") or (os.cpu_count() or 4))
    cap = max(1, int(args.max_concurrency))
    per = max(1, ncpus // cap)
    _info(f"chunk {args.chunk_index}: {len(configs)} configs x {len(runs)} runs = "
          f"{len(work)} replays; concurrency={cap}; {per} threads/worker")

    running = []  # (proc, logf, label)
    results = []  # (label, rc)

    def _launch(cfg, run_id):
        label = f"{cfg['tag']}/run{run_id}"
        env = dict(os.environ)
        env.update(OMP_NUM_THREADS=str(per), MKL_NUM_THREADS=str(per),
                   OPENBLAS_NUM_THREADS=str(per), NUMEXPR_NUM_THREADS=str(per))
        cmd = _worker_cmd(cfg, run_id, args.config, out_root,
                          args.source_method, args.dest_method)
        logf = open(log_dir / f"sweep_{cfg['tag']}_run{run_id:02d}.log", "w")
        _info(f"launch {label} -> {logf.name}")
        p = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env,
                             stdout=logf, stderr=subprocess.STDOUT)
        running.append((p, logf, label))

    idx = 0
    while idx < len(work) or running:
        while idx < len(work) and len(running) < cap:
            cfg, run_id = work[idx]; idx += 1
            _launch(cfg, run_id)
        # reap finished
        still = []
        for p, logf, label in running:
            rc = p.poll()
            if rc is None:
                still.append((p, logf, label))
            else:
                logf.close(); results.append((label, rc))
                _info(f"done {label} rc={rc}  ({len(results)}/{len(work)})")
        running = still
        if idx < len(work) or running:
            time.sleep(5)

    fails = [l for l, rc in results if rc != 0]
    _info(f"chunk {args.chunk_index} DONE: {len(results)} replays, "
          f"{len(fails)} nonzero rc{(' -> ' + ', '.join(fails)) if fails else ''}")


if __name__ == "__main__":
    main()
