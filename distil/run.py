"""DISTIL — the consolidated run entrypoint (09_..md acceptance-test CLI).

    python -m distil.run --task Lift --modality state --ablation full --seed 1 \
        --budget 20 --bootstrap-dir results/shared_bootstrap --output-dir results/...

Reads OpenRouter creds from `.env` ($DISTIL_ROOT/.env or the repo root), needs 1 GPU
(diffusion-policy train + rollout), and writes one deterministic leaf:

    <output-dir>/
        result.json     # curve + per-round history + resolved config metadata
        config.yaml      # the full resolved config (every flag)
        run.log
        kag/             # the EXACT KAG doc used (golden rule 3)
        prompts/         # the EXACT prompts used, per round (written by the LLM client)
        telemetry/       # per-round jsonl (clusters, descriptors, decision, confidence)

Budget counts SUCCESSFUL demos added on top of the byte-identical shared bootstrap
(09_..md): stop when the dataset reaches n_init + budget.

Also: `--make-bootstrap` builds the shared bootstrap for a (task, modality) cell and exits.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch


def _find_root() -> Path:
    """$DISTIL_ROOT if set, else the repo root (parent of this package)."""
    if os.environ.get("DISTIL_ROOT"):
        return Path(os.environ["DISTIL_ROOT"]).resolve()
    return Path(__file__).resolve().parent.parent


def _load_dotenv(root: Path, log=print) -> None:
    """Load KEY=VALUE from <root>/.env (and $DISTIL_ROOT/.env) without clobbering
    already-set env vars. No hardcoded /weka path (golden rule 8)."""
    for env_path in {root / ".env", Path(__file__).resolve().parent / ".env"}:
        if not env_path.is_file():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            if k and k not in os.environ:
                os.environ[k] = v.strip().strip('"').strip("'")
        log(f"[env] loaded {env_path}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _logger(path):
    f = open(path, "a") if path else None

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if f:
            f.write(line + "\n"); f.flush()
    return log


def _bootstrap(task, modality, n, bootstrap_dir, make_env_fn, make_expert_fn, env_h, log):
    """Byte-identical shared bootstrap per (task, modality): the first arm collects
    n demos (seed_base=0, deterministic) and pickles them; every arm/seed of the cell
    loads that exact file (09_..md)."""
    from .collect import collect_demos
    os.makedirs(bootstrap_dir, exist_ok=True)
    path = Path(bootstrap_dir) / f"{task}_{modality}_ni{n}.pkl"
    if path.is_file():
        trajs = pickle.loads(path.read_bytes())
        log(f"[bootstrap] loaded {len(trajs)} shared demos from {path}")
        return trajs
    trajs = collect_demos(make_env_fn(env_h), make_expert_fn(), n, seed_base=0,
                          max_steps=env_h, log_fn=log)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_bytes(pickle.dumps(trajs))
    os.replace(tmp, path)             # atomic publish (no torn read on races)
    log(f"[bootstrap] collected + saved {len(trajs)} shared demos to {path}")
    return trajs


def _copy_kag(task, out_dir, log):
    """Copy the EXACT KAG json + rendered text into the run's kag/ (golden rule 3)."""
    from .p4.kag import load_kag_graph, format_kag_context
    src = Path(__file__).resolve().parent / "p4" / "kag" / f"{task}.json"
    dst_dir = Path(out_dir) / "kag"
    dst_dir.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        shutil.copy2(src, dst_dir / f"{task}.json")
        (dst_dir / f"{task}.kag.txt").write_text(format_kag_context(load_kag_graph(task)))
        log(f"[kag] copied {task}.json + rendered .kag.txt into {dst_dir}")
    else:
        log(f"[kag] WARNING: no KAG graph for {task} at {src}")


def _write_config_yaml(cfg, out_dir):
    try:
        import yaml
        with open(Path(out_dir) / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=True, default_flow_style=False)
    except Exception:
        with open(Path(out_dir) / "config.yaml", "w") as f:
            f.write(json.dumps(cfg, indent=2, default=str))


def _gw_bootstrap(n, bootstrap_dir, log):
    """Byte-identical shared GridWorld bootstrap (deterministic BFS demos, pickled)."""
    from .gridworld.loop import bootstrap_demos
    os.makedirs(bootstrap_dir, exist_ok=True)
    path = Path(bootstrap_dir) / f"GridWorld_state_ni{n}.pkl"
    if path.is_file():
        demos = pickle.loads(path.read_bytes())
        log(f"[bootstrap] loaded {len(demos)} shared GW demos from {path}")
        return demos
    demos = bootstrap_demos(n, log, seed_base=0)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_bytes(pickle.dumps(demos)); os.replace(tmp, path)
    log(f"[bootstrap] saved {len(demos)} shared GW demos to {path}")
    return demos


def _run_baseline(cfg, args, out, device, n_init, make_env_fn, make_expert_fn, log):
    env_h = cfg["env_horizon"]
    init = _bootstrap(cfg["task"], cfg["modality"], n_init, args.bootstrap_dir,
                      make_env_fn, make_expert_fn, env_h, log)
    if args.make_bootstrap:
        log(f"[make-bootstrap] done: {len(init)} demos")
        return
    cfg["num_init_demos"] = len(init)
    cfg["final_demos"] = len(init) + int(args.budget)
    if args.max_rounds is None:
        cfg["max_rounds"] = max(cfg["max_rounds"], 3 * int(args.budget) + 2)
    if out:
        _write_config_yaml(cfg, out)
    from .baselines import run_baseline
    t0 = time.time()
    result = run_baseline(cfg, args.ablation, make_env_fn, make_expert_fn, device,
                          log_fn=log, init_trajs=init)
    result["wall_sec"] = round(time.time() - t0, 1)
    result["task"], result["modality"] = cfg["task"], cfg["modality"]
    result["ablation"], result["seed"] = cfg["ablation"], cfg["seed"]
    log(f"DONE in {result['wall_sec']}s | final_success={result.get('final_success')}")
    if out:
        with open(os.path.join(out, "result.json"), "w") as f:
            json.dump(result, f, indent=2, default=str)
        log(f"[write] {os.path.join(out, 'result.json')}")


def _run_gridworld(cfg, args, out, device, n_init, log):
    init = _gw_bootstrap(n_init, args.bootstrap_dir, log)
    if args.make_bootstrap:
        log(f"[make-bootstrap] done: {len(init)} GW demos")
        return
    # enough rounds to actually SPEND the budget (1 demo/round + slack for budget-free
    # rounds) — else the BASE max_rounds=12 caps GridWorld before the budget is spent.
    if args.max_rounds is None:
        cfg["max_rounds"] = max(cfg["max_rounds"], 3 * int(args.budget) + 2)
    if out:
        _copy_kag("GridWorld", out, log)
        _write_config_yaml(cfg, out)
    from .gridworld.loop import run_distil_gridworld
    t0 = time.time()
    result = run_distil_gridworld(cfg, device, log_fn=log, init_demos=init, work_dir=(out or "."))
    result["wall_sec"] = round(time.time() - t0, 1)
    result["task"], result["modality"] = cfg["task"], cfg["modality"]
    result["ablation"], result["seed"] = cfg["ablation"], cfg["seed"]
    log(f"DONE in {result['wall_sec']}s | final_success={result.get('final_success')}")
    if out:
        with open(os.path.join(out, "result.json"), "w") as f:
            json.dump(result, f, indent=2, default=str)
        log(f"[write] {os.path.join(out, 'result.json')}")


def main():
    p = argparse.ArgumentParser(description="DISTIL consolidated runner")
    p.add_argument("--task", required=True, choices=["Lift", "Wipe", "Door", "GridWorld"],
                   help="robot tasks (robosuite) + GridWorld (equivariant classifier). PushT = Phase 2.")
    p.add_argument("--modality", default="state", choices=["state", "image"])
    p.add_argument("--ablation", default="full")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--budget", type=int, default=20,
                   help="successful demos added on top of the bootstrap (stop rule)")
    p.add_argument("--num-init-demos", type=int, default=None,
                   help="bootstrap size Ni (EXCLUDED from budget); default = per-task config Ni "
                        "(Lift 8, Wipe 12, Door 4, GridWorld 20)")
    p.add_argument("--bootstrap-dir", type=str, default="results/shared_bootstrap")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--make-bootstrap", action="store_true",
                   help="build the shared bootstrap for this (task,modality) cell and exit")
    p.add_argument("--no-llm", action="store_true",
                   help="debug: skip the LLM entirely (geometric planner only)")
    p.add_argument("--max-rounds", type=int, default=None)
    args = p.parse_args()

    root = _find_root()
    out = args.output_dir
    if out:
        os.makedirs(out, exist_ok=True)
    log = _logger(os.path.join(out, "run.log") if out else None)
    _load_dotenv(root, log)

    from .config import get_config
    cfg = get_config(args.task, modality=args.modality, ablation=args.ablation,
                     budget=args.budget, smoke=args.smoke)
    cfg["seed"] = args.seed
    if args.max_rounds is not None:
        cfg["max_rounds"] = args.max_rounds
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    log(f"===== DISTIL | task={args.task} modality={args.modality} "
        f"ablation={args.ablation} seed={args.seed} budget={args.budget} "
        f"smoke={args.smoke} device={device} =====")

    # Ni: explicit --num-init-demos wins; else the per-task config default (Lift 8,
    # Wipe 12, Door 4, GridWorld 20; smoke override = 3). EXCLUDED from budget.
    n_init = args.num_init_demos if args.num_init_demos is not None else cfg["num_init_demos"]

    # ── GridWorld branch (equivariant classifier; no robosuite import) ────────
    if cfg["task"] == "GridWorld":
        _run_gridworld(cfg, args, out, device, n_init, log)
        return

    from .envs import make_env
    from .experts import make_expert
    from .config import BASELINE_ARMS
    env_h = cfg["env_horizon"]
    ek = dict(wipe_marker_obs_k=cfg.get("wipe_marker_obs_k", 0))
    make_env_fn = lambda h: make_env(cfg["task"], horizon=h, **ek)
    make_expert_fn = lambda: make_expert(cfg["task"])

    # baseline arm (Safe/Dropout/Ensemble/Thrifty/Stagger/Diff-DAgger) — robot diffusion only
    if args.ablation in BASELINE_ARMS:
        _run_baseline(cfg, args, out, device, n_init, make_env_fn, make_expert_fn, log)
        return

    # ── make-bootstrap mode ──────────────────────────────────────────────────
    if args.make_bootstrap:
        trajs = _bootstrap(cfg["task"], cfg["modality"], n_init,
                           args.bootstrap_dir, make_env_fn, make_expert_fn, env_h, log)
        log(f"[make-bootstrap] done: {len(trajs)} demos")
        return

    # ── full DISTIL run ──────────────────────────────────────────────────────
    init = _bootstrap(cfg["task"], cfg["modality"], n_init,
                      args.bootstrap_dir, make_env_fn, make_expert_fn, env_h, log)
    cfg["num_init_demos"] = len(init)
    # budget = successful demos ADDED on top of the bootstrap (09_..md).
    cfg["final_demos"] = len(init) + int(args.budget)
    # enough rounds to spend the budget (1 demo/round) + slack for budget-free rounds.
    if args.max_rounds is None:
        cfg["max_rounds"] = max(cfg["max_rounds"], 3 * int(args.budget) + 2)

    # LLM: full DISTIL uses the multi-stage OpenRouter client; decision=heuristic /
    # fallback_only / --no-llm degrade to the geometric planner.
    p4f = cfg["p4"]
    use_llm = ((p4f.get("decision") == "llm") and not p4f.get("fallback_only")
               and p4f.get("allocation") != "random" and not args.no_llm)
    llm = None
    if use_llm:
        from .p4.llm import make_llm
        llm = make_llm(use_vlm=bool(p4f.get("vlm", True)), use_kag=bool(p4f.get("kag", True)))
        log(f"[llm] OpenRouter client " + ("READY" if llm else
            "UNAVAILABLE (no OPENROUTER creds); geometric fallback")
            + f" | vlm={p4f.get('vlm')} kag={p4f.get('kag')}")

    if out:
        _copy_kag(cfg["task"], out, log)
        _write_config_yaml(cfg, out)

    from .p4.loop import run_distil
    ckpt = os.path.join(out, "policy.pt") if out else None
    t0 = time.time()
    result = run_distil(cfg, make_env_fn, make_expert_fn, device=device, log_fn=log,
                        ckpt_path=ckpt, init_trajs=init, work_dir=(out or "."), llm=llm)
    result["wall_sec"] = round(time.time() - t0, 1)
    result["task"], result["modality"] = cfg["task"], cfg["modality"]
    result["ablation"], result["seed"] = cfg["ablation"], cfg["seed"]
    log(f"DONE in {result['wall_sec']}s | final_success={result.get('final_success')}")

    if out:
        with open(os.path.join(out, "result.json"), "w") as f:
            json.dump(result, f, indent=2, default=str)
        log(f"[write] {os.path.join(out, 'result.json')}")


if __name__ == "__main__":
    main()
