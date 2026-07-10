"""Diff-DAgger on robosuite UR5e — CLI entrypoint.

Modes:
  expert-check : run the scripted expert oracle and report success rate
  bc           : behavior-cloning baseline (collect, train once, eval)
  diffdagger   : full Diff-DAgger interactive imitation-learning loop

Examples:
  python -m diffdagger_rs.run --task Lift --mode expert-check --smoke
  python -m diffdagger_rs.run --task Lift --mode diffdagger --smoke
  python -m diffdagger_rs.run --task Wipe --mode diffdagger --output-dir results/wipe
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch


def set_seed(seed: int):
    """Seed every RNG that affects the run so a given --seed is reproducible:
    weight init, training noise, inference/uncertainty noise, threshold
    calibration, and DataLoader shuffling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

from .collect import collect_demos, rollout_expert
from .config import get_config
from .dataset import DemoDataset
from .diffdagger import (build_policy, calibrate_ni, run_diffdagger,
                         train_and_calibrate)
from .envs import make_env
from .eval import evaluate_policy
from .experts import make_expert


def _shared_bootstrap(args, cfg, make_env_fn, make_expert_fn, log):
    """Byte-identical shared bootstrap for a fair comparison: the first arm to run
    collects N expert demos (seed_base=0) and pickles them to a canonical path
    keyed by (task, Ni); every other arm/seed LOADS that exact file. This removes
    the tiny collection nondeterminism (global-RNG-dependent robot init noise) so
    all arms start from literally the same demos. Data is seed-independent
    (seed_base=0), so the key omits the experiment seed."""
    import pickle
    n = int(args.num_init_demos)
    os.makedirs(args.bootstrap_dir, exist_ok=True)
    path = os.path.join(args.bootstrap_dir, f"{cfg['task']}_ni{n}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            trajs = pickle.load(f)
        log(f"[bootstrap] loaded {len(trajs)} shared demos from {path}")
        return trajs
    trajs = collect_demos(make_env_fn(cfg["env_horizon"]), make_expert_fn(), n,
                          seed_base=0, max_steps=cfg["env_horizon"], log_fn=log)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(trajs, f)
    os.replace(tmp, path)                 # atomic publish (no torn read on races)
    log(f"[bootstrap] collected + saved {len(trajs)} shared demos to {path}")
    return trajs


def cmd_make_bootstrap(cfg, args, log, device):
    """Create the shared bootstrap file and exit (run this first, then the arms)."""
    assert args.bootstrap_dir and args.num_init_demos, "need --bootstrap-dir and --num-init-demos"
    make_env_fn = lambda h: make_env(cfg["task"], horizon=h, wipe_marker_obs_k=cfg.get("wipe_marker_obs_k", 0))
    trajs = _shared_bootstrap(args, cfg, make_env_fn, lambda: make_expert(cfg["task"]), log)
    return {"bootstrap_demos": len(trajs)}


def _logger(path):
    f = open(path, "a") if path else None

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if f:
            f.write(line + "\n"); f.flush()
    return log


def cmd_expert_check(cfg, args, log):
    env = make_env(cfg["task"], horizon=cfg["env_horizon"],
                   wipe_marker_obs_k=cfg.get("wipe_marker_obs_k", 0))
    expert = make_expert(cfg["task"])
    n = args.episodes or 10
    succ, lens, covs = 0, [], []
    for ep in range(n):
        _, _, success, t = rollout_expert(env, expert, seed=2000 + ep, max_steps=cfg["env_horizon"])
        succ += int(success); lens.append(t)
        if cfg["task"] == "Wipe":
            covs.append(len(env.raw.wiped_markers) / env.raw.num_markers)
        log(f"  expert ep{ep}: success={success} len={t}"
            + (f" coverage={covs[-1]:.2f}" if covs else ""))
    env.close()
    sr = succ / n
    log(f"[expert-check {cfg['task']}] success={sr:.3f} ({succ}/{n}) mean_len={np.mean(lens):.0f}"
        + (f" mean_coverage={np.mean(covs):.2f}" if covs else ""))
    return {"expert_success_rate": sr}


def cmd_bc(cfg, args, log, device):
    env_h = cfg["env_horizon"]
    ek = dict(wipe_marker_obs_k=cfg.get("wipe_marker_obs_k", 0))
    collect_env = make_env(cfg["task"], horizon=env_h, **ek)
    eval_env = make_env(cfg["task"], horizon=env_h, **ek)
    expert = make_expert(cfg["task"])
    trajs = collect_demos(collect_env, expert, cfg["final_demos"], seed_base=0,
                          max_steps=env_h, log_fn=log)
    policy, dataset = train_and_calibrate(cfg, trajs, device, cfg["initial_train_steps"], log)
    ev = evaluate_policy(policy, eval_env, num_episodes=cfg["eval_episodes"],
                         obs_horizon=cfg["obs_horizon"], act_horizon=cfg["act_horizon"],
                         device=device, base_seed=cfg["eval_seed_base"], max_steps=env_h)
    log(f"[BC {cfg['task']}] success={ev['success_rate']:.3f}")
    collect_env.close(); eval_env.close()
    return {"bc_success_rate": ev["success_rate"]}


def cmd_diffdagger(cfg, args, log, device):
    out = args.output_dir
    ckpt = os.path.join(out, "policy.pt") if out else None
    history = run_diffdagger(
        cfg,
        make_env_fn=lambda horizon: make_env(cfg["task"], horizon=horizon, wipe_marker_obs_k=cfg.get("wipe_marker_obs_k", 0)),
        make_expert_fn=lambda: make_expert(cfg["task"]),
        device=device, log_fn=log, ckpt_path=ckpt,
    )
    return {"history": history}


def cmd_bc_sweep(cfg, args, log, device):
    ni, curve, _pool = calibrate_ni(
        cfg,
        make_env_fn=lambda horizon: make_env(cfg["task"], horizon=horizon, wipe_marker_obs_k=cfg.get("wipe_marker_obs_k", 0)),
        make_expert_fn=lambda: make_expert(cfg["task"]),
        device=device, log_fn=log,
    )
    return {"recommended_ni": ni, "bc_sweep": [[n, s] for n, s in curve],
            "target_success": cfg["target_success"]}


def cmd_experiment(cfg, args, log, device):
    """Calibrate Ni to ~50% via a BC data-scaling sweep, then run Diff-DAgger
    from that Ni (reusing the swept demo pool as the initial dataset)."""
    out = args.output_dir
    ckpt = os.path.join(out, "policy.pt") if out else None
    make_env_fn = lambda horizon: make_env(cfg["task"], horizon=horizon, wipe_marker_obs_k=cfg.get("wipe_marker_obs_k", 0))
    make_expert_fn = lambda: make_expert(cfg["task"])

    # Byte-identical shared bootstrap (fair comparison): load/make the shared demos
    # and run Diff-DAgger from EXACTLY those (same file the P4 arms load).
    if args.bootstrap_dir and args.num_init_demos is not None:
        init_trajs = _shared_bootstrap(args, cfg, make_env_fn, make_expert_fn, log)
        cfg["num_init_demos"] = len(init_trajs)
        log(f"[experiment] shared bootstrap Ni={len(init_trajs)}; running Diff-DAgger")
        history = run_diffdagger(cfg, make_env_fn, make_expert_fn, device=device,
                                 log_fn=log, ckpt_path=ckpt, init_trajs=init_trajs)
        return {"recommended_ni": len(init_trajs), "bc_sweep": None, "history": history}

    # If the user pinned --num-init-demos, honor it and skip the (expensive) sweep.
    if args.num_init_demos is not None:
        log(f"[experiment] --num-init-demos={args.num_init_demos} given; "
            f"skipping BC sweep and collecting that many initial demos")
        history = run_diffdagger(
            cfg, make_env_fn, make_expert_fn, device=device, log_fn=log, ckpt_path=ckpt,
        )
        return {"recommended_ni": cfg["num_init_demos"], "bc_sweep": None,
                "history": history}

    ni, curve, pool = calibrate_ni(cfg, make_env_fn, make_expert_fn, device, log)
    cfg["num_init_demos"] = ni
    log(f"[experiment] starting Diff-DAgger from calibrated Ni={ni} "
        f"(round-0 BC ~= {dict(curve).get(ni):.2f})")
    history = run_diffdagger(
        cfg, make_env_fn, make_expert_fn, device=device, log_fn=log,
        ckpt_path=ckpt, init_trajs=pool[:ni],
    )
    return {"recommended_ni": ni, "bc_sweep": [[n, s] for n, s in curve],
            "history": history}


def cmd_p4hybrid(cfg, args, log, device):
    """P4-LLM V3 hybrid: calibrate Ni (shared bootstrap with Diff-DAgger), then run
    the hybrid loop (screen -> cluster -> SELECT/BRIDGE -> one demo/round)."""
    from .p4.loop import run_p4_hybrid
    out = args.output_dir
    ckpt = os.path.join(out, "policy.pt") if out else None
    make_env_fn = lambda horizon: make_env(cfg["task"], horizon=horizon, wipe_marker_obs_k=cfg.get("wipe_marker_obs_k", 0))
    make_expert_fn = lambda: make_expert(cfg["task"])
    # P4 adds ONE demo/round (nd=1) -> needs enough rounds to reach Nf (+ slack).
    # Only auto-bump when the user did NOT pin --max-rounds, so an explicit cap
    # (e.g. to bound the post-plateau screening grind) is honored.
    if args.max_rounds is None:
        cfg["max_rounds"] = max(cfg["max_rounds"], 3 * cfg["final_demos"])
    if args.bootstrap_dir and args.num_init_demos is not None:   # byte-identical shared bootstrap
        init = _shared_bootstrap(args, cfg, make_env_fn, make_expert_fn, log)
        ni, curve = len(init), None
    elif args.num_init_demos is not None:    # fixed-Ni bootstrap (skip the sweep)
        init = collect_demos(make_env_fn(cfg["env_horizon"]), make_expert_fn(),
                             args.num_init_demos, seed_base=0, max_steps=cfg["env_horizon"], log_fn=log)
        ni, curve = args.num_init_demos, None
    else:
        ni, curve, pool = calibrate_ni(cfg, make_env_fn, make_expert_fn, device, log)
        init = pool[:ni]
    llm = None
    use_vlm = args.use_vlm or bool(cfg.get("p4", {}).get("use_vlm"))
    if use_vlm:
        from .p4.llm import make_llm
        llm = make_llm()
        log(f"[p4] use_vlm=True -> VLM client "
            + ("READY" if llm else "UNAVAILABLE (OPENAI_BASE_URL unset); geometric fallback"))
    log(f"[p4] starting V3 hybrid from calibrated Ni={ni}")
    history = run_p4_hybrid(cfg, make_env_fn, make_expert_fn, device=device, log_fn=log,
                            ckpt_path=ckpt, init_trajs=init, work_dir=(out or "."), llm=llm)
    return {"recommended_ni": ni, "bc_sweep": ([[n, s] for n, s in curve] if curve else None),
            "history": history}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["Lift", "Wipe", "Door"])
    p.add_argument("--mode", default="experiment",
                   choices=["expert-check", "bc", "bc-sweep", "diffdagger", "experiment",
                            "p4_hybrid", "make-bootstrap"])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--use-vlm", action="store_true",
                   help="p4_hybrid: use the VLM advisor (needs OPENAI_BASE_URL); else geometric")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=None, help="for expert-check")
    p.add_argument("--output-dir", type=str, default=None)
    # selective overrides
    p.add_argument("--num-init-demos", type=int, default=None)
    p.add_argument("--bootstrap-dir", type=str, default=None,
                   help="shared byte-identical bootstrap dir (fair comparison; with --num-init-demos)")
    p.add_argument("--final-demos", type=int, default=None)
    p.add_argument("--max-rounds", type=int, default=None)
    p.add_argument("--initial-train-steps", type=int, default=None)
    p.add_argument("--round-train-steps", type=int, default=None)
    p.add_argument("--eval-episodes", type=int, default=None)
    args = p.parse_args()

    cfg = get_config(args.task, smoke=args.smoke)
    cfg["seed"] = args.seed
    for k in ["num_init_demos", "final_demos", "max_rounds", "initial_train_steps",
              "round_train_steps", "eval_episodes"]:
        v = getattr(args, k.replace("-", "_"))
        if v is not None:
            cfg[k] = v

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    log = _logger(os.path.join(args.output_dir, "run.log") if args.output_dir else None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True  # speed; non-bitwise-deterministic

    log(f"===== Diff-DAgger robosuite | task={args.task} mode={args.mode} "
        f"smoke={args.smoke} device={device} seed={args.seed} =====")
    log("config: " + json.dumps({k: v for k, v in cfg.items()}, default=str))

    t0 = time.time()
    if args.mode == "expert-check":
        result = cmd_expert_check(cfg, args, log)
    elif args.mode == "bc":
        result = cmd_bc(cfg, args, log, device)
    elif args.mode == "bc-sweep":
        result = cmd_bc_sweep(cfg, args, log, device)
    elif args.mode == "experiment":
        result = cmd_experiment(cfg, args, log, device)
    elif args.mode == "p4_hybrid":
        result = cmd_p4hybrid(cfg, args, log, device)
    elif args.mode == "make-bootstrap":
        result = cmd_make_bootstrap(cfg, args, log, device)
    else:
        result = cmd_diffdagger(cfg, args, log, device)
    result["wall_sec"] = round(time.time() - t0, 1)
    log(f"DONE in {result['wall_sec']}s")

    if args.output_dir:
        with open(os.path.join(args.output_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=2, default=str)


if __name__ == "__main__":
    main()
