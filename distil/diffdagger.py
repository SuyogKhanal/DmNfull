"""Diff-DAgger interactive imitation-learning loop (Algorithm 1, adapted).

  1. collect Ni initial expert demonstrations
  2. each round: retrain a fresh diffusion policy on the aggregated data, then
     calibrate the OOD threshold from the diffusion loss over the training set
  3. roll out the policy with receding-horizon chunking; at every step estimate
     the diffusion-loss uncertainty of the current plan and query the expert
     when the loss exceeds the alpha-quantile threshold for K steps (patience)
  4. on a query (or a timeout with no success) the expert oracle takes over from
     the current state to completion; its (state, action) pairs are aggregated
  5. repeat until Nf trajectories collected / max_rounds reached

Pure-policy success rate is evaluated (no expert) after every round.
"""
import gc
from collections import deque

import numpy as np
import torch

from .collect import collect_demos
from .dataset import DemoDataset
from .eval import evaluate_policy, make_obs_seq
from .models import DiffusionPolicy
from .train import train_diffusion_policy


def build_policy(cfg, normalization, state_dim, act_dim, device):
    return DiffusionPolicy(
        act_dim=act_dim,
        state_dim=state_dim,
        normalization=normalization,
        obs_horizon=cfg["obs_horizon"],
        pred_horizon=cfg["pred_horizon"],
        act_horizon=cfg["act_horizon"],
        num_diffusion_iters=cfg["num_diffusion_iters"],
        prediction_type=cfg["prediction_type"],
        scheduler_type=cfg["scheduler_type"],
        diffusion_step_embed_dim=cfg["diffusion_step_embed_dim"],
        down_dims=cfg["down_dims"],
        alpha=cfg["alpha"],
        patience_window=cfg["patience_window"],
        patience=cfg["patience"],
        batch_multiplier=cfg["batch_multiplier"],
        num_per_batch=cfg["num_per_batch"],
        modality=cfg.get("modality", "state"),
        keypoints=cfg.get("keypoints", 32),
    ).to(device)


def train_and_calibrate(cfg, trajs, device, train_steps, log_fn):
    dataset = DemoDataset(trajs, cfg["obs_horizon"], cfg["pred_horizon"])
    policy = build_policy(cfg, dataset.get_normalization(), dataset.state_dim, dataset.act_dim, device)
    n_params = sum(p.numel() for p in policy.parameters())
    log_fn(f"  [train] {dataset.num_traj} trajs, {len(dataset)} windows, "
           f"state_dim={dataset.state_dim} act_dim={dataset.act_dim} params={n_params/1e6:.2f}M "
           f"steps={train_steps}")
    train_diffusion_policy(
        policy, dataset, train_steps=train_steps, batch_size=cfg["batch_size"],
        lr=cfg["lr"], weight_decay=cfg["weight_decay"], device=device, log_fn=log_fn,
    )
    # Calibrate over a representative shuffled sample of the WHOLE dataset so the
    # newly aggregated intervention windows contribute to the OOD threshold.
    windows = list(dataset.iter_windows(max_windows=cfg.get("calib_windows", 3000),
                                        shuffle=True, seed=cfg["seed"]))
    losses = policy.set_threshold_from_dataset(windows, num_iter=cfg["threshold_num_iter"])
    log_fn(f"  [calibrate] threshold(alpha={cfg['alpha']})={policy.loss_threshold:.5f} "
           f"over {len(losses)} samples (median={np.median(losses):.5f})")
    return policy, dataset


def train_bc(cfg, trajs, device, train_steps, log_fn):
    """Train a plain behavior-cloning diffusion policy (no threshold)."""
    dataset = DemoDataset(trajs, cfg["obs_horizon"], cfg["pred_horizon"])
    policy = build_policy(cfg, dataset.get_normalization(), dataset.state_dim, dataset.act_dim, device)
    train_diffusion_policy(
        policy, dataset, train_steps=train_steps, batch_size=cfg["batch_size"],
        lr=cfg["lr"], weight_decay=cfg["weight_decay"], device=device, log_fn=log_fn,
    )
    return policy


def calibrate_ni(cfg, make_env_fn, make_expert_fn, device, log_fn):
    """BC data-scaling sweep: find the initial demo count Ni whose round-0 BC
    success is closest to `target_success` (~50%), so Diff-DAgger has headroom
    to demonstrate improvement (the paper's RQ1/RQ2 setup).

    Returns (recommended_ni, curve=[(N, success), ...], demo_pool).
    """
    env_h = cfg["env_horizon"]
    collect_env = make_env_fn(env_h)
    eval_env = make_env_fn(env_h)
    expert = make_expert_fn()
    sweep = sorted(cfg["ni_sweep"])
    pool_n = max(sweep)

    log_fn(f"[sweep] collecting a pool of {pool_n} expert demos (nested prefixes) ...")
    pool = collect_demos(collect_env, expert, pool_n, seed_base=0,
                         max_steps=env_h, log_fn=log_fn)
    if len(pool) < min(sweep):
        raise RuntimeError(
            f"BC sweep: demo pool ({len(pool)}) is smaller than the smallest "
            f"ni_sweep value ({min(sweep)}); expert may be failing to produce "
            f"successful demos. Aborting before training.")
    if len(pool) < pool_n:
        log_fn(f"[sweep] WARNING: pool under-delivered ({len(pool)}/{pool_n}); "
               f"sweep will only cover N<={len(pool)}")

    curve = []
    for N in sweep:
        if N > len(pool):
            break
        policy = train_bc(cfg, pool[:N], device, cfg["sweep_train_steps"], log_fn)
        ev = evaluate_policy(
            policy, eval_env, num_episodes=cfg["sweep_eval_episodes"],
            obs_horizon=cfg["obs_horizon"], act_horizon=cfg["act_horizon"],
            device=device, base_seed=cfg["eval_seed_base"], max_steps=env_h,
        )
        curve.append((N, ev["success_rate"]))
        cov = f" coverage={ev['mean_coverage']:.2f}" if "mean_coverage" in ev else ""
        log_fn(f"[sweep] Ni={N}: BC success={ev['success_rate']:.3f}{cov}")
        del policy
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    if not curve:
        raise RuntimeError(f"BC sweep produced no points (pool={len(pool)}).")
    target = cfg["target_success"]
    ni = min(curve, key=lambda nc: abs(nc[1] - target))[0]
    log_fn(f"[sweep] target~{target:.2f} -> recommended Ni={ni} | curve={curve}")
    collect_env.close()
    eval_env.close()
    return ni, curve, pool


@torch.no_grad()
def dagger_episode(policy, env, expert, cfg, *, seed, device, policy_phase_cap, log_fn,
                   image_size=None):
    """One DAgger episode. Returns metrics + (optional) intervention trajectory."""
    from .eval import frame
    obs_horizon = cfg["obs_horizon"]
    s = env.reset(seed=seed)
    expert.reset(env)
    policy.reset_query()
    dq = deque([s] * obs_horizon, maxlen=obs_horizon)
    idq = deque([frame(env, image_size)] * obs_horizon, maxlen=obs_horizon) if image_size else None

    t, done, success, queried = 0, False, False, False
    max_loss = 0.0

    # ---- policy phase (Algorithm 1) ----
    # Re-plan every step: denoise a fresh action under the current observation,
    # score its expected diffusion loss (the Diff-DAgger uncertainty), query on
    # the K-patience rule, otherwise execute the first action (receding horizon).
    while t < policy_phase_cap and not done:
        obs_seq = make_obs_seq(dq, obs_horizon, device, idq)
        action_chunk, diff_loss = policy.get_action_and_uncertainty(obs_seq)
        max_loss = max(max_loss, diff_loss)
        if policy.register_and_query(diff_loss):
            queried = True
            break
        a = action_chunk[0, 0].cpu().numpy()
        s, r, done, success, info = env.step(a)
        dq.append(s)
        if idq is not None:
            idq.append(frame(env, image_size))
        t += 1

    # ---- auto-intervention on timeout (no query, no success) ----
    auto = False
    if not success and not done and not queried:
        auto = True

    # ---- expert takeover from the current state ----
    interv_states, interv_actions, interv_images = [], [], []
    if (queried or auto) and not success:
        while not done:
            a = np.asarray(expert.get_action(env, s), dtype=np.float32)
            interv_states.append(np.asarray(s, dtype=np.float32))
            interv_actions.append(a)
            if image_size:
                interv_images.append(frame(env, image_size))
            s, r, done, success, info = env.step(a)
            dq.append(s)
            t += 1

    traj = None
    if interv_states:
        traj = {"state": np.stack(interv_states), "action": np.stack(interv_actions)}
        if image_size and interv_images:
            traj["image"] = np.stack(interv_images)
    return dict(
        success=bool(success), queried=queried, auto=auto, length=t,
        intervention=traj, max_loss=float(max_loss),
        n_expert_steps=len(interv_states),
    )


def run_diffdagger(cfg, make_env_fn, make_expert_fn, device, log_fn=print, ckpt_path=None,
                   init_trajs=None, image_size=None):
    """Full Diff-DAgger run. `make_env_fn(horizon)` and `make_expert_fn()` build
    fresh env/expert instances. If `init_trajs` is given it is used as the
    initial dataset (e.g. the first Ni demos from a BC-calibration pool);
    otherwise Ni demos are collected. Returns per-round metrics."""
    rng = np.random.default_rng(cfg["seed"])
    env_horizon = cfg["env_horizon"]
    dagger_horizon = env_horizon + cfg["expert_max_steps"]
    eval_seed_base = cfg["eval_seed_base"]
    eval_eps = cfg["eval_episodes"]

    collect_env = make_env_fn(env_horizon)
    dagger_env = make_env_fn(dagger_horizon)
    eval_env = make_env_fn(env_horizon)
    expert = make_expert_fn()

    # ---- 1. initial expert demonstrations ----
    if init_trajs is not None:
        trajs = list(init_trajs)
        log_fn(f"[init] using {len(trajs)} pre-collected (calibrated Ni) demos")
    else:
        log_fn(f"[init] collecting {cfg['num_init_demos']} initial expert demos ...")
        trajs = collect_demos(
            collect_env, expert, cfg["num_init_demos"],
            seed_base=int(rng.integers(0, 1_000_000)), max_steps=env_horizon, log_fn=log_fn,
            image_size=image_size,
        )

    history = {"round": [], "eval_success": [], "eval_coverage": [], "n_trajs": [],
               "query_rate": [], "policy_phase_success": [], "interventions": []}
    seed_counter = int(rng.integers(1_000_000, 2_000_000))
    total_interventions = 0

    for rnd in range(cfg["max_rounds"]):
        train_steps = cfg["initial_train_steps"] if rnd == 0 else cfg["round_train_steps"]
        log_fn(f"\n===== Round {rnd} | dataset={len(trajs)} trajs =====")
        policy, dataset = train_and_calibrate(cfg, trajs, device, train_steps, log_fn)

        # ---- evaluate the pure policy (no expert) on the FIXED held-out set ----
        ev = evaluate_policy(
            policy, eval_env, num_episodes=eval_eps,
            obs_horizon=cfg["obs_horizon"], act_horizon=cfg["act_horizon"],
            device=device, base_seed=eval_seed_base, max_steps=env_horizon,
            image_size=image_size,
        )
        cov_str = f" coverage={ev['mean_coverage']:.3f}" if "mean_coverage" in ev else ""
        log_fn(f"  [eval] pure-policy success={ev['success_rate']:.3f} "
               f"mean_len={ev['mean_len']:.0f}{cov_str} (fixed {eval_eps}-ep set)")
        n_at_eval = len(trajs)   # dataset size THIS eval used (curve x-axis; match P4 loop)

        # ---- DAgger rollouts: gather Nd interventions ----
        new_interventions = 0
        n_episodes, phase_success, n_query = 0, 0, 0
        while (new_interventions < cfg["interventions_per_round"]
               and n_episodes < cfg["max_rollouts_per_round"]):
            res = dagger_episode(
                policy, dagger_env, expert, cfg,
                seed=seed_counter, device=device, policy_phase_cap=env_horizon, log_fn=log_fn,
                image_size=image_size,
            )
            seed_counter += 1
            n_episodes += 1
            phase_success += int(res["success"] and not res["queried"] and not res["auto"])
            n_query += int(res["queried"])
            if res["intervention"] is not None:
                trajs.append(res["intervention"])
                new_interventions += 1
                total_interventions += 1
            log_fn(f"  [dagger ep{n_episodes}] success={res['success']} query={res['queried']} "
                   f"auto={res['auto']} maxloss={res['max_loss']:.4f} "
                   f"len={res['length']} expert_steps={res['n_expert_steps']} "
                   f"(interv {new_interventions}/{cfg['interventions_per_round']})")

        history["round"].append(rnd)
        history["eval_success"].append(ev["success_rate"])
        history["eval_coverage"].append(ev.get("mean_coverage"))
        history["n_trajs"].append(n_at_eval)
        history["query_rate"].append(n_query / max(1, n_episodes))
        history["policy_phase_success"].append(phase_success / max(1, n_episodes))
        history["interventions"].append(total_interventions)

        if ckpt_path is not None:
            torch.save({"policy_state_dict": policy.state_dict(),
                        "cfg": cfg, "round": rnd, "history": history,
                        "normalization": dataset.get_normalization(),
                        # persist the calibrated Diff-DAgger OOD state so a reloaded
                        # checkpoint can gate/query without recalibrating
                        "loss_threshold": float(policy.loss_threshold),
                        "loss_cdf_samples": (policy.loss_cdf.sorted
                                             if policy.loss_cdf is not None else None),
                        "alpha": cfg["alpha"]}, ckpt_path)
            log_fn(f"  [ckpt] saved -> {ckpt_path}")

        if len(trajs) >= cfg["final_demos"]:
            log_fn(f"[stop] reached final_demos={cfg['final_demos']} ({len(trajs)} trajs)")
            break

    # ---- final evaluation on the same FIXED held-out set ----
    final = evaluate_policy(
        policy, eval_env, num_episodes=eval_eps,
        obs_horizon=cfg["obs_horizon"], act_horizon=cfg["act_horizon"],
        device=device, base_seed=eval_seed_base, max_steps=env_horizon,
        image_size=image_size,
    )
    cov_str = f" coverage={final['mean_coverage']:.3f}" if "mean_coverage" in final else ""
    log_fn(f"\n[FINAL] pure-policy success={final['success_rate']:.3f}{cov_str} "
           f"over {final['num_episodes']} eps")
    history["final_success"] = final["success_rate"]
    history["final_coverage"] = final.get("mean_coverage")

    for e in (collect_env, dagger_env, eval_env):
        e.close()
    return history
