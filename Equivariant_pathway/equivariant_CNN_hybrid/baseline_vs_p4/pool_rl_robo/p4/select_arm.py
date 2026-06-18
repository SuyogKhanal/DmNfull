"""p4_select — P4-LLM demonstration SELECTION on top of detect-then-correct.

Per round: roll ``n_cand`` diffusion-policy episodes; in each, a per-planning-step
divergence signal marks the step t* and whether the episode failed. Take the TOP-3
failures by PEAK signal; the P4 LLM SELECTS which ONE to correct (compressing 3
candidates → 1 choice — the maze-P4 "selection" idea); the expert then corrects
ON-POLICY from that candidate's divergence state (replay the policy prefix
deterministically, expert finishes the episode). 1 demo/round, retrain every Nd,
early-stop at target_sr.

Two detection signals (gated on the task's expert_kind):
  * PushT (PPO expert)        → SafeDAgger action-discrepancy ‖a_nov − a_exp‖ in
                                joint-delta space.
  * Stack/Pick/Plug (motion   → the policy's OWN diffusion loss
    planner, no per-state π_exp)  (get_action(dagger=True)["diffusion_loss"]) — the
                                SAME signal diff_dagger uses, so BOTH arms share one
                                detector and differ ONLY in WHICH failure is
                                corrected (diff_dagger: first CDF(loss)>α-quantile
                                run; p4_select: LLM picks among top-3 peak-loss
                                FAILED candidates). Decision 2026-06-08; see
                                claude_context.md.

On-policy correction (which beats Diff-DAgger on PushT) is preserved. TEXT-ONLY
selection (no rendering) → robust against SAPIEN/Vulkan render fragility. Uses the
LLM; reported alongside diff_dagger.
"""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from ..envs import env_setup as E

E.bootstrap_fork_path()

from diffdagger.main_pipeline.diffdagger_baseline import (  # noqa: E402
    _append_episode_to_dataset, _to_bool,
)
from diffdagger.main_pipeline.sim_bridge import (  # noqa: E402
    collect_initial_demos_shared, train_policy, evaluate_heldout,
)

from ..selection.iil_baselines import (  # noqa: E402
    _full_obs, _expert_delta, _policy_delta, _l2,
)
from . import kag as KAG  # noqa: E402
from . import vlm as VLM  # noqa: E402
from .analysis import AnalysisClass  # noqa: E402
from .prescription import CubePrescriptionClass, PrescriptionEmptyError  # noqa: E402
from .runner import make_client, extract_json  # noqa: E402


def _info(env_id: str, msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [p4_select|{env_id}] {msg}", flush=True)


def _obs_seq(obs_deque, cfg):
    return {k: torch.stack([obs_deque[i][k] for i in range(cfg.obs_horizon)]
                           ).swapaxes(0, 1) for k in obs_deque[0].keys()}


def _policy_action_seq(policy, obs_seq):
    out = policy.get_action(obs_seq, dagger=False, return_dict=False)
    return out["action"] if isinstance(out, dict) else out


_LOSS_FALLBACK_WARNED = False


def _policy_loss_seq(policy, obs_seq):
    """(action_seq, diffusion_loss) via the policy's NATIVE diffusion-loss query —
    the SAME per-step signal the diff_dagger arm reads
    (get_action(dagger=True)["diffusion_loss"]). Falls back to a plain forward +
    loss=0.0 if the loss path is unavailable (e.g. threshold/cdf not set). The
    fallback is LOUD (one-time warning): a silent flat-zero loss would make every
    candidate's peak signal 0.0 → arbitrary top-3 ranking → meaningless selection,
    which would invalidate the head-to-head (_calibrate_diffusion guards against
    the usual cause by recomputing the cdf if absent)."""
    global _LOSS_FALLBACK_WARNED
    try:
        d = policy.get_action(obs_seq, dagger=True, return_dict=True)
        if isinstance(d, dict):
            aseq = d["action"]
            loss = d.get("diffusion_loss", 0.0)
            return aseq, float(loss.item() if hasattr(loss, "item") else loss)
    except Exception as exc:
        if not _LOSS_FALLBACK_WARNED:
            print(f"[p4_select] WARN: diffusion-loss query failed "
                  f"({type(exc).__name__}: {exc}); falling back to loss=0.0 — "
                  f"DETECTOR DEGRADED, selection may be arbitrary this round", flush=True)
            _LOSS_FALLBACK_WARNED = True
    out = policy.get_action(obs_seq, dagger=False, return_dict=False)
    return (out["action"] if isinstance(out, dict) else out), 0.0


def _safe_rollout_one(env, expert, policy, cfg, seed, num_joints, *,
                      signal_mode: str = "discrepancy") -> Dict[str, Any]:
    """Roll ONE policy episode (no intervention). Record executed actions + the
    per-planning-step divergence signal so the divergence step t* and the episode
    can be replayed deterministically later.

    signal_mode:
      "discrepancy" — SafeDAgger action-discrepancy ‖a_nov − a_exp‖ (PPO experts).
      "diffusion"   — the policy's OWN diffusion loss (motion-planner tasks; NO
                      expert.get_action — identical detection signal to diff_dagger,
                      per the 2026-06-08 decision in claude_context.md)."""
    obs, info = env.reset(seed=int(seed))
    policy.reset()
    env.set_action_space("joint_pos")
    expert.reset(env)
    obs_deque = deque([obs] * cfg.pred_horizon, maxlen=cfg.obs_horizon)
    timestep = 0
    done = False
    exec_actions: List[torch.Tensor] = []
    discs: List[float] = []
    diffusion = signal_mode == "diffusion"

    env.set_action_space(cfg.action_space)
    if diffusion:
        action_seq, sig = _policy_loss_seq(policy, _obs_seq(obs_deque, cfg))
    else:
        action_seq, sig = _policy_action_seq(policy, _obs_seq(obs_deque, cfg)), None
    while timestep < cfg.max_episode_steps and not done:
        actions = env.post_process_action(action_seq)
        if diffusion:
            disc = float(sig or 0.0)
        else:
            try:
                full = _full_obs(env)
                a_nov = _policy_delta(action_seq, policy, num_joints)
                a_exp = _expert_delta(env, expert, full, int(seed), num_joints)
                disc = _l2(a_nov, a_exp)
            except Exception:
                disc = 0.0
        for i in range(cfg.action_horizon):
            action = actions[:, policy.obs_horizon - 1 + i]
            exec_actions.append(action.detach().clone())
            discs.append(disc)
            obs, reward, _, _, info = env.step({"action": action})
            done = _to_bool(info.get("success", False))
            obs_deque.append(obs)
            timestep += 1
            if done or timestep >= cfg.max_episode_steps:
                break
        if done:
            break
        if diffusion:
            action_seq, sig = _policy_loss_seq(policy, _obs_seq(obs_deque, cfg))
        else:
            action_seq = _policy_action_seq(policy, _obs_seq(obs_deque, cfg))

    t_star = int(max(range(len(discs)), key=lambda j: discs[j])) if discs else 0
    return {"seed": int(seed), "exec_actions": exec_actions, "t_star": t_star,
            "peak_disc": float(discs[t_star]) if discs else 0.0,
            "success": bool(done), "n_steps": int(timestep),
            "signal_mode": signal_mode}


def _correct_onpolicy_from(env, expert, policy, cfg, cand) -> (List[Any], bool, int):
    """Replay the policy prefix to the divergence step, then the expert corrects
    ON-POLICY to completion (SafeDAgger demo). Returns (ep_tds, success, steps)."""
    obs, info = env.reset(seed=int(cand["seed"]))
    policy.reset()
    env.set_action_space(cfg.action_space)
    for i in range(min(int(cand["t_star"]), len(cand["exec_actions"]))):
        obs, reward, _, _, info = env.step({"action": cand["exec_actions"][i]})
        if _to_bool(info.get("success", False)):
            return [], False, 0          # solved during replay → nothing to correct
    ep_info = dict(seed=int(cand["seed"]))
    env.set_action_space(cfg.expert.action_space)
    expert.reset(env)
    expert.setup_task()
    ep_tds: List[Any] = []
    done = False
    total = 0
    hard_limit = cfg.max_episode_steps + cfg.expert.max_episode_steps
    while not done and total < hard_limit:
        td = expert.move_to_next_goal(ep_info)
        td.update(dict(episode=torch.ones(len(td["episode"])) * int(cand["seed"])))
        ep_tds.append(td)
        done = _to_bool(td["done"][-1])
        total += len(td["episode"])
    success = bool(done) and 10 <= total <= cfg.expert.max_episode_steps
    return ep_tds, success, total


def _llm_select(client, task_desc: str, kag_text: str, top: List[Dict]) -> int:
    """P4 LLM picks WHICH of the top-3 candidate failures to correct. Returns the
    index into `top`. Falls back to 0 (highest peak signal) on any failure."""
    if client is None or len(top) <= 1:
        return 0
    sig_label = ("peak_diffusion_loss"
                 if top[0].get("signal_mode") == "diffusion"
                 else "peak_action_discrepancy")
    rows = "\n".join(
        f"  [{i}] {sig_label}={c['peak_disc']:.4f} "
        f"success={c['success']} length={c['n_steps']} steps" for i, c in enumerate(top))
    sig_phrase = ("the policy's own diffusion loss"
                  if top[0].get("signal_mode") == "diffusion"
                  else "action-discrepancy from the expert")
    prompt = (
        f"TASK: {task_desc}\n\n{kag_text}\n\n"
        f"The novice policy failed these {len(top)} episodes (the top-3 by "
        f"{sig_phrase}). Pick the ONE whose expert on-policy "
        f"correction is the MOST INFORMATIVE demonstration to add next — the "
        f"failure that, once corrected, closes the most impactful failure mode.\n"
        f"CANDIDATES:\n{rows}\n\n"
        'Output ONLY JSON: {"selected_index": <0-based int>, "rationale": "<one sentence>"}')
    try:
        txt = client.reason(prompt, system_prompt=(
            "You select the single most informative failure to correct for "
            "imitation learning on a robot. Output strict JSON only."))
        obj = extract_json(txt) or {}
        i = int(obj.get("selected_index", 0))
        return i if 0 <= i < len(top) else 0
    except Exception as exc:  # pragma: no cover - server path
        print(f"[p4_select] LLM select failed ({type(exc).__name__}); fallback to #0")
        return 0


def run_p4_select_arm(*, method: str, env, eval_env, expert, cfg, make_policy,
                      make_dataset, init_ckpt: str, init_sr: float, work_dir: str,
                      budget: int, nd_retrain: int, target_sr: float, max_rounds: int,
                      heldout_seed_base: int, heldout_num_ep: int,
                      round_epoch: Optional[int], max_train_steps: Optional[int],
                      hp: Dict, seed: int, suite_env_id: str,
                      llm_client_on: bool) -> Dict[str, Any]:
    import random
    work = Path(work_dir); work.mkdir(parents=True, exist_ok=True)
    lc = work / "learning_curve.json"
    ah, mes = cfg.action_horizon, cfg.env.max_episode_steps
    num_joints = int(env.num_joints)
    n_cand = int(hp.get("n_cand", 6))
    rng = random.Random(0xC0FFEE + seed)
    spec = E.task_spec(suite_env_id)
    task_desc = spec.task_description
    kag_text = KAG.load_kag_text(suite_env_id)
    client = make_client() if llm_client_on else None
    # Motion-planner tasks (Stack/Pick/Plug) have no per-state π_exp(s): detect via
    # the policy's OWN diffusion loss — the SAME signal diff_dagger uses (only WHICH
    # failure is corrected differs). PushT keeps action-discrepancy detection.
    signal_mode = "diffusion" if spec.expert_kind == "motionplanner" else "discrepancy"

    def _calibrate_diffusion(pol):
        """Set the native diffusion-loss threshold/cdf so get_action(dagger=True)
        yields a valid per-step loss (the SAME signal diff_dagger uses). The cdf is
        normally persisted in the bootstrap ckpt and refreshed on every retrain; if
        it is ever ABSENT we recompute it from the current dataset rather than let
        the detector silently degrade to flat-zero loss (which would make the top-3
        ranking arbitrary). Uses the same alpha/patience as the hydra cfg (=
        diff_dagger's), keeping both arms' detector identical."""
        if signal_mode != "diffusion":
            return
        pol.alpha = float(cfg.policy.get("alpha", 0.99))
        pol.patience = int(cfg.policy.get("patience", 1))
        pol.patience_window = pol.patience
        _cdf = getattr(pol, "diffusion_loss_cdf", None)
        if _cdf is None:
            print("[p4_select] WARN: policy missing diffusion_loss_cdf; "
                  "recomputing from dataset so the detector is not silently dead",
                  flush=True)
            try:
                pol.update_diffusion_threshold(dataset)
                _cdf = getattr(pol, "diffusion_loss_cdf", None)
            except Exception as exc:
                print(f"[p4_select] ERROR recomputing diffusion cdf: "
                      f"{type(exc).__name__}: {exc}", flush=True)
        if _cdf is not None:
            pol.diffusion_loss_threshold = _cdf.get_quantile(pol.alpha)
        else:
            print("[p4_select] WARN: diffusion_loss_cdf STILL None after recompute "
                  "— check the bootstrap ckpt; selection may degrade", flush=True)
        pol.reset()

    _info(suite_env_id, f"LLM selector {'ON' if client else 'OFF (highest-signal fallback)'}; "
                        f"signal={signal_mode} n_cand={n_cand} nd_retrain={nd_retrain}")

    dataset = make_dataset()
    collect_initial_demos_shared(env, expert, dataset, cfg,
                                 list(range(int(hp.get("num_initial_demos", 20)))))
    policy = make_policy(); policy.load(init_ckpt); policy.to(cfg.device); policy.reset()
    _calibrate_diffusion(policy)

    curve: List[Dict[str, Any]] = []
    t0 = time.time()

    def _point(rnd, q, sr, added, reason):
        try:
            cum = int(len(dataset.rb))
        except Exception:
            cum = -1
        tec = getattr(expert, "total_expert_calls", None)   # secondary cost axis
        curve.append({"round": rnd, "n_queries": q, "success_rate": sr,
                      "demos_added": added, "n_demos": cum, "stop_reason": reason,
                      "total_expert_calls": tec})
        lc.write_text(json.dumps({"method": method, "env": suite_env_id, "seed": seed,
                                  "budget": budget, "backbone": "diffusion",
                                  "variant": "p4_select_on_safedagger", "history": curve,
                                  "final_performance": {"success_rate": sr, "n_queries": q},
                                  "stopped_reason": reason}, indent=2, default=str))
        _info(suite_env_id, f"round {rnd}: sr={sr:.3f} q={q}/{budget} added={added} stop={reason}")

    if hasattr(expert, "reset_counts"):   # count only DAgger-phase expert calls
        expert.reset_counts()
    _point(0, 0, float(init_sr), 0, None)

    def _retrain(rnd):
        nonlocal policy
        if round_epoch is not None:
            cfg.epoch = int(round_epoch)
        if max_train_steps is not None:
            cfg.max_train_steps = int(max_train_steps)
        res = train_policy(policy, dataset, cfg,
                           checkpoint_dir=str(work / f"checkpoints/round_{rnd:03d}"),
                           log_dir=None, fine_tune=False)
        if res.get("status") != "ok":
            raise RuntimeError(f"train_policy failed: {res.get('error')}")
        policy = res["policy"]
        _calibrate_diffusion(policy)

    nd = max(1, int(nd_retrain))
    interventions = 0
    last_trained = 0
    ep_seed = 0
    episodes = 0
    max_episodes = int(hp.get("max_episodes", 1000))
    stopped = "max_rounds"

    while interventions < budget and episodes < max_episodes:
        rnd_no = len(curve)
        cands = []
        for _j in range(n_cand):
            cands.append(_safe_rollout_one(env, expert, policy, cfg, ep_seed,
                                           num_joints, signal_mode=signal_mode))
            ep_seed += 1
            episodes += 1
        fails = [c for c in cands if not c["success"]]
        # Select ONLY among genuine FAILURES (the method = "pick which failure to
        # correct"). If every candidate succeeded, add NO demo this round and roll
        # again — same as diff_dagger, which only intervenes on policy failures.
        # (The old `fails or cands` fallback let the expert "correct" a SUCCESS,
        # producing an easy reset-state demo diff_dagger cannot make → unfair.)
        if not fails:
            continue
        pool = sorted(fails, key=lambda c: -c["peak_disc"])
        top = pool[:3]
        sel = top[_llm_select(client, task_desc, kag_text, top)]
        ep_tds, ok, esteps = _correct_onpolicy_from(env, expert, policy, cfg, sel)
        added = 0
        if ok and ep_tds:
            _append_episode_to_dataset(dataset, ep_tds)
            interventions += 1
            added = 1
        if added and (interventions % nd == 0 or interventions >= budget):
            n_added_since = interventions - last_trained
            _retrain(interventions)
            sr = evaluate_heldout(policy, eval_env, heldout_seed_base, heldout_num_ep, ah, mes)
            last_trained = interventions
            reason = "budget_exhausted" if interventions >= budget else None
            _point(interventions, interventions, sr, n_added_since, reason)
            if sr >= target_sr:
                stopped = "target_hit"; break
            if interventions >= budget:
                stopped = "budget_exhausted"; break

    if stopped == "max_rounds" and episodes >= max_episodes:
        stopped = "max_episodes"          # honest label: backstop hit, not a real stop
    if interventions > last_trained:
        _retrain(interventions)
        sr = evaluate_heldout(policy, eval_env, heldout_seed_base, heldout_num_ep, ah, mes)
        _point(interventions, interventions, sr, interventions - last_trained,
               "budget_exhausted" if interventions >= budget else stopped)

    final = curve[-1] if curve else {}
    return {"method": method, "env": suite_env_id, "stopped_reason": stopped,
            "final_performance": {"success_rate": final.get("success_rate"),
                                  "n_queries": interventions},
            "history": curve, "elapsed_seconds": time.time() - t0,
            "total_expert_calls": interventions}


# ======================================================================== #
# p4_top3 — the FULL P4-LLM method (VLM → reason → prescribe cube layout →  #
# reposition env → motion-planner solve → corrective demo). Suite-native    #
# (the fork's LLMGuidedDAggerPipeline is PushT-hardcoded). Reuses this       #
# module's diffusion-loss detection + retrain/eval/stop skeleton; the new    #
# parts are rendering, VLM-prescription of cube positions, and the           #
# reposition-and-solve corrective demo.                                      #
# ======================================================================== #
def _safe_render(env):
    try:
        return env.render()
    except Exception as exc:  # pragma: no cover - render path
        print(f"[p4_top3] render error: {type(exc).__name__}: {exc}", flush=True)
        return None


def _save_png(frame, path) -> Optional[str]:
    """Save a (1,H,W,3)/(H,W,3) RGB frame (cuda tensor or ndarray) as PNG. Returns
    the path, or None if the frame was None / unsavable (the VLM tolerates a
    missing role)."""
    if frame is None:
        return None
    try:
        import numpy as np
        from PIL import Image
        arr = frame.detach().cpu().numpy() if hasattr(frame, "detach") else np.asarray(frame)
        if arr.ndim == 4:
            arr = arr[0]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr.astype("uint8")).save(str(path))
        return str(path)
    except Exception as exc:  # pragma: no cover
        print(f"[p4_top3] PNG save failed: {type(exc).__name__}: {exc}", flush=True)
        return None


def _render_failure_frames(env, cfg, cand, out_dir, ep_id) -> Dict[str, str]:
    """Reproduce a failed candidate on the render-enabled policy env (same env id +
    seed + recorded actions → same scene + trajectory) and capture start / t* /
    end RGB frames for the VLM. Returns {role: png_path} (roles with a None frame
    are dropped)."""
    out = Path(out_dir)
    pre = f"ep{int(ep_id):05d}_seed{int(cand['seed'])}"
    paths: Dict[str, Optional[str]] = {}
    env.reset(seed=int(cand["seed"]))
    env.set_action_space(cfg.action_space)
    paths["start_frame"] = _save_png(_safe_render(env), out / f"{pre}_start.png")
    exec_actions = cand["exec_actions"]
    n = len(exec_actions)
    t_star = min(int(cand["t_star"]), max(0, n - 1))
    for i in range(n):
        env.step({"action": exec_actions[i]})
        if i == t_star:
            paths["high_loss_frame"] = _save_png(_safe_render(env), out / f"{pre}_tstar.png")
    paths["end_frame"] = _save_png(_safe_render(env), out / f"{pre}_end.png")
    if not paths.get("high_loss_frame"):
        paths["high_loss_frame"] = paths.get("end_frame")
    return {k: v for k, v in paths.items() if v}


def _infeasible_addendum(history: List[Dict]) -> str:
    """Bank recently-infeasible cube layouts (the motion planner could not solve)
    so the re-prescription avoids them (mirrors the fork's infeasible feedback)."""
    if not history:
        return ""
    rows = "\n".join(
        f"  - cubeA={h.get('cubeA_xyz')} cubeB={h.get('cubeB_xyz')} "
        f"(unsolved: {h.get('reason', 'expert could not stack')})"
        for h in history[-8:])
    return ("AVOID THESE LAYOUTS — the expert could NOT produce a successful stack "
            f"demonstration from them; prescribe a clearly DIFFERENT, graspable & "
            f"stackable layout:\n{rows}")


def _collect_prescribed_demo(reposition_env, expert, cfg, pres, seed):
    """Set the prescribed cube layout on StackCube-Start-v0, reset, and run the
    motion-planner expert to completion → one corrective demo. pres=None ⇒ random
    layout (the --no-llm / env-sampled fallback). Returns (ep_tds, success, steps,
    applied_config)."""
    base = reposition_env.unwrapped       # StackCubeStartEnv (set_prescription lives here)
    if pres is None:
        base.clear_prescription()
        applied = {"cubeA_xyz": None, "cubeB_xyz": None}
    else:
        applied = base.set_prescription(
            cubeA_xyz=pres["cubeA_xyz"], cubeB_xyz=pres["cubeB_xyz"],
            cubeA_zrot=pres.get("cubeA_zrot"), cubeB_zrot=pres.get("cubeB_zrot"))
    reposition_env.reset(seed=int(seed))
    reposition_env.set_action_space(cfg.expert.action_space)   # on the WRAPPER
    expert.reset(reposition_env)
    expert.setup_task()
    ep_info = dict(seed=int(seed))
    ep_tds: List[Any] = []
    done = False
    total = 0
    hard_limit = cfg.max_episode_steps + cfg.expert.max_episode_steps
    while not done and total < hard_limit:
        td = expert.move_to_next_goal(ep_info)
        td.update(dict(episode=torch.ones(len(td["episode"])) * int(seed)))
        ep_tds.append(td)
        done = _to_bool(td["done"][-1])
        total += len(td["episode"])
    success = bool(done) and 10 <= total <= cfg.expert.max_episode_steps
    base.clear_prescription()
    return ep_tds, success, total, applied


def _stackcube_descriptor(env, cfg, cand):
    """V3-hybrid: replay a failed candidate's recorded actions to its t* and
    snapshot the cube/grasp state (the clustering descriptor). Mirrors the prefix
    replay in ``_correct_onpolicy_from``; reads cubeA/cubeB pose from the live env
    (the select_arm engine has no meta.json). Returns None on any failure."""
    from .stackcube_hybrid import StackCubeFailureDescriptor, yaw_from_quat_wxyz
    try:
        env.reset(seed=int(cand["seed"]))
        env.set_action_space(cfg.action_space)
        t_star = min(int(cand["t_star"]), len(cand["exec_actions"]))
        for i in range(t_star):
            env.step({"action": cand["exec_actions"][i]})
        base = env.unwrapped
        aP = base.cubeA.pose.p.reshape(-1)[:3].detach().cpu().numpy().tolist()
        aQ = base.cubeA.pose.q.reshape(-1)[:4].detach().cpu().numpy().tolist()
        bP = base.cubeB.pose.p.reshape(-1)[:3].detach().cpu().numpy().tolist()
        bQ = base.cubeB.pose.q.reshape(-1)[:4].detach().cpu().numpy().tolist()
        try:
            g = base.agent.is_grasping(base.cubeA)
            grasp = 1.0 if float(torch.as_tensor(g).reshape(-1)[0].item()) > 0.5 else 0.0
        except Exception:
            grasp = 0.0
        return StackCubeFailureDescriptor(
            episode_id=int(cand["seed"]), seed=int(cand["seed"]),
            peak_loss=float(cand.get("peak_disc", 0.0)),
            t_star=int(cand["t_star"]), T=max(1, int(cand.get("n_steps", 1))),
            cubeA_xyz=[float(v) for v in aP], cubeA_zrot=yaw_from_quat_wxyz(aQ),
            cubeB_xyz=[float(v) for v in bP], cubeB_zrot=yaw_from_quat_wxyz(bQ),
            grasp=grasp, cand=cand)
    except Exception as exc:
        print(f"[p4_top3|hybrid] descriptor build failed "
              f"({type(exc).__name__}: {exc})", flush=True)
        return None


def _hybrid_text_analysis(planner, cand):
    """Text-only failure analysis for the StackCube hybrid — NO env.render(). The
    suite policy env is built headless (no render camera), so VLM frame rendering
    (_render_failure_frames → env.render()) DEADLOCKS on StackCube (SAPIEN/Vulkan).
    p4_select avoids rendering for exactly this reason. The hybrid's actual decision
    is driven by the cube cluster geometry + the prescriber decision context, so the
    VLM analysis is only qualitative seasoning — derive a coarse root_cause/phase
    from the cube + grasp state at t* instead."""
    from .stackcube_hybrid import StackCubeHybridPlanner
    d = planner._member_by_episode_id(int(cand.get("seed", -1))) if planner else None
    prog = int(cand.get("t_star", 0)) / max(1, int(cand.get("n_steps", 1)))
    if d is None:
        return {"root_cause": "approach_failure", "phase": "pre_grasp",
                "rationale": "text-only (no descriptor)"}
    if d.grasp > 0.5:
        rc, ph = "placement_error", "placement"
    elif not StackCubeHybridPlanner._select_feasible(d):
        rc, ph = "contact_instability", "transport"
    elif prog < 0.5:
        rc, ph = "grasp_failure", "grasp"
    else:
        rc, ph = "approach_failure", "pre_grasp"
    return {"root_cause": rc, "phase": ph,
            "rationale": (f"text-only: grasp={'yes' if d.grasp > 0.5 else 'no'}, "
                          f"cubeA=({round(d.cubeA_xyz[0], 2)},{round(d.cubeA_xyz[1], 2)}), "
                          f"progress={round(prog, 2)}")}


def run_p4_top3_arm(*, method: str, env, eval_env, reposition_env, expert, cfg,
                    make_policy, make_dataset, init_ckpt: str, init_sr: float,
                    work_dir: str, budget: int, nd_retrain: int, target_sr: float,
                    max_rounds: int, heldout_seed_base: int, heldout_num_ep: int,
                    round_epoch: Optional[int], max_train_steps: Optional[int],
                    hp: Dict, seed: int, suite_env_id: str,
                    llm_client_on: bool) -> Dict[str, Any]:
    """p4_top3 for a motion-planner task (StackCube). Per round: roll n_cand
    candidates (diffusion-loss detection); take the top-3 FAILURES by peak loss;
    render + VLM-analyze + reason each; the LLM prescribes ONE corrective cube
    layout; StackCube-Start-v0 realises it and the motion planner solves it → one
    corrective demo (infeasible/empty re-prescribe guards); retrain every Nd; stop
    at target_sr/budget. Budget accounting = one KEPT corrective demo per unit
    (identical to diff_dagger/p4_select), so the head-to-head stays apples-to-apples."""
    import random
    if reposition_env is None:
        raise RuntimeError(f"p4_top3 needs a reposition env for {suite_env_id}")
    work = Path(work_dir); work.mkdir(parents=True, exist_ok=True)
    lc = work / "learning_curve.json"
    render_dir = work / "frames"
    ah, mes = cfg.action_horizon, cfg.env.max_episode_steps
    num_joints = int(env.num_joints)
    n_cand = int(hp.get("n_cand", 6))
    represcribe_attempts = int(hp.get("represcribe_attempts", 5))
    infeasible_attempts = int(hp.get("infeasible_attempts", 5))
    rng = random.Random(0xC0FFEE + seed)

    # V3 HYBRID (StackCube): when p4.subtask.collect == "hybrid", the LLM DECIDES
    # per round between SELECT (on-policy correct a real failure) and BRIDGE
    # (prescribe a middle-ground cube layout). Absent ⇒ planner is None ⇒ p4_top3
    # stays byte-identical (pure prescription), preserving the apples-to-apples bar.
    subtask_cfg = dict(hp.get("subtask", {}) or {})
    hybrid_planner = None
    # VLM frame rendering deadlocks on the headless StackCube policy env, so the
    # hybrid defaults to TEXT-ONLY failure analysis (p4_select's robust approach).
    # Set p4.subtask.use_vlm: true to opt back into rendering (only if a render
    # camera is wired). Plain p4_top3 (planner None) keeps its original behaviour.
    use_vlm = True
    if str(subtask_cfg.get("collect", "")).lower() == "hybrid":
        from .stackcube_hybrid import StackCubeHybridPlanner
        hybrid_planner = StackCubeHybridPlanner(work_dir=work_dir, cfg=subtask_cfg)
        use_vlm = bool(subtask_cfg.get("use_vlm", False))
    spec = E.task_spec(suite_env_id)
    task_desc = spec.task_description
    kag_text = KAG.load_kag_text(suite_env_id)
    client = make_client() if llm_client_on else None
    signal_mode = "diffusion" if spec.expert_kind == "motionplanner" else "discrepancy"
    analyzer = AnalysisClass(client, task_desc, kag_text)
    prescriber = CubePrescriptionClass(client, task_desc, kag_text,
                                       max_retries=represcribe_attempts)

    def _calibrate_diffusion(pol):
        if signal_mode != "diffusion":
            return
        pol.alpha = float(cfg.policy.get("alpha", 0.99))
        pol.patience = int(cfg.policy.get("patience", 1))
        pol.patience_window = pol.patience
        _cdf = getattr(pol, "diffusion_loss_cdf", None)
        if _cdf is None:
            print("[p4_top3] WARN: policy missing diffusion_loss_cdf; recomputing",
                  flush=True)
            try:
                pol.update_diffusion_threshold(dataset)
                _cdf = getattr(pol, "diffusion_loss_cdf", None)
            except Exception as exc:
                print(f"[p4_top3] ERROR recomputing cdf: {type(exc).__name__}: {exc}",
                      flush=True)
        if _cdf is not None:
            pol.diffusion_loss_threshold = _cdf.get_quantile(pol.alpha)
        pol.reset()

    _info(suite_env_id, f"p4_top3 LLM {'ON' if client else 'OFF (random-config fallback)'}; "
                        f"signal={signal_mode} n_cand={n_cand} nd_retrain={nd_retrain} "
                        f"represcribe={represcribe_attempts} infeasible={infeasible_attempts}")

    dataset = make_dataset()
    collect_initial_demos_shared(env, expert, dataset, cfg,
                                 list(range(int(hp.get("num_initial_demos", 20)))))
    policy = make_policy(); policy.load(init_ckpt); policy.to(cfg.device); policy.reset()
    _calibrate_diffusion(policy)

    curve: List[Dict[str, Any]] = []
    t0 = time.time()

    def _point(rnd, q, sr, added, reason):
        try:
            cum = int(len(dataset.rb))
        except Exception:
            cum = -1
        tec = getattr(expert, "total_expert_calls", None)
        curve.append({"round": rnd, "n_queries": q, "success_rate": sr,
                      "demos_added": added, "n_demos": cum, "stop_reason": reason,
                      "total_expert_calls": tec})
        lc.write_text(json.dumps({"method": method, "env": suite_env_id, "seed": seed,
                                  "budget": budget, "backbone": "diffusion",
                                  "variant": "p4_top3", "history": curve,
                                  "final_performance": {"success_rate": sr, "n_queries": q},
                                  "stopped_reason": reason}, indent=2, default=str))
        _info(suite_env_id, f"round {rnd}: sr={sr:.3f} q={q}/{budget} added={added} stop={reason}")

    if hasattr(expert, "reset_counts"):
        expert.reset_counts()
    _point(0, 0, float(init_sr), 0, None)

    def _retrain(rnd):
        nonlocal policy
        if round_epoch is not None:
            cfg.epoch = int(round_epoch)
        if max_train_steps is not None:
            cfg.max_train_steps = int(max_train_steps)
        res = train_policy(policy, dataset, cfg,
                           checkpoint_dir=str(work / f"checkpoints/round_{rnd:03d}"),
                           log_dir=None, fine_tune=False)
        if res.get("status") != "ok":
            raise RuntimeError(f"train_policy failed: {res.get('error')}")
        policy = res["policy"]
        _calibrate_diffusion(policy)

    nd = max(1, int(nd_retrain))
    interventions = 0
    last_trained = 0
    ep_seed = 0
    episodes = 0
    rnd_idx = 0                       # hybrid: round index for cluster/coverage memory
    max_episodes = int(hp.get("max_episodes", 1000))
    stopped = "max_rounds"

    def _prescribe_and_collect():
        """Render+VLM+reason the top-3 failures, then LLM-prescribe ONE cube layout
        and collect a demo; re-prescribe (banking infeasible layouts) until a demo
        is kept or infeasible_attempts is exhausted. Returns (ep_tds|None,
        applied|None).

        V3-HYBRID: when ``hybrid_planner`` is set, a decision-context block is
        appended to the prescriber prompt and the prescriber's rationale carries a
        SELECT/BRIDGE tag. SELECT → on-policy correction of the cited REAL failure
        (``_correct_onpolicy_from``); BRIDGE → motion-planner solve of the
        prescribed middle-ground layout (``_collect_prescribed_demo``). Each retry
        escalates to SELECT (safety floor). Plain p4_top3 (planner None) is
        unchanged."""
        # render + VLM analyze + reason each of the top-3
        analyses, fsum = [], []
        for ci, cand in enumerate(top):
            an = None
            if use_vlm and client is not None:
                try:
                    frames = _render_failure_frames(env, cfg, cand, render_dir,
                                                    ep_id=ep_seed * 10 + ci)
                    vr = VLM.analyze_failure(client, task_desc, frames, int(cand["t_star"]))
                    an = analyzer.run(vr)
                except Exception as exc:
                    print(f"[p4_top3] analyze failed ({type(exc).__name__}: {exc})", flush=True)
                    an = None
            if an is None:   # text-only (no render) — default for the StackCube hybrid
                an = _hybrid_text_analysis(hybrid_planner, cand)
            analyses.append(an)
            fsum.append({"t_star": int(cand["t_star"]),
                         "peak_loss": round(float(cand["peak_disc"]), 4),
                         "final_dist": None, "steps": int(cand["n_steps"]),
                         "root_cause": an.get("root_cause"), "phase": an.get("phase")})
        analysis_json = json.dumps(analyses)
        decision_block = hybrid_planner.decision_addendum() if hybrid_planner else ""
        infeasible_history: List[Dict] = []
        for attempt in range(max(1, infeasible_attempts)):
            demo_seed = 90000 + interventions * 100 + attempt
            if client is None:
                pres = None    # env-sampled (random) corrective config
            else:
                try:
                    add = _infeasible_addendum(infeasible_history)
                    if decision_block:
                        add = (add + "\n\n" + decision_block) if add else decision_block
                    pres = prescriber.prescribe(analysis_json, fsum, extra_addendum=add)
                except PrescriptionEmptyError as exc:
                    print(f"[p4_top3] empty prescription ({exc}); "
                          f"falling back to random config", flush=True)
                    pres = None

            if hybrid_planner is not None:
                spec = hybrid_planner.decide(pres, attempt=attempt)
                if spec.mode == "onpolicy_correction":
                    ep_tds, ok, esteps = _correct_onpolicy_from(
                        env, expert, policy, cfg, spec.cand)
                    applied = {"mode": "select", "choice": spec.choice,
                               "seed": spec.seed, "episode_id": spec.episode_id}
                else:  # bridge → prescribed middle-ground cube layout
                    ep_tds, ok, esteps, layout = _collect_prescribed_demo(
                        reposition_env, expert, cfg, spec.layout, demo_seed)
                    applied = {"mode": "bridge", "choice": spec.choice,
                               "cited": spec.cited, **(layout or {})}
                hybrid_planner.note_collect(
                    spec, {"success": ok, "episode_length": esteps, "applied": applied},
                    attempt=attempt)
                if ok and ep_tds:
                    return ep_tds, applied
                infeasible_history.append({**(applied or {}),
                                           "reason": f"unsolved in {esteps} steps"})
                continue

            ep_tds, ok, esteps, applied = _collect_prescribed_demo(
                reposition_env, expert, cfg, pres, demo_seed)
            if ok and ep_tds:
                return ep_tds, applied
            infeasible_history.append({**(applied or {}),
                                       "reason": f"unsolved in {esteps} steps"})
            if pres is None:      # random already failed; retrying random rarely helps
                continue
        return None, None

    while interventions < budget and episodes < max_episodes:
        cands = []
        for _j in range(n_cand):
            cands.append(_safe_rollout_one(env, expert, policy, cfg, ep_seed,
                                           num_joints, signal_mode=signal_mode))
            ep_seed += 1
            episodes += 1
        fails = [c for c in cands if not c["success"]]
        if not fails:
            continue
        if hybrid_planner is not None:
            descs = [d for d in (_stackcube_descriptor(env, cfg, c) for c in fails)
                     if d is not None]
            hybrid_planner.set_round(rnd_idx, descs)
            rnd_idx += 1
        top = sorted(fails, key=lambda c: -c["peak_disc"])[:3]
        ep_tds, applied = _prescribe_and_collect()
        added = 0
        if ep_tds:
            _append_episode_to_dataset(dataset, ep_tds)
            interventions += 1
            added = 1
        if added and (interventions % nd == 0 or interventions >= budget):
            n_added_since = interventions - last_trained
            _retrain(interventions)
            sr = evaluate_heldout(policy, eval_env, heldout_seed_base, heldout_num_ep, ah, mes)
            last_trained = interventions
            reason = "budget_exhausted" if interventions >= budget else None
            _point(interventions, interventions, sr, n_added_since, reason)
            if sr >= target_sr:
                stopped = "target_hit"; break
            if interventions >= budget:
                stopped = "budget_exhausted"; break

    if stopped == "max_rounds" and episodes >= max_episodes:
        stopped = "max_episodes"
    if interventions > last_trained:
        _retrain(interventions)
        sr = evaluate_heldout(policy, eval_env, heldout_seed_base, heldout_num_ep, ah, mes)
        _point(interventions, interventions, sr, interventions - last_trained,
               "budget_exhausted" if interventions >= budget else stopped)

    final = curve[-1] if curve else {}
    return {"method": method, "env": suite_env_id, "stopped_reason": stopped,
            "final_performance": {"success_rate": final.get("success_rate"),
                                  "n_queries": interventions},
            "history": curve, "elapsed_seconds": time.time() - t0,
            "total_expert_calls": interventions}
