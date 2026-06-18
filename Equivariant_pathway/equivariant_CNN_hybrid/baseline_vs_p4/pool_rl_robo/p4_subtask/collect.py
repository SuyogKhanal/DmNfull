"""collect_subtask_demo — materialize a sub-task ResetSpec into a PPO-expert demo.

A faithful clone of the fork's ``collect_prescribed_demo``
(``diffdagger/main_pipeline/sim_bridge.py``), extended to ALSO set the robot qpos
(``_agent_init_qpos``) on a PushT-Subtask-v0 reposition env so the expert starts
mid-task at the failure sub-task. The fork's sim_bridge is never edited; this
lives suite-side and is INJECTED into the pipeline (``pipe._subtask_collect``),
so the fork carries no suite import.

Success semantics, budget accounting, and dataset augmentation are IDENTICAL to
``collect_prescribed_demo`` (the budget unit stays "one successful demo"), so the
equal-to-equal comparison with diff_dagger is preserved.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch


def collect_subtask_demo(reposition_env, expert, cfg,
                         tee_xyz: List[float], tee_zrot: float,
                         tcp_xyz: List[float],
                         agent_qpos: Optional[List[float]] = None,
                         seed: int = 9999) -> Dict[str, Any]:
    # Imported lazily: the fork is on sys.path via env_setup.bootstrap_fork_path().
    from diffdagger.main_pipeline.sim_bridge import _move_tcp_to_target

    base_env = reposition_env.unwrapped          # PushTSubtaskEnv

    # 1. Prescribed tee pose (the primary lever) + optional arm sub-task entry.
    base_env._tee_init_xyz = [float(v) for v in tee_xyz]
    base_env._tee_init_zrot = float(tee_zrot)
    base_env._agent_init_qpos = ([float(v) for v in agent_qpos]
                                 if agent_qpos is not None else None)

    # 2. Reset → _initialize_episode applies the tee pose AND (if set) the arm qpos.
    obs, info = reposition_env.reset(seed=seed)

    # 3. TCP nudge ONLY when the arm was NOT placed at a failure config
    #    (when agent_qpos is set the arm already starts near the contact; nudging
    #    would step physics 120× and disturb the carefully placed sub-task state).
    if agent_qpos is None:
        try:
            _move_tcp_to_target(reposition_env, base_env, tcp_xyz)
        except Exception as e:
            print(f"[SubtaskDemo] TCP move skipped ({e}) — expert starts from reset")

    achieved_tee = base_env.tee.pose.p.detach().cpu().numpy().flatten().tolist()
    try:
        achieved_q = base_env.agent.robot.qpos.detach().cpu().numpy().flatten().tolist()
    except Exception:
        achieved_q = None
    print(f"[SubtaskDemo] target tee={[round(float(v),3) for v in tee_xyz]} "
          f"achieved tee={[round(float(v),3) for v in achieved_tee[:3]]} "
          f"agent_qpos={'set' if agent_qpos is not None else 'canonical'}")

    # 4. PPO expert (joint_pos), identical to collect_prescribed_demo.
    reposition_env.set_action_space("joint_pos")
    ep_info = dict(seed=seed)
    expert.reset(reposition_env)
    expert.setup_task()

    ep_tds = []
    done = False
    timestep = 0
    limit = cfg.max_episode_steps + cfg.expert.max_episode_steps
    while not done and timestep < limit:
        td = expert.move_to_next_goal(ep_info)
        td.update(dict(episode=torch.ones(len(td["episode"])) * seed))
        ep_tds.append(td)
        done = td["done"][-1] if "done" in td else False
        timestep += len(td.get("episode", []))
        if done:
            break

    total_steps = sum(len(td.get("episode", [])) for td in ep_tds)
    success = (bool(done)
               and total_steps <= cfg.expert.max_episode_steps
               and total_steps >= 10)

    return {
        "success": success,
        "trajectory": ep_tds if success else [],
        "episode_length": timestep,
        "target_tee_xyz": [float(v) for v in tee_xyz],
        "achieved_tee_xyz": [float(v) for v in achieved_tee[:3]],
        "target_agent_qpos": ([float(v) for v in agent_qpos]
                              if agent_qpos is not None else None),
        "achieved_agent_qpos": achieved_q,
    }


# ---------------------------------------------------------------------------
# v2 — DAgger-faithful ON-POLICY correction (p4_select's winning mechanism)
# ---------------------------------------------------------------------------

def collect_onpolicy_correction(env, policy, expert, cfg, scene_seed: int,
                                *, signal_mode: str = "diffusion") -> Dict[str, Any]:
    """Collect ONE on-policy expert correction at a REAL failure's scene.

    Mirrors how Diff-DAgger's internal states work while collecting (the user's
    hard requirement): (1) reset the POLICY env to the failure's seed — the same
    scene; (2) roll the CURRENT policy recording its own per-step diffusion loss
    (the SAME detector diff_dagger queries); (3) replay the prefix to the argmax-
    loss step t*; (4) the EXPERT takes over from that visited mid-episode state
    and finishes — only the expert segment becomes the demo. If the re-roll
    SUCCEEDS, there is nothing to correct → budget-free skip (caller escalates
    to the next real failure). Reuses p4_select's battle-tested helpers
    (`_safe_rollout_one` / `_correct_onpolicy_from`) — the variant that already
    BEAT diff_dagger 5/5 on PushT.
    """
    from ..p4.select_arm import _safe_rollout_one, _correct_onpolicy_from

    num_joints = int(env.num_joints)
    cand = _safe_rollout_one(env, expert, policy, cfg, int(scene_seed), num_joints,
                             signal_mode=signal_mode)
    if cand["success"]:
        # The current policy now solves this scene — no correction possible.
        try:
            env.set_action_space(cfg.action_space)
        except Exception:
            pass
        print(f"[OnPolicyDemo] seed={scene_seed}: re-roll SUCCEEDED in "
              f"{cand['n_steps']} steps — nothing to correct (skip, budget-free)")
        return {"success": False, "trajectory": [], "episode_length": 0,
                "skipped": "solved_on_reroll", "scene_seed": int(scene_seed),
                "t_star": int(cand["t_star"]), "reroll_steps": int(cand["n_steps"]),
                "peak_signal": float(cand["peak_disc"])}

    ep_tds, ok, total = _correct_onpolicy_from(env, expert, policy, cfg, cand)
    try:
        env.set_action_space(cfg.action_space)   # restore the policy action space
    except Exception:
        pass
    print(f"[OnPolicyDemo] seed={scene_seed}: failure re-rolled "
          f"({cand['n_steps']} steps, t*={cand['t_star']}, "
          f"peak_loss={cand['peak_disc']:.5f}) → expert correction "
          f"{'OK' if ok else 'FAILED'} ({total} steps)")
    return {"success": bool(ok), "trajectory": ep_tds if ok else [],
            "episode_length": int(total), "skipped": None,
            "scene_seed": int(scene_seed), "t_star": int(cand["t_star"]),
            "reroll_steps": int(cand["n_steps"]),
            "peak_signal": float(cand["peak_disc"])}


def collect_subtask_demo_v2(*, reposition_env, env, policy, expert, cfg, spec,
                            seed: int) -> Dict[str, Any]:
    """Mode dispatcher for the fork's collect hook (hook C).

    spec.mode == "onpolicy_correction" → v2 DAgger-faithful path on the POLICY
    env at the real failure's scene seed. Any other mode → the legacy v1
    teleport-reset path on the reposition env (kept for ablations)."""
    if spec.mode == "onpolicy_correction" and spec.seed is not None:
        return collect_onpolicy_correction(env, policy, expert, cfg, spec.seed)
    return collect_subtask_demo(
        reposition_env, expert, cfg, spec.tee_xyz, spec.tee_zrot, spec.tcp_xyz,
        agent_qpos=spec.agent_qpos, seed=seed)
