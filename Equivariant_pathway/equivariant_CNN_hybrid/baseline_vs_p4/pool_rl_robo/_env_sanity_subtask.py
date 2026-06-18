"""GPU env-sanity for PushT-Subtask-v0 + collect_subtask_demo (no LLM).

Validates the riskiest new mechanism: setting the robot qpos at reset so the
expert starts MID-TASK at a failure config, while the prescribed tee pose also
applies. Builds the REAL cfg + PPO expert + reposition env exactly as the
orchestrator does, then runs one sub-task demo and checks achieved≈target.
"""
import types
from types import SimpleNamespace

PKG = "Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo"
import importlib
C = importlib.import_module(PKG + ".orchestrator._common")
MS = importlib.import_module(PKG + ".envs.maniskill_env")
X = importlib.import_module(PKG + ".envs.experts")
E = importlib.import_module(PKG + ".envs.env_setup")
collect = importlib.import_module(PKG + ".p4_subtask.collect")

import math, json

ENV = "PushT-v1"
cfg_path = E.SUITE_ROOT / "config_p4_subtask.yaml"
scfg = C.load_suite_cfg(cfg_path)
args = SimpleNamespace(initial_demos=None, budget=None, target_sr=None,
                       max_rounds=None, seed=1, smoke=True)
k = C.resolve_knobs(scfg, args)
k = C.apply_smoke(k)
E.register_envs()
cfg = C.build_cfg(ENV, k)

print("[sanity] building reposition env (PushT-Subtask-v0) + expert ...", flush=True)
repo = MS.make_reposition_env(cfg, ENV, prefer_subtask=True)
expert = X.load_expert(cfg, ENV)
print("[sanity] repo env class:", type(repo.unwrapped).__name__, flush=True)
assert type(repo.unwrapped).__name__ == "PushTSubtaskEnv", "wrong reposition env class"

# canonical panda_stick keyframe (WhiteTableSceneBuilder) + a clear joint-0 shift
CANON = [0.662, 0.212, 0.086, -2.685, -0.115, 2.898, 1.673]
agent_qpos = [CANON[0] + 0.30] + CANON[1:] + [0.0, 0.0]   # 9-dof; joint0 +0.30
tee_xyz = [0.05, -0.12, 0.021]
tee_zrot = 0.7
tcp_xyz = [0.10, -0.15, 0.04]

# ---- control: agent_qpos=None must land the arm at the CANONICAL keyframe ----
base = repo.unwrapped
base._tee_init_xyz = tee_xyz
base._tee_init_zrot = tee_zrot
base._agent_init_qpos = None
repo.reset(seed=123)
q_canon = base.agent.robot.qpos.detach().cpu().numpy().flatten().tolist()
print(f"[sanity] control (qpos=None) joint0={q_canon[0]:.4f} (expect ~{CANON[0]})", flush=True)

# ---- test: collect_subtask_demo with the failure-config arm qpos -------------
print("[sanity] running collect_subtask_demo with agent_qpos set ...", flush=True)
res = collect.collect_subtask_demo(repo, expert, cfg, tee_xyz, tee_zrot, tcp_xyz,
                                   agent_qpos=agent_qpos, seed=4242)
ach_tee = res["achieved_tee_xyz"]
ach_q = res["achieved_agent_qpos"]
tee_err = math.dist(ach_tee[:2], tee_xyz[:2])
j0 = ach_q[0] if ach_q else None
print(json.dumps({
    "success": res["success"], "episode_length": res["episode_length"],
    "target_tee_xyz": tee_xyz, "achieved_tee_xyz": [round(v,4) for v in ach_tee],
    "tee_xy_err": round(tee_err, 5),
    "achieved_joint0_at_reset_NOTE": "achieved_q is post-rollout, not at reset",
}, indent=2), flush=True)

# The decisive check is the CONTROL vs OVERRIDE at RESET time. Re-do a reset-only
# probe to read qpos immediately after reset (before the expert moves the arm).
base._tee_init_xyz = tee_xyz
base._tee_init_zrot = tee_zrot
base._agent_init_qpos = agent_qpos
repo.reset(seed=4242)
q_set = base.agent.robot.qpos.detach().cpu().numpy().flatten().tolist()
tee_set = base.tee.pose.p.detach().cpu().numpy().flatten().tolist()
j0_set = q_set[0]
tee_err2 = math.dist(tee_set[:2], tee_xyz[:2])
print(f"[sanity] OVERRIDE reset: joint0={j0_set:.4f} (expect ~{agent_qpos[0]:.3f}), "
      f"tee={[round(v,4) for v in tee_set[:3]]} tee_xy_err={tee_err2:.5f}", flush=True)

ok_qpos = abs(j0_set - agent_qpos[0]) < 0.05
ok_canon = abs(q_canon[0] - CANON[0]) < 0.05
ok_tee = tee_err2 < 0.03
print(f"\n[sanity] qpos_override_holds={ok_qpos}  canonical_control_ok={ok_canon}  "
      f"tee_applied={ok_tee}", flush=True)
if ok_qpos and ok_canon and ok_tee:
    print("ENV_SANITY_PASS", flush=True)
else:
    print("ENV_SANITY_FAIL", flush=True)
