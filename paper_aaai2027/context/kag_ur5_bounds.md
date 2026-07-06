# KAG ground documents and prescription bounds — UR5 RoboSuite tasks (Lift, Wipe, Door)

**Status: all three KAG docs were FOUND in the author's repo** (local checkout of
`SuyogKhanal/diff-dagger-ur5` at `/weka/s226137394/diff-dagger-ur5`, commit `974775d`), at
`diffdagger_rs/p4/kag/{Lift,Door,Wipe}.json`. Nothing was authored from scratch; the found
docs were copied verbatim to `/weka/s226137394/DmNfull/paper_aaai2027/context/kag_ur5/` for
persistence. (The GitHub remote is SSH-only/private; the local checkout is the source.)

## Important naming/semantics note

The UR5 implementation does **not** use the PushT-style *relative* perturbation parameters
(`perturb_max_xy = 0.06 m`, `perturb_max_theta = 0.4 rad` in
`pool_rl_robo/config_p4_subtask.yaml`). Instead, the BRIDGE prescription places the object at
the **mean xy of the cited failures**, hard-clamped to an **absolute placement range** in
`diffdagger_rs/p4/bounds.py` (`TASK_BOUNDS` + `clamp_obj_xy`, applied in
`p4/bridge.py:set_object_pose` and `p4/planner.py:_bridge_spec`). These ranges mirror
robosuite's own `UniformRandomSampler` reset ranges, so a prescribed start can never leave the
task's native reset distribution. The paper-facing `delta_max` is therefore the **half-width of
the clamp range** (max deviation from the range centre), and `theta_max` is the max yaw
deviation the prescription can introduce.

## Bounds table

| Task | delta_max (xy, m) | theta_max (rad) | Source |
|------|-------------------|-----------------|--------|
| Lift | 0.03 (x), 0.03 (y) — clamp range x,y ∈ [-0.03, 0.03] about the table centre | 0.0 (no yaw randomisation: `rotation=None`; prescription yaw fixed at 0) | FOUND: `diffdagger_rs/p4/kag/Lift.json` (ws_cube node); bounds: `diffdagger_rs/p4/bounds.py` L18; env source: `repo/robosuite/environments/manipulation/lift.py` L328-330 |
| Wipe | n/a — SELECT-only, no scene prescription (BRIDGE infeasible) | n/a | FOUND: `diffdagger_rs/p4/kag/Wipe.json` (select_only implication); rationale: `diffdagger_rs/p4/bounds.py` L24-26 |
| Door | 0.0135 (x), 0.013 (y) — absolute frame clamp x ∈ [-0.135, -0.108], y ∈ [-0.366, -0.340] (world) | 0.0 for prescription (BRIDGE reuses the representative failure's quat, shifts xy only); native sampler yaw span 0.25 (yaw ∈ [-π/2-0.25, -π/2], i.e. ±0.125 about centre) | FOUND: `diffdagger_rs/p4/kag/Door.json` (ws_door node); bounds: `diffdagger_rs/p4/bounds.py` L23; quat-reuse: `diffdagger_rs/p4/planner.py:_bridge_spec`; env source: `repo/robosuite/environments/manipulation/door.py` L310-313 |

All repo paths are relative to `/weka/s226137394/diff-dagger-ur5/`.

## Per-task KAG content

### Lift (`diffdagger_rs/p4/kag/Lift.json`) — FOUND
UR5e + Robotiq85 parallel-jaw gripper under an OSC_POSE (6-DOF) controller grasps a small cube
resting on a table at z ≈ 0.831 m and lifts it above a height threshold while grasped. The
reliable cube-init workspace (`ws_cube`) is x ∈ [-0.03, 0.03], y ∈ [-0.03, 0.03], z = 0.831
(world-frame metres) — exactly robosuite's `UniformRandomSampler` range, confirmed empirically
over 12 seeds per the bounds.py comment. Failure modes covered: grasp_failure,
approach_failure, contact_instability, timeout; the `workspace_constraint` implication mandates
every prescribed cube xy stay inside the range.

### Wipe (`diffdagger_rs/p4/kag/Wipe.json`) — FOUND
UR5e with a 0-DOF wiping pad (WipingGripper) under OSC_POSE presses down and sweeps a trail of
~100 arena-randomised dirt markers until all are cleared (coverage = 1.0). Because the
randomised quantity is a whole marker *path* rather than a single object pose, the KAG's
`select_only` implication makes BRIDGE infeasible: the demonstration choice is always SELECT of
a representative failed episode, so no perturbation bound exists or is needed. Failure modes:
contact_instability, approach_failure, placement_error, timeout.

### Door (`diffdagger_rs/p4/kag/Door.json`) — FOUND
UR5e + Robotiq85 under OSC_POSE reaches a door handle and pulls the door open past a hinge
angle of 0.3 rad; the door frame's position and yaw are randomised each episode. The reliable
door-frame workspace (`ws_door`) is the measured absolute body_xpos range x ∈ [-0.135, -0.108],
y ∈ [-0.366, -0.340], z = 1.10, yaw ∈ [-1.82, -1.57] rad (padded empirical measurement of the
sampler range x_range=[0.07,0.09], y_range=[-0.01,0.01], rotation=(-π/2-0.25, -π/2) relative to
table_offset (-0.2, -0.35, 0.8)). The door has no free joint, so BRIDGE sets the frame body's
model pose directly, keeping the representative failure's z and quaternion (no rotational
prescription) and only shifting xy to the clamped middle ground. Failure modes:
approach_failure, grasp_failure, pose_mismatch, contact_instability, timeout.
