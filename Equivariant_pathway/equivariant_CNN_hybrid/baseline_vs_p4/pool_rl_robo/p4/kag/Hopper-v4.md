# KAG — Hopper-v4

domain: MuJoCo continuous locomotion (planar, single leg)
env_id: Hopper-v4
body: one-legged hopper (torso + thigh + leg + foot), 3 actuated joints
goal_type: forward progress with early termination on falling

## Task
- objective: hop forward (+x) as far as possible WITHOUT falling
- horizon: up to 1000 steps; episode TERMINATES early if the hopper becomes unhealthy
  (torso height too low or torso angle too tilted) — i.e. it fell

## Observation (Box, 11 dims, flattened directly)
- semantics: qpos[1:] (5 positional: torso z-height, torso angle, thigh/leg/foot joint angles)
  followed by qvel (6 velocities)
- torso_height: obs[0]   torso_angle: obs[1]   forward_velocity: obs[5]
- note: indices are approximate; obs[0]=height and obs[1]=angle govern "is it upright".

## Action (Box, 3 dims, range [-1, 1])
- joint torques: [thigh, leg, foot]

## Reward
- reward = forward_reward (root x-velocity) + healthy/alive bonus - 0.001 * control cost
- staying upright (alive) and moving forward both matter; falling ends the episode (no more reward)

## Success criterion (this suite)
- survival proxy: episode ended by truncation (survived full horizon) = success;
  early termination (fell) = failure. So success_rate here = "fraction of episodes that stayed up".

## What a good demonstration teaches
- a stable, continuous, periodic hopping cycle that maintains torso height and angle
- recovery from a near-fall (low torso height / large tilt) back to a balanced hop

## Common novice failure modes (where an expert demo is most corrective)
- falling: torso height collapses or torso angle exceeds the healthy range -> early termination
- toppling forward/backward after a single large hop
- stalling upright without forward progress

## Selecting an informative state
- prefer HIGH-discrepancy states where the torso is near the unhealthy boundary
  (low obs[0] height or large |obs[1]| angle) or just before an early termination —
  expert guidance there teaches recovery and prevents the fall.
