# KAG — Walker2d-v4

domain: MuJoCo continuous locomotion (planar, bipedal)
env_id: Walker2d-v4
body: 2D bipedal walker (torso + 2 legs, each thigh/leg/foot), 6 actuated joints
goal_type: forward walking with early termination on falling

## Task
- objective: walk forward (+x) steadily and as far as possible WITHOUT falling
- horizon: up to 1000 steps; episode TERMINATES early when unhealthy
  (torso height out of [0.8, 2.0] or torso angle out of [-1, 1] rad) — i.e. it fell

## Observation (Box, 17 dims, flattened directly)
- semantics: qpos[1:] (8 positional: torso z-height, torso angle, 6 leg joint angles)
  followed by qvel (9 velocities)
- torso_height: obs[0]   torso_angle: obs[1]   forward_velocity: obs[8]
- note: indices approximate; obs[0]=height and obs[1]=angle govern balance.

## Action (Box, 6 dims, range [-1, 1])
- joint torques: [right_thigh, right_leg, right_foot, left_thigh, left_leg, left_foot]

## Reward
- reward = forward_reward (root x-velocity) + healthy/alive bonus - 0.001 * control cost
- balance (alive) AND forward speed both matter; falling ends the episode

## Success criterion (this suite)
- survival proxy: truncation (walked the full horizon upright) = success; early
  termination (fell) = failure. success_rate = fraction of episodes that stayed upright.

## What a good demonstration teaches
- smooth, balanced, alternating bipedal gait with steady forward velocity
- recovery from an off-balance pose (torso tilting / height dropping) before a fall

## Common novice failure modes (where an expert demo is most corrective)
- falling: torso height < 0.8 or |angle| > 1 -> early termination
- hopping on one leg / dragging a foot instead of alternating
- stalling upright with little forward progress

## Selecting an informative state
- prefer HIGH-discrepancy states where balance is marginal (obs[0] near the height
  bounds or |obs[1]| large) or right before a fall — expert torques there teach the
  balancing correction that keeps the walker alive and moving.
