# KAG — HalfCheetah-v4

domain: MuJoCo continuous locomotion (planar)
env_id: HalfCheetah-v4
body: 2D cheetah-like robot, 9 links / 6 actuated joints
goal_type: non-terminating velocity maximization (no goal set, no early termination)

## Task
- objective: run forward (+x) as fast as possible while limiting control effort
- horizon: 1000 steps, episode always truncates (the robot never "falls"/terminates)

## Observation (Box, 17 dims, flattened directly)
- semantics: qpos[1:] (8 positional: root z-height, root angle, 6 joint angles)
  followed by qvel (9 velocities: root x-vel, root z-vel, root ang-vel, 6 joint vels)
- root_x_velocity: obs[8]  (this is the forward speed — the single most important feature)
- the global x-position is EXCLUDED from the observation by default
- note: indices are approximate; treat obs[8] as forward velocity, obs[0] as body height

## Action (Box, 6 dims, range [-1, 1])
- joint torques: [back_thigh, back_shin, back_foot, front_thigh, front_shin, front_foot]

## Reward
- reward = forward_reward (proportional to root_x_velocity) - 0.1 * sum(action^2) control cost
- there is NO sparse success signal; higher cumulative reward = faster sustained running

## Success criterion (this suite)
- survival proxy: episode reaches truncation without early termination -> always True here
- implication: ThriftyDAgger task-risk saturates (~0) on this env; it then behaves like EnsembleDAgger

## What a good demonstration teaches
- sustained high forward velocity with a coordinated, periodic galloping gait
- recovery from a low-speed / stalled / tumbling body pose back into a fast gait

## Common novice failure modes (where an expert demo is most corrective)
- flipping or tumbling (large |root angle|, body height collapse)
- stalling near zero or negative forward velocity (obs[8] small or negative)
- thrashing torques (high control cost without forward progress)

## Selecting an informative state
- prefer states with HIGH action discrepancy where the novice's forward velocity is
  low/negative or its body pose is unstable — those are where expert torque guidance
  changes the gait the most.
