# KAG — FetchReach-v4

domain: gymnasium-robotics Fetch manipulation (goal-conditioned, sparse reward)
env_id: FetchReach-v4
body: 7-DoF Fetch arm; control is end-effector Cartesian displacement
goal_type: goal-reaching with a real binary success signal (info['is_success'])

## Task
- objective: move the end-effector (gripper) to a randomly sampled 3D goal position
- horizon: 50 steps (short), episode truncates at the cap

## Observation (Dict -> flattened as observation + achieved_goal + desired_goal = 16 dims)
- observation (10): grip_pos(3), gripper_state(2), grip_velp(3), gripper_velocity(2)
- achieved_goal (3): current end-effector position
- desired_goal (3): target position to reach
- KEY FEATURE: gripper_to_goal_distance = ||achieved_goal - desired_goal|| (provided per
  candidate state as 'gripper_to_goal_dist'); success when this < 0.05 m (5 cm)

## Action (Box, 4 dims, range [-1, 1])
- [dx, dy, dz, gripper] Cartesian end-effector displacement; the gripper dim is unused for reaching

## Reward
- sparse: 0.0 when within 5 cm of the goal, -1.0 otherwise (per step). Episode return is
  roughly -(steps until reached); success_rate is the meaningful metric here.

## Success criterion (this suite)
- info['is_success'] == 1 (end-effector within 5 cm of desired_goal). Real, not a proxy.

## What a good demonstration teaches
- move the end-effector directly and efficiently toward the goal, reducing
  gripper_to_goal_distance to < 5 cm quickly

## Common novice failure modes (where an expert demo is most corrective)
- moving away from / orthogonal to the goal (gripper_to_goal_dist stays high or grows)
- overshooting or oscillating around the goal without settling inside 5 cm
- stalling far from the goal

## Selecting an informative state
- prefer HIGH-discrepancy states with LARGE gripper_to_goal_dist (the novice is far from
  the goal and disagrees with the expert) — an expert reach-demonstration from there
  teaches the corrective direction toward the target.
