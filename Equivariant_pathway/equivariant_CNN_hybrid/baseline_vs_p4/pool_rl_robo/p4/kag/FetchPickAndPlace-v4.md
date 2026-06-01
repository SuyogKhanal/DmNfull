# KAG — FetchPickAndPlace-v4

domain: gymnasium-robotics Fetch manipulation (goal-conditioned, sparse reward)
env_id: FetchPickAndPlace-v4
body: 7-DoF Fetch arm with a parallel gripper; control is end-effector displacement + gripper
goal_type: pick-and-place with a real binary success signal (info['is_success'])

## Task
- objective: grasp a box on the table and place it at a randomly sampled 3D target
- horizon: 50 steps, episode truncates at the cap
- multi-phase: reach-to-object -> grasp (close gripper) -> lift -> transport -> release at target

## Observation (Dict -> flattened as observation + achieved_goal + desired_goal = 31 dims)
- observation (25): grip_pos(3), object_pos(3), object_rel_pos(3 = object - grip),
  gripper_state(2), object_rotation(3), object_velp(3), object_velr(3),
  grip_velp(3), gripper_velocity(2)
- achieved_goal (3): current OBJECT position (not the gripper!)
- desired_goal (3): target position for the object
- KEY FEATURE: object_to_target_distance = ||achieved_goal - desired_goal|| (provided per
  candidate state as 'gripper_to_goal_dist'); success when this < 0.05 m (5 cm).
  object_rel_pos (object - gripper) ~ 0 indicates the gripper is at the object (graspable).

## Action (Box, 4 dims, range [-1, 1])
- [dx, dy, dz, gripper]: Cartesian end-effector displacement + gripper open/close (last dim ACTIVE here)

## Reward
- sparse: 0.0 when the object is within 5 cm of the target, -1.0 otherwise (per step).
  success_rate is the meaningful metric; this is the hardest of the five tasks.

## Success criterion (this suite)
- info['is_success'] == 1 (object within 5 cm of desired_goal). Real, not a proxy.

## What a good demonstration teaches
- the full grasp-and-place chain: align gripper over the object, close gripper to grasp,
  lift, transport the object to the target, and release within 5 cm
- the gripper dimension matters: a demo must show closing the gripper at the object

## Common novice failure modes (where an expert demo is most corrective)
- failing to grasp: gripper never closes on the object (object_rel_pos stays large)
- knocking the object away / dropping it mid-transport
- moving the empty gripper toward the target without the object
- object_to_target_distance never drops below 5 cm

## Selecting an informative state
- prefer HIGH-discrepancy states at a critical phase boundary: gripper near the object but
  not grasping (object_rel_pos small, gripper still open), or object lifted but drifting
  from the target (object_to_target_distance large). Expert demos at these phase
  transitions are the most corrective for this long-horizon task.
