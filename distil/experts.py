"""Scripted expert oracles for robosuite UR5e Lift and Wipe.

These are *reactive* closed-loop controllers: given the current simulator state
they return a good action toward task completion, so they can be queried at any
state during a Diff-DAgger rollout (when the policy hands over control mid-episode
the expert resumes correctly without any episode history). They read privileged
ground-truth state (cube pose, dirt-marker positions) from the raw env.

Both use the UR5e default OSC_POSE controller. Action layout:
    [dx, dy, dz, dax, day, daz, gripper]   (Lift, 7-dim, last = gripper)
    [dx, dy, dz, dax, day, daz]            (Wipe, 6-dim, WipingGripper has no DOF)
Position deltas in [-1, 1] map to +/-0.05 m/step; we keep orientation fixed
(dax=day=daz=0) since the UR5e init pose already points the tool/gripper down.
"""
import numpy as np


def _clip(x):
    return np.clip(x, -1.0, 1.0)


class LiftExpert:
    """Pick-and-lift oracle for the Lift task (UR5e + Robotiq85, OSC_POSE).

    Reactive phase machine driven purely by geometry + grasp detection:
      APPROACH : move above the cube, gripper open
      DESCEND  : lower onto the cube, gripper open
      CLOSE    : at the cube, command the gripper shut until a grasp is detected
      LIFT     : grasp detected -> raise the cube above the success margin
    """

    def __init__(
        self,
        kp: float = 12.0,
        hover_height: float = 0.06,
        lift_height: float = 0.18,
        align_tol: float = 0.012,
        grasp_z_tol: float = 0.020,
        grasp_target_dz: float = 0.0,
    ):
        self.kp = kp
        self.hover_height = hover_height
        self.lift_height = lift_height
        self.align_tol = align_tol
        self.grasp_z_tol = grasp_z_tol
        self.grasp_target_dz = grasp_target_dz
        self.action_dim = 7

    def reset(self, env=None):
        pass

    def get_action(self, env, state=None) -> np.ndarray:
        raw = env.raw
        eef = np.asarray(raw.sim.data.site_xpos[raw.robots[0].eef_site_id["right"]], dtype=np.float64)
        cube = np.asarray(raw.sim.data.body_xpos[raw.cube_body_id], dtype=np.float64)
        table_h = float(raw.model.mujoco_arena.table_offset[2])
        grasped = bool(raw._check_grasp(gripper=raw.robots[0].gripper, object_geoms=raw.cube))

        xy_err = cube[:2] - eef[:2]
        xy_dist = float(np.linalg.norm(xy_err))
        z_gap = eef[2] - cube[2]

        if grasped or cube[2] > table_h + 0.045:
            # LIFT (also latches once the cube is clearly off the table)
            target = np.array([cube[0], cube[1], table_h + self.lift_height])
            grip = 1.0
        elif xy_dist < self.align_tol and z_gap < self.grasp_z_tol:
            # CLOSE: hold over the cube and shut the gripper
            target = np.array([cube[0], cube[1], cube[2] + self.grasp_target_dz])
            grip = 1.0
        elif xy_dist < self.align_tol:
            # DESCEND with open gripper
            target = np.array([cube[0], cube[1], cube[2] + self.grasp_target_dz])
            grip = -1.0
        else:
            # APPROACH above the cube, gripper open
            target = np.array([cube[0], cube[1], cube[2] + self.hover_height])
            grip = -1.0

        pos_act = _clip(self.kp * (target - eef))
        return np.array([pos_act[0], pos_act[1], pos_act[2], 0.0, 0.0, 0.0, grip])


class WipeExpert:
    """Table-wiping oracle for the Wipe task (UR5e + WipingGripper, OSC_POSE).

    Reactive greedy controller: press the wiping pad onto the table and move
    toward the nearest not-yet-wiped dirt marker. As markers under the pad are
    cleared they leave `wiped_markers`, so the nearest-marker target advances
    and the pad sweeps across the dirt. Marker world positions are read from the
    simulator (they are not in the observation).
    """

    def __init__(
        self,
        kxy: float = 15.0,
        kz: float = 20.0,
        press_depth: float = 0.015,
        descend_xy_gate: float = 0.05,
    ):
        self.kxy = kxy
        self.kz = kz
        self.press_depth = press_depth
        self.descend_xy_gate = descend_xy_gate
        self.action_dim = 6

    def reset(self, env=None):
        pass

    @staticmethod
    def _remaining_marker_positions(raw):
        wiped = set(id(m) for m in raw.wiped_markers)
        pos = []
        for m in raw.model.mujoco_arena.markers:
            if id(m) in wiped:
                continue
            bid = raw.sim.model.body_name2id(m.root_body)
            pos.append(raw.sim.data.body_xpos[bid])
        return np.asarray(pos, dtype=np.float64) if pos else np.zeros((0, 3))

    def get_action(self, env, state=None) -> np.ndarray:
        raw = env.raw
        eef = np.asarray(raw.sim.data.site_xpos[raw.robots[0].eef_site_id["right"]], dtype=np.float64)
        table_h = float(np.asarray(raw.table_offset)[2])
        markers = self._remaining_marker_positions(raw)

        if markers.shape[0] == 0:
            return np.zeros(6)  # all clear, hold

        # nearest remaining marker in the table (x, y) plane
        d = np.linalg.norm(markers[:, :2] - eef[:2], axis=1)
        target_xy = markers[int(np.argmin(d)), :2]
        target_z = table_h - self.press_depth

        xy_err = target_xy - eef[:2]
        xy_dist = float(np.linalg.norm(xy_err))

        dz = _clip(self.kz * (target_z - eef[2]))
        # if still high above the table and not yet over a marker, prioritise
        # descending straight down before sweeping sideways
        if (eef[2] - table_h) > 0.03 and xy_dist > self.descend_xy_gate:
            dx = dy = 0.0
        else:
            dxy = _clip(self.kxy * xy_err)
            dx, dy = float(dxy[0]), float(dxy[1])
        return np.array([dx, dy, dz, 0.0, 0.0, 0.0])


class DoorExpert:
    """Door-opening oracle (UR5e + Robotiq85, OSC_POSE, use_latch=False).

    Reactive controller:
      APPROACH : move the end-effector onto the handle, gripper open
      ENGAGE   : at the handle, close the gripper
      OPEN     : drive the handle along the hinge's opening arc while servoing
                 back onto the handle to keep contact, until hinge_qpos > 0.3

    The opening direction is computed exactly from the hinge joint's world
    anchor + axis (tangent = axis x radial), so there is no guess about which
    way to pull/push: cross(axis, radial) points along increasing hinge angle.
    """

    def __init__(self, kp: float = 10.0, lead: float = 0.20, hover: float = 0.16,
                 press_z: float = 0.0, align_tol: float = 0.02,
                 grasp_trigger: float = 0.12, settle: int = 18,
                 no_progress_steps: int = 20, progress_eps: float = 0.008):
        self.kp = kp
        self.lead = lead                 # how far ahead along the arc to aim
        self.hover = hover               # align height above the handle
        self.press_z = press_z           # grasp/drag target height above handle
        self.align_tol = align_tol       # in-plane tolerance before descending
        self.grasp_trigger = grasp_trigger  # eef within handle_z+this -> grasp
        self.settle = settle             # steps to hold the closed grasp
        # closed-loop retry: if pulling makes no hinge progress for this many
        # steps the grasp missed/slipped -> lift and re-grasp
        self.no_progress_steps = no_progress_steps
        self.progress_eps = progress_eps
        self.action_dim = 7

    def reset(self, env=None):
        self.phase = "approach"   # approach -> descend -> grasp -> pull (retry)
        self.timer = 0
        self.pull_ref_hinge = 0.0
        self.no_progress = 0

    @staticmethod
    def _hinge_frame(raw):
        jid = raw.sim.model.joint_name2id(raw.door.hinge_joint)
        anchor = np.asarray(raw.sim.data.xanchor[jid], dtype=np.float64)
        axis = np.asarray(raw.sim.data.xaxis[jid], dtype=np.float64)
        axis = axis / (np.linalg.norm(axis) + 1e-9)
        return anchor, axis

    def get_action(self, env, state=None) -> np.ndarray:
        raw = env.raw
        eef = np.asarray(raw.sim.data.site_xpos[raw.robots[0].eef_site_id["right"]], dtype=np.float64)
        handle = np.asarray(raw._handle_xpos, dtype=np.float64)
        hinge = float(raw.sim.data.qpos[raw.hinge_qpos_addr])

        anchor, axis = self._hinge_frame(raw)
        radial = handle - anchor
        radial = radial - np.dot(radial, axis) * axis      # project into hinge plane
        tangent = np.cross(axis, radial)                   # +ve hinge-angle direction
        tangent = tangent / (np.linalg.norm(tangent) + 1e-9)

        xy_dist = float(np.linalg.norm((handle - eef)[:2]))
        grasp_z = handle[2] + self.press_z

        # ---- closed-loop phase machine with grasp-retry ----
        if self.phase == "approach":
            target = np.array([handle[0], handle[1], handle[2] + self.hover])
            grip = -1.0
            if xy_dist < self.align_tol:
                self.phase = "descend"
        elif self.phase == "descend":
            target = np.array([handle[0], handle[1], grasp_z])
            grip = -1.0
            if (eef[2] - handle[2]) < self.grasp_trigger:
                self.phase, self.timer = "grasp", 0
        elif self.phase == "grasp":
            target = np.array([handle[0], handle[1], grasp_z])
            grip = 1.0
            self.timer += 1
            if self.timer > self.settle:
                self.phase = "pull"
                self.pull_ref_hinge, self.no_progress = hinge, 0
        else:  # "pull": drag along the arc; retry the grasp if no hinge progress
            drag_xy = (handle + self.lead * tangent)[:2]
            target = np.array([drag_xy[0], drag_xy[1], grasp_z])
            grip = 1.0
            if hinge > self.pull_ref_hinge + self.progress_eps:
                self.pull_ref_hinge, self.no_progress = hinge, 0   # progressing
            else:
                self.no_progress += 1
            if self.no_progress > self.no_progress_steps:
                self.phase = "approach"        # grasp slipped/missed -> re-grasp

        pos = _clip(self.kp * (target - eef))
        return np.array([pos[0], pos[1], pos[2], 0.0, 0.0, 0.0, grip])


EXPERTS = {"Lift": LiftExpert, "Wipe": WipeExpert, "Door": DoorExpert}


def make_expert(task: str, **kwargs):
    return EXPERTS[task](**kwargs)
