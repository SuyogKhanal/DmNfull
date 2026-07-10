"""robosuite UR5e environments for Diff-DAgger (Lift, Wipe).

A thin wrapper around `robosuite.make` that:
  * binds the UR5e robot with its default OSC_POSE composite controller,
  * runs head-less and state-only (no GL needed for physics; offscreen render
    only enabled if you ask for it),
  * flattens the observation dict into a fixed-order float32 state vector,
  * exposes a uniform step() returning (state, reward, done, success, info),
  * supports reproducible per-episode reseeding of the object/robot placement.

The *raw* robosuite env is kept on `.env` so the privileged scripted experts
can read ground-truth simulator state (cube pose, dirt-marker positions, ...).
"""
import os
from typing import List, Optional

import numpy as np

# Pure physics stepping needs no GL context; default to egl so the same code
# can also render offscreen on a GPU node. Override with MUJOCO_GL=osmesa on
# CPU-only nodes. Set before importing robosuite.
os.environ.setdefault("MUJOCO_GL", "egl")

import robosuite as suite  # noqa: E402
from robosuite import load_composite_controller_config  # noqa: E402


# Per-task flat-state observation keys (ordered). Scalars are reshaped to (1,).
LIFT_STATE_KEYS = [
    "robot0_eef_pos",        # 3
    "robot0_eef_quat",       # 4
    "robot0_gripper_qpos",   # 6
    "cube_pos",              # 3
    "cube_quat",             # 4
    "gripper_to_cube_pos",   # 3
]
WIPE_STATE_KEYS = [
    "robot0_eef_pos",                 # 3
    "robot0_eef_quat",                # 4
    "robot0_joint_pos_cos",           # 6
    "robot0_joint_pos_sin",           # 6
    "wipe_centroid",                  # 3
    "gripper_to_wipe_centroid",       # 3
    "wipe_radius",                    # 1
    "proportion_wiped",               # 1
    "robot0_contact",                 # 1
]
DOOR_STATE_KEYS = [
    "robot0_eef_pos",                 # 3
    "robot0_eef_quat",                # 4
    "robot0_gripper_qpos",            # 6
    "handle_pos",                     # 3  (door object obs, like cube_pos)
    "door_pos",                       # 3
    "hinge_qpos",                     # 1  (door opening angle)
]

TASK_STATE_KEYS = {"Lift": LIFT_STATE_KEYS, "Wipe": WIPE_STATE_KEYS, "Door": DOOR_STATE_KEYS}


def _reseed_placement(env, seed: int):
    """Reseed the env master RNG and every (possibly nested) placement sampler
    so a given seed reproduces the same initial configuration.

    Note: with hard_reset=False the arena is built once and holds its OWN rng
    reference (e.g. WipeArena samples the dirt-marker layout from arena.rng in
    reset_arena), so we must reseed that object too -- rebinding env.rng alone
    does NOT make Wipe resets reproducible.
    """
    rng = np.random.default_rng(seed)
    env.rng = rng
    pi = getattr(env, "placement_initializer", None)
    if pi is not None:
        pi.rng = rng
        samplers = getattr(pi, "samplers", None)
        if isinstance(samplers, dict):
            for s in samplers.values():
                s.rng = rng
        elif isinstance(samplers, (list, tuple)):
            for s in samplers:
                s.rng = rng
    arena = getattr(getattr(env, "model", None), "mujoco_arena", None)
    if arena is not None and hasattr(arena, "rng"):
        arena.rng = rng


class RobosuiteStateEnv:
    """State-only robosuite env with a flat observation vector."""

    def __init__(
        self,
        task: str,                      # "Lift" | "Wipe"
        robots: str = "UR5e",
        horizon: int = 250,
        control_freq: int = 20,
        reward_shaping: bool = True,
        state_keys: Optional[List[str]] = None,
        task_config: Optional[dict] = None,
        seed: Optional[int] = None,
        ignore_done: bool = True,       # we manage termination ourselves
        wipe_marker_obs_k: int = 0,     # Wipe: append nearest-K remaining markers
        offscreen: bool = False,        # enable an offscreen GL context for RGB render
        render_camera: str = "agentview",
        render_height: int = 512,
        render_width: int = 512,
    ):
        assert task in ("Lift", "Wipe", "Door"), task
        self.task = task
        self.wipe_marker_obs_k = wipe_marker_obs_k
        self.state_keys = state_keys or TASK_STATE_KEYS[task]
        self.offscreen = offscreen
        self.render_camera = render_camera
        self.render_height = render_height
        self.render_width = render_width
        controller = load_composite_controller_config(robot=robots)

        kwargs = dict(
            robots=robots,
            controller_configs=controller,
            has_renderer=False,
            # An offscreen GL context is created only when this is True; render()
            # then works. Physics/placement are unchanged, so a render env built
            # from the same wrapper reproduces the state env's dynamics exactly.
            has_offscreen_renderer=offscreen,
            use_camera_obs=False,        # we render manually via sim.render
            use_object_obs=True,
            reward_shaping=reward_shaping,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=False,
            seed=seed,
        )
        if task == "Wipe":
            kwargs["gripper_types"] = "WipingGripper"
            if task_config is not None:
                kwargs["task_config"] = task_config
        if task == "Door":
            # No spring latch -> the door opens by pulling the handle (the latch
            # variant requires turning a sprung handle first). Still the real
            # "open the door past 0.3 rad" task. The door's POSITION and YAW are
            # randomized each episode by robosuite's default sampler -- this real
            # covariate shift is essential for a valid Diff-DAgger experiment, so
            # the scripted oracle must be robust to it (closed-loop grasp retry).
            kwargs["use_latch"] = False
        self.env = suite.make(task, **kwargs)

        self.horizon = horizon
        self.action_dim = self.env.action_dim
        self._t = 0
        # determine state_dim from a probe reset
        obs = self.env.reset()
        self.state_dim = self._flatten(obs).shape[0]

    # ------------------------------------------------------------------
    def _flatten(self, obs) -> np.ndarray:
        parts = []
        for k in self.state_keys:
            v = np.asarray(obs[k], dtype=np.float32).reshape(-1)
            parts.append(v)
        if self.task == "Wipe" and self.wipe_marker_obs_k > 0:
            parts.append(self._wipe_marker_features())
        return np.concatenate(parts).astype(np.float32)

    def _wipe_marker_features(self) -> np.ndarray:
        """Nearest-K not-yet-wiped dirt markers, as positions relative to the
        end-effector (sorted by in-plane distance, repeat-last padded).

        The default condensed Wipe obs only exposes a single dirt CENTROID, so
        the policy cannot localize the last scattered markers. This gives it a
        compact local map of remaining dirt (object state, like cube_pos in
        Lift) so it can finish coverage instead of plateauing.
        """
        raw = self.env
        k = self.wipe_marker_obs_k
        eef = np.asarray(raw.sim.data.site_xpos[raw.robots[0].eef_site_id["right"]], dtype=np.float32)
        wiped = set(id(m) for m in raw.wiped_markers)
        pos = []
        for m in raw.model.mujoco_arena.markers:
            if id(m) in wiped:
                continue
            bid = raw.sim.model.body_name2id(m.root_body)
            pos.append(raw.sim.data.body_xpos[bid])
        feat = np.zeros((k, 3), dtype=np.float32)
        if pos:
            rel = np.asarray(pos, dtype=np.float32) - eef
            order = np.argsort(np.linalg.norm(rel[:, :2], axis=1))[:k]
            sel = rel[order]
            feat[: len(sel)] = sel
            if len(sel) < k:               # repeat-last padding (no fake "at-eef" dirt)
                feat[len(sel):] = sel[-1]
        return feat.reshape(-1)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            _reseed_placement(self.env, seed)
        obs = self.env.reset()
        self._t = 0
        return self._flatten(obs)

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        obs, reward, _done, info = self.env.step(action)
        self._t += 1
        success = bool(self.env._check_success())
        done = success or (self._t >= self.horizon)
        return self._flatten(obs), float(reward), done, success, info

    def render(self, camera: Optional[str] = None, height: Optional[int] = None,
               width: Optional[int] = None) -> np.ndarray:
        """Offscreen RGB (H,W,3) uint8 at the CURRENT sim state (needs
        offscreen=True). mjr_readPixels is bottom-up, so flip vertically."""
        if not self.offscreen:
            raise RuntimeError("render() needs offscreen=True (no GL context otherwise)")
        cam = camera or self.render_camera
        h = height or self.render_height
        w = width or self.render_width
        frame = self.env.sim.render(width=w, height=h, camera_name=cam, depth=False)
        return np.asarray(frame[::-1], dtype=np.uint8)

    @property
    def raw(self):
        return self.env

    def coverage(self):
        """Wipe: fraction of dirt markers wiped (informative partial-credit
        signal next to the strict all-markers binary success). None otherwise."""
        if self.task == "Wipe":
            return len(self.env.wiped_markers) / max(1, self.env.num_markers)
        return None

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


def make_env(task: str, **kwargs) -> RobosuiteStateEnv:
    return RobosuiteStateEnv(task, **kwargs)
