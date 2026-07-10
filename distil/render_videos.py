"""Render expert demonstration videos for Lift, Wipe, and Door (UR5e).

Builds each robosuite UR5e task with an offscreen renderer, rolls out the
scripted expert oracle, captures camera frames, and writes one MP4 per task.
Only successful episodes are saved (it retries seeds until the expert succeeds),
so teammates see a clean, completed demonstration of each task.
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse

import imageio
import numpy as np
import robosuite as suite
from robosuite import load_composite_controller_config

from .envs import _reseed_placement
from .experts import make_expert

# A thin proxy exposing `.raw` so the experts (which read env.raw) work directly
# on the raw robosuite env we build here with offscreen rendering enabled.
class _Proxy:
    def __init__(self, env):
        self.raw = env


CAMERAS = {"Lift": "frontview", "Wipe": "frontview", "Door": "frontview"}


def build_env(task, robots="UR5e"):
    controller = load_composite_controller_config(robot=robots)
    kwargs = dict(
        robots=robots,
        controller_configs=controller,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        control_freq=20,
        horizon=400,
        ignore_done=True,
        hard_reset=False,
    )
    if task == "Wipe":
        kwargs["gripper_types"] = "WipingGripper"
    if task == "Door":
        kwargs["use_latch"] = False
        kwargs["initialization_noise"] = None
        from robosuite.utils.placement_samplers import UniformRandomSampler
        kwargs["placement_initializer"] = UniformRandomSampler(
            name="ObjectSampler",
            x_range=[0.075, 0.085],
            y_range=[-0.005, 0.005],
            rotation=(-np.pi / 2.0, -np.pi / 2.0),
            rotation_axis="z",
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=(-0.2, -0.35, 0.8),
        )
    return suite.make(task, **kwargs)


def grab_frame(env, camera, w, h):
    rgb = env.sim.render(width=w, height=h, camera_name=camera)
    return np.flipud(rgb).copy()  # MuJoCo renders bottom-up


def rollout_and_record(task, out_path, w=640, h=480, fps=20, max_tries=20):
    env = build_env(task)
    camera = CAMERAS[task]
    expert = make_expert(task)
    proxy = _Proxy(env)

    for attempt in range(max_tries):
        seed = 4000 + attempt
        _reseed_placement(env, seed)
        env.reset()
        expert.reset(proxy)
        frames = []
        success = False
        success_hold = 0
        for t in range(env.horizon):
            action = expert.get_action(proxy)
            env.step(action)
            frames.append(grab_frame(env, camera, w, h))
            if env._check_success():
                success = True
                success_hold += 1
                if success_hold >= 15:  # linger a moment on the completed task
                    break
        if success:
            imageio.mimsave(out_path, frames, fps=fps, quality=8,
                            macro_block_size=None)
            print(f"[{task}] success on seed {seed}: wrote {out_path} "
                  f"({len(frames)} frames)", flush=True)
            env.close()
            return True
        print(f"[{task}] seed {seed} did not succeed, retrying...", flush=True)
    env.close()
    print(f"[{task}] FAILED to get a successful demo in {max_tries} tries", flush=True)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/videos")
    ap.add_argument("--tasks", nargs="+", default=["Lift", "Wipe", "Door"])
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for task in args.tasks:
        out = os.path.join(args.out_dir, f"{task.lower()}_ur5e_expert.mp4")
        rollout_and_record(task, out, w=args.width, h=args.height)


if __name__ == "__main__":
    main()
