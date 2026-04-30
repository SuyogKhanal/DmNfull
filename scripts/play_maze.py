import argparse
import sys
import os
import json
import time
import math
from typing import List, Optional, Tuple

import pygame

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.maze_env import MazeNavEnv, ACTION_NAMES, CELL_SIZE, TILE_FREE, TILE_FIRE, TILE_GOAL

MAZE_NAME          = "multimodal"
DEMO_DIR           = "demos"
FIRE_MODE          = "variable"
NUM_FIRE_TILES     = 3
MIN_MANHATTAN_DIST = 4
TARGET_DEMOS       = 50
DRAG_THRESHOLD     = 15


def count_saved_demos(demo_dir: str) -> int:
    if not os.path.isdir(demo_dir):
        return 0
    return len([f for f in os.listdir(demo_dir) if f.endswith(".json")])


def _parse_pair(s: str) -> Tuple[int, int]:
    parts = [p.strip() for p in s.replace("(", "").replace(")", "").split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected 'r,c', got '{s}'")
    return (int(parts[0]), int(parts[1]))


def _parse_fires(s: str) -> List[Tuple[int, int]]:
    """Parses 'r1,c1;r2,c2;r3,c3' into [(r1,c1),(r2,c2),(r3,c3)]."""
    items = [p.strip() for p in s.split(";") if p.strip()]
    return [_parse_pair(it) for it in items]


def parse_args():
    p = argparse.ArgumentParser(description="Interactive maze demo recorder.")
    p.add_argument("--demo_dir", type=str, default=DEMO_DIR,
                   help="Directory to write demo JSON files into.")
    p.add_argument("--start", type=_parse_pair, default=None,
                   help="Force start position 'r,c' (overrides randomisation).")
    p.add_argument("--goal",  type=_parse_pair, default=None,
                   help="Force goal position 'r,c' (overrides randomisation).")
    p.add_argument("--fires", type=_parse_fires, default=None,
                   help="Force fire positions 'r1,c1;r2,c2;r3,c3'.")
    p.add_argument("--layout_id", type=str, default=None,
                   help="Tag included in saved demo filenames (e.g. 'p3_round2_demo1').")
    p.add_argument("--single_episode", action="store_true",
                   help="Quit pygame after the first saved demo (used by active_loop).")
    return p.parse_args()


def create_env(
    fire_mode: str,
    seed: Optional[int] = None,
    forced_start: Optional[Tuple[int, int]] = None,
    forced_goal: Optional[Tuple[int, int]] = None,
    forced_fires: Optional[List[Tuple[int, int]]] = None,
):
    """Creates a MazeNavEnv. When forced_* are given, randomisation is fully off
    and the env is post-edited so start, goal, and fires match the spec exactly.
    This is the path used by active_loop to launch a demo session on a layout
    the LLM recommended."""
    if forced_start is not None or forced_goal is not None or forced_fires is not None:
        env = MazeNavEnv(
            maze_name=MAZE_NAME,
            render_mode="human",
            randomize_start=False,
            randomize_goal=False,
            randomize_fire=False,
            num_fire_tiles=len(forced_fires) if forced_fires is not None else NUM_FIRE_TILES,
            seed=seed if seed is not None else int(time.time()),
            fire_positions=forced_fires,
        )
        env.reset()
        if forced_goal is not None:
            env.grid[env.grid == TILE_GOAL] = TILE_FREE
            for fr, fc in env.fire_positions:
                if (fr, fc) == tuple(forced_goal):
                    raise ValueError(f"forced goal {forced_goal} overlaps a fire cell")
            env.goal_pos = tuple(forced_goal)
            env.grid[env.goal_pos[0], env.goal_pos[1]] = TILE_GOAL
        if forced_start is not None:
            if tuple(forced_start) in env.fire_positions:
                raise ValueError(f"forced start {forced_start} overlaps a fire cell")
            if forced_goal is not None and tuple(forced_start) == tuple(forced_goal):
                raise ValueError(f"forced start equals forced goal: {forced_start}")
            env.agent_pos = tuple(forced_start)
            env.start_pos = tuple(forced_start)
            import numpy as _np
            env.visited = _np.zeros_like(env.grid, dtype=bool)
            env.visited[env.agent_pos] = True
            env.trajectory = [env.agent_pos]
        env.render()
        return env

    dynamic = fire_mode == "variable"
    return MazeNavEnv(
        maze_name=MAZE_NAME,
        render_mode="human",
        randomize_start=dynamic,
        randomize_goal=dynamic,
        randomize_fire=dynamic,
        num_fire_tiles=NUM_FIRE_TILES,
        seed=seed if seed is not None else int(time.time()),
    )


def is_valid_episode(env):
    """Returns True if agent and goal are at least MIN_MANHATTAN_DIST apart."""
    ar, ac = env.agent_pos
    gr, gc = env.goal_pos
    return abs(ar - gr) + abs(ac - gc) >= MIN_MANHATTAN_DIST


def reset_episode(
    fire_mode,
    episode,
    attempt=0,
    demo_dir: str = DEMO_DIR,
    forced_start=None,
    forced_goal=None,
    forced_fires=None,
):
    """Creates and resets env. When forced_* are provided, the env is built once
    with the exact spec and no retry/randomisation occurs. Otherwise retries
    until start/goal distance meets MIN_MANHATTAN_DIST."""
    if forced_start is not None or forced_goal is not None or forced_fires is not None:
        env = create_env(
            fire_mode,
            seed=int(time.time()) + attempt,
            forced_start=forced_start,
            forced_goal=forced_goal,
            forced_fires=forced_fires,
        )
        config = env.get_dynamic_config()
        dist   = abs(env.agent_pos[0] - env.goal_pos[0]) + abs(env.agent_pos[1] - env.goal_pos[1])
        demos_so_far = count_saved_demos(demo_dir)
        print(f"\nEpisode {episode} | LLM-RECOMMENDED LAYOUT | Demos saved so far: {demos_so_far}")
        print(f"  Start: {config['start_pos']}  Goal: {config['goal_pos']}  "
              f"Fire: {config['fire_positions']}  Manhattan dist: {dist}")
        print(f"\n{env.get_grid_image_description()}\n")
        return env, [{"state": env._get_obs()["state"], "image": env._get_obs()["image"]}], [], []

    max_attempts = 50
    for i in range(max_attempts):
        env = create_env(fire_mode, seed=int(time.time()) + attempt + i)
        env.reset()
        if is_valid_episode(env):
            env.render()
            config = env.get_dynamic_config()
            dist   = abs(env.agent_pos[0] - env.goal_pos[0]) + abs(env.agent_pos[1] - env.goal_pos[1])
            demos_so_far = count_saved_demos(demo_dir)
            print(f"\nEpisode {episode} | Mode: {fire_mode.upper()} | Demos saved so far: {demos_so_far}/{TARGET_DEMOS}")
            print(f"  Start: {config['start_pos']}  Goal: {config['goal_pos']}  "
                  f"Fire: {config['fire_positions']}  Manhattan dist: {dist}")
            print(f"\n{env.get_grid_image_description()}\n")
            return env, [{"state": env._get_obs()["state"], "image": env._get_obs()["image"]}], [], []
        env.close()
    print(f"Warning: could not find valid start/goal pair after {max_attempts} attempts, using last generated.")
    env.render()
    return env, [{"state": env._get_obs()["state"], "image": env._get_obs()["image"]}], [], []


def save_demo(env, obs_seq, action_seq, reward_seq, demo_dir: str = DEMO_DIR, layout_id: Optional[str] = None):
    """Saves the current trajectory as a JSON demo file under demo_dir.
    When layout_id is set, the filename includes it so active_loop demos are
    traceable back to the LLM recommendation that produced them."""
    config    = env.get_dynamic_config()
    timestamp = int(time.time() * 1000)  # ms — avoid collisions when saving fast
    os.makedirs(demo_dir, exist_ok=True)
    suffix = f"_{layout_id}" if layout_id else ""
    filename  = os.path.join(demo_dir, f"demo_{MAZE_NAME}{suffix}_{timestamp}.json")
    demo = {
        "maze_name":      MAZE_NAME,
        "timestamp":      timestamp,
        "layout_id":      layout_id,
        "start_pos":      [int(x) for x in config["start_pos"]],
        "goal_pos":       [int(x) for x in config["goal_pos"]],
        "fire_positions": [[int(x) for x in p] for p in config["fire_positions"]],
        "trajectory":     [[int(x) for x in p] for p in env.get_trajectory()],
        "observations":   [o["state"].tolist() for o in obs_seq],
        "images":         [o["image"].tolist() for o in obs_seq],
        "actions":        [int(a) for a in action_seq],
        "rewards":        [float(r) for r in reward_seq],
        "total_reward":   float(sum(reward_seq)),
        "success":        bool(reward_seq[-1] > 5.0) if reward_seq else False,
    }
    with open(filename, "w") as f:
        json.dump(demo, f, indent=2)
    demos_now = count_saved_demos(demo_dir)
    print(f"\nDemo saved: {filename}")
    print(f"Steps: {len(action_seq)} | Total reward: {sum(reward_seq):.2f}")
    print(f">>> Demos in {demo_dir}: {demos_now} <<<")
    return filename


def draw_trajectory_overlay(env, trajectory):
    """Draws a colour-gradient path overlay on the pygame surface."""
    if env._screen is None:
        return
    for i, (tr, tc) in enumerate(trajectory):
        cx = tc * CELL_SIZE + CELL_SIZE // 2
        cy = tr * CELL_SIZE + CELL_SIZE // 2
        progress = i / max(len(trajectory) - 1, 1)
        r_color  = int(50 + 200 * (1 - progress))
        g_color  = int(50 + 200 * progress)
        pygame.draw.circle(env._screen, (r_color, g_color, 50), (cx, cy), 10)
        if i > 0:
            prev_r, prev_c = trajectory[i - 1]
            px = prev_c * CELL_SIZE + CELL_SIZE // 2
            py = prev_r * CELL_SIZE + CELL_SIZE // 2
            pygame.draw.line(env._screen, (r_color, g_color, 50), (px, py), (cx, cy), 3)
    pygame.display.flip()


def draw_drag_arrow(env, start_pixel, end_pixel):
    """Renders a directional drag arrow on the pygame surface."""
    if env._screen is None:
        return
    env.render()
    dx   = end_pixel[0] - start_pixel[0]
    dy   = end_pixel[1] - start_pixel[1]
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 10:
        pygame.display.flip()
        return
    color       = (255, 200, 0)
    angle       = math.atan2(dy, dx)
    arrow_len   = 12
    arrow_angle = 0.5
    pygame.draw.line(env._screen, color, start_pixel, end_pixel, 3)
    left_x  = end_pixel[0] - arrow_len * math.cos(angle - arrow_angle)
    left_y  = end_pixel[1] - arrow_len * math.sin(angle - arrow_angle)
    right_x = end_pixel[0] - arrow_len * math.cos(angle + arrow_angle)
    right_y = end_pixel[1] - arrow_len * math.sin(angle + arrow_angle)
    pygame.draw.polygon(env._screen, color, [end_pixel, (int(left_x), int(left_y)), (int(right_x), int(right_y))])
    pygame.display.flip()


def step_and_log(env, action, obs_seq, action_seq, reward_seq):
    """Executes one env step, appends to sequences, and prints step info."""
    obs, reward, terminated, truncated, info = env.step(action)
    obs_seq.append(obs)
    action_seq.append(action)
    reward_seq.append(reward)
    print(f"  Step {info['step_count']:3d} | {ACTION_NAMES[action]:5s} | "
          f"Pos: {info['agent_pos']} | Reward: {reward:+.2f} | Dist: {info['manhattan_dist']}")
    if terminated or truncated:
        outcome = "GOAL REACHED!" if info.get("success") else ("Timeout." if truncated else "FIRE! Episode ended.")
        print(f"\n{outcome} Total reward: {sum(reward_seq):.2f}")
        print("Press R to reset or S to save.")
    return terminated, truncated


def main():
    """Entry point: runs the interactive human play loop with dynamic maze support."""
    global FIRE_MODE
    args = parse_args()

    demo_dir = args.demo_dir
    os.makedirs(demo_dir, exist_ok=True)

    demos_at_start = count_saved_demos(demo_dir)
    print(f"Maze Nav — {MAZE_NAME}")
    print(f"Demos already in {demo_dir}: {demos_at_start}")
    if args.start or args.goal or args.fires:
        print(f"FORCED LAYOUT: start={args.start} goal={args.goal} fires={args.fires}")
        if args.layout_id:
            print(f"Layout id: {args.layout_id}")
    if args.single_episode:
        print("SINGLE EPISODE MODE: pygame will close after one save.")
    print("Drag or arrow keys to move | R=reset | S=save | F=toggle fire mode | Q=quit\n")

    episode = 1
    env, obs_seq, action_seq, reward_seq = reset_episode(
        FIRE_MODE, episode,
        demo_dir=demo_dir,
        forced_start=args.start,
        forced_goal=args.goal,
        forced_fires=args.fires,
    )

    drag_active      = False
    drag_start_pixel = None

    KEY_TO_ACTION = {
        pygame.K_UP:    0,
        pygame.K_DOWN:  1,
        pygame.K_LEFT:  2,
        pygame.K_RIGHT: 3,
    }

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if not env.episode_done:
                    mx, my               = event.pos
                    clicked_col          = mx // CELL_SIZE
                    clicked_row          = my // CELL_SIZE
                    agent_row, agent_col = env.agent_pos
                    if clicked_row == agent_row and clicked_col == agent_col:
                        drag_active      = True
                        drag_start_pixel = (mx, my)

            elif event.type == pygame.MOUSEMOTION:
                if drag_active and env._screen is not None:
                    agent_row, agent_col = env.agent_pos
                    agent_cx = agent_col * CELL_SIZE + CELL_SIZE // 2
                    agent_cy = agent_row * CELL_SIZE + CELL_SIZE // 2
                    draw_drag_arrow(env, (agent_cx, agent_cy), event.pos)

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if drag_active:
                    drag_active = False
                    mx, my      = event.pos
                    dx          = mx - drag_start_pixel[0]
                    dy          = my - drag_start_pixel[1]
                    drag_start_pixel = None
                    env.render()
                    pygame.display.flip()
                    if abs(dx) < DRAG_THRESHOLD and abs(dy) < DRAG_THRESHOLD:
                        continue
                    if not env.episode_done:
                        action = (3 if dx > 0 else 2) if abs(dx) >= abs(dy) else (1 if dy > 0 else 0)
                        step_and_log(env, action, obs_seq, action_seq, reward_seq)

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

                elif event.key == pygame.K_f:
                    FIRE_MODE = "fixed" if FIRE_MODE == "variable" else "variable"
                    print(f"Fire mode → {FIRE_MODE.upper()} (takes effect on next reset)")

                elif event.key == pygame.K_r:
                    env.close()
                    episode += 1
                    env, obs_seq, action_seq, reward_seq = reset_episode(
                        FIRE_MODE, episode,
                        demo_dir=demo_dir,
                        forced_start=args.start,
                        forced_goal=args.goal,
                        forced_fires=args.fires,
                    )
                    drag_active      = False
                    drag_start_pixel = None

                elif event.key == pygame.K_s:
                    if action_seq:
                        saved_traj = env.get_trajectory()
                        save_demo(env, obs_seq, action_seq, reward_seq,
                                  demo_dir=demo_dir, layout_id=args.layout_id)
                        draw_trajectory_overlay(env, saved_traj)
                        pygame.time.wait(1500)
                        if args.single_episode:
                            print("[play_maze] --single_episode set; closing pygame.")
                            running = False
                            break
                        env.close()
                        episode += 1
                        env, obs_seq, action_seq, reward_seq = reset_episode(
                            FIRE_MODE, episode, attempt=episode,
                            demo_dir=demo_dir,
                            forced_start=args.start,
                            forced_goal=args.goal,
                            forced_fires=args.fires,
                        )
                        drag_active      = False
                        drag_start_pixel = None
                        print("Auto-reset. Ready for next demo!")
                    else:
                        print("No actions recorded yet.")

                elif event.key in KEY_TO_ACTION:
                    if not env.episode_done:
                        step_and_log(env, KEY_TO_ACTION[event.key], obs_seq, action_seq, reward_seq)

    env.close()
    print(f"\nGoodbye! Total demos in {demo_dir}: {count_saved_demos(demo_dir)}")


if __name__ == "__main__":
    main()