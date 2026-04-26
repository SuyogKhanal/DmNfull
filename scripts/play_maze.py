import sys
import os
import json
import time
import math
import pygame

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.maze_env import MazeNavEnv, ACTION_NAMES, CELL_SIZE

MAZE_NAME          = "multimodal"
DEMO_DIR           = "demos"
FIRE_MODE          = "variable"
NUM_FIRE_TILES     = 3
MIN_MANHATTAN_DIST = 4
TARGET_DEMOS       = 50
DRAG_THRESHOLD     = 15

os.makedirs(DEMO_DIR, exist_ok=True)


def count_saved_demos():
    """Returns the number of JSON demo files currently in DEMO_DIR."""
    return len([f for f in os.listdir(DEMO_DIR) if f.endswith(".json")])


def create_env(fire_mode, seed=None):
    """Creates a MazeNavEnv with dynamic or fixed positions depending on fire_mode."""
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


def reset_episode(fire_mode, episode, attempt=0):
    """Creates and resets env, retrying until start/goal distance meets minimum."""
    max_attempts = 50
    for i in range(max_attempts):
        env = create_env(fire_mode, seed=int(time.time()) + attempt + i)
        env.reset()
        if is_valid_episode(env):
            env.render()
            config = env.get_dynamic_config()
            dist   = abs(env.agent_pos[0] - env.goal_pos[0]) + abs(env.agent_pos[1] - env.goal_pos[1])
            demos_so_far = count_saved_demos()
            print(f"\nEpisode {episode} | Mode: {fire_mode.upper()} | Demos saved so far: {demos_so_far}/{TARGET_DEMOS}")
            print(f"  Start: {config['start_pos']}  Goal: {config['goal_pos']}  "
                  f"Fire: {config['fire_positions']}  Manhattan dist: {dist}")
            print(f"\n{env.get_grid_image_description()}\n")
            return env, [{"state": env._get_obs()["state"], "image": env._get_obs()["image"]}], [], []
        env.close()
    print(f"Warning: could not find valid start/goal pair after {max_attempts} attempts, using last generated.")
    env.render()
    return env, [{"state": env._get_obs()["state"], "image": env._get_obs()["image"]}], [], []


def save_demo(env, obs_seq, action_seq, reward_seq):
    """Saves the current trajectory as a JSON demo file and prints running demo count."""
    config    = env.get_dynamic_config()
    timestamp = int(time.time())
    filename  = os.path.join(DEMO_DIR, f"demo_{MAZE_NAME}_{timestamp}.json")
    demo = {
        "maze_name":      MAZE_NAME,
        "timestamp":      timestamp,
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
    demos_now = count_saved_demos()
    print(f"\nDemo saved: {filename}")
    print(f"Steps: {len(action_seq)} | Total reward: {sum(reward_seq):.2f}")
    print(f">>> Demos saved so far: {demos_now}/{TARGET_DEMOS} <<<")
    if demos_now >= TARGET_DEMOS:
        print(f"Target of {TARGET_DEMOS} demos reached!")
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

    demos_at_start = count_saved_demos()
    print(f"Maze Nav — {MAZE_NAME}")
    print(f"Demos already in {DEMO_DIR}: {demos_at_start}/{TARGET_DEMOS}")
    print("Drag or arrow keys to move | R=reset | S=save | F=toggle fire mode | Q=quit\n")

    episode = 1
    env, obs_seq, action_seq, reward_seq = reset_episode(FIRE_MODE, episode)

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
                    env, obs_seq, action_seq, reward_seq = reset_episode(FIRE_MODE, episode)
                    drag_active      = False
                    drag_start_pixel = None

                elif event.key == pygame.K_s:
                    if action_seq:
                        saved_traj = env.get_trajectory()
                        save_demo(env, obs_seq, action_seq, reward_seq)
                        draw_trajectory_overlay(env, saved_traj)
                        pygame.time.wait(1500)
                        env.close()
                        episode += 1
                        env, obs_seq, action_seq, reward_seq = reset_episode(FIRE_MODE, episode, attempt=episode)
                        drag_active      = False
                        drag_start_pixel = None
                        print("Auto-reset. Ready for next demo!")
                    else:
                        print("No actions recorded yet.")

                elif event.key in KEY_TO_ACTION:
                    if not env.episode_done:
                        step_and_log(env, KEY_TO_ACTION[event.key], obs_seq, action_seq, reward_seq)

    env.close()
    print(f"\nGoodbye! Total demos in {DEMO_DIR}: {count_saved_demos()}")


if __name__ == "__main__":
    main()