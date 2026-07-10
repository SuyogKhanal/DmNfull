"""MazeNavEnv — vendored into distil/ (self-contained, 04_..md golden rule 8).

Ported verbatim from DmNfull/envs/maze_env.py with three portability edits so a
`git clone` on HPC-B runs it with no extra host deps:
  * `pygame` import is LAZY (only for the human/rgb_array window, never for the
    obs image or the DISTIL loop) — the loop uses `obs["image"]`, a pure-numpy
    bird-eye render.
  * dropped the unused `cv2` import.
  * the layout table is vendored inline (no `configs.maze_layouts` dependency), and
    the ctor accepts an explicit `grid`/`start`/`goal` so a sampled layout can be
    forced deterministically (needed for reproducible screening / SELECT re-roll).

State obs = 14-d [agent_rc/4, goal_rc/4, 3x3 nbhd/3, steps_remaining/200]. Image obs
= 80x80x3 uint8 bird-eye. step() returns the Gymnasium 5-tuple (obs, reward,
terminated, truncated, info); the GridWorldEnv adapter remaps it to the robosuite-
style (obs, reward, done, success, info) the DISTIL engine expects.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

TILE_FREE, TILE_WALL, TILE_FIRE, TILE_GOAL = 0, 1, 2, 3

TILE_COLORS = {TILE_FREE: (240, 240, 240), TILE_WALL: (40, 40, 40),
               TILE_FIRE: (220, 60, 20), TILE_GOAL: (50, 200, 80)}
AGENT_COLOR = (30, 100, 220)

CELL_SIZE, FONT_SIZE, IMG_SIZE, RENDER_IMG_SIZE = 80, 18, 80, 80

ACTION_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

REWARD_GOAL, REWARD_FIRE, REWARD_WALL_HIT, REWARD_STEP, REWARD_REVISIT = 10.0, -10.0, -0.5, -0.1, -0.05
MAX_STEPS = 200

# Vendored layouts (only the primary 'multimodal' two-route maze; DISTIL runs use
# sampled layouts via distil.gridworld.layouts, not this table).
MAZE_LAYOUTS = {
    "multimodal": {
        "grid": [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 2, 2, 0, 0],
                 [0, 0, 0, 0, 0], [0, 0, 0, 0, 3]],
        "start": [0, 0], "goal": [4, 4],
    },
}


class MazeNavEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, maze_name: str = "multimodal", render_mode: Optional[str] = None,
                 randomize_start: bool = False, randomize_goal: bool = False,
                 randomize_fire: bool = False, num_fire_tiles: int = 3, seed: int = 42,
                 fire_positions: Optional[List] = None,
                 grid: Optional[List] = None, start: Optional[List] = None,
                 goal: Optional[List] = None):
        super().__init__()
        self.render_mode = render_mode
        self.randomize_start = randomize_start
        self.randomize_goal = randomize_goal
        self.randomize_fire = randomize_fire
        self.num_fire_tiles = num_fire_tiles
        self.np_rng = np.random.default_rng(seed)

        # explicit layout override (a sampled layout) takes precedence over the table.
        if grid is not None:
            layout = {"grid": grid, "start": list(start), "goal": list(goal)}
        else:
            assert maze_name in MAZE_LAYOUTS, f"Unknown maze: {maze_name}"
            layout = MAZE_LAYOUTS[maze_name]
        self.grid_template = np.array(layout["grid"], dtype=np.int32)
        self.grid_size = self.grid_template.shape[0]
        self.start_pos = tuple(layout["start"])
        self.goal_pos = tuple(layout["goal"])
        self.maze_name = maze_name

        if fire_positions is not None:
            self.grid_template[self.grid_template == TILE_FIRE] = TILE_FREE
            for r, c in fire_positions:
                self.grid_template[int(r), int(c)] = TILE_FIRE

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict({
            "state": spaces.Box(0.0, 1.0, (14,), np.float32),
            "image": spaces.Box(0, 255, (IMG_SIZE, IMG_SIZE, 3), np.uint8)})

        self._screen = self._clock = self._font = None
        self.grid = self.agent_pos = self.visited = None
        self.step_count = 0
        self.episode_done = self._reached_goal = False
        self.trajectory: List = []
        self.fire_positions: List = []

    # DISTIL layout override: force a fixed (grid, start, goal, fires) scene.
    def configure_layout(self, grid, start, goal, fire_positions) -> None:
        self.grid_template = np.array(grid, dtype=np.int32)
        self.grid_size = self.grid_template.shape[0]
        self.start_pos = tuple(start)
        self.goal_pos = tuple(goal)
        self.grid_template[self.grid_template == TILE_FIRE] = TILE_FREE
        for r, c in fire_positions:
            self.grid_template[int(r), int(c)] = TILE_FIRE
        # sampled layouts carry the goal as a separate coord, not baked into the grid;
        # paint TILE_GOAL so step()'s `tile==TILE_GOAL` success check actually fires.
        self.grid_template[int(goal[0]), int(goal[1])] = TILE_GOAL
        self.randomize_start = self.randomize_goal = self.randomize_fire = False

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        self.grid = self.grid_template.copy()
        self.step_count = 0
        self.episode_done = self._reached_goal = False
        self.visited = np.zeros_like(self.grid, dtype=bool)
        self.trajectory = []
        self.goal_pos = tuple(self.goal_pos)

        if self.randomize_fire:
            self.grid[self.grid == TILE_FIRE] = TILE_FREE
            cand = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)
                    if self.grid[r, c] == TILE_FREE and (r, c) != self.start_pos and (r, c) != self.goal_pos]
            chosen = self.np_rng.choice(len(cand), size=min(self.num_fire_tiles, len(cand)), replace=False)
            self.fire_positions = [cand[i] for i in chosen]
            for r, c in self.fire_positions:
                self.grid[r, c] = TILE_FIRE
        else:
            self.fire_positions = [(r, c) for r in range(self.grid_size)
                                   for c in range(self.grid_size) if self.grid[r, c] == TILE_FIRE]

        if self.randomize_goal:
            self.grid[self.grid == TILE_GOAL] = TILE_FREE
            cand = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)
                    if self.grid[r, c] == TILE_FREE and (r, c) != self.start_pos and (r, c) not in self.fire_positions]
            self.goal_pos = cand[self.np_rng.integers(len(cand))]
            self.grid[self.goal_pos[0], self.goal_pos[1]] = TILE_GOAL

        if self.randomize_start:
            free = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)
                    if self.grid[r, c] == TILE_FREE and (r, c) != self.goal_pos and (r, c) not in self.fire_positions]
            self.agent_pos = free[self.np_rng.integers(len(free))]
        else:
            self.agent_pos = self.start_pos

        self.visited[self.agent_pos] = True
        self.trajectory.append(self.agent_pos)
        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        assert not self.episode_done, "Episode done. Call reset()."
        assert self.action_space.contains(action), f"Invalid action: {action}"
        dr, dc = ACTION_DELTAS[action]
        new_r, new_c = self.agent_pos[0] + dr, self.agent_pos[1] + dc
        reward, terminated, truncated = REWARD_STEP, False, False
        if not self._in_bounds(new_r, new_c) or self.grid[new_r, new_c] == TILE_WALL:
            reward += REWARD_WALL_HIT
        else:
            self.agent_pos = (new_r, new_c)
            tile = self.grid[new_r, new_c]
            if tile == TILE_FIRE:
                reward += REWARD_FIRE; terminated = True
            elif tile == TILE_GOAL:
                reward += REWARD_GOAL; terminated = True; self._reached_goal = True
            elif self.visited[new_r, new_c]:
                reward += REWARD_REVISIT
            self.visited[new_r, new_c] = True
            self.trajectory.append(self.agent_pos)
        self.step_count += 1
        if self.step_count >= MAX_STEPS:
            truncated = True
        self.episode_done = terminated or truncated
        return self._get_obs(), reward, terminated, truncated, self._get_info(action=action)

    def _get_bird_eye_view(self) -> np.ndarray:
        gs = self.grid_size
        cell_px = RENDER_IMG_SIZE // gs
        img = np.zeros((RENDER_IMG_SIZE, RENDER_IMG_SIZE, 3), dtype=np.uint8)
        for r in range(gs):
            for c in range(gs):
                y0, x0 = r * cell_px, c * cell_px
                tile = self.grid[r, c]
                if (r, c) == self.agent_pos:
                    color = AGENT_COLOR
                elif tile == TILE_GOAL:
                    color = (50, 200, 80)
                elif tile == TILE_FIRE:
                    color = (220, 60, 20)
                else:
                    color = TILE_COLORS.get(tile, (100, 100, 100))
                img[y0:y0 + cell_px, x0:x0 + cell_px] = color
        return img

    def _get_obs(self) -> Dict:
        gs = self.grid_size
        agent_norm = np.array(self.agent_pos, dtype=np.float32) / (gs - 1)
        goal_norm = np.array(self.goal_pos, dtype=np.float32) / (gs - 1)
        local = np.ones((3, 3), dtype=np.float32)
        for di in range(-1, 2):
            for dj in range(-1, 2):
                r, c = self.agent_pos[0] + di, self.agent_pos[1] + dj
                if self._in_bounds(r, c):
                    local[di + 1, dj + 1] = self.grid[r, c] / 3.0
        steps_remaining = np.array([(MAX_STEPS - self.step_count) / MAX_STEPS], dtype=np.float32)
        state = np.concatenate([agent_norm, goal_norm, local.flatten(), steps_remaining])
        return {"state": state, "image": self._get_bird_eye_view()}

    def _get_info(self, action: Optional[int] = None) -> Dict[str, Any]:
        ar, ac = self.agent_pos
        gr, gc = self.goal_pos
        return {
            "agent_pos": self.agent_pos, "goal_pos": self.goal_pos,
            "fire_positions": [list(p) for p in zip(*np.where(self.grid == TILE_FIRE))] if np.any(self.grid == TILE_FIRE) else [],
            "manhattan_dist": abs(ar - gr) + abs(ac - gc),
            "step_count": self.step_count, "steps_remaining": MAX_STEPS - self.step_count,
            "maze_name": self.maze_name, "success": self._reached_goal,
        }

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

    def render(self):
        if self.render_mode is None:
            return None
        import pygame  # lazy: only the human/rgb_array window needs it
        if self._screen is None:
            pygame.init()
            w, h = self.grid_size * CELL_SIZE, self.grid_size * CELL_SIZE + 60
            self._screen = pygame.display.set_mode((w, h)) if self.render_mode == "human" \
                else pygame.Surface((w, h))
        # (window drawing omitted — DISTIL uses obs["image"], not this path)
        return None

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None

    def get_grid_image_description(self) -> str:
        symbols = {TILE_FREE: "·", TILE_WALL: "#", TILE_FIRE: "F", TILE_GOAL: "G"}
        lines = []
        for r in range(self.grid_size):
            row = ""
            for c in range(self.grid_size):
                row += "A " if (r, c) == self.agent_pos else symbols.get(self.grid[r, c], "?") + " "
            lines.append(row.strip())
        return "\n".join(lines)
