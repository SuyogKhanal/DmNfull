import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import cv2
import json
import os
from typing import Optional, Dict, Any, Tuple, List
from configs.maze_layouts import MAZE_LAYOUTS


TILE_FREE  = 0
TILE_WALL  = 1
TILE_FIRE  = 2
TILE_GOAL  = 3

TILE_COLORS = {
    TILE_FREE: (240, 240, 240),
    TILE_WALL: (40,  40,  40),
    TILE_FIRE: (220, 60,  20),
    TILE_GOAL: (50,  200, 80),
}
AGENT_COLOR     = (30,  100, 220)
VISITED_COLOR   = (200, 220, 255)
GRID_LINE_COLOR = (180, 180, 180)

CELL_SIZE       = 80
FONT_SIZE       = 18
IMG_SIZE        = 80
RENDER_IMG_SIZE = 80

ACTION_DELTAS = {
    0: (-1,  0),
    1: ( 1,  0),
    2: ( 0, -1),
    3: ( 0,  1),
}
ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

REWARD_GOAL     =  10.0
REWARD_FIRE     = -10.0
REWARD_WALL_HIT =  -0.5
REWARD_STEP     =  -0.1
REWARD_REVISIT  =  -0.05

MAX_STEPS       = 200


class MazeNavEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        maze_name: str = "medium_1",
        render_mode: Optional[str] = "human",
        randomize_start: bool = False,
        randomize_goal: bool = False,
        randomize_fire: bool = False,
        num_fire_tiles: int = 3,
        seed: int = 42,
        fire_positions: Optional[List] = None,
    ):
        """Initialises the MazeNavEnv with optional dynamic randomisation."""
        super().__init__()
        self.render_mode     = render_mode
        self.randomize_start = randomize_start
        self.randomize_goal  = randomize_goal
        self.randomize_fire  = randomize_fire
        self.num_fire_tiles  = num_fire_tiles
        self.np_rng          = np.random.default_rng(seed)

        assert maze_name in MAZE_LAYOUTS, f"Unknown maze: {maze_name}. Available: {list(MAZE_LAYOUTS.keys())}"
        layout = MAZE_LAYOUTS[maze_name]
        self.grid_template = np.array(layout["grid"], dtype=np.int32)
        self.grid_size     = self.grid_template.shape[0]
        self.start_pos     = tuple(layout["start"])
        self.goal_pos      = tuple(layout["goal"])
        self.maze_name     = maze_name

        if fire_positions is not None:
            self.grid_template[self.grid_template == TILE_FIRE] = TILE_FREE
            for r, c in fire_positions:
                self.grid_template[r, c] = TILE_FIRE

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict({
            "state": spaces.Box(low=0.0, high=1.0, shape=(14,), dtype=np.float32),
            "image": spaces.Box(low=0, high=255, shape=(IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8),
        })

        self._screen = None
        self._clock  = None
        self._font   = None

        self.grid           = None
        self.agent_pos      = None
        self.step_count     = 0
        self.visited        = None
        self.episode_done   = False
        self.trajectory     = []
        self._reached_goal  = False
        self.fire_positions = []

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict, Dict]:
        """Resets the environment, optionally randomising start, goal, and fire positions."""
        super().reset(seed=seed)
        self.grid          = self.grid_template.copy()
        self.step_count    = 0
        self.episode_done  = False
        self._reached_goal = False
        self.visited       = np.zeros_like(self.grid, dtype=bool)
        self.trajectory    = []
        self.goal_pos      = tuple(MAZE_LAYOUTS[self.maze_name]["goal"])

        if self.randomize_fire:
            self.grid[self.grid == TILE_FIRE] = TILE_FREE
            candidates = [
                (r, c)
                for r in range(self.grid_size)
                for c in range(self.grid_size)
                if self.grid[r, c] == TILE_FREE
                and (r, c) != self.start_pos
                and (r, c) != self.goal_pos
            ]
            chosen = self.np_rng.choice(
                len(candidates),
                size=min(self.num_fire_tiles, len(candidates)),
                replace=False,
            )
            self.fire_positions = [candidates[i] for i in chosen]
            for r, c in self.fire_positions:
                self.grid[r, c] = TILE_FIRE
        else:
            self.fire_positions = [
                (r, c)
                for r in range(self.grid_size)
                for c in range(self.grid_size)
                if self.grid[r, c] == TILE_FIRE
            ]

        if self.randomize_goal:
            self.grid[self.grid == TILE_GOAL] = TILE_FREE
            candidates = [
                (r, c)
                for r in range(self.grid_size)
                for c in range(self.grid_size)
                if self.grid[r, c] == TILE_FREE
                and (r, c) != self.start_pos
                and (r, c) not in self.fire_positions
            ]
            idx = self.np_rng.integers(len(candidates))
            self.goal_pos = candidates[idx]
            self.grid[self.goal_pos[0], self.goal_pos[1]] = TILE_GOAL

        if self.randomize_start:
            free_cells = [
                (r, c)
                for r in range(self.grid_size)
                for c in range(self.grid_size)
                if self.grid[r, c] == TILE_FREE
                and (r, c) != self.goal_pos
                and (r, c) not in self.fire_positions
            ]
            idx = self.np_rng.integers(len(free_cells))
            self.agent_pos = free_cells[idx]
        else:
            self.agent_pos = self.start_pos

        self.visited[self.agent_pos] = True
        self.trajectory.append(self.agent_pos)

        obs  = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Steps the environment forward by one action."""
        assert not self.episode_done, "Episode done. Call reset()."
        assert self.action_space.contains(action), f"Invalid action: {action}"

        dr, dc = ACTION_DELTAS[action]
        new_r  = self.agent_pos[0] + dr
        new_c  = self.agent_pos[1] + dc

        reward     = REWARD_STEP
        terminated = False
        truncated  = False

        if not self._in_bounds(new_r, new_c) or self.grid[new_r, new_c] == TILE_WALL:
            reward += REWARD_WALL_HIT
        else:
            self.agent_pos = (new_r, new_c)
            tile = self.grid[new_r, new_c]

            if tile == TILE_FIRE:
                reward     += REWARD_FIRE
                terminated  = True
            elif tile == TILE_GOAL:
                reward     += REWARD_GOAL
                terminated  = True
                self._reached_goal = True
            elif self.visited[new_r, new_c]:
                reward += REWARD_REVISIT

            self.visited[new_r, new_c] = True
            self.trajectory.append(self.agent_pos)

        self.step_count += 1
        if self.step_count >= MAX_STEPS:
            truncated = True

        self.episode_done = terminated or truncated
        obs  = self._get_obs()
        info = self._get_info(action=action)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Renders the current frame to screen or returns an rgb_array."""
        if self.render_mode is None:
            return None

        if self._screen is None:
            self._init_pygame()

        self._draw_frame()

        if self.render_mode == "human":
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
            return None
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self._screen)), axes=(1, 0, 2)
            )

    def close(self):
        """Closes the pygame window if open."""
        if self._screen is not None:
            pygame.quit()
            self._screen = None

    def _get_bird_eye_view(self) -> np.ndarray:
        """Renders at 80x80 (integer-aligned cells) then downsamples to 64x64 via cv2."""
        gs = self.grid_size
        cell_px = RENDER_IMG_SIZE // gs
        img = np.zeros((RENDER_IMG_SIZE, RENDER_IMG_SIZE, 3), dtype=np.uint8)

        for r in range(gs):
            for c in range(gs):
                y0 = r * cell_px
                x0 = c * cell_px
                y1 = y0 + cell_px
                x1 = x0 + cell_px
                tile = self.grid[r, c]

                if (r, c) == self.agent_pos:
                    color = AGENT_COLOR
                elif tile == TILE_GOAL:
                    color = (50, 200, 80)
                elif tile == TILE_FIRE:
                    color = (220, 60, 20)
                else:
                    color = TILE_COLORS.get(tile, (100, 100, 100))

                img[y0:y1, x0:x1] = color

        return img


    def _get_obs(self) -> Dict:
        """Constructs the observation dict with state vector and bird's-eye image."""
        gs = self.grid_size

        agent_norm = np.array(self.agent_pos, dtype=np.float32) / (gs - 1)
        goal_norm  = np.array(self.goal_pos,  dtype=np.float32) / (gs - 1)

        local = np.ones((3, 3), dtype=np.float32)
        for di in range(-1, 2):
            for dj in range(-1, 2):
                r = self.agent_pos[0] + di
                c = self.agent_pos[1] + dj
                if self._in_bounds(r, c):
                    local[di + 1, dj + 1] = self.grid[r, c] / 3.0

        steps_remaining = np.array(
            [(MAX_STEPS - self.step_count) / MAX_STEPS], dtype=np.float32
        )

        state = np.concatenate([
            agent_norm,
            goal_norm,
            local.flatten(),
            steps_remaining,
        ])

        return {
            "state": state,
            "image": self._get_bird_eye_view(),
        }

    def _get_info(self, action: Optional[int] = None) -> Dict[str, Any]:
        """Returns a diagnostic info dict for the current step."""
        ar, ac = self.agent_pos
        gr, gc = self.goal_pos
        manhattan = abs(ar - gr) + abs(ac - gc)

        neighbourhood = {}
        for name, (dr, dc) in [("up",(-1,0)),("down",(1,0)),("left",(0,-1)),("right",(0,1))]:
            r, c = ar + dr, ac + dc
            if not self._in_bounds(r, c):
                neighbourhood[name] = "wall"
            else:
                t = self.grid[r, c]
                neighbourhood[name] = {TILE_FREE:"free", TILE_WALL:"wall",
                                        TILE_FIRE:"fire", TILE_GOAL:"goal"}[t]

        return {
            "agent_pos":         self.agent_pos,
            "goal_pos":          self.goal_pos,
            "fire_positions":    [list(p) for p in zip(*np.where(self.grid == TILE_FIRE))] if np.any(self.grid == TILE_FIRE) else [],
            "manhattan_dist":    manhattan,
            "step_count":        self.step_count,
            "steps_remaining":   MAX_STEPS - self.step_count,
            "neighbourhood":     neighbourhood,
            "action_taken":      ACTION_NAMES.get(action, "RESET"),
            "visited_count":     int(self.visited.sum()),
            "trajectory_length": len(self.trajectory),
            "maze_name":         self.maze_name,
            "success":           self._reached_goal,
            "llm_state_description": self._describe_state_text(),
        }

    def _describe_state_text(self) -> str:
        """Returns a natural language description of the current agent state."""
        ar, ac = self.agent_pos
        gr, gc = self.goal_pos
        neighbourhood = {}
        for name, (dr, dc) in [("up",(-1,0)),("down",(1,0)),("left",(0,-1)),("right",(0,1))]:
            r, c = ar + dr, ac + dc
            if not self._in_bounds(r, c):
                neighbourhood[name] = "wall (boundary)"
            else:
                t = self.grid[r, c]
                neighbourhood[name] = {
                    TILE_FREE: "free path",
                    TILE_WALL: "wall",
                    TILE_FIRE: "FIRE (dangerous!)",
                    TILE_GOAL: "GOAL (target!)"
                }[t]

        desc = (
            f"The agent is at position (row={ar}, col={ac}) in a {self.grid_size}x{self.grid_size} maze. "
            f"The goal is at (row={gr}, col={gc}). "
            f"Manhattan distance to goal: {abs(ar-gr)+abs(ac-gc)}. "
            f"Adjacent cells - Up: {neighbourhood['up']}, Down: {neighbourhood['down']}, "
            f"Left: {neighbourhood['left']}, Right: {neighbourhood['right']}. "
            f"Steps taken: {self.step_count}/{MAX_STEPS}."
        )
        return desc

    def _init_pygame(self):
        """Initialises the pygame display, clock, and font."""
        pygame.init()
        w = self.grid_size * CELL_SIZE
        h = self.grid_size * CELL_SIZE + 60
        self._screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption(f"Maze Nav — {self.maze_name}")
        self._clock = pygame.time.Clock()
        self._font  = pygame.font.SysFont("monospace", FONT_SIZE)

    def _draw_frame(self):
        """Draws the current maze state onto the pygame surface."""
        self._screen.fill((20, 20, 20))
        gs = self.grid_size

        for r in range(gs):
            for c in range(gs):
                x = c * CELL_SIZE
                y = r * CELL_SIZE
                tile = self.grid[r, c]

                if (r, c) == self.agent_pos:
                    color = AGENT_COLOR
                elif self.visited[r, c] and tile == TILE_FREE:
                    color = VISITED_COLOR
                else:
                    color = TILE_COLORS.get(tile, (100, 100, 100))

                pygame.draw.rect(self._screen, color, (x, y, CELL_SIZE, CELL_SIZE))
                pygame.draw.rect(self._screen, GRID_LINE_COLOR, (x, y, CELL_SIZE, CELL_SIZE), 1)

                label = None
                if (r, c) == self.agent_pos:
                    label = "A"
                elif tile == TILE_GOAL:
                    label = "G"
                elif tile == TILE_FIRE:
                    label = "F"
                elif tile == TILE_WALL:
                    label = "#"

                if label:
                    surf = self._font.render(label, True, (255, 255, 255))
                    self._screen.blit(surf, (x + CELL_SIZE//2 - 8, y + CELL_SIZE//2 - 10))

                coord = self._font.render(f"{r},{c}", True, (150, 150, 150))
                self._screen.blit(coord, (x + 3, y + 3))

        for (tr, tc) in self.trajectory[:-1]:
            cx = tc * CELL_SIZE + CELL_SIZE // 2
            cy = tr * CELL_SIZE + CELL_SIZE // 2
            pygame.draw.circle(self._screen, (100, 150, 255), (cx, cy), 5)

        ar, ac = self.agent_pos
        gr, gc = self.goal_pos
        status = (
            f"Pos:({ar},{ac})  Goal:({gr},{gc})  "
            f"Dist:{abs(ar-gr)+abs(ac-gc)}  Steps:{self.step_count}/{MAX_STEPS}  "
            f"Maze:{self.maze_name}"
        )
        txt = self._font.render(status, True, (220, 220, 220))
        self._screen.blit(txt, (5, self.grid_size * CELL_SIZE + 10))

    def _in_bounds(self, r: int, c: int) -> bool:
        """Returns True if (r, c) is within the maze grid."""
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

    def get_trajectory(self) -> list:
        """Returns a copy of the agent's position trajectory for this episode."""
        return list(self.trajectory)

    def get_dynamic_config(self) -> dict:
        """Returns current episode's start, goal, and fire positions."""
        return {
            "start_pos":      list(self.agent_pos),
            "goal_pos":       list(self.goal_pos),
            "fire_positions": [list(p) for p in zip(*np.where(self.grid == TILE_FIRE))] if np.any(self.grid == TILE_FIRE) else [],
        }

    def get_grid_image_description(self) -> str:
        """Returns an ASCII art representation of the current grid."""
        symbols = {TILE_FREE: "·", TILE_WALL: "█", TILE_FIRE: "F", TILE_GOAL: "G"}
        lines = []
        for r in range(self.grid_size):
            row_str = ""
            for c in range(self.grid_size):
                if (r, c) == self.agent_pos:
                    row_str += "A "
                else:
                    row_str += symbols.get(self.grid[r, c], "?") + " "
            lines.append(row_str.strip())
        return "\n".join(lines)