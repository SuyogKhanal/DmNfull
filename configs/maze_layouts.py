"""
Maze Layout Definitions
=======================
Grid key:
  0 = Free path
  1 = Wall
  2 = Fire (hazard)
  3 = Goal

Agent start position is defined separately (not encoded in grid).
"""

MAZE_LAYOUTS = {


    "multimodal": {
        "grid": [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 2, 2, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 3],
        ],
        "start": [0, 0],
        "goal":  [4, 4],
        "description": (
            "PRIMARY experiment maze. Open 5x5 grid. "
            "Two fires at (2,1) and (2,2) block the straight middle path. "
            "TOP route: (2,0)→(1,0)→(0,0)→(0,1)→(0,2)→(0,3)→(0,4)→(1,4)→(2,4). "
            "BOTTOM route: (2,0)→(3,0)→(4,0)→(4,1)→(4,2)→(4,3)→(4,4)→(3,4)→(2,4). "
            "Both routes are 8 steps. Record ~5 top + ~5 bottom demos."
        ),
    },

}