"""Per-round reward-curve CSV logger (continuous-control analogue of
pool_x_selector's ``logging_ext/training_log.py``; the maze logged held-out SR,
here we log mean/std episode reward + goal success rate). Append-on-write."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Union

_HEADER = (
    "round", "n_queries", "n_demos",
    "mean_reward", "std_reward", "success_rate",
    "finetune_epochs", "lr",
)


class TrainingLog:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(_HEADER)

    def write_row(self, round_idx: int, n_queries: int, n_demos: int,
                  mean_reward: float, std_reward: float,
                  success_rate: Optional[float], finetune_epochs: int, lr: float) -> None:
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([
                int(round_idx), int(n_queries), int(n_demos),
                round(float(mean_reward), 4), round(float(std_reward), 4),
                ("" if success_rate is None else round(float(success_rate), 4)),
                int(finetune_epochs), float(lr),
            ])
