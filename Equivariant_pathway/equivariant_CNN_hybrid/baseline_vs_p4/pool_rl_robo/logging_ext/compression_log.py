"""Per-round P4 selection-diagnostics CSV (continuous-control analogue of
pool_x_selector's ``logging_ext/compression_log.py``). The maze logged how many
failure modes the LLM compressed into how many demos; here (1 query/round) we
log the candidate-pool size, the picked state's score, and the resulting reward
— a record of what the P4-LLM selector did each round. Append-on-write."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Union

_HEADER = (
    "round", "n_candidates", "n_demos_collected",
    "picked_score", "rollout_success_rate", "mean_reward",
)


class CompressionLog:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(_HEADER)

    def write_row(self, round_idx: int, n_candidates: int, n_demos_collected: int,
                  picked_score: Optional[float], rollout_success_rate: Optional[float],
                  mean_reward: float) -> None:
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([
                int(round_idx), int(n_candidates), int(n_demos_collected),
                ("" if picked_score is None else round(float(picked_score), 6)),
                ("" if rollout_success_rate is None else round(float(rollout_success_rate), 4)),
                round(float(mean_reward), 4),
            ])
