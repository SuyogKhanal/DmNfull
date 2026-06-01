"""Per-round CSV logger: round, demos_added_total, training_rounds_total,
heldout_sr, finetune_epochs, lr, replay_mix.

Append-on-write so a partial run still leaves a usable CSV.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Union

_HEADER = (
    "round",
    "demos_added_total",
    "training_rounds_total",
    "heldout_sr",
    "finetune_epochs",
    "lr",
    "replay_mix",
)


class TrainingLog:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(_HEADER)

    def write_row(
        self,
        round_idx: int,
        demos_added_total: int,
        training_rounds_total: int,
        heldout_sr: float,
        finetune_epochs: int,
        lr: float,
        replay_mix: float,
    ) -> None:
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([
                int(round_idx),
                int(demos_added_total),
                int(training_rounds_total),
                float(heldout_sr),
                int(finetune_epochs),
                float(lr),
                float(replay_mix),
            ])
