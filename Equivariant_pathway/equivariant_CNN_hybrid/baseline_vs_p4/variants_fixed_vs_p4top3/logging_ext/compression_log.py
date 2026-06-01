"""Per-round compression-ratio CSV logger for the P4 variants.

Columns:
  round, n_failures_observed, n_layouts_prescribed, compression_ratio,
  n_demos_collected, n_corridor_infeasible

``compression_ratio = n_failures_observed / max(1, n_layouts_prescribed)``
— how many failure modes did the LLM compress into how many demos. A
value of 1.0 means one prescription per failure (no compression). Higher
values mean the LLM found a way to address several failures with fewer
demos (good); lower than 1.0 means the LLM is prescribing redundantly.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Union

_HEADER = (
    "round",
    "n_failures_observed",
    "n_layouts_prescribed",
    "compression_ratio",
    "n_demos_collected",
    "n_corridor_infeasible",
)


class CompressionLog:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(_HEADER)

    def write_row(
        self,
        round_idx: int,
        n_failures_observed: int,
        n_layouts_prescribed: int,
        n_demos_collected: int,
        n_corridor_infeasible: int = 0,
    ) -> None:
        ratio = (
            float(n_failures_observed) / max(1, n_layouts_prescribed)
            if n_layouts_prescribed > 0
            else 0.0
        )
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([
                int(round_idx),
                int(n_failures_observed),
                int(n_layouts_prescribed),
                round(float(ratio), 4),
                int(n_demos_collected),
                int(n_corridor_infeasible),
            ])
