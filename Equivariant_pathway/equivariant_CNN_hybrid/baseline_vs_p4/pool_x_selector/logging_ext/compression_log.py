"""Per-round compression-ratio CSV logger for the P4 variants.

Columns:
  round, n_failures_observed, n_layouts_prescribed, compression_ratio,
  n_demos_collected, n_corridor_infeasible, prescribed_loss_mean

``compression_ratio = n_failures_observed / max(1, n_layouts_prescribed)``
— how many failure modes did the LLM compress into how many demos. A
value of 1.0 means one prescription per failure (no compression). Higher
values mean the LLM found a way to address several failures with fewer
demos (good); lower than 1.0 means the LLM is prescribing redundantly.

``prescribed_loss_mean`` is the pre-finetune policy's mean per-step BCE
loss on the kept prescribed layouts (info-gain proxy; high = informative
demo, low = the policy already handles this layout). Empty when no
layouts were kept this round.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Union

_HEADER = (
    "round",
    "n_failures_observed",
    "n_layouts_prescribed",
    "compression_ratio",
    "n_demos_collected",
    "n_corridor_infeasible",
    "prescribed_loss_mean",
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
        prescribed_loss_mean: Optional[float] = None,
    ) -> None:
        ratio = (
            float(n_failures_observed) / max(1, n_layouts_prescribed)
            if n_layouts_prescribed > 0
            else 0.0
        )
        loss_cell = "" if prescribed_loss_mean is None else round(float(prescribed_loss_mean), 6)
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([
                int(round_idx),
                int(n_failures_observed),
                int(n_layouts_prescribed),
                round(float(ratio), 4),
                int(n_demos_collected),
                int(n_corridor_infeasible),
                loss_cell,
            ])
