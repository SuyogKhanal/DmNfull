"""Training dataset for the CNN+MLP maze policy.

Loads the human-recorded demo JSON files written by `scripts/play_maze.py`
into CNN_pathway/demos (or any directory passed via the constructor) and
exposes one (image, state, action) triple per recorded transition.

Demo schema (matches play_maze.save_demo):
    {
        "observations": [{"state": [...], "image": [[H][W][3]]}, ...],
        "actions":      [int, ...],
        ...
    }

Order of fields:
    len(observations) == len(actions) + 1
The dataset emits (obs[t], action[t]) — i.e. drops the terminal observation.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_training_layouts(yaml_path: str) -> Dict:
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f) or {}


class CNNMLPDemoDataset(Dataset):
    """In-memory dataset of (image, state, action) triples loaded from
    play_maze JSON demos under `demo_dir`."""

    def __init__(
        self,
        demo_dir: str,
        layouts_yaml: Optional[str] = None,
        augment: bool = True,
        recursive: bool = True,
    ):
        self.augment = augment
        self.layouts_cfg = load_training_layouts(layouts_yaml) if layouts_yaml else {}

        pattern = os.path.join(demo_dir, "**", "*.json") if recursive else os.path.join(demo_dir, "*.json")
        files = sorted(glob.glob(pattern, recursive=recursive))
        if not files:
            raise FileNotFoundError(
                f"No demo JSON files found under {demo_dir}. "
                f"Run play_maze.py first:\n"
                f"  python scripts/play_maze.py "
                f"--layouts-from CNN_pathway/training_layouts.yaml "
                f"--demo_dir {demo_dir}"
            )

        self._records: List[Tuple[np.ndarray, np.ndarray, int]] = []
        per_layout_counts: Dict[str, int] = {}

        for fpath in files:
            with open(fpath, "r") as f:
                demo = json.load(f)
            obs_seq = demo.get("observations") or []
            act_seq = demo.get("actions") or []
            if not obs_seq or not act_seq:
                continue

            n_pairs = min(len(obs_seq) - 1, len(act_seq))
            if n_pairs <= 0:
                continue

            layout_id = demo.get("layout_id") or "unknown"
            per_layout_counts[layout_id] = per_layout_counts.get(layout_id, 0) + n_pairs

            for t in range(n_pairs):
                state = np.asarray(obs_seq[t]["state"], dtype=np.float32)
                image = np.asarray(obs_seq[t]["image"], dtype=np.uint8)
                action = int(act_seq[t])
                self._records.append((image, state, action))

        if not self._records:
            raise ValueError(f"All demos in {demo_dir} were empty.")

        print(f"[cnn-dataset] Loaded {len(files)} demos → "
              f"{len(self._records)} (image, state, action) samples")
        for lid, n in sorted(per_layout_counts.items()):
            print(f"[cnn-dataset]   layout_id={lid:<32s} samples={n}")

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int):
        image, state, action = self._records[idx]
        img = image.astype(np.float32) / 255.0
        if self.augment:
            brightness = np.random.uniform(0.85, 1.15)
            contrast = np.random.uniform(0.90, 1.10)
            img = np.clip(img * brightness * contrast, 0.0, 1.0)
        return (
            torch.from_numpy(img).float(),
            torch.from_numpy(state).float(),
            torch.tensor(action, dtype=torch.long),
        )
