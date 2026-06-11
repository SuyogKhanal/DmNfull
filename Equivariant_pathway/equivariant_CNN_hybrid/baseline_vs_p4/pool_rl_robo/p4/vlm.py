"""Component 1 — VLM failure localisation.

At the end of a rollout, three frames (start / high-loss-t* / end) are passed to
the VLM, which returns a natural-language failure description. t* is the timestep
of peak diffusion loss during the rollout. Failure modes are NOT hard-coded — the
VLM infers them from the frames + the task description.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from . import prompts as P


def analyze_failure(client, task_description: str,
                    frame_paths: Dict[str, str], t_star: int) -> str:
    """frame_paths: {"start_frame": path, "high_loss_frame": path, "end_frame": path}.
    Returns a combined natural-language failure report (one section per frame)."""
    if client is None:
        return "(VLM disabled — no failure description)"
    order = ["start_frame", "high_loss_frame", "end_frame"]
    sections: List[str] = []
    for role in order:
        path = frame_paths.get(role)
        if not path:
            continue
        try:
            txt = client.analyze_frame(path, P.vlm_prompt(task_description, role, t_star),
                                       system_prompt=P.VLM_SYSTEM)
        except Exception as exc:  # pragma: no cover - server path
            txt = f"[VLM error on {role}: {type(exc).__name__}: {exc}]"
        sections.append(f"--- {role} (t*={t_star}) ---\n{txt}")
    return "=== VLM FAILURE REPORT ===\n" + "\n".join(sections) + "\n=== END ==="
