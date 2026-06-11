"""Component 2b — PrescriptionClass (correction-episode configuration).

Given the AnalysisClass output (+ the top-K failure summary + KAG), the Reasoning
LLM prescribes the configuration of the NEXT correction episode:
{gripper_pose_at_start, object_pose, environment_variant}.

LOAD-BEARING empty-prescription guard (maze §6c): a failure is present, so the
prescription MUST be non-empty and fully specified. An empty / malformed / refused
prescription raises PrescriptionEmptyError; ``prescribe`` retries with an explicit
"you MUST prescribe" instruction before giving up. Never let an empty prescription
silently propagate (that collects zero demos and wastes the round).
"""
from __future__ import annotations

from typing import Dict, List

from . import prompts as P
from .runner import extract_json


class PrescriptionEmptyError(RuntimeError):
    """Raised when the LLM returns an empty/malformed/refused prescription."""


def _valid_pose(v, n: int) -> bool:
    return (isinstance(v, (list, tuple)) and len(v) == n
            and all(isinstance(x, (int, float)) for x in v))


def _validate(obj: Dict) -> Dict:
    """Raise PrescriptionEmptyError unless obj is a concrete, full prescription."""
    if not isinstance(obj, dict) or not obj:
        raise PrescriptionEmptyError("prescription is empty / not an object")
    gp = obj.get("gripper_pose_at_start")
    op = obj.get("object_pose")
    if not _valid_pose(gp, 7) or not _valid_pose(op, 7):
        raise PrescriptionEmptyError(
            f"prescription poses missing/invalid (gripper={gp}, object={op})")
    return {"gripper_pose_at_start": [float(x) for x in gp],
            "object_pose": [float(x) for x in op],
            "environment_variant": str(obj.get("environment_variant", "no change")),
            "rationale": str(obj.get("rationale", ""))[:300]}


class PrescriptionClass:
    def __init__(self, client, task_description: str, kag_text: str,
                 max_retries: int = 2):
        self.client = client
        self.task = task_description
        self.kag = kag_text
        self.max_retries = int(max_retries)

    def prescribe(self, analysis_json: str, failures: List[dict]) -> Dict:
        if self.client is None:
            raise PrescriptionEmptyError("LLM client unavailable")
        base = P.prescription_prompt(self.task, self.kag, analysis_json,
                                     P.failures_summary(failures))
        last_err = None
        for attempt in range(self.max_retries + 1):
            prompt = base if attempt == 0 else (
                base + "\n\nYOU MUST PRESCRIBE a concrete, non-empty config now. "
                "Your previous output was empty or invalid; copy a real failure's "
                "poses and adjust minimally toward the goal rather than emitting "
                "nothing.")
            try:
                txt = self.client.reason(prompt, system_prompt=P.PRESCRIPTION_SYSTEM)
                return _validate(extract_json(txt) or {})
            except PrescriptionEmptyError as exc:
                last_err = exc
                continue
            except Exception as exc:  # transport error
                last_err = exc
                continue
        raise PrescriptionEmptyError(
            f"no valid prescription after {self.max_retries + 1} attempts: {last_err}")


def _cube_validate(obj: Dict) -> Dict:
    """Raise PrescriptionEmptyError unless obj is a concrete cube layout
    (cubeA_xyz + cubeB_xyz as 3-floats; z-rotations optional)."""
    if not isinstance(obj, dict) or not obj:
        raise PrescriptionEmptyError("cube prescription empty / not an object")
    a, b = obj.get("cubeA_xyz"), obj.get("cubeB_xyz")
    if not _valid_pose(a, 3) or not _valid_pose(b, 3):
        raise PrescriptionEmptyError(
            f"cube positions missing/invalid (cubeA={a}, cubeB={b})")

    def _optf(v):
        return float(v) if isinstance(v, (int, float)) else None

    return {"cubeA_xyz": [float(x) for x in a],
            "cubeB_xyz": [float(x) for x in b],
            "cubeA_zrot": _optf(obj.get("cubeA_zrot")),
            "cubeB_zrot": _optf(obj.get("cubeB_zrot")),
            "n_demos": int(obj.get("n_demos", 1) or 1),
            "rationale": str(obj.get("rationale", ""))[:300]}


class CubePrescriptionClass:
    """Cube-layout prescription (StackCube/PickCube): the LLM aggregates the top-K
    failure analyses into ONE corrective scene = where to place cube A (pick) and
    cube B (base). Same empty-prescription HARD-FLOOR guard + retry as PushT."""

    def __init__(self, client, task_description: str, kag_text: str,
                 max_retries: int = 2):
        self.client = client
        self.task = task_description
        self.kag = kag_text
        self.max_retries = int(max_retries)

    def prescribe(self, analysis_json: str, failures: List[dict],
                  extra_addendum: str = "") -> Dict:
        if self.client is None:
            raise PrescriptionEmptyError("LLM client unavailable")
        base = P.cube_prescription_prompt(self.task, self.kag, analysis_json,
                                          P.failures_summary(failures))
        if extra_addendum:
            base = base + "\n\n" + extra_addendum
        last_err = None
        for attempt in range(self.max_retries + 1):
            prompt = base if attempt == 0 else (
                base + "\n\nYOU MUST PRESCRIBE a concrete, non-empty cube layout now. "
                "Your previous output was empty or invalid; copy a real failure's "
                "cube positions and adjust minimally toward a graspable, stackable "
                "layout rather than emitting nothing.")
            try:
                txt = self.client.reason(prompt, system_prompt=P.CUBE_PRESCRIPTION_SYSTEM)
                return _cube_validate(extract_json(txt) or {})
            except PrescriptionEmptyError as exc:
                last_err = exc
                continue
            except Exception as exc:  # transport error
                last_err = exc
                continue
        raise PrescriptionEmptyError(
            f"no valid cube prescription after {self.max_retries + 1} attempts: {last_err}")
