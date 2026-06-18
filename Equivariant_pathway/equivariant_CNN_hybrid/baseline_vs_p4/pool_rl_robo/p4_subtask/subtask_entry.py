"""Fix 2 (core) — reconstruct the failure sub-task entry and bound the demo there.

Diff-DAgger corrects surgically at the failure point t*. p4_top3 instead
re-demonstrates a whole fresh episode. This module builds the bridge:

  * ANCHOR  — the representative failure's reconstructed t* state (tee pose +
              full robot qpos + stick TCP), a REAL state the policy reached.
  * PERTURB — a BOUNDED shift of the anchor tee toward the LLM's coverage point
              (the LLM emits an absolute in-bounds tee; we keep only the part of
              that move within `perturb_max_*` of the anchor, so the demo stays
              near a real failure → the recorded arm qpos remains physically
              plausible → the expert demonstrates only the hard remaining push).
              The TCP is re-derived in the tee's frame so the stick stays on the
              same push face after the shift.
  * EXACT   — zero-perturbation replay of the anchor (the SAFE FALLBACK the
              infeasibility loop escalates to; guaranteed penetration-free).
  * FRESH   — last resort: the LLM's absolute pose with a canonical robot
              (== old p4_top3 behaviour), used when even the exact anchor fails.

Everything is clamped to the KAG workspace bounds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .descriptor import FailureDescriptor
from . import kag_bounds as KB


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _rot2d(x: float, y: float, theta: float):
    c, s = math.cos(theta), math.sin(theta)
    return c * x - s * y, s * x + c * y


@dataclass
class Anchor:
    ep_dir: str
    episode_id: int
    t_star: int
    T: int
    tee_xyz: List[float]
    tee_theta: float
    full_qpos: List[float]
    tcp_xyz: List[float]
    peak_loss: float

    def digest(self) -> Dict[str, Any]:
        return {"ep_dir": self.ep_dir, "episode_id": self.episode_id,
                "t_star": self.t_star, "T": self.T,
                "tee_xyz": [round(v, 4) for v in self.tee_xyz],
                "tee_theta": round(self.tee_theta, 4),
                "tcp_xyz": [round(v, 4) for v in self.tcp_xyz],
                "full_qpos": [round(v, 4) for v in self.full_qpos],
                "peak_loss": round(self.peak_loss, 6)}


@dataclass
class ResetSpec:
    tee_xyz: List[float]
    tee_zrot: float
    tcp_xyz: List[float]
    agent_qpos: Optional[List[float]]   # None ⇒ canonical robot (fresh-tee fallback)
    mode: str                           # "onpolicy_correction" | "perturbed" |
                                        # "exact_anchor" | "fresh_tee"
    seed: Optional[int] = None          # v2: scene seed of the REAL failure to
                                        # re-roll + correct on-policy
    provenance: Dict[str, Any] = field(default_factory=dict)

    def digest(self) -> Dict[str, Any]:
        return {"mode": self.mode, "seed": self.seed,
                "tee_xyz": [round(v, 4) for v in self.tee_xyz],
                "tee_zrot": round(self.tee_zrot, 4),
                "tcp_xyz": [round(v, 4) for v in self.tcp_xyz],
                "agent_qpos": ([round(v, 4) for v in self.agent_qpos]
                               if self.agent_qpos is not None else None),
                "provenance": self.provenance}


def onpolicy_correction_spec(member: "FailureDescriptor") -> ResetSpec:
    """v2 (DAgger-faithful): the demo = re-roll the chosen REAL failure's scene
    seed with the CURRENT policy, then the expert corrects on-policy from the
    visited divergence state (p4_select's winning mechanism). No teleport, no
    synthetic pose — the tee fields here are informational (telemetry/memory)."""
    return ResetSpec(
        tee_xyz=KB.clamp_tee(member.tee_xyz), tee_zrot=float(member.tee_theta),
        tcp_xyz=KB.clamp_tcp(member.tcp_xyz), agent_qpos=None,
        mode="onpolicy_correction", seed=int(member.seed),
        provenance={"episode_id": member.episode_id, "ep_dir": member.ep_dir,
                    "t_star_discovery": member.t_star, "T": member.T,
                    "peak_loss": round(member.peak_loss, 6)})


def build_anchor(rep: FailureDescriptor) -> Anchor:
    return Anchor(
        ep_dir=rep.ep_dir, episode_id=rep.episode_id, t_star=rep.t_star, T=rep.T,
        tee_xyz=KB.clamp_tee(rep.tee_xyz), tee_theta=float(rep.tee_theta),
        full_qpos=list(rep.full_qpos), tcp_xyz=KB.clamp_tcp(rep.tcp_xyz),
        peak_loss=rep.peak_loss)


def _rederive_tcp(anchor: Anchor, new_tee_xyz: List[float], new_theta: float) -> List[float]:
    """Keep the stick on the same push face: take the anchor's tcp-relative-to-tee
    offset (in the anchor tee frame) and re-apply it at the shifted tee/orientation."""
    ox = anchor.tcp_xyz[0] - anchor.tee_xyz[0]
    oy = anchor.tcp_xyz[1] - anchor.tee_xyz[1]
    # express offset in the anchor tee frame, then re-apply at the new orientation
    rx, ry = _rot2d(ox, oy, -anchor.tee_theta)
    wx, wy = _rot2d(rx, ry, new_theta)
    tcp = [new_tee_xyz[0] + wx, new_tee_xyz[1] + wy, anchor.tcp_xyz[2]]
    return KB.clamp_tcp(tcp)


def apply_coverage_perturbation(anchor: Anchor, llm_tee_xyz, llm_tee_zrot,
                                *, max_xy: float = 0.06, max_theta: float = 0.4) -> ResetSpec:
    """Shift the anchor tee toward the LLM's coverage point, bounded so the demo
    stays near a real failure (anchor qpos stays plausible). Returns a perturbed
    ResetSpec that starts the expert mid-task at the failure sub-task."""
    dx = float(llm_tee_xyz[0]) - anchor.tee_xyz[0]
    dy = float(llm_tee_xyz[1]) - anchor.tee_xyz[1]
    # L2 cap on the xy shift so the demo stays within max_xy metres of the anchor
    # (a per-axis clamp would allow up to max_xy*sqrt(2)).
    nrm = math.hypot(dx, dy)
    if nrm > max_xy and nrm > 1e-9:
        s = max_xy / nrm
        dx, dy = dx * s, dy * s
    dth = KB.clamp(_wrap_pi(float(llm_tee_zrot) - anchor.tee_theta), -max_theta, max_theta)
    new_tee = KB.clamp_tee([anchor.tee_xyz[0] + dx, anchor.tee_xyz[1] + dy, KB.TEE_Z])
    new_theta = float(anchor.tee_theta + dth)
    tcp = _rederive_tcp(anchor, new_tee, new_theta)
    return ResetSpec(
        tee_xyz=new_tee, tee_zrot=new_theta, tcp_xyz=tcp,
        agent_qpos=list(anchor.full_qpos), mode="perturbed",
        provenance={"anchor": anchor.digest(),
                    "applied_delta": {"dx": round(dx, 4), "dy": round(dy, 4),
                                      "dtheta": round(dth, 4)},
                    "llm_tee_xyz": [round(float(v), 4) for v in llm_tee_xyz[:3]],
                    "llm_tee_zrot": round(float(llm_tee_zrot), 4)})


def exact_anchor_resetspec(anchor: Anchor) -> ResetSpec:
    """Zero-perturbation replay of the recorded failure state (safe fallback)."""
    return ResetSpec(
        tee_xyz=list(anchor.tee_xyz), tee_zrot=float(anchor.tee_theta),
        tcp_xyz=list(anchor.tcp_xyz), agent_qpos=list(anchor.full_qpos),
        mode="exact_anchor", provenance={"anchor": anchor.digest()})


def fresh_tee_resetspec(llm_tee_xyz, llm_tee_zrot, llm_tcp_xyz) -> ResetSpec:
    """Last resort: the LLM's absolute pose with a canonical robot (≡ p4_top3)."""
    return ResetSpec(
        tee_xyz=KB.clamp_tee(llm_tee_xyz), tee_zrot=float(llm_tee_zrot),
        tcp_xyz=KB.clamp_tcp(llm_tcp_xyz), agent_qpos=None, mode="fresh_tee",
        provenance={"note": "fresh-tee fallback (canonical robot)"})
