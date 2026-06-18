"""PushT workspace bounds (single source of truth for clamping + telemetry).

These mirror the injected KAG document
(``pool_rl_robo/p4/kag/PushT-v1.json`` → ``ws_tee`` / ``ws_tcp``). They are the
PPO expert's RELIABLE region; configs outside are dropped/clamped because the
expert was not trained there. We hardcode them here so the planner can clamp
without a file read, and expose a loader so telemetry can verify them against
the live KAG json (provenance).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

# tee (movable T) reliable init range
TEE_X: Tuple[float, float] = (-0.20, 0.20)
TEE_Y: Tuple[float, float] = (-0.25, 0.05)
TEE_Z: float = 0.021

# tcp (stick end-effector) reliable range
TCP_X: Tuple[float, float] = (-0.35, 0.35)
TCP_Y: Tuple[float, float] = (-0.35, 0.35)
TCP_Z: Tuple[float, float] = (0.02, 0.08)


def clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def clamp_tee(xyz) -> list:
    return [clamp(float(xyz[0]), *TEE_X), clamp(float(xyz[1]), *TEE_Y), TEE_Z]


def clamp_tcp(xyz) -> list:
    return [clamp(float(xyz[0]), *TCP_X),
            clamp(float(xyz[1]), *TCP_Y),
            clamp(float(xyz[2]), *TCP_Z)]


def as_dict() -> Dict[str, object]:
    return {
        "tee_x": list(TEE_X), "tee_y": list(TEE_Y), "tee_z": TEE_Z,
        "tcp_x": list(TCP_X), "tcp_y": list(TCP_Y), "tcp_z": list(TCP_Z),
    }


def load_from_kag(kag_json_path) -> Dict[str, object]:
    """Best-effort read of ws_tee/ws_tcp from the suite KAG json (telemetry
    provenance only; returns the hardcoded dict on any failure)."""
    try:
        data = json.loads(Path(kag_json_path).read_text())
        nodes = {n.get("id"): n.get("properties", {}) for n in data.get("nodes", [])}
        tee = nodes.get("ws_tee", {})
        tcp = nodes.get("ws_tcp", {})
        out = as_dict()
        if tee:
            out["tee_x"], out["tee_y"], out["tee_z"] = tee.get("x", out["tee_x"]), tee.get("y", out["tee_y"]), tee.get("z", out["tee_z"])
        if tcp:
            out["tcp_x"], out["tcp_y"], out["tcp_z"] = tcp.get("x", out["tcp_x"]), tcp.get("y", out["tcp_y"]), tcp.get("z", out["tcp_z"])
        out["_source"] = str(kag_json_path)
        return out
    except Exception as e:  # pragma: no cover
        d = as_dict()
        d["_source"] = f"hardcoded (KAG read failed: {e})"
        return d
