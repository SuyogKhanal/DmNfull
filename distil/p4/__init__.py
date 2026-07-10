"""P4-LLM V3 Hybrid for robosuite UR5e (Diff-DAgger comparison).

Adds the P4-LLM method on top of the repo's Diff-DAgger infrastructure: each round
screens the current policy for failures, clusters them into modes, and an LLM
decides how to spend ONE demonstration — SELECT (on-policy correct one real
failure) or BRIDGE (prescribe a middle-ground scene covering several). The
compression core (clustering / diversity / memory / telemetry / parse) is ported
verbatim from the ManiSkill P4-LLM; the descriptor, workspace bounds, BRIDGE
scene-prescription, planner binding, VLM rendering, and the run loop are
robosuite-specific.
"""
from .clustering import cluster_failures, pick_dominant, ClusterResult, Cluster
from .diversity import farthest_point_select
from .memory import CentroidMemory
from .telemetry import Telemetry
from .parse import parse_choice

__all__ = [
    "cluster_failures", "pick_dominant", "ClusterResult", "Cluster",
    "farthest_point_select", "CentroidMemory", "Telemetry", "parse_choice",
]
