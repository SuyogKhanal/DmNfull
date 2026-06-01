"""Per-environment policy backbones (the novice π_nov).

Backbone is SEPARATE from the interactive-IL algorithm (SafeDAgger / DropoutDAgger
/ EnsembleDAgger / ThriftyDAgger / Stagger / P4-LLM): all six methods share the
SAME backbone for a given env; only the expert-query decision rule differs
(policy_backbone_guide.md §1).

Mapping (our 5 envs):
  HalfCheetah-v4 / Hopper-v4 / Walker2d-v4  -> Gaussian MLP on the low-dim state.
  FetchReach-v4 / FetchPickAndPlace-v4      -> Diffusion Policy (single-step DDPM)
                                               on FROZEN-R3M image features (+ goal
                                               / gripper proprioception).

Uncertainty (for Dropout/Ensemble/Thrifty) comes from ``Policy.samples(feat, k)``:
MLP -> MC-dropout or ensemble members; Diffusion -> k denoise samples (variance).
"""
from .base import Policy  # noqa: F401
from .factory import build_policy  # noqa: F401
