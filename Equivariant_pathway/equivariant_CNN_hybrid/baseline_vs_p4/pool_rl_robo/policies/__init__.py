"""Shared diffusion-policy backbone (the novice π_nov).

Backbone is SEPARATE from the demonstration-acquisition rule: all 7 methods
(p4_top3, diff_dagger, and the 5 IIL baselines) share the SAME diffusion policy
for a given task; only the query / prescription rule differs. The policy itself
comes from the fork (a V-objective CNN-UNet DiffDAggerPolicy, instantiated from
the task's Hydra config); ``factory`` is the thin build/dataset wrapper.
"""
from .factory import build_policy, make_dataset  # noqa: F401
