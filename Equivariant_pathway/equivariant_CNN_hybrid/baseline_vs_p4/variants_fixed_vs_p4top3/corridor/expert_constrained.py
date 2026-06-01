"""Thin re-export so callers can write
``from …corridor.expert_constrained import build_constrained_expert``.

The actual implementation lives in :mod:`blocker`. This split keeps the
public entry point named after the experiment concept ("constrained
expert") while the blocker module owns the corridor-mask construction.
"""
from .blocker import (  # noqa: F401
    build_constrained_expert,
    build_constrained_grid,
    feasibility_check,
    parse_steps_string,
)
