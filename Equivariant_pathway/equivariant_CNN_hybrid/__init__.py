"""Equivariant + CNN hybrid pathway.

Combines the visual encoder of RGBCNNPolicy (encoder + adaptive pool, NO
classification head) with a full EquivariantUNetPolicy. Forward takes
both an RGB observation and the 5-channel grid encoding; the gathered
agent-cell logits from the equivariant branch are concatenated with the
CNN feature vector and routed through a small MLP head. Both branches
remain independently trainable.
"""
