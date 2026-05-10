"""CNN_pathway correction-pool-size sweep (RGBCNNPolicy version).

Runs four isolated copies of the (CNN baseline_only, CNN p4_only) pair
in parallel across four GPUs, one per pool size in {20, 30, 40, 50},
and produces a combined super-plot. Self-contained — does not import
from Equivariant_pathway/.
"""
