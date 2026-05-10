"""Hybrid pathway correction-pool-size sweep.

Runs four isolated copies of the (hybrid baseline_only, hybrid p4_only)
pair across pool sizes 20/30/40/50 in parallel, then produces a
combined super-plot. Self-contained — does not depend on
CNN_pathway/pool_sweep/ or Equivariant_pathway/pool_sweep/.
"""
