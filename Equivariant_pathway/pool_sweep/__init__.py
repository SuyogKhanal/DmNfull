"""Correction-pool-size sweep for the baseline_only vs p4_only comparison.

Runs four isolated copies of the (baseline_only, p4_only) pair — one per
correction-pool size in {20, 30, 40, 50} — fully in parallel across four
GPUs, then produces a combined 2x2 super-plot so the supervisor's
question ("does LLM-compression scale with pool size?") can be answered
in a single chart.

Each pool-size run lives under
    Equivariant_pathway/pool_sweep/runs/pool_<N>/{baseline_only,p4_only}/
with its own demos/, checkpoints/, results/, and layout YAMLs. This is
achieved via the BASELINE_ONLY_ROOT / P4_ONLY_ROOT env vars that
baseline_only/pipeline.py and p4_only/pipeline.py already honour.
"""
