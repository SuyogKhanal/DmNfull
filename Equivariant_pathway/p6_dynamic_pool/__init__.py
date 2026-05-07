"""LLM-guided dynamic-pool-expansion equivariant policy pipeline.

Research claim: "LLM-guided dynamic pool expansion overcomes correction
pool saturation in DAgger without test-set leakage."

Differs from p6_only in one critical way: the layouts the LLM prescribes
each round are APPENDED to the correction pool, so the next round rolls
the policy out on the GROWING pool. The held-out 50 layouts remain
completely untouched until the final eval — zero feedback path from them
into any demo collector.
"""
