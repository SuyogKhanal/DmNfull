"""Random-expansion baseline for the LLM-guided dynamic-pool claim.

Same architecture as p6_dynamic_pool except every round's new pool
layouts are sampled UNIFORMLY AT RANDOM from the layout space (with
heldout / training / current-pool signatures blocked). The random
expansion budget per round equals the previous round's heldout failure
count, so this method gets a strictly LARGER expert-query budget than
p6_dynamic_pool's LLM prescriptions — the comparison is "is the LLM's
small targeted set worth more than a much bigger random sample?"

The 50 held-out layouts are still never used as demo sources. Only
their failure COUNT feeds back as a budget signal.
"""
