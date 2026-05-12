"""Budget-constrained baseline-vs-P4 comparison (hybrid pathway).

A single repetition of the comparison runs the EquivariantCNNHybridPolicy
starting from 20 BFS demos and asks: who reaches the 90% heldout SR target
faster, given a hard cap of K_BUDGET extra demonstrations? The two methods
(baseline DAgger; P4 LLM-prescribed) share the same initial model, the same
heldout layouts, and — within a single repetition — the same correction pool.

Across repetitions the correction-pool composition is re-randomised with a
strict guard against contamination from either the training-20 or the
heldout-200 sets. The aggregate chart plots mean +/- std heldout SR over
the K_BUDGET extra demos for both methods, on a fixed common x-grid so
runs are directly comparable.
"""
