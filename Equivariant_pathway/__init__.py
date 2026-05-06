"""Equivariant_pathway — p4m-equivariant maze policy + full active loop.

Mirrors CNN_pathway in shape (model.py, dataset.py, train.py, rollout_test.py,
analyze_p4/p5/p6.py, training_layouts.yaml, test_layouts.yaml) but swaps the
CNN+MLP backbone for the EquivariantUNetPolicy from the dmnpol repo, and
extends the loop with:
  - 50-layout held-out random test set (heldout_test_layouts.yaml)
  - per-layout JSON + image tracking for train/test/heldout
  - baseline DAgger (no profile reasoning, retrain every 4 corrective episodes)
  - profile-isolated demo/checkpoint folders for P4/P5/P6
  - termination by heldout success >= 90% (no 5-round cap)
  - end-of-cycle line charts comparing baseline vs P4/P5/P6.
"""
