"""Top-level model package.

This file exists so `model/` resolves as a regular package even when a
sibling module named `model` (e.g. CNN_pathway/model.py or
Equivariant_pathway/model.py) is on sys.path. Without it, `model/` is
only an implicit namespace package, and PEP 328 makes a regular module
of the same name win — which broke `from model.diffusion_policy import
MazeDiffusionPolicy` whenever pipeline/rollout.py was reached from a
script launched out of one of the pathway folders.
"""
