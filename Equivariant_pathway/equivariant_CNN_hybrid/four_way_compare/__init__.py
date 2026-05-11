"""Four-way comparison wrapper for the hybrid pathway.

Runs baseline_only / p4_only / p5_only / p6_only on the SAME initial
20 BFS demos and the SAME initial hybrid checkpoint, at correction
pool size 40, then renders 3 pair charts (baseline-vs-each) plus a
super 4-way overlay.

RAG isolation: P5's bank lives at runs/p5_only/rag_bank/, P6's at
runs/p6_only/rag_bank/. Different filesystem paths, no contamination.

TKF isolation (P6): tkf.demo_dir = runs/p6_only/demos/, which
contains only BFS demos (initial 20 + per-round LLM-prescribed).
Source-filter at bootstrap rejects any baseline_dagger correction.
"""
