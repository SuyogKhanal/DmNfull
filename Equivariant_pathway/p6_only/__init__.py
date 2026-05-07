"""Minimal P6-only equivariant policy pipeline.

A standalone subset of Equivariant_pathway that runs ONLY the P6 LLM
analysis loop on the equivariant model, sharing layouts / initial demos
/ initial checkpoint with baseline_only/ so the methods are directly
comparable. P6 = VLM + Reasoning + KAG + RAG + TKF + Cross-Episode +
Plain-LLM Aggregator (full stack).

Prompts for the three LLM stages live in p6_only/prompts/*.yml so the
prompting strategy can be edited without touching code.
"""
