"""DISTIL — Demonstration Distillation for Sample-Efficient Imitation Learning.

Consolidated, self-contained module (04_..md): the robosuite UR5e engine (Lift / Wipe
/ Door state modality), the DISTIL compression core (geometric descriptor, silhouette-k
clustering, Eq-9 cluster memory, FPS context set, SELECT/BRIDGE + infeasibility loop),
and the OpenRouter multi-stage LLM decision head. No hardcoded /weka path; the LLM is
a pure OpenRouter API call (no local model weights), so a `git clone` runs anywhere with
a GPU + an `.env` (golden rules 1 & 8).

Entry point (09_..md acceptance test):
    python -m distil.run --task Lift --modality state --ablation full --seed 1 \
        --budget 20 --bootstrap-dir results/shared_bootstrap --output-dir results/...

Phase 1 (this commit) ports the robot tasks in the STATE modality. GridWorld, PushT-v2,
and the image-modality policy encoders are Phase 2.
"""
__all__ = ["envs", "experts", "models", "dataset", "train", "eval", "collect",
           "diffdagger", "config", "run", "p4"]
