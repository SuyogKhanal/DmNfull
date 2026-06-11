"""p4_top3 arm — the P4-LLM method under evaluation.

P4-LLM rolls out the diffusion novice, takes the top-K (=3) highest-loss
failures, and runs the three-component pipeline
  VLM (start / high-loss-t* / end frames)
    → Reasoning LLM (root-cause + phase → prescription)
    → Config LLM (prescription → env reset config)
    → load the config into the simulator → expert solves it → ONE corrective demo,
retrain from scratch every Nd, evaluate the frozen held-out set.

ENGINE (first PushT submission): this is exactly the fork's tested
``LLMGuidedDAggerPipeline`` in SEQUENTIAL mode (one prescribed config/round) with
the top-K failure cap — it already implements the full VLM→reason→prescribe→load
(PushT-Start-v0)→expert-solve→retrain loop with the empty-prescription HARD-FLOOR
guards. We wrap it, point it at the suite's per-task KAG, and write the suite-format
learning_curve.json. The fresh 3-component modules in this package
(``vlm`` / ``analysis`` / ``prescription`` / ``config_generator`` / ``prompts``)
document the architecture to the prompt's exact schema and are the implementation
path for the non-PushT tasks (where no fork pipeline config exists yet).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..envs import env_setup as E

E.bootstrap_fork_path()

from . import kag as KAG  # noqa: E402


def _suite_curve(history, *, method, env_id, seed, budget, stopped, out: Path) -> Dict:
    """Convert the fork p4 history (defensive on key names) → suite schema."""
    def g(r, *names):
        for n in names:
            if n in r and r[n] is not None:
                return r[n]
        return None

    hist = []
    for r in (history or []):
        hist.append({
            "round": g(r, "round"),
            "n_queries": g(r, "expert_calls", "budget_used", "n_queries", "demos_used"),
            "success_rate": g(r, "heldout_success_rate", "success_rate", "sr"),
            "demos_added": g(r, "demos_added", "n_new_demos"),
            "n_demos": g(r, "cumulative_demos", "n_demos"),
            "stop_reason": g(r, "stop_reason"),
        })
    last = hist[-1] if hist else {}
    payload = {"method": method, "env": env_id, "seed": seed, "budget": budget,
               "backbone": "diffusion", "p4_mode": "sequential_top3",
               "history": hist,
               "final_performance": {"success_rate": last.get("success_rate"),
                                     "n_queries": last.get("n_queries")},
               "stopped_reason": stopped or last.get("stop_reason") or "budget_exhausted"}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    return payload


# suite-key → fork AnalyzerConfig attribute (system-prompt + addendum overrides).
# Any key absent/empty in the YAML ⇒ the fork keeps its hardcoded default (zero-diff).
_PROMPT_KEY_TO_ATTR = {
    "analysis_system":     "analysis_system_override",
    "prescription_system": "prescription_system_override",
    "cross_system":        "cross_system_override",
    "aggregator_system":   "aggregator_system_override",
    "vlm_system":          "vlm_system_override",
    "reasoning_addendum":  "reasoning_addendum_override",
    "aggregator_addendum": "aggregator_addendum_override",
    "sequential_addendum": "sequential_addendum_override",
}


def _apply_prompt_overrides(pcfg, suite_env_id: str, prompts_dir) -> None:
    """Load per-env prompt file (<dir>/<env>.yaml, fallback <dir>/default.yaml) and set
    the fork AnalyzerConfig *_override fields. No-op when prompts_dir is unset/missing
    (→ fork's hardcoded prompts, i.e. zero behavioural change)."""
    if not prompts_dir:
        return
    import yaml
    d = Path(prompts_dir)
    if not d.is_absolute():
        d = Path(__file__).resolve().parent.parent / prompts_dir
    f = d / f"{suite_env_id}.yaml"
    if not f.is_file():
        f = d / "default.yaml"
    if not f.is_file():
        print(f"[p4_top3] no prompt file under {d}; using fork default prompts")
        return
    spec = yaml.safe_load(f.read_text()) or {}
    n = 0
    for key, attr in _PROMPT_KEY_TO_ATTR.items():
        val = spec.get(key)
        if isinstance(val, str) and val.strip():
            setattr(pcfg.analyzer, attr, val)
            n += 1
    print(f"[p4_top3] loaded {n} prompt override(s) from {f}")


def run_p4_arm(*, method: str, top_k: int, cfg, env, eval_env, reposition_env,
               expert, make_policy, make_dataset, init_ckpt: str, init_sr: float,
               work_dir: str, k: Dict[str, Any], suite_env_id: str,
               llm_client_on: bool) -> Dict[str, Any]:
    """Run p4_top3 for one env. Returns the suite summary dict; writes
    learning_curve.json into work_dir."""
    from diffdagger.main_pipeline.config import (
        PipelineConfig, ProfileConfig, BudgetConfig,
    )
    from diffdagger.main_pipeline.pipeline import LLMGuidedDAggerPipeline

    if reposition_env is None:
        raise RuntimeError(
            f"p4_top3 needs a reposition env for {suite_env_id}; only PushT is "
            f"wired for the first submission (PushT-Start-v0).")
    if not llm_client_on:
        print("[p4_top3] WARNING: LLM disabled (--no-llm); the pipeline will fall "
              "back to env-sampled failure configs (no prescription).")

    # Speed up the per-step diffusion-loss in the failure-discovery rollout
    # (set before make_policy + retrains, which re-instantiate from cfg.policy).
    if int(k.get("p4_batch_multiplier", 0)) > 0:
        cfg.policy.batch_multiplier = int(k["p4_batch_multiplier"])

    # EARLY-STOP at target (queries-to-90% is the headline metric — same as the
    # baselines). The fork stops the p4 cycle when the per-round failure-discovery
    # ROLLOUT SR ≥ target; that estimate must be LOW-NOISE or it false-triggers
    # (a 30-episode rollout hit 0.90 by luck at true SR 0.68). config.yaml sets
    # rollout_episodes=60 so the rollout SR reliably reflects the true ~90%.
    # The reported queries-to-90% uses the frozen HELD-OUT curve (uniform across
    # all 7 methods). max_rounds is raised so infeasible-prescription rounds don't
    # starve the budget before the target is reached.
    target = float(k["target_sr"])
    work = Path(work_dir); work.mkdir(parents=True, exist_ok=True)
    pcfg = PipelineConfig(
        dataset_dir=cfg.dataset_dir,
        work_dir=str(work / "p4_engine"),
        num_initial_demos=int(k["initial_demos"]),
        max_dagger_iterations=max(int(k["max_rounds"]), 3 * int(k["budget"])),
        target_success_rate=target,
    )
    pcfg.profile = ProfileConfig.from_name(str((cfg.get("profile", "P4")) or "P4"))
    pcfg.budget = BudgetConfig(
        budget_total=int(k["budget"]),
        target_success_rate=target,
        max_rounds=max(int(k["max_rounds"]), 3 * int(k["budget"])),
        heldout_seed_base=7777,
        heldout_num_ep=int(k["heldout_n"]),
        rollout_episodes=int(k.get("rollout_episodes", 100)),
        phase_b_max_workers=int(k["phase_b_workers"]),
        mode="sequential",                  # one prescribed config/round
        nd_retrain=int(k["nd_retrain"]),
    )
    # top-K failure analysis: cap the analysed failures per round to K=3.
    pcfg.budget.max_failures_per_round = int(top_k)
    kp = KAG.kag_text_path(suite_env_id)    # per-task KAG text doc (fork-readable)
    if kp:
        pcfg.analyzer.kag_path = str(kp)
    # Frames per failed episode = start + SINGLE highest-loss + end (P4-top3 spec).
    pcfg.analyzer.frames.top_k_high_loss = 1
    # In-round retry caps: empty-prescription retry + infeasibility loop (the expert
    # must be able to solve the prescribed config before the demo is kept).
    pcfg.budget.represcribe_attempts = int(k.get("p4_represcribe_attempts", 5))
    pcfg.budget.infeasible_attempts = int(k.get("p4_infeasible_attempts", 5))
    # Per-env configurable prompts (override the fork's hardcoded systems/addenda).
    _apply_prompt_overrides(pcfg, suite_env_id, k.get("p4_prompts_dir"))
    pcfg.analyzer.sync_openai()

    pipe = LLMGuidedDAggerPipeline(pcfg)
    pipe.setup(env, expert, make_policy(), make_dataset(), cfg,
               eval_env=eval_env, reposition_env=reposition_env)
    summ = pipe.run_budget_cycle(
        collect_initial=True, initial_seeds=list(range(int(k["initial_demos"]))),
        init_ckpt=init_ckpt, init_sr=init_sr,
        round_epoch=int(k["round_epochs"]), max_train_steps=int(k["max_train_steps"]),
        high_loss_percentile=int(k["p4_high_loss_percentile"]))

    history = (summ or {}).get("history", [])
    stopped = (summ or {}).get("stop_reason") or (summ or {}).get("stopped_reason")
    payload = _suite_curve(history, method=method, env_id=suite_env_id,
                           seed=int(k["seed"]), budget=int(k["budget"]),
                           stopped=stopped, out=work / "learning_curve.json")
    return {"method": method, "env": suite_env_id,
            "final_performance": payload["final_performance"],
            "stopped_reason": payload["stopped_reason"], "history": payload["history"],
            "total_expert_calls": payload["final_performance"].get("n_queries")}
