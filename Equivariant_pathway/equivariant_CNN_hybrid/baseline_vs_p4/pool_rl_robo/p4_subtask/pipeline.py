"""p4_subtask arm — the improved P4-LLM method (replaces the losing p4_top3).

Same engine as p4_top3 (the fork's ``LLMGuidedDAggerPipeline`` in sequential
mode: VLM → reason → prescribe → infeasibility loop → feedback loop), but we
INJECT a ``SubtaskPlanner`` + the ``collect_subtask_demo`` callable so each
prescribed demo:
  * is chosen by code-level clustering + diversity (Fix 1 + 4),
  * starts MID-TASK at the reconstructed failure sub-task (Fix 2), and
  * rotates coverage via a cross-round centroid memory.

The fork carries only three default-OFF gated hooks; with the planner absent the
p4_top3 / diff_dagger paths are byte-identical. All guidance reaches the LLM via
the always-effective per-round addendum (fork hook B), so ``prompts_dir`` stays
null (the fork's *_override path is the deferred Phase-B, not relied on here).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from ..envs import env_setup as E

E.bootstrap_fork_path()

from . import kag_bounds as KB            # noqa: E402
from .planner import SubtaskPlanner       # noqa: E402
from .collect import collect_subtask_demo_v2  # noqa: E402

# reuse the p4_top3 wrapper's suite-curve writer + KAG resolver (no behaviour change)
from ..p4 import kag as KAG               # noqa: E402
from ..p4.pipeline import _suite_curve, _apply_prompt_overrides  # noqa: E402

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _strict_llm() -> bool:
    """DISEIL_STRICT_LLM=1 => an unusable LLM aborts the run instead of degrading."""
    return os.environ.get("DISEIL_STRICT_LLM", "0") == "1"


def _preflight_openrouter() -> None:
    """Assert OpenRouter is configured AND actually answering, before the loop starts.

    The fork swallows every LLM error (llm_clients._oai_client raises lazily inside the
    pipeline's try/except), so a missing key / wrong base_url / 404 does NOT crash: the
    run finishes with n_queries=0 and writes a plausible learning_curve.json that is not
    DISEIL at all. There is no fallback flag to catch it after the fact. So we make one
    real call up front and refuse to start if it does not come back.
    """
    base = os.environ.get("OPENAI_BASE_URL", "")
    if base.rstrip("/") != OPENROUTER_BASE_URL:
        raise RuntimeError(
            f"DISEIL_STRICT_LLM=1: OPENAI_BASE_URL must be exactly {OPENROUTER_BASE_URL}, "
            f"got {base!r}. (run_pool_rl_robo.sh clobbers it with a local vLLM proxy URL — "
            f"use the OpenRouter launcher instead.)")
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("DISEIL_STRICT_LLM=1: OPENAI_API_KEY is not set.")
    # The fork reads OPENAI_API_KEY only. An OpenAI sk-proj- key against OpenRouter is a
    # 401, and an OpenRouter key against api.openai.com would be a silent model 404.
    if not key.startswith("sk-or-"):
        raise RuntimeError(
            "DISEIL_STRICT_LLM=1: OPENAI_API_KEY is not an OpenRouter key (expected the "
            "sk-or-... OPENROUTER_API_KEY). The fork reads OPENAI_API_KEY only.")

    from types import SimpleNamespace
    from diffdagger.main_analysis.llm_clients import PlainClient
    llm_model = os.environ.get("LLM_MODEL_NAME", "qwen/qwen3-32b")
    cfg = SimpleNamespace(model=llm_model, effort="low", max_output_tokens=512)
    r = PlainClient(cfg).structure('Return ONLY this JSON: {"ok":true}')
    text, ptok, ctok = r.get("text", ""), r.get("prompt_tokens", 0), r.get("completion_tokens", 0)
    if not text.strip() or (ptok + ctok) <= 0:
        raise RuntimeError(
            f"DISEIL_STRICT_LLM=1: OpenRouter pre-flight returned no usable output "
            f"(text={text!r}, prompt_tokens={ptok}, completion_tokens={ctok}). Aborting "
            f"rather than running the deterministic no-LLM fallback.")
    print(f"[llm-guard] OpenRouter pre-flight OK: base_url={base} model={llm_model} "
          f"tokens={ptok}+{ctok} reply={text.strip()[:60]!r}", flush=True)


def run_p4_subtask_arm(*, method: str, top_k: int, cfg, env, eval_env, reposition_env,
                       expert, make_policy, make_dataset, init_ckpt: str, init_sr: float,
                       work_dir: str, k: Dict[str, Any], suite_env_id: str,
                       llm_client_on: bool) -> Dict[str, Any]:
    """Run p4_subtask for one env. Returns the suite summary dict; writes
    learning_curve.json into work_dir."""
    from diffdagger.main_pipeline.config import (
        PipelineConfig, ProfileConfig, BudgetConfig,
    )
    from diffdagger.main_pipeline.pipeline import LLMGuidedDAggerPipeline

    if reposition_env is None:
        raise RuntimeError(
            f"p4_subtask needs a PushT-Subtask-v0 reposition env for {suite_env_id}; "
            f"only PushT is wired (set need_repo + prefer_subtask in the orchestrator).")
    if _strict_llm():
        if not llm_client_on:
            raise RuntimeError(
                "DISEIL_STRICT_LLM=1: llm_client_on is False (--no-llm). Refusing to run — "
                "a prescription from env-sampled configs is NOT DISEIL.")
        _preflight_openrouter()
    elif not llm_client_on:
        print("[p4_subtask] WARNING: LLM disabled (--no-llm); prescription falls back "
              "to env-sampled configs (the sub-task anchor still applies).")

    if int(k.get("p4_batch_multiplier", 0)) > 0:
        cfg.policy.batch_multiplier = int(k["p4_batch_multiplier"])

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
    # Cluster over ALL round failures; analyse top_k diverse ones with the VLM/LLM.
    pcfg.budget.max_failures_per_round = int(top_k)
    kp = KAG.kag_text_path(suite_env_id)
    if kp:
        pcfg.analyzer.kag_path = str(kp)
    pcfg.analyzer.frames.top_k_high_loss = 1
    pcfg.budget.represcribe_attempts = int(k.get("p4_represcribe_attempts", 5))
    pcfg.budget.infeasible_attempts = int(k.get("p4_infeasible_attempts", 5))
    _apply_prompt_overrides(pcfg, suite_env_id, k.get("p4_prompts_dir"))  # null ⇒ no-op
    pcfg.analyzer.sync_openai()

    pipe = LLMGuidedDAggerPipeline(pcfg)
    pipe.setup(env, expert, make_policy(), make_dataset(), cfg,
               eval_env=eval_env, reposition_env=reposition_env)

    # ── INJECT the sub-task machinery (the only difference from run_p4_arm) ──
    kag_json = E.SUITE_ROOT / "p4" / "kag" / f"{suite_env_id}.json"
    planner = SubtaskPlanner(
        work_dir=str(work),
        cfg=dict(k.get("p4_subtask", {}) or {}),
        kag_json_path=str(kag_json) if kag_json.is_file() else None)
    pipe._subtask_planner = planner
    pipe._subtask_collect = collect_subtask_demo_v2
    print(f"[p4_subtask] planner injected (collect={planner.collect_mode}, "
          f"signal={planner.signal_mode}, max_clusters={planner.max_k}, "
          f"cluster_features={planner.cluster_features}"
          f"{f'/{planner.embed_model}' if planner.cluster_features == 'visual' else ''}, "
          f"heldout_confirm={planner.confirm_target_with_heldout}); "
          f"telemetry → {work/'telemetry'}")

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
