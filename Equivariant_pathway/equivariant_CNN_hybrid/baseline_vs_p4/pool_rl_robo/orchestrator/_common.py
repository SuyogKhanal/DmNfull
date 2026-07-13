"""Shared driver for the 7-method ManiSkill comparison.

One SLURM array task = one ENV. All 7 methods run sequentially into
``results/{ENV}/run_{id}/{method}/results/`` against ONE shared initial diffusion
policy (the fork's apples-to-apples bootstrap), so only the demonstration-
acquisition rule differs between methods.

Dispatch (run_one.py::METHOD_SPEC):
  p4_top3       → p4.pipeline.run_p4_arm           (fresh 3-component LLM pipeline)
  diff_dagger   → fork run_diffdagger_baseline      (native diffusion-loss CDF rule)
  safe/dropout/ensemble/thrifty/stagger → selection.iil_baselines.run_iil_arm
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..envs import env_setup as E

E.bootstrap_fork_path()

import yaml  # noqa: E402

from ..envs import maniskill_env as MS  # noqa: E402
from ..envs import experts as X  # noqa: E402
from ..policies import factory as PF  # noqa: E402
from . import workspace as W  # noqa: E402


# ---- config + smoke ----------------------------------------------------
def load_suite_cfg(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def resolve_knobs(scfg: Dict[str, Any], args) -> Dict[str, Any]:
    """CLI flag (if set) > config.yaml > hardcoded default."""
    def pick(cli, key, default):
        if cli is not None:
            return cli
        return scfg.get(key, default)

    k = {
        "initial_demos": int(pick(args.initial_demos, "initial_demos", 50)),
        "budget": int(pick(args.budget, "budget", 100)),
        "demos_per_round": int(scfg.get("demos_per_round", 1)),
        "target_sr": float(pick(args.target_sr, "target_sr", 0.90)),
        "max_rounds": int(pick(args.max_rounds, "max_rounds", 100)),
        "max_steps": int(scfg.get("max_steps", 200)),
        "heldout_n": int(scfg.get("heldout_n", 100)),
        "seed": int(pick(args.seed, "seed", 0)),
        # training cadence (continuous protocol uses the fork's from-scratch retrain)
        "nd_retrain": int(scfg.get("nd_retrain", 4)),
        "initial_epochs": int(scfg.get("initial_epochs", scfg.get("trainer", {}).get("finetune_epochs", 300))),
        "round_epochs": int(scfg.get("round_epochs", scfg.get("trainer", {}).get("finetune_epochs", 200))),
        "max_train_steps": int(scfg.get("max_train_steps", 30000)),
        # p4
        "p4_high_loss_percentile": int(scfg.get("p4", {}).get("high_loss_percentile", 95)),
        "phase_b_workers": int(scfg.get("p4", {}).get("phase_b_workers", 8)),
        "rollout_episodes": int(scfg.get("rollout_episodes", 100)),
        "p4_batch_multiplier": int(scfg.get("p4", {}).get("batch_multiplier", 0)),
        # P4-top3 in-round retry caps (empty-prescription retry / infeasibility loop).
        "p4_represcribe_attempts": int(scfg.get("p4", {}).get("represcribe_attempts", 5)),
        "p4_infeasible_attempts": int(scfg.get("p4", {}).get("infeasible_attempts", 5)),
        "p4_prompts_dir": scfg.get("p4", {}).get("prompts_dir"),
        # p4_subtask: cluster-anchored sub-task-entry knobs (absent ⇒ inert).
        "p4_subtask": dict(scfg.get("p4", {}).get("subtask", {}) or {}),
        "p4_select_n_cand": int(scfg.get("p4_select", {}).get("n_cand", 6)),
        # cap on episodes/arm — raise it when nd=1 + run-to-budget so an arm can
        # actually spend the full demo budget (diff_dagger needs many rollouts to
        # find enough interventions once the policy is good).
        "max_episodes_per_arm": int(scfg.get("max_episodes_per_arm", 400)),
        # eval vectorisation (heldout_n must be a multiple of eval_num_envs)
        "eval_num_envs": int(scfg.get("eval_num_envs", 100)),
        "baselines": scfg.get("baselines", {}) or {},
    }
    return k


def apply_smoke(k: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal 1-round-per-method smoke (every method, incl. P4 LLM, runs once)."""
    k.update(initial_demos=2, budget=1, target_sr=0.99, max_rounds=1, nd_retrain=1,
             initial_epochs=2, round_epochs=2, max_train_steps=300,
             heldout_n=10, eval_num_envs=10, phase_b_workers=1)
    # shrink ensemble/thrifty member counts for the smoke
    bl = k["baselines"]
    for kind in ("ensemble_dagger", "thrifty_dagger"):
        bl.setdefault(kind, {})
        bl[kind]["M"] = 2
    bl.setdefault("thrifty_dagger", {})["q_epochs"] = 5
    bl.setdefault("dropout_dagger", {})["N"] = 3
    return k


def build_cfg(suite_env: str, k: Dict[str, Any]):
    """Compose the fork Hydra cfg + apply suite eval knobs. The fork cfg owns the
    policy/env/dataset/expert; we only adjust eval vectorisation + step caps and
    absolutize the fork's RELATIVE asset paths (normalizers / expert ckpts) — the
    fork resolves them from its own root, but we run from DmNfull."""
    cfg = MS.load_cfg(suite_env)
    cfg.eval_env.num_envs = int(k["eval_num_envs"])
    if int(k["heldout_n"]) % int(k["eval_num_envs"]) != 0:
        ne = int(k["eval_num_envs"])
        k["heldout_n"] = max(ne, (k["heldout_n"] // ne) * ne or ne)
    cfg.max_train_steps = int(k["max_train_steps"])

    def _abs(p):
        p = str(p)
        if Path(p).is_absolute():
            return p
        suite_p = E.SUITE_ROOT / p          # suite-local assets (e.g. StackCube)
        if suite_p.exists():
            return str(suite_p)
        return str(E.FORK_ROOT / p)         # fork assets (PushT normalizers/ckpts)

    if "normalizers_path" in cfg and cfg.normalizers_path:
        cfg.normalizers_path = _abs(cfg.normalizers_path)
    if cfg.get("expert") is not None and "ckpts" in cfg.expert:
        cfg.expert.ckpts = [_abs(c) for c in cfg.expert.ckpts]
    return cfg


# ---- learning-curve schema converter (fork → suite) --------------------
def _fork_curve_to_suite(fork_lc: Path, suite_lc: Path, *, method: str,
                         env_id: str, seed: int, budget: int) -> None:
    data = json.loads(Path(fork_lc).read_text())
    hist = []
    for r in data.get("history", []):
        hist.append({"round": r.get("round"),
                     "n_queries": r.get("expert_calls"),
                     "success_rate": r.get("heldout_success_rate"),
                     "demos_added": r.get("demos_added"),
                     "n_demos": r.get("cumulative_demos"),
                     "stop_reason": r.get("stop_reason")})
    last = hist[-1] if hist else {}
    suite_lc.parent.mkdir(parents=True, exist_ok=True)
    suite_lc.write_text(json.dumps({
        "method": method, "env": env_id, "seed": seed, "budget": budget,
        "backbone": "diffusion", "history": hist,
        "final_performance": {"success_rate": last.get("success_rate"),
                              "n_queries": last.get("n_queries")},
        "stopped_reason": (hist[-1].get("stop_reason") if hist else None)
        or "budget_exhausted"}, indent=2, default=str))


class ExpertCallCounter:
    """Transparent proxy around the expert that counts get_action / move_to_next_goal
    calls. Lets each arm report a SECONDARY cost axis — total expert interactions —
    alongside the primary IIL metric (demonstrations/corrections added).

    Why it matters for the comparison: p4_select screens n_cand candidate rollouts
    by calling expert.get_action at every step (an automated oracle discrepancy
    check), whereas diff_dagger's native loss-query makes ~0 expert calls during
    screening. Demonstrations-added is the fair, standard 'expert query' metric, but
    logging the raw call count makes the screening cost explicit and auditable.
    Delegates every other attribute/method to the wrapped expert unchanged."""
    def __init__(self, e):
        self.__dict__["_e"] = e
        self.__dict__["n_get_action"] = 0
        self.__dict__["n_corrections"] = 0

    def reset_counts(self):
        self.__dict__["n_get_action"] = 0
        self.__dict__["n_corrections"] = 0

    @property
    def total_expert_calls(self):
        return self.__dict__["n_get_action"] + self.__dict__["n_corrections"]

    def get_action(self, *a, **kw):
        self.__dict__["n_get_action"] += 1
        return self.__dict__["_e"].get_action(*a, **kw)

    def move_to_next_goal(self, *a, **kw):
        self.__dict__["n_corrections"] += 1
        return self.__dict__["_e"].move_to_next_goal(*a, **kw)

    def __getattr__(self, name):           # only for attrs not found on the proxy
        return getattr(self.__dict__["_e"], name)

    def __setattr__(self, name, value):    # forward attribute writes to the expert
        setattr(self.__dict__["_e"], name, value)


# ---- per-method arms ---------------------------------------------------
def run_method(*, method: str, spec, cfg, env, eval_env, reposition_env, expert,
               make_policy, make_dataset, init_ckpt: float, init_sr: float,
               ws, k: Dict[str, Any], llm_client_on: bool) -> Dict[str, Any]:
    kind, sel = spec  # ("p4",3) | ("baseline","<diff_dagger|iil>")
    results_dir = ws.method_dirs(method)["results"]
    results_dir.mkdir(parents=True, exist_ok=True)
    env_id = ws.env_id
    seed, budget = k["seed"], k["budget"]

    # D5 compute telemetry (paper D5_Compute table). INERT unless D5_TELEMETRY=1:
    # install() returns immediately, so every existing run is unchanged. When on,
    # it only wraps stage functions (call-through) and appends per-round wall-clock
    # + per-LLM-call token usage to a NEW side file results/<method>/telemetry/
    # d5_events.jsonl. No prompt, threshold, default or output schema is touched.
    try:
        from .. import telemetry_d5 as _D5
        _D5.install(str(results_dir), method)
    except Exception:
        pass

    if method == "p4_top3":
        # Motion-planner tasks (StackCube/…) use the SUITE-SIDE p4_top3 arm
        # (VLM→reason→prescribe CUBE layout→StackCube-Start-v0→motion-planner solve);
        # the fork's LLMGuidedDAggerPipeline is PushT-hardcoded (tee prescription).
        if E.task_spec(env_id).expert_kind == "motionplanner":
            from ..p4 import select_arm as PS
            # detector parity with diff_dagger/p4_select (identical diffusion-loss).
            dd = dict(k.get("baselines", {}).get("diff_dagger", {}) or {})
            cfg.policy.alpha = float(dd.get("alpha", cfg.policy.get("alpha", 0.99)))
            cfg.policy.patience = int(dd.get("patience", cfg.policy.get("patience", 1)))
            cfg.policy.patience_window = cfg.policy.patience
            if int(dd.get("batch_multiplier", 0)) > 0:
                cfg.policy.batch_multiplier = int(dd["batch_multiplier"])
            hp = {"num_initial_demos": k["initial_demos"],
                  "n_cand": int(k.get("p4_select_n_cand", 6)),
                  "max_episodes": int(k["max_episodes_per_arm"]),
                  "represcribe_attempts": int(k.get("p4_represcribe_attempts", 5)),
                  "infeasible_attempts": int(k.get("p4_infeasible_attempts", 5)),
                  # V3 hybrid (StackCube): p4.subtask.collect=="hybrid" turns this
                  # arm into the SELECT/BRIDGE hybrid; absent ⇒ plain p4_top3.
                  "subtask": dict(k.get("p4_subtask", {}) or {})}
            return PS.run_p4_top3_arm(
                method=method, env=env, eval_env=eval_env, reposition_env=reposition_env,
                expert=expert, cfg=cfg, make_policy=make_policy, make_dataset=make_dataset,
                init_ckpt=init_ckpt, init_sr=init_sr, work_dir=str(results_dir),
                budget=budget, nd_retrain=k["nd_retrain"], target_sr=k["target_sr"],
                max_rounds=k["max_rounds"], heldout_seed_base=7777,
                heldout_num_ep=k["heldout_n"], round_epoch=k["round_epochs"],
                max_train_steps=k["max_train_steps"], hp=hp, seed=seed,
                suite_env_id=env_id, llm_client_on=llm_client_on)
        from ..p4 import pipeline as P4
        return P4.run_p4_arm(
            method=method, top_k=int(sel), cfg=cfg, env=env, eval_env=eval_env,
            reposition_env=reposition_env, expert=expert, make_policy=make_policy,
            make_dataset=make_dataset, init_ckpt=init_ckpt, init_sr=init_sr,
            work_dir=str(results_dir), k=k, suite_env_id=env_id,
            llm_client_on=llm_client_on)

    if method == "p4_subtask":
        # Improved P4-LLM (PushT only): same fork engine as p4_top3 + an injected
        # SubtaskPlanner (cluster → dominant cluster → coverage centroid → anchor)
        # that starts the expert MID-TASK at the reconstructed failure sub-task.
        from ..p4_subtask import pipeline as P4S
        return P4S.run_p4_subtask_arm(
            method=method, top_k=int(sel), cfg=cfg, env=env, eval_env=eval_env,
            reposition_env=reposition_env, expert=expert, make_policy=make_policy,
            make_dataset=make_dataset, init_ckpt=init_ckpt, init_sr=init_sr,
            work_dir=str(results_dir), k=k, suite_env_id=env_id,
            llm_client_on=llm_client_on)

    if method == "p4_select":
        # P4-LLM selection on top of SafeDAgger (detect + on-policy correct).
        from ..p4 import select_arm as PS
        # On motion-planner tasks p4_select detects via the policy's diffusion loss
        # — the SAME signal diff_dagger uses. Make the detector IDENTICAL by sourcing
        # alpha/patience/batch_multiplier from the diff_dagger baseline config (not
        # the hydra default), so both arms compute the loss the same way (fairness).
        if E.task_spec(env_id).expert_kind == "motionplanner":
            dd = dict(k.get("baselines", {}).get("diff_dagger", {}) or {})
            cfg.policy.alpha = float(dd.get("alpha", cfg.policy.get("alpha", 0.99)))
            cfg.policy.patience = int(dd.get("patience", cfg.policy.get("patience", 1)))
            cfg.policy.patience_window = cfg.policy.patience
            if int(dd.get("batch_multiplier", 0)) > 0:
                cfg.policy.batch_multiplier = int(dd["batch_multiplier"])
        hp = {"num_initial_demos": k["initial_demos"],
              "n_cand": int(k.get("p4_select_n_cand", 6)),
              "max_episodes": int(k["max_episodes_per_arm"])}
        return PS.run_p4_select_arm(
            method=method, env=env, eval_env=eval_env, expert=expert, cfg=cfg,
            make_policy=make_policy, make_dataset=make_dataset, init_ckpt=init_ckpt,
            init_sr=init_sr, work_dir=str(results_dir), budget=budget,
            nd_retrain=k["nd_retrain"], target_sr=k["target_sr"],
            max_rounds=k["max_rounds"], heldout_seed_base=7777,
            heldout_num_ep=k["heldout_n"], round_epoch=k["round_epochs"],
            max_train_steps=k["max_train_steps"], hp=hp, seed=seed,
            suite_env_id=env_id, llm_client_on=llm_client_on)

    # diff_dagger + the 5 IIL baselines: ONE unified arm loop; only the query
    # rule differs, and ALL early-stop at target_sr (the apples-to-apples fix).
    from ..selection import iil_baselines as IIL
    kindname = IIL.KIND_OF[method]
    hp = dict(k["baselines"].get(method, {}) or {})
    hp["num_initial_demos"] = k["initial_demos"]
    hp["max_episodes"] = int(k["max_episodes_per_arm"])
    # diff_dagger's native query reads the policy's diffusion-loss threshold, so
    # put alpha/K on cfg.policy → every from-scratch retrain recalibrates at alpha.
    _orig = None
    if kindname == "diff":
        _orig = {kk: cfg.policy.get(kk)
                 for kk in ("alpha", "patience", "patience_window", "batch_multiplier")}
        cfg.policy.alpha = float(hp.get("alpha", 0.99))
        cfg.policy.patience = int(hp.get("patience", 1))
        cfg.policy.patience_window = int(hp.get("patience", 1))
        if int(hp.get("batch_multiplier", 0)) > 0:   # cheaper per-step loss estimate
            cfg.policy.batch_multiplier = int(hp["batch_multiplier"])
    try:
        return IIL.run_iil_arm(
            method=method, kind=kindname, env=env, eval_env=eval_env, expert=expert,
            cfg=cfg, make_policy=make_policy, make_dataset=make_dataset,
            init_ckpt=init_ckpt, init_sr=init_sr, work_dir=str(results_dir),
            budget=budget, nd_retrain=k["nd_retrain"], target_sr=k["target_sr"],
            max_rounds=k["max_rounds"], heldout_seed_base=7777,
            heldout_num_ep=k["heldout_n"], round_epoch=k["round_epochs"],
            max_train_steps=k["max_train_steps"], hp=hp, seed=seed)
    finally:
        if _orig is not None:
            for kk, v in _orig.items():
                cfg.policy[kk] = v


# ---- the driver --------------------------------------------------------
def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--env", required=True)
    ap.add_argument("--run_id", type=int, default=0)
    ap.add_argument("--methods", default=None,
                    help="comma list or 'all' (default: config methods)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--initial-demos", dest="initial_demos", type=int, default=None)
    ap.add_argument("--target-sr", dest="target_sr", type=float, default=None)
    ap.add_argument("--max-rounds", dest="max_rounds", type=int, default=None)
    ap.add_argument("--obs-mode", default=None)        # accepted, cfg-driven
    ap.add_argument("--control-mode", default=None)    # accepted, cfg-driven
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-llm", dest="no_llm", action="store_true")


def run_suite(args, method_spec: Dict[str, Any], config_path: Path,
              summary_filename: str = "run_summary.json") -> int:
    import torch
    from diffdagger.main_pipeline.comparison_harness import (
        _bootstrap_shared_init, _seed_everything,
    )

    scfg = load_suite_cfg(config_path)
    all_methods = list(scfg.get("methods", list(method_spec.keys())))
    if args.methods in (None, "all"):
        methods = all_methods
    else:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    methods = [m for m in methods if m in method_spec]

    k = resolve_knobs(scfg, args)
    if args.smoke:
        k = apply_smoke(k)
    E.register_envs()
    cfg = build_cfg(args.env, k)
    ws = W.workspace_for(args.env, args.run_id)

    # Parallel-split safety: when only a SUBSET of methods runs (e.g. p4 in its
    # own job, baselines in another), use a per-job shared-bootstrap dir + summary
    # filename so concurrent jobs don't race/clobber. Each subset job re-runs the
    # (deterministic, same-seed) bootstrap independently — same recipe, so still
    # apples-to-apples; for a byte-identical shared init, run all 7 in one job.
    if set(methods) == set(method_spec):
        tag, shared_dir, summary_filename = "all", ws.shared, summary_filename
    elif methods == ["p4_select"]:
        tag = "p4_select"
        shared_dir = ws.root / "shared_p4_select"
        summary_filename = "run_summary_p4_select.json"
    elif methods == ["p4_subtask"]:
        # Own shared dir + summary so a p4_subtask job never clobbers the
        # baselines'/diff_dagger's shared_baselines or run_summary_baselines.json.
        tag = "p4_subtask"
        shared_dir = ws.root / "shared_p4_subtask"
        summary_filename = "run_summary_p4_subtask.json"
    elif "p4_top3" not in methods:
        tag = "baselines"
        shared_dir = ws.root / "shared_baselines"
        summary_filename = "run_summary_baselines.json"
    elif methods == ["p4_top3"]:
        tag = "p4"
        shared_dir = ws.root / "shared_p4"
        summary_filename = "run_summary_p4.json"
    else:
        tag = "_".join(methods)[:40]
        shared_dir = ws.root / f"shared_{tag}"
        summary_filename = f"run_summary_{tag}.json"
    shared_dir = Path(shared_dir)
    shared_dir.mkdir(parents=True, exist_ok=True)

    env = MS.make_policy_env(cfg)
    eval_env = MS.make_eval_env(cfg, num_envs=k["eval_num_envs"])
    # prescribe-and-load methods need a reposition env; p4_subtask needs the
    # superset PushT-Subtask-v0 (honours _agent_init_qpos for the sub-task entry).
    need_subtask = any(m == "p4_subtask" for m in methods)
    need_repo = any(m in ("p4_top3", "p4_subtask") for m in methods)
    reposition_env = (MS.make_reposition_env(cfg, args.env, prefer_subtask=need_subtask)
                      if need_repo else None)
    expert = ExpertCallCounter(X.load_expert(cfg, args.env))  # secondary cost axis
    make_policy = PF.policy_factory(cfg)
    make_dataset = PF.dataset_factory(cfg)
    # Any LLM method enables the client: p4_top3 / p4_subtask (prescribe) or p4_select.
    llm_client_on = (not args.no_llm) and any(
        m in ("p4_top3", "p4_select", "p4_subtask") for m in methods)

    # ── shared bootstrap: ONE initial diffusion policy for all arms ──
    _seed_everything(k["seed"])
    # EXACT-bootstrap reuse (opt-in, for equal-to-equal vs an existing arm): if
    # P4_REUSE_INIT_CKPT points at an existing init_ckpt.pth (e.g. the diff_dagger
    # arm's run_<seed>/shared_baselines/init_ckpt.pth), load it verbatim and read its
    # frozen init_sr from the sibling init_meta.json — do NOT rebuild (bootstrap
    # training + heldout eval are GPU-nondeterministic, so a rebuild would not match).
    import os
    _reuse = os.environ.get("P4_REUSE_INIT_CKPT", "").strip()
    if _reuse and Path(_reuse).is_file():
        _meta = Path(_reuse).with_name("init_meta.json")
        _isr = json.loads(_meta.read_text()).get("init_sr") if _meta.is_file() else None
        boot = {"init_ckpt": _reuse, "init_sr": _isr}
        print(f"[bootstrap] REUSING shared init {_reuse} (init_sr={_isr}) — "
              f"skipping rebuild", flush=True)
    else:
        boot = _bootstrap_shared_init(
            cfg=cfg, make_policy=make_policy, make_dataset=make_dataset,
            env=env, eval_env=eval_env, expert=expert, shared_dir=str(shared_dir),
            initial_seeds=list(range(k["initial_demos"])),
            heldout_seed_base=7777, heldout_num_ep=k["heldout_n"],
            checkpoint_fractions=list(cfg.get("checkpoint_fractions", [0.8, 0.9, 1.0])),
            epoch=k["initial_epochs"], max_train_steps=k["max_train_steps"])
    init_ckpt, init_sr = boot["init_ckpt"], boot["init_sr"]

    results: Dict[str, Any] = {}
    for m in methods:
        t0 = time.time()
        print(f"\n{'#'*70}\n# METHOD: {m}\n{'#'*70}", flush=True)
        try:
            summ = run_method(method=m, spec=method_spec[m], cfg=cfg, env=env,
                              eval_env=eval_env, reposition_env=reposition_env,
                              expert=expert, make_policy=make_policy,
                              make_dataset=make_dataset, init_ckpt=init_ckpt,
                              init_sr=init_sr, ws=ws, k=k,
                              llm_client_on=llm_client_on)
            results[m] = {"final_performance": (summ or {}).get("final_performance"),
                          "stopped_reason": (summ or {}).get("stopped_reason"),
                          "wall_sec": round(time.time() - t0, 1)}
        except Exception as exc:  # keep the array task alive across methods
            import traceback
            traceback.print_exc()
            results[m] = {"error": f"{type(exc).__name__}: {exc}",
                          "wall_sec": round(time.time() - t0, 1)}
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {"env": args.env, "fork_env": E.fork_env_id(args.env),
               "run_id": args.run_id, "seed": k["seed"], "budget": k["budget"],
               "smoke": bool(args.smoke), "methods": methods, "results": results}
    out = ws.root / summary_filename
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[run] wrote {out}")
    return 0
