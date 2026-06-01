"""Per-run driver for the 5 IIL baselines (rotated-only), with the 5 methods
running IN PARALLEL inside one SLURM job.

Sibling of ``run_one.py`` (the 8-method pool×selector grid). Runs ONLY the
interactive-IL baselines used in the P4-LLM comparison — SafeDAgger*,
DropoutDAgger, EnsembleDAgger, ThriftyDAgger, Stagger — each on the rotated
correction pool. P4 is NOT re-run; the baselines reuse P4's per-round pools.

Two modes:

* DRIVER (default) — for a single ``--run_id`` it (1) bootstraps shared assets
  ONCE, (2) mirrors per-run init demos/checkpoint, (3) builds a small eval
  held-out subset (``--heldout_n``) while keeping the FULL held-out for pool
  blocking, (4) PRE-GENERATES round_001..round_{max_rounds} correction pools
  once (so the parallel methods only read them — race-free, and all methods
  share byte-identical pools), then (5) launches one worker subprocess PER
  METHOD concurrently (CPU cores divided across them via OMP/MKL threads),
  waits for all, and merges their results into
  ``results/run_{id}/run_summary_baselines.json`` (P4's run_summary.json is
  never touched).

* WORKER (``--single_method M``) — runs exactly ONE baseline (the driver did
  the bootstrap / pool pre-gen / heldout subset) and writes its result to
  ``results/run_{id}/.baseline_summaries/{M}.json``.

``--smoke`` shrinks to budget=2 / max_rounds=2 / 1 epoch / heldout_n=4 and a
2-member ensemble for a fast wiring + contamination check. (max_rounds=2 is the
minimum that lets the contamination check verify intra-run pool disjointness —
with a single round there is nothing to compare.)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from ..layouts.layout_setup import (
    _layouts_in_yaml,
    ensure_shared_layouts,
    pregenerate_round_pools,
)
from . import bootstrap as bootstrap_mod
from .workspace import RunWorkspace, logs_dir, workspace_for

SUITE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_CONFIG_PATH = SUITE_ROOT / "config_baselines.yaml"

# method name -> kind (the decision rule iil_baselines dispatches on).
METHOD_SPEC = {
    "safe_dagger": "safe",
    "dropout_dagger": "dropout",
    "ensemble_dagger": "ensemble",
    "thrifty_dagger": "thrifty",
    "stagger": "stagger",
}


def _info(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [run_baselines] {msg}", flush=True)


def _section(title: str, char: str = "=") -> None:
    print(f"\n{char * 80}\n{title}\n{char * 80}", flush=True)


def _load_config(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _smoke_overrides(cfg: Dict) -> Dict:
    cfg = dict(cfg)
    cfg["n_runs"] = 1
    cfg["budget"] = 2
    cfg["max_rounds"] = 2          # >=2 so intra-run pool disjointness is testable
    cfg["heldout_n"] = 4           # minimum eval set
    cfg["correction_n"] = 8        # small pool -> fast rollouts
    cfg["max_consecutive_empty"] = 2
    cfg.setdefault("trainer", {})
    cfg["trainer"] = dict(cfg["trainer"])
    cfg["trainer"]["finetune_epochs"] = 1
    cfg["methods"] = list(METHOD_SPEC.keys())
    b = dict(cfg.get("baselines", {}) or {})
    b["ensemble"] = {**(b.get("ensemble", {}) or {}), "M": 2}
    b["dropout"] = {**(b.get("dropout", {}) or {}), "N": 2}
    b["thrifty"] = {**(b.get("thrifty", {}) or {}), "M": 2, "q_epochs": 1}
    cfg["baselines"] = b
    return cfg


def _apply_cli_overrides(cfg: Dict, args: argparse.Namespace) -> Dict:
    trainer = dict(cfg.get("trainer", {}) or {})
    top = {
        "budget": args.budget, "target_sr": args.target_sr,
        "correction_n": args.correction_n, "heldout_n": args.heldout_n,
        "initial_demos": args.initial_demos, "initial_epochs": args.initial_epochs,
        "max_rounds": args.max_rounds, "max_steps": args.max_steps,
        "seed": args.seed, "max_consecutive_empty": args.max_consecutive_empty,
        "baseline_finetune_epochs": args.finetune_epochs,
    }
    for k, v in top.items():
        if v is not None:
            cfg[k] = v
    tr = {"lr": args.lr, "batch_size": args.batch_size,
          "weight_decay": args.weight_decay, "replay_mix": args.replay_mix,
          "replay_mix_floor": args.replay_mix_floor}
    if args.finetune_epochs is not None:
        tr["finetune_epochs"] = args.finetune_epochs
    for k, v in tr.items():
        if v is not None:
            trainer[k] = v
    if trainer:
        cfg["trainer"] = trainer
    return cfg


def _resolve_baseline_hp(cfg: Dict, args: argparse.Namespace) -> Dict:
    b = cfg.get("baselines", {}) or {}
    safe = b.get("safe", {}) or {}
    dropout = b.get("dropout", {}) or {}
    ensemble = b.get("ensemble", {}) or {}
    thrifty = b.get("thrifty", {}) or {}

    def pick(cli, *cfg_vals, default):
        if cli is not None:
            return cli
        for v in cfg_vals:
            if v is not None:
                return v
        return default

    M = int(pick(args.ensemble_M, ensemble.get("M"), thrifty.get("M"), default=5))
    return dict(
        tau_safe=float(pick(args.tau_safe, safe.get("tau"), default=0.10)),
        mc_N=int(pick(args.mc_N, dropout.get("N"), default=16)),
        dropout_d=float(pick(args.dropout_d, dropout.get("dropout_d"), default=0.2)),
        p_thresh=float(pick(args.p_thresh, dropout.get("p"), default=0.5)),
        tau_drop=float(pick(args.tau_drop, dropout.get("tau"), default=0.5)),
        M=M,
        tau_ens=float(pick(args.tau_ens, ensemble.get("tau"), default=0.10)),
        sigma_ens=float(pick(args.sigma_ens, ensemble.get("sigma"), default=0.10)),
        ensemble_eval_mode=str(pick(args.ensemble_eval_mode, ensemble.get("eval_mode"), default="mean")),
        alpha_h=float(pick(args.alpha_h, thrifty.get("alpha_h"), default=0.10)),
        gamma=float(pick(args.gamma, thrifty.get("gamma"), default=0.95)),
        thrifty_risk_agg=str(pick(args.thrifty_risk_agg, thrifty.get("risk_agg"), default="mean")),
        q_epochs=int(pick(args.q_epochs, thrifty.get("q_epochs"), default=50)),
        q_lr=float(pick(args.q_lr, thrifty.get("q_lr"), default=1e-3)),
    )


def _run_one_baseline(ws: RunWorkspace, cfg: Dict, hp: Dict,
                      train_yaml: Path, heldout_eval_yaml: Path,
                      heldout_block_yaml: Path,
                      method_name: str, kind: str) -> Dict:
    from ..selection import iil_baselines
    trainer_cfg = cfg.get("trainer", {}) or {}
    epochs = int(cfg.get("baseline_finetune_epochs")
                 or trainer_cfg.get("finetune_epochs", 90))
    return iil_baselines.run(
        run_id=ws.run_id,
        method_name=method_name,
        method_kind=kind,
        bo_root=ws.method_root(method_name),
        shared_demo_dir=ws.init_demos,
        shared_ckpt_dir=ws.init_checkpoints,
        train_yaml=train_yaml,
        heldout_yaml=heldout_eval_yaml,        # eval (possibly truncated)
        heldout_block_yaml=heldout_block_yaml, # pool blocking (full)
        run_shared_dir=ws.shared,
        correction_n=int(cfg.get("correction_n", 20)),
        budget=int(cfg.get("budget", 15)),
        target_sr=float(cfg.get("target_sr", 0.90)),
        max_rounds=int(cfg.get("max_rounds", 50)),
        max_steps=int(cfg.get("max_steps", 60)),
        seed=int(cfg.get("seed", 0)) + int(ws.run_id),
        finetune_epochs=epochs,
        lr=float(trainer_cfg.get("lr", 5e-5)),
        batch_size=int(trainer_cfg.get("batch_size", 64)),
        weight_decay=float(trainer_cfg.get("weight_decay", 1e-4)),
        replay_mix=float(trainer_cfg.get("replay_mix", 0.5)),
        replay_mix_floor=float(trainer_cfg.get("replay_mix_floor", 0.2)),
        max_consecutive_empty=int(cfg.get("max_consecutive_empty", 8)),
        **hp,
    )


# --------------------------------------------------------------------------- #
# Held-out eval subset
# --------------------------------------------------------------------------- #
def _build_eval_heldout(heldout_full: Path, heldout_n: Optional[int],
                        out_path: Path) -> Path:
    """Return a path to the held-out yaml used for EVAL. If ``heldout_n`` is
    set and smaller than the full set, write a truncated copy (first
    ``heldout_n`` layouts) and return it; otherwise return the full path.
    Pool sampling always blocks against the FULL set, so truncating eval does
    not weaken the contamination guard.
    """
    if heldout_n is None:
        return heldout_full
    full = _layouts_in_yaml(heldout_full)
    if heldout_n >= len(full) or heldout_n <= 0:
        return heldout_full
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump({"img_size": 80, "grid_size": 5, "cell_px": 16,
                        "heldout_test_layouts": full[:int(heldout_n)]},
                       f, sort_keys=False)
    return out_path


# --------------------------------------------------------------------------- #
# Subprocess argument forwarding (driver -> per-method workers)
# --------------------------------------------------------------------------- #
def _forward_flags(args: argparse.Namespace) -> List[str]:
    """Rebuild the CLI flags that must be forwarded to a worker subprocess so
    it re-derives the identical cfg + hp.
    """
    out: List[str] = ["--config", str(args.config)]
    pairs = [
        ("--budget", args.budget), ("--target_sr", args.target_sr),
        ("--correction_n", args.correction_n), ("--heldout_n", args.heldout_n),
        ("--initial_demos", args.initial_demos), ("--initial_epochs", args.initial_epochs),
        ("--max_rounds", args.max_rounds), ("--max_steps", args.max_steps),
        ("--seed", args.seed), ("--max_consecutive_empty", args.max_consecutive_empty),
        ("--finetune_epochs", args.finetune_epochs), ("--lr", args.lr),
        ("--batch_size", args.batch_size), ("--weight_decay", args.weight_decay),
        ("--replay_mix", args.replay_mix), ("--replay_mix_floor", args.replay_mix_floor),
        ("--tau_safe", args.tau_safe), ("--mc_N", args.mc_N),
        ("--dropout_d", args.dropout_d), ("--p_thresh", args.p_thresh),
        ("--tau_drop", args.tau_drop), ("--ensemble_M", args.ensemble_M),
        ("--tau_ens", args.tau_ens), ("--sigma_ens", args.sigma_ens),
        ("--ensemble_eval_mode", args.ensemble_eval_mode), ("--alpha_h", args.alpha_h),
        ("--gamma", args.gamma), ("--thrifty_risk_agg", args.thrifty_risk_agg),
        ("--q_epochs", args.q_epochs), ("--q_lr", args.q_lr),
    ]
    for flag, val in pairs:
        if val is not None:
            out += [flag, str(val)]
    if args.smoke:
        out += ["--smoke"]
    return out


def _summary_dir(ws: RunWorkspace) -> Path:
    d = ws.root / ".baseline_summaries"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# WORKER mode: run exactly one method
# --------------------------------------------------------------------------- #
def _run_single(args: argparse.Namespace, cfg: Dict, hp: Dict) -> None:
    method = args.single_method
    kind = METHOD_SPEC.get(method)
    if kind is None:
        raise SystemExit(f"unknown baseline '{method}'")
    if not args.skip_bootstrap:
        bootstrap_mod.ensure_upstream_bootstrap(
            seed=int(cfg.get("seed", 0)),
            initial_epochs=int(cfg.get("initial_epochs", 500)),
            initial_demos=int(cfg.get("initial_demos", 20)),
            heldout_n=int(cfg.get("heldout_n", 200)),
        )
    shared = ensure_shared_layouts(
        initial_demos=int(cfg.get("initial_demos", 20)),
        heldout_n=int(cfg.get("heldout_n", 200)),
    )
    ws = workspace_for(args.run_id)
    heldout_eval = (Path(args.heldout_eval_path) if args.heldout_eval_path
                    else shared["heldout"])
    _section(f"WORKER run_id={args.run_id} method={method} kind={kind}", char="-")
    rec: Dict = {}
    try:
        result = _run_one_baseline(ws, cfg, hp, shared["train"], heldout_eval,
                                   shared["heldout"], method_name=method, kind=kind)
        rec = {"stopped_reason": result.get("stopped_reason"),
               "n_history": len(result.get("history", []) or [])}
    except Exception as exc:  # noqa: BLE001
        _info(f"worker '{method}' FAILED: {exc!r}")
        traceback.print_exc()
        rec = {"stopped_reason": "error", "error": repr(exc)}
    with open(_summary_dir(ws) / f"{method}.json", "w") as f:
        json.dump(rec, f, indent=2, default=str)
    _info(f"worker '{method}' DONE: {rec}")


# --------------------------------------------------------------------------- #
# DRIVER mode: bootstrap + pre-gen + spawn parallel workers
# --------------------------------------------------------------------------- #
def _run_driver(args: argparse.Namespace, cfg: Dict, hp: Dict,
                methods: List[str]) -> None:
    _section(
        f"RUN_ID={args.run_id}  baselines={methods}  "
        f"budget={cfg.get('budget')} target_sr={cfg.get('target_sr')} "
        f"max_rounds={cfg.get('max_rounds')} parallel={not args.no_parallel}  hp={hp}"
    )

    _section("PHASE 1: BOOTSTRAP", char="-")
    bootstrap_mod.ensure_upstream_bootstrap(
        seed=int(cfg.get("seed", 0)),
        initial_epochs=int(cfg.get("initial_epochs", 500)),
        initial_demos=int(cfg.get("initial_demos", 20)),
        heldout_n=int(cfg.get("heldout_n", 200)),
    )

    _section("PHASE 2: LAYOUTS + POOL PRE-GENERATION", char="-")
    shared = ensure_shared_layouts(
        initial_demos=int(cfg.get("initial_demos", 20)),
        heldout_n=int(cfg.get("heldout_n", 200)),
    )
    ws = workspace_for(args.run_id)
    bootstrap_mod.mirror_upstream_into_run(
        upstream_demos_dir=shared["train"].parent / "init_demos",
        upstream_ckpt_dir=shared["train"].parent / "init_checkpoints",
        per_run_demos_dir=ws.init_demos,
        per_run_ckpt_dir=ws.init_checkpoints,
    )
    # Eval held-out subset (truncate for speed; pool blocking stays on full).
    heldout_n = cfg.get("heldout_n")
    heldout_eval = _build_eval_heldout(
        shared["heldout"], int(heldout_n) if heldout_n is not None else None,
        ws.shared / f"heldout_eval_{heldout_n}.yaml")
    _info(f"eval held-out: {heldout_eval}  (full block set: {shared['heldout']})")
    # Pre-generate round pools ONCE (reuse P4's, seed-bump-fill the rest) so
    # the parallel methods only READ them — race-free + byte-identical pools.
    # Bound to the rounds a run can realistically reach (budget demos + a
    # trailing empty streak + margin); flock in the helper guards any round a
    # method reaches beyond this (essentially never, given the demo budget).
    max_rounds = int(cfg.get("max_rounds", 50))
    n_pregen = min(max_rounds,
                   int(cfg.get("budget", 15)) + int(cfg.get("max_consecutive_empty", 8)) + 5)
    pregenerate_round_pools(
        run_id=ws.run_id, n_rounds=n_pregen,
        correction_n=int(cfg.get("correction_n", 20)),
        train_yaml=shared["train"], heldout_yaml=shared["heldout"],
        run_shared_dir=ws.shared)
    _info(f"pre-generated up to round_{n_pregen:03d} correction pools "
          f"(max_rounds={max_rounds}; flock guards any beyond)")

    _section("PHASE 3: RUN BASELINES (parallel)" if not args.no_parallel
             else "PHASE 3: RUN BASELINES (sequential)", char="-")
    sd = _summary_dir(ws)
    for m in methods:  # clear stale per-method summaries from a prior run
        (sd / f"{m}.json").unlink(missing_ok=True)

    parallel = (not args.no_parallel) and len(methods) > 1
    if parallel:
        ncpus = int(os.environ.get("SLURM_CPUS_PER_TASK")
                    or os.environ.get("SLURM_CPUS_ON_NODE")
                    or (os.cpu_count() or 4))
        per = max(1, ncpus // len(methods))
        _info(f"cores: {ncpus} total -> {per} threads/method across {len(methods)} methods")
        ldir = logs_dir()
        procs = []
        for m in methods:
            env = dict(os.environ)
            env.update(OMP_NUM_THREADS=str(per), MKL_NUM_THREADS=str(per),
                       OPENBLAS_NUM_THREADS=str(per), NUMEXPR_NUM_THREADS=str(per))
            cmd = [sys.executable, "-u", "-m",
                   "Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4."
                   "pool_x_selector.orchestrator.run_baselines",
                   "--run_id", str(args.run_id),
                   "--single_method", m, "--skip_bootstrap",
                   "--heldout_eval_path", str(heldout_eval),
                   *_forward_flags(args)]
            logf = open(ldir / f"baseline_run_{int(args.run_id):02d}_{m}.log", "w")
            _info(f"spawn worker [{m}] -> {logf.name}")
            procs.append((m, subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env,
                                              stdout=logf, stderr=subprocess.STDOUT), logf))
        for m, p, logf in procs:
            rc = p.wait()
            logf.close()
            _info(f"worker [{m}] exited rc={rc}")
    else:
        for m in methods:
            _section(f"method: {m}", char=".")
            kind = METHOD_SPEC[m]
            rec: Dict = {}
            try:
                result = _run_one_baseline(ws, cfg, hp, shared["train"], heldout_eval,
                                           shared["heldout"], method_name=m, kind=kind)
                rec = {"stopped_reason": result.get("stopped_reason"),
                       "n_history": len(result.get("history", []) or [])}
            except Exception as exc:  # noqa: BLE001
                _info(f"method '{m}' FAILED: {exc!r}")
                traceback.print_exc()
                rec = {"stopped_reason": "error", "error": repr(exc)}
            with open(sd / f"{m}.json", "w") as f:
                json.dump(rec, f, indent=2, default=str)

    # ---- collect per-method summaries -> run_summary_baselines.json --------
    results: Dict[str, Dict] = {}
    for m in methods:
        p = sd / f"{m}.json"
        if p.exists():
            try:
                results[m] = json.load(open(p))
            except (OSError, json.JSONDecodeError):
                results[m] = {"stopped_reason": "missing_summary"}
        else:
            results[m] = {"stopped_reason": "no_summary_written"}

    summary = {
        "run_id": ws.run_id,
        "budget": int(cfg.get("budget", 15)),
        "target_sr": float(cfg.get("target_sr", 0.90)),
        "max_rounds": int(cfg.get("max_rounds", 50)),
        "correction_yaml_dir": str(ws.shared),
        "heldout_eval_yaml": str(heldout_eval),
        "heldout_block_yaml": str(shared["heldout"]),
        "training_yaml": str(shared["train"]),
        "correction_n": int(cfg.get("correction_n", 20)),
        "methods": methods,
        "baseline_hp": hp,
        "parallel": bool(parallel),
        "results": results,
    }
    out_path = ws.root / "run_summary_baselines.json"
    existing: Dict = {}
    if out_path.exists():
        try:
            existing = json.load(open(out_path))
        except (OSError, json.JSONDecodeError):
            existing = {}
    merged = dict((existing.get("results") or {}))
    merged.update(results)
    summary["results"] = merged
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    _section(f"RUN_ID={ws.run_id} (baselines) DONE")
    print(json.dumps(summary, indent=2, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", type=int, required=True)
    ap.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--methods", type=str, default=None,
                    help="Comma-sep subset of the 5 baselines. Default: config methods.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no_parallel", action="store_true",
                    help="Run the methods sequentially in-process (default: parallel subprocesses).")
    # WORKER-mode flags (set by the driver when spawning per-method subprocesses).
    ap.add_argument("--single_method", type=str, default=None)
    ap.add_argument("--skip_bootstrap", action="store_true")
    ap.add_argument("--heldout_eval_path", type=str, default=None)
    # Standard knobs (default None -> fall through to config).
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--target_sr", type=float, default=None)
    ap.add_argument("--correction_n", type=int, default=None)
    ap.add_argument("--heldout_n", type=int, default=None)
    ap.add_argument("--initial_demos", type=int, default=None)
    ap.add_argument("--initial_epochs", type=int, default=None)
    ap.add_argument("--max_rounds", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max_consecutive_empty", type=int, default=None)
    ap.add_argument("--finetune_epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--weight_decay", type=float, default=None)
    ap.add_argument("--replay_mix", type=float, default=None)
    ap.add_argument("--replay_mix_floor", type=float, default=None)
    # Baseline-specific knobs.
    ap.add_argument("--tau_safe", type=float, default=None)
    ap.add_argument("--mc_N", type=int, default=None)
    ap.add_argument("--dropout_d", type=float, default=None)
    ap.add_argument("--p_thresh", type=float, default=None)
    ap.add_argument("--tau_drop", type=float, default=None)
    ap.add_argument("--ensemble_M", type=int, default=None)
    ap.add_argument("--tau_ens", type=float, default=None)
    ap.add_argument("--sigma_ens", type=float, default=None)
    ap.add_argument("--ensemble_eval_mode", type=str, default=None,
                    choices=[None, "mean", "member0"])
    ap.add_argument("--alpha_h", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--thrifty_risk_agg", type=str, default=None,
                    choices=[None, "mean", "max"])
    ap.add_argument("--q_epochs", type=int, default=None)
    ap.add_argument("--q_lr", type=float, default=None)
    args = ap.parse_args()

    cfg = _load_config(Path(args.config))
    cfg = _apply_cli_overrides(cfg, args)
    if args.smoke:
        cfg = _smoke_overrides(cfg)
    hp = _resolve_baseline_hp(cfg, args)

    if args.single_method:
        _run_single(args, cfg, hp)
        return

    methods: List[str] = (
        [m.strip() for m in args.methods.split(",") if m.strip()]
        if args.methods
        else list(cfg.get("methods", list(METHOD_SPEC.keys())) or [])
    )
    _run_driver(args, cfg, hp, methods)


if __name__ == "__main__":
    main()
