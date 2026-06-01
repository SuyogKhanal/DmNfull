"""Per-run driver for the pool_x_selector suite.

Invoked by ``run_all.sh`` (once per run, in parallel across all N runs).
For a single ``--run_id`` it:

  1. Triggers the upstream-shared bootstrap (idempotent across runs).
  2. Materializes the per-run fixed correction pool (when any *_fixed
     method is selected); *_rotate methods sample per-round pools instead.
  3. Runs the requested methods (see METHOD_SPEC) sequentially in this
     same process: baseline highest-loss/random and P4 top3/all, each on a
     fixed or rotated pool. Dispatch is table-driven (METHOD_SPEC).
  4. Writes results/run_{id}/run_summary.json.

Each method writes into its own ``results/run_{id}/{method}/`` subtree so
they don't interfere with each other.

The ``--smoke`` flag shrinks everything to a 1-run, 2-budget, 3-round
smoke test across all 8 methods, used to verify orchestration end-to-end
before launching production.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import yaml

from ..layouts import contamination as cross_run_contam
from ..layouts.layout_setup import (
    ensure_correction_layouts_for_run,
    ensure_shared_layouts,
)
from . import bootstrap as bootstrap_mod
from .workspace import RunWorkspace, workspace_for


SUITE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = SUITE_ROOT / "config.yaml"


# The 2-axis method grid. Each value is (kind, selector, pool):
#   kind     "baseline" -> _run_baseline; "p4" -> _run_p4
#   selector baseline: "highest_loss"|"random"; p4: top_k (3=top-3, None=all)
#   pool     "fixed" (one pool/run) | "rotate" (fresh pool/round)
# Adding/removing a method is one line here (+ config.yaml methods +
# workspace.METHOD_DIR_NAMES, which derives from these keys).
METHOD_SPEC = {
    "baseline_highloss_fixed":  ("baseline", "highest_loss", "fixed"),
    "baseline_random_fixed":    ("baseline", "random",       "fixed"),
    "baseline_highloss_rotate": ("baseline", "highest_loss", "rotate"),
    "baseline_random_rotate":   ("baseline", "random",       "rotate"),
    "p4_top3_fixed":            ("p4", 3,    "fixed"),
    "p4_top3_rotate":           ("p4", 3,    "rotate"),
    "p4_all_fixed":             ("p4", None, "fixed"),
    "p4_all_rotate":            ("p4", None, "rotate"),
}


def _info(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [run_one] {msg}", flush=True)


def _section(title: str, char: str = "=") -> None:
    print(f"\n{char * 80}\n{title}\n{char * 80}", flush=True)


def _load_config(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _smoke_overrides(cfg: Dict) -> Dict:
    """Apply smoke-test overrides. Keep the schema; just shrink the work."""
    cfg = dict(cfg)
    cfg["n_runs"] = 1
    cfg["budget"] = 2
    cfg["max_rounds"] = 3
    cfg.setdefault("trainer", {})
    cfg["trainer"]["finetune_epochs"] = 3
    cfg["methods"] = list(METHOD_SPEC.keys())
    return cfg


def _run_baseline(
    ws: RunWorkspace, cfg: Dict, train_yaml: Path, heldout_yaml: Path,
    method_name: str, selection: str, fixed_pool_yaml: Path,
) -> Dict:
    from ..selection import baseline_dagger
    trainer_cfg = cfg.get("trainer", {}) or {}
    baseline_epochs = int(
        cfg.get("baseline_finetune_epochs")
        or trainer_cfg.get("finetune_epochs", 20)
    )
    return baseline_dagger.run(
        run_id=ws.run_id,
        bo_root=ws.method_root(method_name),
        shared_demo_dir=ws.init_demos,
        shared_ckpt_dir=ws.init_checkpoints,
        train_yaml=train_yaml,
        heldout_yaml=heldout_yaml,
        run_shared_dir=ws.shared,
        correction_n=int(cfg.get("correction_n", 20)),
        budget=int(cfg.get("budget", 15)),
        target_sr=float(cfg.get("target_sr", 0.90)),
        max_rounds=int(cfg.get("max_rounds", 50)),
        max_steps=int(cfg.get("max_steps", 60)),
        seed=int(cfg.get("seed", 0)) + int(ws.run_id),
        finetune_epochs=baseline_epochs,
        lr=float(trainer_cfg.get("lr", 5e-5)),
        batch_size=int(trainer_cfg.get("batch_size", 64)),
        weight_decay=float(trainer_cfg.get("weight_decay", 1e-4)),
        replay_mix=float(trainer_cfg.get("replay_mix", 0.5)),
        replay_mix_floor=float(trainer_cfg.get("replay_mix_floor", 0.2)),
        fixed_pool_yaml=fixed_pool_yaml,
        selection=selection,
        method_label=method_name,
    )


def _run_p4(
    ws: RunWorkspace, cfg: Dict, train_yaml: Path, heldout_yaml: Path,
    method_name: str, top_k, fixed_pool_yaml,
) -> Dict:
    """Run a P4 variant via the unified pipeline. ``top_k`` selects how
    many failures the LLM sees (3 = top-3; None = all); ``fixed_pool_yaml``
    set = fixed pool, None = rotate. Both prescribe exactly 1 layout/round.
    """
    from ..p4 import pipeline_p4
    p4_cfg = dict(cfg)
    p4_epochs = cfg.get("p4_finetune_epochs")
    if p4_epochs is not None:
        tc = dict(cfg.get("trainer", {}) or {})
        tc["finetune_epochs"] = int(p4_epochs)
        p4_cfg["trainer"] = tc
    return pipeline_p4.run(
        run_id=ws.run_id,
        method_root=ws.method_root(method_name),
        shared_demo_dir=ws.init_demos,
        shared_ckpt_dir=ws.init_checkpoints,
        train_yaml=train_yaml,
        heldout_yaml=heldout_yaml,
        run_shared_dir=ws.shared,
        correction_n=int(cfg.get("correction_n", 20)),
        config=p4_cfg,
        top_k=top_k,
        fixed_pool_yaml=fixed_pool_yaml,
        method_name=method_name,
    )


def _apply_cli_overrides(cfg: Dict, args: argparse.Namespace) -> Dict:
    """Mutate ``cfg`` in-place with any CLI override that was actually
    supplied (i.e. not ``None``). Top-level scalars land at the top;
    trainer knobs land under ``cfg["trainer"]``.
    """
    trainer = dict(cfg.get("trainer", {}) or {})
    overrides_top = {
        "budget": args.budget,
        "target_sr": args.target_sr,
        "correction_n": args.correction_n,
        "heldout_n": args.heldout_n,
        "initial_demos": args.initial_demos,
        "initial_epochs": args.initial_epochs,
        "max_rounds": args.max_rounds,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "corridor_blocking": args.corridor_blocking,
        # Per-method finetune-epoch overrides (read by _run_baseline / _run_p4).
        "baseline_finetune_epochs": args.baseline_finetune_epochs,
        "p4_finetune_epochs": args.p4_finetune_epochs,
    }
    for k, v in overrides_top.items():
        if v is not None:
            cfg[k] = v

    overrides_trainer = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "replay_mix": args.replay_mix,
        "replay_mix_floor": args.replay_mix_floor,
    }
    if args.finetune_epochs is not None:
        # Shared default; per-method --baseline_finetune_epochs /
        # --p4_finetune_epochs above take precedence when set.
        overrides_trainer["finetune_epochs"] = args.finetune_epochs
    for k, v in overrides_trainer.items():
        if v is not None:
            trainer[k] = v
    if trainer:
        cfg["trainer"] = trainer
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", type=int, required=True)
    ap.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument(
        "--methods", type=str, default=None,
        help="Comma-sep subset of {baseline,p4_sequential,p4_batch}. "
             "Default: config.yaml methods.",
    )
    ap.add_argument(
        "--smoke", action="store_true",
        help="Apply smoke-test overrides: budget=2, max_rounds=3, "
             "epochs=3, methods=[baseline]. Useful for end-to-end verification.",
    )
    # Top-level config overrides (mirror swatch.sh's knob block).
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--target_sr", type=float, default=None)
    ap.add_argument("--correction_n", type=int, default=None)
    ap.add_argument("--heldout_n", type=int, default=None)
    ap.add_argument("--initial_demos", type=int, default=None)
    ap.add_argument("--initial_epochs", type=int, default=None)
    ap.add_argument("--max_rounds", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--corridor_blocking", type=lambda s: str(s).lower() in ("1", "true", "yes", "on"),
        default=None, help="Force corridor blocking on/off (overrides config).",
    )
    # Per-method epoch overrides (mirror swatch.sh ROUND_EPOCHS / BASELINE_ROUND_EPOCHS).
    ap.add_argument("--baseline_finetune_epochs", type=int, default=None,
                    help="Fine-tune epochs for the baseline DAgger round.")
    ap.add_argument("--p4_finetune_epochs", type=int, default=None,
                    help="Fine-tune epochs for each P4 round.")
    # Shared trainer overrides.
    ap.add_argument("--finetune_epochs", type=int, default=None,
                    help="Shared default finetune_epochs (overridden per-method by the two flags above).")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--weight_decay", type=float, default=None)
    ap.add_argument("--replay_mix", type=float, default=None)
    ap.add_argument("--replay_mix_floor", type=float, default=None)
    args = ap.parse_args()

    cfg = _load_config(Path(args.config))
    cfg = _apply_cli_overrides(cfg, args)
    if args.smoke:
        cfg = _smoke_overrides(cfg)

    methods: List[str] = (
        [m.strip() for m in args.methods.split(",") if m.strip()]
        if args.methods
        else list(cfg.get("methods", ["baseline"]) or [])
    )
    cfg["methods"] = methods

    _section(
        f"RUN_ID={args.run_id}  methods={methods}  "
        f"budget={cfg.get('budget')} target_sr={cfg.get('target_sr')}"
    )

    # ---- Phase 1: bootstrap shared assets (idempotent) -------------------
    _section("PHASE 1: BOOTSTRAP", char="-")
    bootstrap_mod.ensure_upstream_bootstrap(
        seed=int(cfg.get("seed", 0)),
        initial_epochs=int(cfg.get("initial_epochs", 500)),
        initial_demos=int(cfg.get("initial_demos", 20)),
        heldout_n=int(cfg.get("heldout_n", 200)),
    )

    # ---- Phase 2: per-run layouts ----------------------------------------
    _section("PHASE 2: LAYOUTS", char="-")
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
    # The *_fixed methods share ONE fixed correction pool per run
    # (apples-to-apples); sample it once here. Seed = CORRECTION_SEED_BASE +
    # run_id (distinct per run, identical across methods). The *_rotate
    # methods sample a fresh pool per round instead, so skip this entirely
    # when no fixed method is selected.
    needs_fixed = any(spec[2] == "fixed" for m in methods
                      if (spec := METHOD_SPEC.get(m)) is not None)
    fixed_pool_yaml = None
    if needs_fixed:
        fixed_pool_yaml = ensure_correction_layouts_for_run(
            run_id=ws.run_id,
            correction_n=int(cfg.get("correction_n", 20)),
            train_yaml=shared["train"],
            heldout_yaml=shared["heldout"],
            out_dir=ws.shared / "fixed_pool",
        )
        _info(f"fixed correction pool ({cfg.get('correction_n', 20)} layouts) -> {fixed_pool_yaml}")

    # ---- Phase 3: run methods sequentially -------------------------------
    summary: Dict = {
        "run_id": ws.run_id,
        "budget": int(cfg.get("budget", 15)),
        "target_sr": float(cfg.get("target_sr", 0.90)),
        "correction_yaml_dir": str(ws.shared),
        "heldout_yaml": str(shared["heldout"]),
        "training_yaml": str(shared["train"]),
        "correction_n": int(cfg.get("correction_n", 20)),
        "methods": methods,
        "results": {},
    }

    for m in methods:
        _section(f"PHASE 3: {m}", char="-")
        try:
            spec = METHOD_SPEC.get(m)
            if spec is None:
                _info(f"unknown method '{m}', skipping")
                continue
            kind, sel_or_k, pool = spec
            pool_yaml = fixed_pool_yaml if pool == "fixed" else None
            if pool == "fixed" and pool_yaml is None:
                raise SystemExit(f"{m} needs the fixed pool but it wasn't sampled")
            if kind == "baseline":
                result = _run_baseline(ws, cfg, shared["train"], shared["heldout"],
                                       method_name=m, selection=sel_or_k,
                                       fixed_pool_yaml=pool_yaml)
            else:
                result = _run_p4(ws, cfg, shared["train"], shared["heldout"],
                                 method_name=m, top_k=sel_or_k,
                                 fixed_pool_yaml=pool_yaml)
            summary["results"][m] = {
                "stopped_reason": result.get("stopped_reason"),
                "n_history": len(result.get("history", []) or []),
            }
        except Exception as exc:  # noqa: BLE001
            _info(f"method '{m}' FAILED: {exc!r}")
            traceback.print_exc()
            summary["results"][m] = {
                "stopped_reason": "error",
                "error": repr(exc),
            }

    with open(ws.root / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    _section(f"RUN_ID={ws.run_id} DONE")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
