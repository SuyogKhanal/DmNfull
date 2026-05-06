"""End-to-end orchestrator for the equivariant policy full cycle.

What this script does (one invocation = full cycle)
====================================================
1. **Generate** heldout_test_layouts.yaml (50 random 5x5 layouts) if it
   doesn't already exist; persist the spec + per-layout PNGs under
   results/equivariant_pathway/cycle_<ts>/heldout/.
2. **Persist** training_layouts.yaml + test_layouts.yaml into the cycle
   dir for archival (JSON + PNGs).
3. **Collect** the initial 10 BFS-expert demos using training_layouts.yaml
   (5 layouts x 2 reps = 10) -> Equivariant_pathway/demos/.
4. **Train** the equivariant policy on those 10 demos -> save
   `Equivariant_pathway/checkpoints/best_eq_policy.pth`. This is the
   shared starting checkpoint every learner copies.
5. For each method in {baseline_dagger, p4, p5, p6} run an isolated loop:
     - copy the initial 10 demos into <method_dir>/demos/
     - copy best_eq_policy.pth into <method_dir>/checkpoints/
     - until heldout_success_rate >= 0.90 OR no progress:
         * rollout on test_layouts.yaml (writes full_output.json)
         * methods p4/p5/p6: run the LLM analysis -> recommended_layouts.json
                             -> BFS expert collects those -> retrain
         * baseline_dagger: rollout-and-correct in-line (BFS recovers each
                             failure), retrain after every 4 corrective
                             demos
         * eval on heldout_test_layouts.yaml (50 layouts) and record the
           round metrics (sr_test, sr_heldout, additional_demos_collected,
           cumulative_demos)
6. **Plot** the four required line charts under cycle_<ts>/charts/.

NB on parallelism
-----------------
The user asked for P4/P5/P6 to "run in parallel". Real CPU/GPU parallelism
is the caller's job — running this script with --methods p4,p5,p6 and
piping each into a separate slurm job is the recommended setup. When all
methods are passed in one invocation, this script runs them sequentially
in the same process; the per-method directory isolation (separate demo
dir, separate checkpoint dir, separate RAG bank) is what guarantees
correctness regardless of how the caller schedules them.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PATHWAY_ROOT = REPO_ROOT / "Equivariant_pathway"
DEFAULT_INIT_DEMO_DIR = PATHWAY_ROOT / "demos"
DEFAULT_INIT_CKPT_DIR = PATHWAY_ROOT / "checkpoints"

TARGET_SR = 0.90
MAX_ROUNDS_GUARD = 50  # not a "5-round cap" — only protects against runaway loops


def _info(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [cycle] {msg}")


def _section(title: str, char: str = "=", width: int = 80):
    print(f"\n{char * width}\n{title}\n{char * width}")


def _run(cmd: List[str], cwd: Optional[Path] = None) -> int:
    _info(f"$ {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(cwd or REPO_ROOT))


def _glob_demo_count(demo_dir: Path) -> int:
    if not demo_dir.exists():
        return 0
    return sum(1 for _ in demo_dir.rglob("*.json"))


# ---------------------------------------------------------------------------
# Setup phase: layouts, initial demos, initial checkpoint.
# ---------------------------------------------------------------------------
def _list_dir(p: Path, pattern: str = "*") -> List[str]:
    if not p.exists():
        return []
    return sorted(x.name for x in p.glob(pattern) if x.is_file())


def setup_cycle(args, cycle_dir: Path) -> Dict:
    """Phase 1 of the cycle. Three sub-phases, each loudly logged so
    nothing is "magic" — the user always sees:
       1A. Layout YAMLs (training, test, held-out 50 random)
       1B. Initial expert demo collection
       1C. Initial training -> shared checkpoint

    Decisions about whether to re-collect demos / retrain are made
    EXPLICITLY: every code path prints a [setup] line stating what it
    decided and WHY, so an empty run dir is never confused with a
    "skipping because previous artefacts exist" run.
    """
    from Equivariant_pathway.layout_tracker import save_layouts_from_yaml
    from Equivariant_pathway.layout_sampler import (
        sample_layouts, write_yaml, _load_blocked_signatures,
    )

    train_yaml   = PATHWAY_ROOT / "training_layouts.yaml"
    test_yaml    = PATHWAY_ROOT / "test_layouts.yaml"
    heldout_yaml = PATHWAY_ROOT / "heldout_test_layouts.yaml"

    _section("PHASE 1A: LAYOUT YAMLS", char="-")
    _info(f"  training_layouts.yaml    : {train_yaml}")
    _info(f"  test_layouts.yaml        : {test_yaml}")
    _info(f"  heldout_test_layouts.yaml: {heldout_yaml}")

    # ----- training layouts (20 random, dmnpol-style) ----------------
    # Generate when the file is missing, when its `training_layouts:` list
    # is shorter than --initial_demos, when the user explicitly asked
    # for --regen_training, or when --force_retrain is set. We
    # IMPORTANTLY exclude test_layouts.yaml + heldout_test_layouts.yaml
    # signatures so the training distribution never overlaps the
    # evaluation sets.
    def _yaml_layout_count(path: Path) -> int:
        if not path.exists():
            return 0
        try:
            with open(path, "r") as f:
                spec = yaml.safe_load(f) or {}
            for k in ("training_layouts", "test_layouts",
                      "heldout_test_layouts", "layouts"):
                lst = spec.get(k)
                if isinstance(lst, list):
                    return len(lst)
        except Exception:
            return 0
        return 0

    train_existing = _yaml_layout_count(train_yaml)
    need_train = (
        args.force_retrain
        or args.regen_training
        or not train_yaml.exists()
        or train_existing < args.initial_demos
    )
    if need_train:
        why = []
        if args.force_retrain:    why.append("--force_retrain")
        if args.regen_training:   why.append("--regen_training")
        if not train_yaml.exists(): why.append("file missing")
        if train_existing < args.initial_demos:
            why.append(f"file has {train_existing} layouts < {args.initial_demos}")
        _info(f"  DECISION: regenerating training layouts (because: {', '.join(why)})")
        # Heldout must already be sampled BEFORE we can exclude its
        # signatures; if it doesn't exist yet we'll generate that first.
        # The block below handles it.
    else:
        _info(f"  DECISION: using existing training file ({train_existing} layouts, "
              f"{train_yaml.stat().st_size} bytes)")

    # ----- heldout layouts (50 random) -------------------------------
    if (not heldout_yaml.exists()) or args.regen_heldout or args.force_retrain:
        why = []
        if args.force_retrain:           why.append("--force_retrain")
        if args.regen_heldout:           why.append("--regen_heldout")
        if not heldout_yaml.exists():    why.append("file missing")
        _info(f"  DECISION: regenerating held-out {args.heldout_n} layouts "
              f"(because: {', '.join(why)})")
        # Held-out is sampled WITHOUT the training file as exclusion
        # (the training file may not exist yet on first run); we rely
        # on disjoint sampling later (training will exclude heldout).
        blocked = _load_blocked_signatures([str(test_yaml)])
        h_layouts = sample_layouts(
            n=args.heldout_n, grid_size=5, num_fires=3,
            min_manhattan=4, seed=args.seed + 7919, blocked_signatures=blocked,
        )
        write_yaml(h_layouts, heldout_yaml, grid_size=5)
        _info(f"  wrote {len(h_layouts)} held-out layouts -> {heldout_yaml}")
    else:
        _info(f"  using existing held-out file ({heldout_yaml.stat().st_size} bytes)")

    if need_train:
        # Now that heldout is on disk, training generation can safely
        # exclude both test AND heldout signatures so the training
        # distribution is fully disjoint from evaluation.
        blocked = _load_blocked_signatures([str(test_yaml), str(heldout_yaml)])
        t_layouts = sample_layouts(
            n=args.initial_demos, grid_size=5, num_fires=3,
            min_manhattan=4, seed=args.seed, blocked_signatures=blocked,
        )
        # n_repetitions = 1 per layout — exactly one demo per random
        # layout, per the user's spec ("no repetitions").
        for L in t_layouts:
            L["n_repetitions"] = 1
        # Write under the `training_layouts:` key (which both
        # collect_demos.py and dmnpol's loader recognise). We use
        # write_yaml with our own key by rewriting the payload.
        payload = {
            "img_size":          80,
            "grid_size":         5,
            "cell_px":           16,
            "num_fires":         3,
            "min_manhattan":     4,
            "n_repetitions":     1,
            "training_layouts":  t_layouts,
        }
        with open(train_yaml, "w") as f:
            yaml.safe_dump(payload, f, sort_keys=False)
        _info(f"  wrote {len(t_layouts)} random training layouts (each with "
              f"unique start/goal/fires) -> {train_yaml}")

    _info(f"  persisting layout JSON+PNG snapshots into {cycle_dir}/")
    save_layouts_from_yaml(train_yaml,   cycle_dir / "training",   "training")
    save_layouts_from_yaml(test_yaml,    cycle_dir / "test",       "test")
    save_layouts_from_yaml(heldout_yaml, cycle_dir / "heldout",    "heldout")

    # ------------------------------------------------------------------
    # 1B: Initial expert demo collection.
    # ------------------------------------------------------------------
    _section("PHASE 1B: INITIAL EXPERT (A*) DEMO COLLECTION", char="-")
    _info(f"  layouts source : {train_yaml}")
    _info(f"  demos target   : {DEFAULT_INIT_DEMO_DIR}")
    _info(f"  target count   : {args.initial_demos} demos")
    _info(f"  expert         : A* / BFS optimal-action mask (Equivariant_pathway/expert.py)")

    if args.force_retrain:
        _info("  --force_retrain set: deleting existing demos to redo from scratch.")
        for f in DEFAULT_INIT_DEMO_DIR.glob("*.json"):
            f.unlink()

    # Compute these AFTER the deletion block so they reflect the
    # post-deletion state of DEFAULT_INIT_DEMO_DIR. Otherwise --force_retrain
    # would delete the demos but leave `enough_already` stuck at True from
    # the pre-deletion snapshot, causing collection to be skipped and every
    # method to start from an empty initial demo set.
    pre_existing = _list_dir(DEFAULT_INIT_DEMO_DIR, "*.json")
    _info(f"  files already in target: {len(pre_existing)}")
    for n in pre_existing[:5]:
        _info(f"    - {n}")
    if len(pre_existing) > 5:
        _info(f"    ... and {len(pre_existing) - 5} more")
    enough_already = len(pre_existing) >= args.initial_demos

    if args.skip_initial_collect:
        _info(f"  DECISION: SKIP collection (--skip_initial_collect=True). "
              f"Using {len(pre_existing)} pre-existing demos.")
    elif enough_already and not args.force_retrain:
        _info(f"  DECISION: SKIP collection — folder already has "
              f"{len(pre_existing)} >= {args.initial_demos} demos. "
              f"To force fresh collection, pass --force_retrain.")
    else:
        _info(f"  DECISION: COLLECT {args.initial_demos} demos from {train_yaml}")
        rc = _run([
            sys.executable, "-u", "-m", "Equivariant_pathway.collect_demos",
            "--layouts", str(train_yaml),
            "--demo_dir", str(DEFAULT_INIT_DEMO_DIR),
            "--num_demos", str(args.initial_demos),
            "--seed", str(args.seed),
        ])
        if rc != 0:
            raise RuntimeError(f"initial collect_demos exited rc={rc}")
    final_demos = _list_dir(DEFAULT_INIT_DEMO_DIR, "*.json")
    _info(f"  final demo count: {len(final_demos)}  (target was {args.initial_demos})")
    _info(f"  files in {DEFAULT_INIT_DEMO_DIR}/:")
    for n in final_demos:
        _info(f"    - {n}")

    # ------------------------------------------------------------------
    # 1C: Initial training of the shared checkpoint.
    # ------------------------------------------------------------------
    _section("PHASE 1C: INITIAL TRAINING -> SHARED CHECKPOINT", char="-")
    pre_ckpts = _list_dir(DEFAULT_INIT_CKPT_DIR)
    best_path = DEFAULT_INIT_CKPT_DIR / "best_eq_policy.pth"
    _info(f"  demo dir       : {DEFAULT_INIT_DEMO_DIR}")
    _info(f"  checkpoint dir : {DEFAULT_INIT_CKPT_DIR}")
    _info(f"  epochs         : {args.initial_epochs}")
    _info(f"  files already in checkpoint dir: {pre_ckpts}")

    if args.force_retrain:
        _info("  --force_retrain set: deleting existing checkpoints to retrain from scratch.")
        for f in DEFAULT_INIT_CKPT_DIR.glob("*"):
            if f.is_file():
                f.unlink()
        pre_ckpts = []

    if args.skip_initial_train:
        _info(f"  DECISION: SKIP training (--skip_initial_train=True). "
              f"Using whatever is in {DEFAULT_INIT_CKPT_DIR}.")
    elif best_path.exists() and not args.force_retrain:
        _info(f"  DECISION: SKIP training — best_eq_policy.pth already exists at "
              f"{best_path} ({best_path.stat().st_size} bytes). To force retraining, "
              f"pass --force_retrain (or delete the file manually).")
    else:
        _info(f"  DECISION: TRAIN from scratch on {len(final_demos)} demos for "
              f"{args.initial_epochs} epochs")
        rc = _run([
            sys.executable, "-u", "-m", "Equivariant_pathway.train",
            "--demo_dir", str(DEFAULT_INIT_DEMO_DIR),
            "--checkpoint_dir", str(DEFAULT_INIT_CKPT_DIR),
            "--epochs", str(args.initial_epochs),
            "--seed", str(args.seed),
            "--max_demos", str(args.initial_demos),
        ])
        if rc != 0:
            raise RuntimeError(f"initial train exited rc={rc}")
    final_ckpts = _list_dir(DEFAULT_INIT_CKPT_DIR)
    _info(f"  files in {DEFAULT_INIT_CKPT_DIR}/:")
    for n in final_ckpts:
        sz = (DEFAULT_INIT_CKPT_DIR / n).stat().st_size
        _info(f"    - {n}  ({sz} bytes)")

    return {
        "train_yaml":   str(train_yaml),
        "test_yaml":    str(test_yaml),
        "heldout_yaml": str(heldout_yaml),
        "init_demo_dir":   str(DEFAULT_INIT_DEMO_DIR),
        "init_ckpt_dir":   str(DEFAULT_INIT_CKPT_DIR),
        "n_initial_demos": _glob_demo_count(DEFAULT_INIT_DEMO_DIR),
    }


# ---------------------------------------------------------------------------
# Per-method setup: copy initial demos + initial checkpoint into the
# method's isolated subfolder. Only P5/P6 get a rag_bank — neither
# baseline DAgger nor P4 use RAG (their profile YAMLs have use_rag=false),
# so creating the directory for them is just clutter.
# ---------------------------------------------------------------------------
USES_RAG = {"p5", "p6"}


def _bootstrap_method_dir(method: str, cycle_dir: Path,
                          setup: Dict) -> Dict[str, Path]:
    _section(f"BOOTSTRAP method={method}", char="-")
    method_dir = cycle_dir / method
    demo_dir   = method_dir / "demos"
    ckpt_dir   = method_dir / "checkpoints"
    rounds_dir = method_dir / "rounds"
    for p in (demo_dir, ckpt_dir, rounds_dir):
        p.mkdir(parents=True, exist_ok=True)

    # rag_bank is only created (and only used) by methods that actually
    # consult it. Profile P4 has use_rag=false; baseline DAgger uses no
    # LLM at all. Only P5 and P6 need a per-method bank.
    rag_bank: Optional[Path] = None
    if method in USES_RAG:
        rag_bank = method_dir / "rag_bank"
        rag_bank.mkdir(parents=True, exist_ok=True)
        _info(f"  rag_bank: created at {rag_bank} (method uses RAG)")
    else:
        _info(f"  rag_bank: SKIPPED (method '{method}' does not use RAG)")

    # 1. Copy initial demos. Skipping anything already present in the
    #    target (which would only happen if a previous run left them
    #    behind), but logging the copied count so the user always
    #    knows how many demos this method's loop is starting with.
    init_demo_dir = Path(setup["init_demo_dir"])
    init_demos = sorted(init_demo_dir.rglob("*.json"))
    n_pre = sum(1 for _ in demo_dir.rglob("*.json"))
    n_copied = 0
    for src in init_demos:
        dst = demo_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            n_copied += 1
    n_now = sum(1 for _ in demo_dir.rglob("*.json"))
    _info(f"  demos: src={init_demo_dir}  -> dst={demo_dir}")
    _info(f"         pre-existing in dst: {n_pre};  copied {n_copied} files;  now total: {n_now}")

    # 2. Copy initial checkpoint(s) — BOTH last_eq_policy.pth (which
    #    has the optim/sched state needed for warm-start training)
    #    AND best_eq_policy.pth (which has just model weights for
    #    rollout / eval).  The previous version copied best.pt to both
    #    slots, which made train.py --resume crash with KeyError on
    #    optim_state_dict.
    init_ckpt_dir = Path(setup["init_ckpt_dir"])
    init_best = init_ckpt_dir / "best_eq_policy.pth"
    init_last = init_ckpt_dir / "last_eq_policy.pth"
    copied = []
    if init_best.exists():
        dst = ckpt_dir / "best_eq_policy.pth"
        shutil.copy2(init_best, dst)
        copied.append(("best_eq_policy.pth", dst.stat().st_size))
    else:
        _info(f"  WARNING: source best_eq_policy.pth not found at {init_best}")
    if init_last.exists():
        dst = ckpt_dir / "last_eq_policy.pth"
        shutil.copy2(init_last, dst)
        copied.append(("last_eq_policy.pth", dst.stat().st_size))
    elif init_best.exists():
        # Fallback: only best is available (e.g. user ran with
        # --skip_initial_train and never produced last.pt). Copy best
        # under the last name so train.py --resume has SOMETHING to
        # warm-start from. train.py's warm-start logic now handles a
        # last.pt missing optim/sched gracefully.
        dst = ckpt_dir / "last_eq_policy.pth"
        shutil.copy2(init_best, dst)
        copied.append(("last_eq_policy.pth (=best fallback)", dst.stat().st_size))
    _info(f"  checkpoints: src={init_ckpt_dir}  -> dst={ckpt_dir}")
    for name, sz in copied:
        _info(f"               copied {name}  ({sz} bytes)")

    return {
        "method_dir": method_dir,
        "demo_dir":   demo_dir,
        "ckpt_dir":   ckpt_dir,
        "rag_bank":   rag_bank,  # may be None for methods that don't use RAG
        "rounds_dir": rounds_dir,
    }


# ---------------------------------------------------------------------------
# Held-out eval: run rollout_test.py on heldout_test_layouts.yaml and
# return success rate (we read full_output.json metadata).
# ---------------------------------------------------------------------------
def _eval_heldout(checkpoint: Path, heldout_yaml: Path,
                  out_dir: Path, seed: int = 0) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rc = _run([
        sys.executable, "-u", "-m", "Equivariant_pathway.rollout_test",
        "--checkpoint", str(checkpoint),
        "--layouts",    str(heldout_yaml),
        "--out_dir",    str(out_dir),
        "--seed",       str(seed),
    ])
    if rc != 0:
        raise RuntimeError(f"heldout eval rollout exited rc={rc}")
    full = json.load(open(out_dir / "full_output.json"))
    md = full.get("metadata", {}) or {}
    n  = int(md.get("n_episodes", 0) or 0)
    ns = int(md.get("n_successes", 0) or 0)
    return {
        "n_episodes":  n,
        "n_successes": ns,
        "success_rate": ns / n if n else 0.0,
        "out_dir":     str(out_dir),
    }


def _eval_test_set(checkpoint: Path, test_yaml: Path, out_dir: Path,
                   seed: int = 0, n_episodes: Optional[int] = None) -> Dict:
    """Run the active-loop test rollout. When `n_episodes` is set,
    rollout_test.py round-robins through the test_yaml's layouts so
    every method (baseline DAgger, P4, P5, P6) sees the SAME
    `(layout, seed)` schedule per round. The schedule is determined
    entirely by (test_yaml, seed, n_episodes) — passing identical
    values across methods guarantees identical rollout starts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u", "-m", "Equivariant_pathway.rollout_test",
        "--checkpoint", str(checkpoint),
        "--layouts",    str(test_yaml),
        "--out_dir",    str(out_dir),
        "--seed",       str(seed),
    ]
    if n_episodes is not None and n_episodes > 0:
        cmd += ["--n_episodes", str(n_episodes)]
    rc = _run(cmd)
    if rc != 0:
        raise RuntimeError(f"test rollout exited rc={rc}")
    full = json.load(open(out_dir / "full_output.json"))
    md = full.get("metadata", {}) or {}
    n  = int(md.get("n_episodes", 0) or 0)
    ns = int(md.get("n_successes", 0) or 0)
    return {
        "n_episodes":  n,
        "n_successes": ns,
        "success_rate": ns / n if n else 0.0,
        "out_dir":     str(out_dir),
        "full_output": str(out_dir / "full_output.json"),
    }


# ---------------------------------------------------------------------------
# Per-method round implementations.
# ---------------------------------------------------------------------------
def _run_profile_round(
    method: str, paths: Dict[str, Path], setup: Dict,
    rnd: int, args, n_initial_demos: int,
    cumulative_demos_added: int,
) -> Dict:
    """One round for a profile method (p4/p5/p6).

    1. Rollout on test_layouts -> rollout_dir
    2. Run analyze_p<N>.py -> recommended_layouts.json
    3. BFS-collect those layouts into <method>/demos/round_<N>/
    4. Retrain the policy on (initial 10 + everything in <method>/demos)
    5. Eval on heldout_test_layouts -> success_rate
    """
    method_dir = paths["method_dir"]
    demo_dir   = paths["demo_dir"]
    ckpt_dir   = paths["ckpt_dir"]
    rag_bank   = paths["rag_bank"]  # Optional[Path] — None for non-RAG methods

    round_dir = paths["rounds_dir"] / f"round_{rnd}"
    round_dir.mkdir(parents=True, exist_ok=True)

    n_demos_now = sum(1 for _ in demo_dir.rglob("*.json"))
    best_pt = ckpt_dir / "best_eq_policy.pth"
    _info(f"[{method}] round={rnd}  demos_in_method_dir={n_demos_now}  "
          f"checkpoint={best_pt.name} ({best_pt.stat().st_size if best_pt.exists() else 0} bytes)")

    # 1. Rollout on the fixed test set with the SAME 50-episode schedule
    # the baseline DAgger uses, so the comparison across methods is
    # apples-to-apples (same layouts, same per-episode seeds, same
    # number of rollouts). The seed formula matches _run_baseline_round
    # exactly.
    test_rollout_dir = round_dir / "test_rollout"
    test_metrics = _eval_test_set(
        checkpoint=ckpt_dir / "best_eq_policy.pth",
        test_yaml=Path(setup["test_yaml"]),
        out_dir=test_rollout_dir,
        seed=args.seed + rnd * 1000,
        n_episodes=args.dagger_episodes,
    )

    # 2. Analyze with the LLM pipeline.
    # Launch via `-m Equivariant_pathway.analyze_p<N>` (NOT as a file path).
    # File-path launch puts Equivariant_pathway/ on sys.path[0], which
    # makes Equivariant_pathway/model.py shadow the top-level model/
    # package — `from model.diffusion_policy import ...` then fails with
    # "model is not a package". `-m` invocation keeps only the CWD on
    # sys.path[0], so the right `model/` package wins.
    analyze_module = {
        "p4": "Equivariant_pathway.analyze_p4",
        "p5": "Equivariant_pathway.analyze_p5",
        "p6": "Equivariant_pathway.analyze_p6",
    }[method]
    analysis_dir = round_dir / f"{method}_analysis"
    cmd = [
        sys.executable, "-u", "-m", analyze_module,
        "--rollout_dir", str(test_rollout_dir),
        "--out_dir",     str(analysis_dir),
    ]
    if method in ("p5", "p6"):
        # rag_bank was only created for these methods (USES_RAG); for
        # safety re-check that paths actually has it before using.
        if rag_bank is None:
            raise RuntimeError(
                f"{method} expects a rag_bank but _bootstrap_method_dir didn't "
                f"create one — bug in USES_RAG / _bootstrap_method_dir."
            )
        cmd += ["--rag_bank", str(rag_bank / f"round_{rnd}")]
    if method in ("p4", "p5", "p6"):
        cmd += ["--demo_dir", str(demo_dir)]
    rc = _run(cmd)
    if rc != 0:
        _info(f"{method} analysis exited rc={rc}; skipping demo collection this round")

    # 3. Pull recommended_layouts.json out of the analysis run dir; the
    # pipeline runner writes it under analysis_dir/recommended_layouts.json
    # via the aggregator. If absent, fall back to building it from the
    # report's prescriptions.
    rec_path = analysis_dir / "recommended_layouts.json"
    if not rec_path.exists():
        report_path = analysis_dir / f"{method}_prescription_report.json"
        if report_path.exists():
            report = json.load(open(report_path))
            flat = []
            for pres in report.get("prescription", {}).get("demonstration_prescriptions", []) or []:
                for li, lay in enumerate(pres.get("recommended_layouts", []) or []):
                    flat.append({
                        "parent_demo_id": pres.get("cluster", "?"),
                        "layout_index":   li,
                        "repetition":     1,
                        "n_repetitions":  1,
                        "start_pos":      list(lay.get("start_pos", [])),
                        "goal_pos":       list(lay.get("goal_pos", [])),
                        "fire_positions": [list(p) for p in lay.get("fire_positions", []) or []],
                        "rationale":      lay.get("rationale", ""),
                    })
            rec_path.parent.mkdir(parents=True, exist_ok=True)
            with open(rec_path, "w") as f:
                json.dump({"layouts": flat,
                           "demo_dir": str(demo_dir / f"round_{rnd}"),
                           "n_layouts": len(flat),
                           "round": rnd}, f, indent=2, default=str)

    # 4. BFS-collect the prescribed layouts.
    round_demo_dir = demo_dir / f"round_{rnd}"
    round_demo_dir.mkdir(parents=True, exist_ok=True)
    if rec_path.exists():
        rc = _run([
            sys.executable, "-u", "-m", "Equivariant_pathway.collect_demos",
            "--layouts_from", str(rec_path),
            "--demo_dir",     str(round_demo_dir),
            "--seed",         str(args.seed + 1000 + rnd),
        ])
        if rc != 0:
            _info(f"BFS collect exited rc={rc}; round may add 0 demos")

    # Snapshot the prescribed and failure layouts as JSON+PNG so the
    # cycle dir is browseable.
    from Equivariant_pathway.layout_tracker import (
        save_layouts, save_failures_from_full_output,
    )
    if rec_path.exists():
        spec = json.load(open(rec_path))
        save_layouts(
            spec.get("layouts", []) or [],
            base_dir=round_dir / "prescribed_layouts",
            label=f"{method}_round_{rnd}_prescribed",
            extra_meta={"method": method, "round": rnd},
        )
    save_failures_from_full_output(
        Path(test_metrics["full_output"]),
        base_dir=round_dir / "test_failures",
        label=f"{method}_round_{rnd}_test_failures",
    )

    # 5. Retrain on (initial demos UNION method demos).
    demos_after = _glob_demo_count(demo_dir)
    if demos_after > n_initial_demos + cumulative_demos_added:
        rc = _run([
            sys.executable, "-u", "-m", "Equivariant_pathway.train",
            "--demo_dir",      str(demo_dir),
            "--checkpoint_dir", str(ckpt_dir),
            "--epochs",         str(args.round_epochs),
            "--seed",           str(args.seed),
            "--resume",
        ])
        if rc != 0:
            raise RuntimeError(f"{method} round {rnd} retrain exited rc={rc}")
    else:
        _info(f"{method} round {rnd}: no new demos collected; skipping retrain")

    # 6. Held-out eval.
    heldout_dir = round_dir / "heldout_eval"
    heldout_metrics = _eval_heldout(
        checkpoint=ckpt_dir / "best_eq_policy.pth",
        heldout_yaml=Path(setup["heldout_yaml"]),
        out_dir=heldout_dir,
        seed=args.seed + rnd,
    )
    save_failures_from_full_output(
        heldout_dir / "full_output.json",
        base_dir=round_dir / "heldout_failures",
        label=f"{method}_round_{rnd}_heldout_failures",
    )

    return {
        "round":                       rnd,
        "method":                      method,
        "test_success_rate":           test_metrics["success_rate"],
        "test_n_failures":             test_metrics["n_episodes"] - test_metrics["n_successes"],
        "heldout_success_rate":        heldout_metrics["success_rate"],
        "heldout_n_failures":          heldout_metrics["n_episodes"] - heldout_metrics["n_successes"],
        "demos_after_round":           demos_after,
        "additional_demos_collected":  demos_after - n_initial_demos,
        "round_dir":                   str(round_dir),
    }


def _run_baseline_round(
    paths: Dict[str, Path], setup: Dict, rnd: int, args,
    n_initial_demos: int, cumulative_demos_added: int,
) -> Dict:
    """One round for the baseline DAgger method.

    The "round" here is one rollout-and-correct PASS over the test set.
    During the pass, retraining happens every 4 corrective demos via
    baseline_dagger.run_dagger_correction_pass. The held-out eval at the
    end gives us the round's success_rate.
    """
    from Equivariant_pathway.baseline_dagger import run_dagger_correction_pass
    from Equivariant_pathway.layout_tracker import save_failures_from_full_output

    demo_dir = paths["demo_dir"]
    ckpt_dir = paths["ckpt_dir"]
    round_dir = paths["rounds_dir"] / f"round_{rnd}"
    round_dir.mkdir(parents=True, exist_ok=True)

    n_demos_now = sum(1 for _ in demo_dir.rglob("*.json"))
    best_pt = ckpt_dir / "best_eq_policy.pth"
    _info(f"[baseline_dagger] round={rnd}  demos_in_method_dir={n_demos_now}  "
          f"checkpoint={best_pt.name} ({best_pt.stat().st_size if best_pt.exists() else 0} bytes)")

    pass_summary = run_dagger_correction_pass(
        layouts_yaml=Path(setup["test_yaml"]),
        checkpoint=ckpt_dir / "best_eq_policy.pth",
        demo_dir=demo_dir,
        out_dir=round_dir / "dagger_pass",
        seed=args.seed + rnd * 1000,
        max_steps=60,
        train_every_n=4,
        train_demo_paths=None,  # use --demo_dir
        epochs=args.round_epochs,
        n_episodes=args.dagger_episodes,
    )

    demos_after = _glob_demo_count(demo_dir)
    heldout_dir = round_dir / "heldout_eval"
    heldout_metrics = _eval_heldout(
        checkpoint=ckpt_dir / "best_eq_policy.pth",
        heldout_yaml=Path(setup["heldout_yaml"]),
        out_dir=heldout_dir,
        seed=args.seed + rnd,
    )
    save_failures_from_full_output(
        heldout_dir / "full_output.json",
        base_dir=round_dir / "heldout_failures",
        label=f"baseline_dagger_round_{rnd}_heldout_failures",
    )

    return {
        "round":                       rnd,
        "method":                      "baseline_dagger",
        "test_success_rate":           pass_summary["success_rate"],
        "test_n_failures":             pass_summary["n_failures"],
        "n_corrected":                 pass_summary["n_corrected_in_pass"],
        "cum_corrections_before":      pass_summary["cum_corrections_before"],
        "cum_corrections_after":       pass_summary["cum_corrections_after"],
        "n_train_calls_in_round":      pass_summary["n_train_calls"],
        "heldout_success_rate":        heldout_metrics["success_rate"],
        "heldout_n_failures":          heldout_metrics["n_episodes"] - heldout_metrics["n_successes"],
        "demos_after_round":           demos_after,
        "additional_demos_collected":  demos_after - n_initial_demos,
        "round_dir":                   str(round_dir),
    }


# ---------------------------------------------------------------------------
# Per-method full loop (until target SR).
# ---------------------------------------------------------------------------
def run_method(method: str, cycle_dir: Path, setup: Dict, args) -> Dict:
    _section(f"METHOD = {method}", char="=")
    paths = _bootstrap_method_dir(method, cycle_dir, setup)
    n_initial_demos = setup["n_initial_demos"]

    rounds: List[Dict] = []
    cumulative_demos_added = 0
    sr_history: List[float] = []
    rnd = 0
    while rnd < args.max_rounds:
        rnd += 1
        _section(f"{method} ROUND {rnd}", char="-")
        if method == "baseline_dagger":
            r = _run_baseline_round(paths, setup, rnd, args,
                                    n_initial_demos, cumulative_demos_added)
        else:
            r = _run_profile_round(method, paths, setup, rnd, args,
                                   n_initial_demos, cumulative_demos_added)
        cumulative_demos_added = r["additional_demos_collected"]
        rounds.append(r)
        sr_history.append(r["heldout_success_rate"])
        _info(
            f"{method} round {rnd}: "
            f"test_sr={r['test_success_rate']:.2f}  "
            f"heldout_sr={r['heldout_success_rate']:.2f}  "
            f"add_demos={r['additional_demos_collected']}"
        )

        # Save the per-method curve every round so charts.py works on a
        # cycle that didn't reach the target.
        curve = {
            "method": method,
            "n_initial_demos": n_initial_demos,
            "target_sr": TARGET_SR,
            "rounds": rounds,
        }
        with open(paths["method_dir"] / "learning_curve.json", "w") as f:
            json.dump(curve, f, indent=2, default=str)

        if r["heldout_success_rate"] >= TARGET_SR:
            _info(f"{method}: TARGET_SR={TARGET_SR} reached at round {rnd} "
                  f"with {cumulative_demos_added} additional demos.")
            break
        # Cheap stagnation detector — five consecutive rounds with no
        # change to additional_demos_collected and no SR improvement
        # means the loop is spinning. Cuts wasted compute.
        if len(rounds) >= 5:
            last5 = rounds[-5:]
            if (all(rd["additional_demos_collected"] == cumulative_demos_added
                    for rd in last5) and
                max(rd["heldout_success_rate"] for rd in last5) <
                rounds[-6]["heldout_success_rate"] + 0.01):
                _info(f"{method}: stalled for 5 rounds; stopping.")
                break

    return {
        "method": method,
        "rounds": rounds,
        "n_initial_demos": n_initial_demos,
        "target_sr": TARGET_SR,
        "method_dir": str(paths["method_dir"]),
    }


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Equivariant policy full cycle.")
    p.add_argument("--methods", type=str, default="baseline_dagger,p4,p5,p6",
                   help="Comma-separated method list. Order matters only when "
                        "running them sequentially in this process.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--initial_demos", type=int, default=20,
                   help="Number of expert demos collected to start the cycle "
                        "(matched to training_layouts.yaml: 5 layouts x 4 reps "
                        "= 20). Every method (baseline + P4/P5/P6) starts from "
                        "the same set, copied verbatim into the per-method "
                        "demos/ subfolder during bootstrap.")
    p.add_argument("--force_retrain", action="store_true",
                   help="Wipe Equivariant_pathway/demos/, "
                        "Equivariant_pathway/checkpoints/, AND "
                        "training_layouts.yaml (so it gets re-sampled with "
                        "fresh random layouts) before setup. Use this when "
                        "you want a 100% from-scratch run, with no risk of "
                        "stale artefacts from a previous invocation.")
    p.add_argument("--initial_epochs", type=int, default=200,
                   help="Epochs for the initial training run on the 10 demos.")
    p.add_argument("--round_epochs", type=int, default=60,
                   help="Epochs for retraining within each round of every method.")
    p.add_argument("--heldout_n", type=int, default=50,
                   help="Number of held-out random layouts (default 50, as "
                        "specified by the project goal).")
    p.add_argument("--max_rounds", type=int, default=MAX_ROUNDS_GUARD,
                   help="Hard guard against infinite loops. There is NO 5-round "
                        "cap — this is the explicit anti-runaway, default 50.")
    p.add_argument("--dagger_episodes", type=int, default=50,
                   help="Episodes per round, applied UNIFORMLY to every "
                        "method (baseline DAgger AND P4/P5/P6). Layouts are "
                        "drawn round-robin from test_layouts.yaml with "
                        "seed = args.seed + rnd*1000 + ep_idx, so all four "
                        "methods see the EXACT same (layout, seed) schedule "
                        "per round and the only thing that differs across "
                        "methods is what each one does with the failures it "
                        "observes (BFS-correct in-line for baseline; LLM "
                        "analyse + prescribe + BFS-collect for profiles). "
                        "For baseline DAgger, retraining additionally fires "
                        "every 4 corrective demos accumulated GLOBALLY.")
    p.add_argument("--cycle_dir", type=str, default=None,
                   help="Top-level dir for this cycle's artefacts. Defaults to "
                        "results/equivariant_pathway/cycle_<timestamp>/.")
    p.add_argument("--regen_heldout", action="store_true",
                   help="Force-regenerate heldout_test_layouts.yaml even if it exists.")
    p.add_argument("--regen_training", action="store_true",
                   help="Force-regenerate training_layouts.yaml even if it has "
                        ">= --initial_demos entries. The 20 training layouts are "
                        "sampled fresh (each with random start/goal/fire), "
                        "deduplicated by signature, and excluded against "
                        "test_layouts.yaml + heldout_test_layouts.yaml.")
    p.add_argument("--skip_initial_collect", action="store_true")
    p.add_argument("--skip_initial_train", action="store_true")
    p.add_argument("--skip_charts", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.cycle_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cycle_dir = REPO_ROOT / "results" / "equivariant_pathway" / f"cycle_{ts}"
    else:
        cycle_dir = Path(args.cycle_dir)
    cycle_dir.mkdir(parents=True, exist_ok=True)
    _info(f"cycle dir: {cycle_dir}")

    setup = setup_cycle(args, cycle_dir)
    with open(cycle_dir / "setup.json", "w") as f:
        json.dump({**setup, "args": vars(args)}, f, indent=2, default=str)

    summaries: Dict[str, Dict] = {}
    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        try:
            summaries[method] = run_method(method, cycle_dir, setup, args)
        except Exception as e:
            _info(f"{method} crashed: {e!r}")
            summaries[method] = {"error": str(e)}

    with open(cycle_dir / "cycle_summary.json", "w") as f:
        json.dump(summaries, f, indent=2, default=str)

    if not args.skip_charts:
        from Equivariant_pathway.charts import make_charts
        chart_summary = make_charts(cycle_dir)
        _info(f"charts: {chart_summary}")

    _section("CYCLE DONE", char="=")
    for m, s in summaries.items():
        rnds = (s or {}).get("rounds") or []
        last = rnds[-1] if rnds else None
        if last:
            print(f"  {m:<20s}: rounds={len(rnds)}  "
                  f"final_heldout_sr={last['heldout_success_rate']:.2f}  "
                  f"add_demos={last['additional_demos_collected']}")
        else:
            print(f"  {m:<20s}: NO ROUNDS  ({(s or {}).get('error','?')})")


if __name__ == "__main__":
    main()
