"""DISTIL ablation-matrix orchestrator (05_..md triage + 08_..md prioritized split).

Enumerates the (task, modality, arm, seed) cells, tags each with a priority tier and
a runnable/pending status (a cell is runnable only if its component is built + smoked),
and either prints the plan (--dry-run), writes the RUN_STATE ledger, or submits jobs
(--submit) with GPU/CPU routing. One arm+seed = one sbatch job (golden rule 5); robot
tasks go to a GPU partition, GridWorld runs CPU.

Priority tiers (08_..md):
  P0  full DISTIL, all built (task, modality) cells x 5 seeds        (headline table)
  P1  Tier-1 knockouts (allocation thesis) on all built cells        (crown jewel)
  P2  drawer ablations + budget sweep
Examples:
  python -m distil.matrix --dry-run
  python -m distil.matrix --priority P0 P1 --modality state --submit
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

SEEDS = [1, 2, 3, 4, 5]
# Submission priority order (05_..md): GridWorld first (cheapest -> fastest COMPLETE
# allocation-thesis + sign-test signal), then the high-headroom tasks the crown jewel
# lives on (Wipe, PushT), then Door, then Lift (near-saturated in the paper). Cells are
# emitted in this order so the cheap/high-value work is submitted first.
TASKS = ["GridWorld", "Wipe", "PushT", "Door", "Lift"]
MODALITIES = ["state", "image"]

# arm -> priority tier. 'full' + Tier-1 knockouts are the crown jewel.
ARMS_P0 = ["full"]
ARMS_P1 = ["memory_off", "allocation_random", "clustering_off", "decision_heuristic", "vlm_off"]
ARMS_P2 = ["kag_off", "bridge_off", "fallback_only", "near_dominant_off", "fixed_k3", "llm_effort_low"]
BASELINE_ARMS = ["diffdagger"]  # gate family {safe,dropout,ensemble,thrifty,stagger} pending port

# Which (task, modality) cells are BUILT + SMOKED (runnable now).
RUNNABLE = {
    ("GridWorld", "state"): True,
    ("Lift", "state"): True,
    ("Wipe", "state"): True,
    ("Door", "state"): True,
    # image modality: policy+dataset built; per-step render threading pending -> not yet.
    ("GridWorld", "image"): False,
    ("Lift", "image"): False, ("Wipe", "image"): False, ("Door", "image"): False,
    ("PushT", "state"): False, ("PushT", "image"): False,   # vendoring pending
}

# rough per-cell wall-clock weight (for the 2-HPC load-balance; 08_..md split by cost).
COST = {"GridWorld": 1, "Lift": 3, "Door": 4, "Wipe": 6, "PushT": 6}


def _cells(priorities, modalities, seeds):
    arms = []
    if "P0" in priorities:
        arms += ARMS_P0
    if "P1" in priorities:
        arms += ARMS_P1
    if "P2" in priorities:
        arms += ARMS_P2 + BASELINE_ARMS
    for task in TASKS:
        for mod in modalities:
            if not RUNNABLE.get((task, mod)):
                continue
            for arm in arms:
                # baselines only defined for robot diffusion tasks currently
                if arm in BASELINE_ARMS and task == "GridWorld":
                    continue
                for s in seeds:
                    yield {"task": task, "modality": mod, "arm": arm, "seed": s,
                           "cost": COST[task]}


def _sbatch_cmd(cell, budget, bootstrap_dir, results_dir):
    task, mod, arm, seed = cell["task"], cell["modality"], cell["arm"], cell["seed"]
    out = f"{results_dir}/{task}/{mod}/{arm}/seed{seed}"
    # every arm can run on any GPU node (GridWorld is CPU-light but has no CPU-only
    # partition here, so it fills idle GPU slots on ANY partition rather than queuing
    # behind robots on the 3 a100 nodes).
    part = "gpu,gpu-large"
    name = f"distil_{task}_{mod}_{arm}_s{seed}"
    env = (f"ALL,TASK={task},MODALITY={mod},ABLATION={arm},SEED={seed},"
           f"BUDGET={budget},OUTPUT_DIR={out},BOOTSTRAP_DIR={bootstrap_dir}/{task}_{mod}")
    return name, [
        "sbatch", "--parsable", f"--job-name={name}",
        f"--partition={part}", "--qos=batch-long", "--time=1-00:00:00",
        "--nodes=1", "--gpus-per-node=1", "--cpus-per-gpu=8", "--mem=32G",
        f"--export={env}", "distil/scripts/run_distil.sbatch"], out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--priority", nargs="+", default=["P0", "P1"])
    ap.add_argument("--modality", nargs="+", default=["state"], choices=MODALITIES)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--bootstrap-dir", default="distil/results/shared_bootstrap")
    ap.add_argument("--results-dir", default="distil/results")
    ap.add_argument("--run-state", default="distil/RUN_STATE.md")
    ap.add_argument("--submit", action="store_true", help="actually sbatch (else dry-run)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cells = list(_cells(a.priority, a.modality, a.seeds))
    total_cost = sum(c["cost"] for c in cells)
    print(f"[matrix] {len(cells)} runnable cells | priorities={a.priority} "
          f"modalities={a.modality} | total cost-weight={total_cost}")

    ledger = ["# RUN_STATE — DISTIL matrix ledger (08_..md source of truth)", "",
              "| task | modality | arm | seed | hpc | jobid | status | result_path |",
              "|---|---|---|---|---|---|---|---|"]
    submitted = 0
    for c in cells:
        name, cmd, out = _sbatch_cmd(c, a.budget, a.bootstrap_dir, a.results_dir)
        jobid, status = "-", "planned"
        if a.submit:
            try:
                jobid = subprocess.check_output(cmd, cwd="/weka/s226137394/DmNfull").decode().strip()
                status = "submitted"
                submitted += 1
            except Exception as e:
                status = f"submit_error:{str(e)[:40]}"
        else:
            print("  " + " ".join(cmd[:8]) + f" ... {name}")
        ledger.append(f"| {c['task']} | {c['modality']} | {c['arm']} | {c['seed']} | "
                      f"HPC-A | {jobid} | {status} | {out} |")

    Path(a.run_state).write_text("\n".join(ledger) + "\n")
    print(f"[matrix] wrote {a.run_state} ({'submitted ' + str(submitted) if a.submit else 'dry-run'})")


if __name__ == "__main__":
    main()
