# PROVENANCE — σ calibration

## SLURM jobs

| attempt | job ID | outcome |
|---|---|---|
| 1 | `110493` | CANCELLED — hung walking `pool_x_selector` (~835k `episode_data.json`, zero centroid artifacts). Pruned that walk. |
| 2 | `110494` | FAILED — `matplotlib.axvline(None)` on a cell with one cluster (no NN pair). Guarded. |
| 3 | `110495` | COMPLETED — but adversarial audit of its output found real defects (see below). **Superseded.** |
| 4 | **`110500`** | **COMPLETED — the run every number in `sigma_report.md` comes from.** |

```bash
cd /weka/s226137394/DmNfull
sbatch distil/scripts/run_sigma_analysis.sbatch          # -> job 110500
/home/s226137394/.conda/envs/maze/bin/python distil/scripts/write_sigma_report.py
```

Partition `gpu` with `--gres=none` (pure CPU: this cluster exposes **no** `cpu` partition — `sinfo`
lists only `gpu` and `gpu-large`), 4 CPUs, 16 GB, no GPU, **no env rollouts, no LLM calls**.
Interpreter `/home/s226137394/.conda/envs/maze/bin/python` (numpy 1.26.4, matplotlib 3.10.9,
openpyxl 3.1.5) — the `diffdagger` env has no matplotlib/openpyxl.

## What job 110495 got wrong, and what 110500 fixed

The first completed run was audited by three independent adversarial checkers. They found defects
that were **real**, not cosmetic, so the script was fixed and re-run rather than the report patched:

| defect | fix in `analyze_sigma.py` |
|---|---|
| Penalty columns computed over a **pooled** distance set (NN ∪ target↔memory) while σ was calibrated on the NN set alone — a 55-order-of-magnitude self-contradiction on GridWorld | penalty stats now computed **separately** per distance set; headline uses the NN set (the one σ is fitted to) |
| GridWorld length scale used the grid **pitch** (1 cell) where every robot entry is a spawn **extent** — this manufactured the "GridWorld is 5–10× off" case against the spawn-extent rule | `SPAWN_EXTENT["GridWorld"] = 4.0` (5×5 lattice index span). The claim is **withdrawn** in the report. |
| `n_pairwise` printed against NN quantiles | `n_nn` reported distinctly |
| `_smoke` runs silently folded into 5 cells | `_smoke` **excluded** |
| The decision-level statistic was computed and then dropped | §3.5: `select_target` replayed with the **true γ-weighted sum**, counting argmax flips at current vs recommended σ |
| Single-entry kernel modelled where the deployed code uses a γ-weighted **sum** | `recency_penalty()` now implements the real sum |
| 4 hand-transcription rounding slips | report is **generated from the CSV** (`write_sigma_report.py`) — hand-transcription is now structurally impossible |

## Scripts (all NEW/additive; no pipeline code and no default was modified)

- `distil/scripts/analyze_sigma.py` — harvest, distances, diagnosis, decision replay, α·L_task, figures
- `distil/scripts/write_sigma_report.py` — emits `sigma_report.md` straight from the CSV
- `distil/scripts/run_sigma_analysis.sbatch` — the SLURM wrapper

**`distil/config.py:71` (`memory_sigma = 0.06`) was NOT changed.** Analysis + recommendation only.

## Outputs

- `sigma_report.md` — distances, diagnosis, decision replay, α/L_task derivation, recommendation, limits
- `sigma_per_task.csv` — machine-readable, one row per task × modality
- `sigma_rows.json`, `harvest_raw.json` — full per-cell distributions + every file touched
- `figures/<Task>_<modality>.{pdf,png}` — NN-distance histogram + kernel curve at current vs recommended σ

## Sources harvested

**115 run directories, 1,751 cluster-carrying rounds, 115 `centroid_memory.json` files.** `_smoke` excluded.

`pool_x_selector` deliberately NOT walked — verified to hold zero centroid artifacts:
```
find .../pool_x_selector -maxdepth 6 -name centroid_memory.json  -> 0
find .../pool_x_selector -maxdepth 6 -type d -name telemetry     -> 0
```

### Run directories harvested, per cell

**Door/image** — 1 run dirs, 6 telemetry files

- `/weka/s226137394/DmNfull/distil/results/_compute/Door/image/full/seed1`

**Door/state** — 1 run dirs, 6 telemetry files

- `/weka/s226137394/DmNfull/distil/results/_compute/Door/state/full/seed1`

**GridWorld/image** — 1 run dirs, 2 telemetry files

- `/weka/s226137394/DmNfull/distil/results/_compute/GridWorld/image/full/seed1`

**GridWorld/state** — 30 run dirs, 630 telemetry files

- `/weka/s226137394/DmNfull/distil/results/GridWorld/state/allocation_random/seed1`
- `/weka/s226137394/DmNfull/distil/results/GridWorld/state/allocation_random/seed2`
- `/weka/s226137394/DmNfull/distil/results/GridWorld/state/allocation_random/seed3`
- `/weka/s226137394/DmNfull/distil/results/GridWorld/state/allocation_random/seed4`
- `/weka/s226137394/DmNfull/distil/results/GridWorld/state/allocation_random/seed5`
- `/weka/s226137394/DmNfull/distil/results/GridWorld/state/clustering_off/seed1`
- `/weka/s226137394/DmNfull/distil/results/GridWorld/state/clustering_off/seed2`
- `/weka/s226137394/DmNfull/distil/results/GridWorld/state/clustering_off/seed3`
- `/weka/s226137394/DmNfull/distil/results/GridWorld/state/clustering_off/seed4`
- `/weka/s226137394/DmNfull/distil/results/GridWorld/state/clustering_off/seed5`
- `/weka/s226137394/DmNfull/distil/results/GridWorld/state/decision_heuristic/seed1`
- `/weka/s226137394/DmNfull/distil/results/GridWorld/state/decision_heuristic/seed2`
- `/weka/s226137394/DmNfull/distil/results/GridWorld/state/decision_heuristic/seed3`
- `/weka/s226137394/DmNfull/distil/results/GridWorld/state/decision_heuristic/seed4`
- `/weka/s226137394/DmNfull/distil/results/GridWorld/state/decision_heuristic/seed5`
- `/weka/s226137394/DmNfull/distil/results/GridWorld/state/full/seed1`
- `/weka/s226137394/DmNfull/distil/results/GridWorld/state/full/seed2`
- `/weka/s226137394/DmNfull/distil/results/GridWorld/state/full/seed3`
- `/weka/s226137394/DmNfull/distil/results/GridWorld/state/full/seed4`
- `/weka/s226137394/DmNfull/distil/results/GridWorld/state/full/seed5`
- `/weka/s226137394/DmNfull/distil/results/GridWorld/state/memory_off/seed1`
- `/weka/s226137394/DmNfull/distil/results/GridWorld/state/memory_off/seed2`
- `/weka/s226137394/DmNfull/distil/results/GridWorld/state/memory_off/seed3`
- `/weka/s226137394/DmNfull/distil/results/GridWorld/state/memory_off/seed4`
- `/weka/s226137394/DmNfull/distil/results/GridWorld/state/memory_off/seed5`
- `/weka/s226137394/DmNfull/distil/results/GridWorld/state/vlm_off/seed1`
- `/weka/s226137394/DmNfull/distil/results/GridWorld/state/vlm_off/seed2`
- `/weka/s226137394/DmNfull/distil/results/GridWorld/state/vlm_off/seed3`
- `/weka/s226137394/DmNfull/distil/results/GridWorld/state/vlm_off/seed4`
- `/weka/s226137394/DmNfull/distil/results/GridWorld/state/vlm_off/seed5`

**Lift/image** — 28 run dirs, 586 telemetry files

- `/weka/s226137394/DmNfull/distil/results/Lift/image/allocation_random/seed3`
- `/weka/s226137394/DmNfull/distil/results/Lift/image/allocation_random/seed4`
- `/weka/s226137394/DmNfull/distil/results/Lift/image/allocation_random/seed5`
- `/weka/s226137394/DmNfull/distil/results/Lift/image/clustering_off/seed1`
- `/weka/s226137394/DmNfull/distil/results/Lift/image/clustering_off/seed2`
- `/weka/s226137394/DmNfull/distil/results/Lift/image/clustering_off/seed3`
- `/weka/s226137394/DmNfull/distil/results/Lift/image/clustering_off/seed4`
- `/weka/s226137394/DmNfull/distil/results/Lift/image/clustering_off/seed5`
- `/weka/s226137394/DmNfull/distil/results/Lift/image/decision_heuristic/seed1`
- `/weka/s226137394/DmNfull/distil/results/Lift/image/decision_heuristic/seed2`
- `/weka/s226137394/DmNfull/distil/results/Lift/image/decision_heuristic/seed3`
- `/weka/s226137394/DmNfull/distil/results/Lift/image/decision_heuristic/seed4`
- `/weka/s226137394/DmNfull/distil/results/Lift/image/decision_heuristic/seed5`
- `/weka/s226137394/DmNfull/distil/results/Lift/image/full/seed1`
- `/weka/s226137394/DmNfull/distil/results/Lift/image/full/seed2`
- `/weka/s226137394/DmNfull/distil/results/Lift/image/full/seed3`
- `/weka/s226137394/DmNfull/distil/results/Lift/image/full/seed4`
- `/weka/s226137394/DmNfull/distil/results/Lift/image/full/seed5`
- `/weka/s226137394/DmNfull/distil/results/Lift/image/memory_off/seed1`
- `/weka/s226137394/DmNfull/distil/results/Lift/image/memory_off/seed2`
- `/weka/s226137394/DmNfull/distil/results/Lift/image/memory_off/seed3`
- `/weka/s226137394/DmNfull/distil/results/Lift/image/memory_off/seed4`
- `/weka/s226137394/DmNfull/distil/results/Lift/image/memory_off/seed5`
- `/weka/s226137394/DmNfull/distil/results/Lift/image/vlm_off/seed1`
- `/weka/s226137394/DmNfull/distil/results/Lift/image/vlm_off/seed2`
- `/weka/s226137394/DmNfull/distil/results/Lift/image/vlm_off/seed3`
- `/weka/s226137394/DmNfull/distil/results/Lift/image/vlm_off/seed4`
- `/weka/s226137394/DmNfull/distil/results/Lift/image/vlm_off/seed5`

**Lift/state** — 30 run dirs, 191 telemetry files

- `/weka/s226137394/DmNfull/distil/results/Lift/state/allocation_random/seed1`
- `/weka/s226137394/DmNfull/distil/results/Lift/state/allocation_random/seed2`
- `/weka/s226137394/DmNfull/distil/results/Lift/state/allocation_random/seed3`
- `/weka/s226137394/DmNfull/distil/results/Lift/state/allocation_random/seed4`
- `/weka/s226137394/DmNfull/distil/results/Lift/state/allocation_random/seed5`
- `/weka/s226137394/DmNfull/distil/results/Lift/state/clustering_off/seed1`
- `/weka/s226137394/DmNfull/distil/results/Lift/state/clustering_off/seed2`
- `/weka/s226137394/DmNfull/distil/results/Lift/state/clustering_off/seed3`
- `/weka/s226137394/DmNfull/distil/results/Lift/state/clustering_off/seed4`
- `/weka/s226137394/DmNfull/distil/results/Lift/state/clustering_off/seed5`
- `/weka/s226137394/DmNfull/distil/results/Lift/state/decision_heuristic/seed1`
- `/weka/s226137394/DmNfull/distil/results/Lift/state/decision_heuristic/seed2`
- `/weka/s226137394/DmNfull/distil/results/Lift/state/decision_heuristic/seed3`
- `/weka/s226137394/DmNfull/distil/results/Lift/state/decision_heuristic/seed4`
- `/weka/s226137394/DmNfull/distil/results/Lift/state/decision_heuristic/seed5`
- `/weka/s226137394/DmNfull/distil/results/Lift/state/full/seed1`
- `/weka/s226137394/DmNfull/distil/results/Lift/state/full/seed2`
- `/weka/s226137394/DmNfull/distil/results/Lift/state/full/seed3`
- `/weka/s226137394/DmNfull/distil/results/Lift/state/full/seed4`
- `/weka/s226137394/DmNfull/distil/results/Lift/state/full/seed5`
- `/weka/s226137394/DmNfull/distil/results/Lift/state/memory_off/seed1`
- `/weka/s226137394/DmNfull/distil/results/Lift/state/memory_off/seed2`
- `/weka/s226137394/DmNfull/distil/results/Lift/state/memory_off/seed3`
- `/weka/s226137394/DmNfull/distil/results/Lift/state/memory_off/seed4`
- `/weka/s226137394/DmNfull/distil/results/Lift/state/memory_off/seed5`
- `/weka/s226137394/DmNfull/distil/results/Lift/state/vlm_off/seed1`
- `/weka/s226137394/DmNfull/distil/results/Lift/state/vlm_off/seed2`
- `/weka/s226137394/DmNfull/distil/results/Lift/state/vlm_off/seed3`
- `/weka/s226137394/DmNfull/distil/results/Lift/state/vlm_off/seed4`
- `/weka/s226137394/DmNfull/distil/results/Lift/state/vlm_off/seed5`

**PlugCharger/state** — 3 run dirs, 3 telemetry files

- `/weka/s226137394/DmNfull/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_rl_robo/results/PlugCharger-v1/run_1/p4_top3/results`
- `/weka/s226137394/DmNfull/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_rl_robo/results/PlugCharger-v1/run_2/p4_top3/results`
- `/weka/s226137394/DmNfull/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_rl_robo/results/PlugCharger-v1/run_991/p4_top3/results`

**PushT/state** — 12 run dirs, 425 telemetry files

- `/weka/s226137394/DmNfull/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_rl_robo/results/PushT-v1/_archived_p4_subtask_v1/run_1/p4_subtask/results`
- `/weka/s226137394/DmNfull/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_rl_robo/results/PushT-v1/_archived_p4_subtask_v1/run_2/p4_subtask/results`
- `/weka/s226137394/DmNfull/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_rl_robo/results/PushT-v1/_archived_p4_subtask_v1/run_990/p4_subtask/results`
- `/weka/s226137394/DmNfull/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_rl_robo/results/PushT-v1/run_1/p4_subtask/results`
- `/weka/s226137394/DmNfull/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_rl_robo/results/PushT-v1/run_11/p4_subtask/results`
- `/weka/s226137394/DmNfull/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_rl_robo/results/PushT-v1/run_111/p4_subtask/results`
- `/weka/s226137394/DmNfull/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_rl_robo/results/PushT-v1/run_112/p4_subtask/results`
- `/weka/s226137394/DmNfull/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_rl_robo/results/PushT-v1/run_12/p4_subtask/results`
- `/weka/s226137394/DmNfull/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_rl_robo/results/PushT-v1/run_2/p4_subtask/results`
- `/weka/s226137394/DmNfull/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_rl_robo/results/PushT-v1/run_511/p4_subtask/results`
- `/weka/s226137394/DmNfull/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_rl_robo/results/PushT-v1/run_991/p4_subtask/results`
- `/weka/s226137394/DmNfull/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_rl_robo/results/PushT-v1/run_992/p4_subtask/results`

**StackCube/state** — 5 run dirs, 5 telemetry files

- `/weka/s226137394/DmNfull/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_rl_robo/results/StackCube-v1/run_1/p4_top3/results`
- `/weka/s226137394/DmNfull/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_rl_robo/results/StackCube-v1/run_2/p4_top3/results`
- `/weka/s226137394/DmNfull/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_rl_robo/results/StackCube-v1/run_993/p4_top3/results`
- `/weka/s226137394/DmNfull/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_rl_robo/results/StackCube-v1/run_995/p4_top3/results`
- `/weka/s226137394/DmNfull/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_rl_robo/results/StackCube-v1/run_996/p4_top3/results`

**Wipe/image** — 1 run dirs, 6 telemetry files

- `/weka/s226137394/DmNfull/distil/results/_compute/Wipe/image/full/seed1`

**Wipe/state** — 3 run dirs, 6 telemetry files

- `/weka/s226137394/DmNfull/distil/results/Wipe/state/full/seed1`
- `/weka/s226137394/DmNfull/distil/results/Wipe/state/full/seed2`
- `/weka/s226137394/DmNfull/distil/results/Wipe/state/full/seed3`

## Reference cross-checks (all read-only)

- `paper_aaai2027/COC_REPORT/ablations_results/DISTIL_ablation_results.xlsx` sheet **A13** (σ sweep) — the recommendation reproduces its σ≈0.02 (Lift) and σ≈0.005 (Door) analytic estimates from measured data. The report notes the two routes are partly **collinear**, so this is weaker confirmation than it looks.
- Same workbook, sheet **GT_SR** — confirms DISEIL = 100.0 ± 0.0 on Lift (the ceiling caveat).
- `paper_aaai2027/context/kag_ur5_bounds.md` + `distil/p4/bounds.py` `TASK_BOUNDS` — spawn extents. **Wipe is absent from both** (no reset bounds exist).

**This study opened the workbook read-only and did not modify it.**
