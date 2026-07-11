# RUN_STATE — DISTIL matrix (live ledger)

Two clusters, one ledger (HANDOFF_HPC2 §3, §6). Per-cluster jobids.

- **HPC-A (weka):** GridWorld (all arms) + Lift-state.
- **HPC-B (rohan, this session):** Wipe + Door state — WHOLE cells, 6 arms × 5 seeds = 60 jobs.

> Wipe/Door reassigned to HPC-B (HANDOFF §3). Per-task Ni from config (upstream b5485c09): Wipe=12, Door=4. Blackwell node rtxp6000l-f-01 excluded (no cu121 kernels). budget=20.

## HPC-A — GridWorld + Lift (weka)
| cell | seed | jobid | state | final_SR |
|---|---|---|---|---|
| GridWorld_state_allocation_random | 1 | 109014 | PENDING | - |
| GridWorld_state_allocation_random | 2 | 109015 | PENDING | - |
| GridWorld_state_allocation_random | 3 | 109016 | PENDING | - |
| GridWorld_state_allocation_random | 4 | 109017 | PENDING | - |
| GridWorld_state_allocation_random | 5 | 109018 | PENDING | - |
| GridWorld_state_clustering_off | 1 | 109019 | PENDING | - |
| GridWorld_state_clustering_off | 2 | 109020 | PENDING | - |
| GridWorld_state_clustering_off | 3 | 109021 | PENDING | - |
| GridWorld_state_clustering_off | 4 | 109022 | PENDING | - |
| GridWorld_state_clustering_off | 5 | 109023 | PENDING | - |
| GridWorld_state_decision_heuristic | 1 | 109024 | PENDING | - |
| GridWorld_state_decision_heuristic | 2 | 109025 | PENDING | - |
| GridWorld_state_decision_heuristic | 3 | 109026 | PENDING | - |
| GridWorld_state_decision_heuristic | 4 | 109027 | PENDING | - |
| GridWorld_state_decision_heuristic | 5 | 109028 | PENDING | - |
| GridWorld_state_full | 1 | 109003 | PENDING | - |
| GridWorld_state_full | 2 | 109004 | PENDING | - |
| GridWorld_state_full | 3 | 109005 | PENDING | - |
| GridWorld_state_full | 4 | 109006 | PENDING | - |
| GridWorld_state_full | 5 | 109007 | PENDING | - |
| GridWorld_state_memory_off | 1 | 109009 | PENDING | - |
| GridWorld_state_memory_off | 2 | 109010 | PENDING | - |
| GridWorld_state_memory_off | 3 | 109011 | PENDING | - |
| GridWorld_state_memory_off | 4 | 109012 | PENDING | - |
| GridWorld_state_memory_off | 5 | 109013 | PENDING | - |
| GridWorld_state_vlm_off | 1 | 109029 | PENDING | - |
| GridWorld_state_vlm_off | 2 | 109030 | PENDING | - |
| GridWorld_state_vlm_off | 3 | 109031 | PENDING | - |
| GridWorld_state_vlm_off | 4 | 109032 | PENDING | - |
| GridWorld_state_vlm_off | 5 | 109033 | PENDING | - |
| Lift_state_full | 1 | 108986 | RUNNING | - |
| Lift_state_full | 2 | 108987 | RUNNING | - |
| Lift_state_full | 3 | 108988 | RUNNING | - |
| Lift_state_full | 4 | 108989 | RUNNING | - |
| Lift_state_full | 5 | 108990 | RUNNING | - |

## HPC-B — Wipe + Door state (rohan), Ni=config, budget=20, Blackwell-excluded
| cell | seed | hpc | jobid | state | final_SR |
|---|---|---|---|---|---|
| Wipe_state_full | 1 | B | 3361395 | COMPLETED | 0.92 |
| Wipe_state_full | 2 | B | 3361396 | COMPLETED | 0.82 |
| Wipe_state_full | 3 | B | 3361397 | COMPLETED | 0.89 |
| Wipe_state_full | 4 | B | 3361398 | COMPLETED | 0.8 |
| Wipe_state_full | 5 | B | 3361399 | COMPLETED | 0.88 |
| Wipe_state_memory_off | 1 | B | 3361400 | COMPLETED | 0.95 |
| Wipe_state_memory_off | 2 | B | 3361401 | COMPLETED | 0.94 |
| Wipe_state_memory_off | 3 | B | 3361402 | COMPLETED | 0.92 |
| Wipe_state_memory_off | 4 | B | 3361403 | COMPLETED | 0.87 |
| Wipe_state_memory_off | 5 | B | 3361404 | COMPLETED | 0.9 |
| Wipe_state_allocation_random | 1 | B | 3361405 | COMPLETED | 0.86 |
| Wipe_state_allocation_random | 2 | B | 3361406 | COMPLETED | 0.91 |
| Wipe_state_allocation_random | 3 | B | 3361407 | COMPLETED | 0.81 |
| Wipe_state_allocation_random | 4 | B | 3361408 | COMPLETED | 0.91 |
| Wipe_state_allocation_random | 5 | B | 3361409 | COMPLETED | 0.91 |
| Wipe_state_clustering_off | 1 | B | 3361410 | COMPLETED | 0.91 |
| Wipe_state_clustering_off | 2 | B | 3361411 | COMPLETED | 0.81 |
| Wipe_state_clustering_off | 3 | B | 3361412 | COMPLETED | 0.92 |
| Wipe_state_clustering_off | 4 | B | 3361413 | COMPLETED | 0.85 |
| Wipe_state_clustering_off | 5 | B | 3361414 | COMPLETED | 0.78 |
| Wipe_state_decision_heuristic | 1 | B | 3361415 | COMPLETED | 0.81 |
| Wipe_state_decision_heuristic | 2 | B | 3361416 | COMPLETED | 0.9 |
| Wipe_state_decision_heuristic | 3 | B | 3361417 | COMPLETED | 0.86 |
| Wipe_state_decision_heuristic | 4 | B | 3361418 | COMPLETED | 0.95 |
| Wipe_state_decision_heuristic | 5 | B | 3361419 | COMPLETED | 0.88 |
| Wipe_state_vlm_off | 1 | B | 3361420 | COMPLETED | 0.79 |
| Wipe_state_vlm_off | 2 | B | 3361421 | COMPLETED | 0.93 |
| Wipe_state_vlm_off | 3 | B | 3361422 | COMPLETED | 0.94 |
| Wipe_state_vlm_off | 4 | B | 3361423 | COMPLETED | 0.89 |
| Wipe_state_vlm_off | 5 | B | 3361424 | COMPLETED | 0.9 |
| Door_state_full | 1 | B | 3361425 | COMPLETED | 0.92 |
| Door_state_full | 2 | B | 3361426 | COMPLETED | 0.97 |
| Door_state_full | 3 | B | 3361427 | COMPLETED | 0.92 |
| Door_state_full | 4 | B | 3361428 | COMPLETED | 0.96 |
| Door_state_full | 5 | B | 3361429 | COMPLETED | 0.91 |
| Door_state_memory_off | 1 | B | 3361430 | COMPLETED | 0.97 |
| Door_state_memory_off | 2 | B | 3361431 | COMPLETED | 0.95 |
| Door_state_memory_off | 3 | B | 3361432 | COMPLETED | 0.97 |
| Door_state_memory_off | 4 | B | 3361433 | COMPLETED | 0.94 |
| Door_state_memory_off | 5 | B | 3361434 | COMPLETED | 0.97 |
| Door_state_allocation_random | 1 | B | 3361435 | COMPLETED | 0.97 |
| Door_state_allocation_random | 2 | B | 3361436 | COMPLETED | 0.96 |
| Door_state_allocation_random | 3 | B | 3361437 | COMPLETED | 0.98 |
| Door_state_allocation_random | 4 | B | 3361438 | COMPLETED | 0.94 |
| Door_state_allocation_random | 5 | B | 3361439 | COMPLETED | 0.95 |
| Door_state_clustering_off | 1 | B | 3361440 | COMPLETED | 0.96 |
| Door_state_clustering_off | 2 | B | 3361441 | COMPLETED | 0.97 |
| Door_state_clustering_off | 3 | B | 3361442 | COMPLETED | 0.93 |
| Door_state_clustering_off | 4 | B | 3361443 | COMPLETED | 0.93 |
| Door_state_clustering_off | 5 | B | 3361444 | COMPLETED | 0.94 |
| Door_state_decision_heuristic | 1 | B | 3361445 | COMPLETED | 0.96 |
| Door_state_decision_heuristic | 2 | B | 3361446 | SUBMITTED | - |
| Door_state_decision_heuristic | 3 | B | 3361447 | COMPLETED | 1.0 |
| Door_state_decision_heuristic | 4 | B | 3361448 | COMPLETED | 0.98 |
| Door_state_decision_heuristic | 5 | B | 3361449 | COMPLETED | 0.96 |
| Door_state_vlm_off | 1 | B | 3361450 | COMPLETED | 0.87 |
| Door_state_vlm_off | 2 | B | 3361451 | COMPLETED | 0.99 |
| Door_state_vlm_off | 3 | B | 3361452 | COMPLETED | 0.99 |
| Door_state_vlm_off | 4 | B | 3361453 | COMPLETED | 0.92 |
| Door_state_vlm_off | 5 | B | 3361454 | SUBMITTED | - |

