# DISTIL aggregation summary

Cells found: 10 | results_dir: `distil/results`

## Headline (full DISTIL) SR by cell
| task | modality | SR mean±std | seeds | demos→90% | bridge% |
|---|---|---|---|---|---|
| Door | state | 0.000±0.000 | 1 | None | 1.0 |
| GridWorld | state | 0.759±0.340 | 6 | 34.0 | 0.624 |
| Lift | state | 0.000±0.000 | 1 | None | 1.0 |
| Wipe | state | 0.000±0.000 | 1 | None | 0.0 |

## Ablation deltas vs full DISTIL (Tier-1 knockouts negative-if-important)
| task | modality | ablation | SR | Δ vs full |
|---|---|---|---|---|
| GridWorld | state | allocation_random | 0.912 | +0.153 |
| GridWorld | state | clustering_off | 0.916 | +0.157 |
| GridWorld | state | decision_heuristic | 0.911 | +0.152 |
| GridWorld | state | memory_off | 0.777 | +0.018 |
| GridWorld | state | vlm_off | 0.907 | +0.148 |
| Lift | state | safe | 0.000 | +0.000 |

## Tier-5 statistics (DISTIL full vs best baseline per cell)
- paired cells: 1 | DISTIL wins: 0/1
- sign test (coin-flip null): p ≈ 1.0000

| task | modality | DISTIL | best baseline | win |
|---|---|---|---|---|
| Lift | state | 0.000 | 0.000 | ✗ |