<!--
Replacement for the deleted Figure 20 (F15_cluster_purity.pdf), per spec item D/Fig-20.
Source: ablations_results/DISTIL_ablation_results.xlsx, sheet D1_Cluster_Purity (study A15
under the canonical renumbering of spec item E4).
Restricted to the three ablation settings (spec item E1). Numbers are copied verbatim from
the workbook; do not alter them.
-->

**Table: Root-cause label purity per geometric cluster (study A15).**
Mean cluster purity is the fraction of a geometric cluster's failures that share the dominant
root cause assigned by the language model. Mean root causes per cluster counts the distinct
root causes present in a cluster. Mean silhouette measures the separation of the geometric
clusters.

| Setting | Mean cluster purity | Mean root causes per cluster | Mean silhouette |
|---|---|---|---|
| GridWorld 5x5 (image) | 0.89 | 1.62 | 0.58 |
| Push-T (state) | 0.91 | 1.35 | 0.64 |
| Door (image) | 0.84 | 1.86 | 0.56 |
