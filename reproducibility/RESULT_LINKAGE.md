# Manuscript-to-result linkage

| Manuscript contribution | Canonical files | Convention / scope |
|---|---|---|
| Main RF–MLP–CNN comparison | `results/main_scores_by_repeat.csv`, `results/main_summary.csv` | 663 points, 2 years, 2 targets, 2 protocols; patchCNN = original_fresh; mean of pooled OOF metrics per repetition |
| Same-capacity context / arrangement controls | `results/control_effects.csv`, `experiments/cnn_controls/results/ablation_summary.csv` | Original_fresh vs centre-tiled vs joint-vector shuffle; separate retraining, shared architecture and original-training normalization |
| Buffer / first-three-repetition checks | `experiments/cnn_controls/results/matched_first_three_repeats.csv` | Pair only the shared repetitions; descriptive comparisons |
| Second permutation | `experiments/cnn_controls/results/permutation_sensitivity_seed0.csv` | Shuffle seed 271828 on outer repetition 0 only |
| Target-scale / high-SOC diagnostics | Main repeat-score metric columns, `experiments/main/results/performance_summary.csv`, `paired_changes_vs_naive.csv` | Retransformation is an exploratory RF diagnostic, separate from patch benchmark; main score columns retain the >=150 g/kg diagnostic |
| Conditional loss-gate pilot | `experiments/cnn_controls/loss_gate/paired_deltas.csv`, `scores.csv`, `selected_hyperparameters.csv`, `inner_cv.csv` | Conditional on earlier disjoint A/B/C expert fits; public runner does not refit this pilot |
| Coordinate metadata | `experiments/coordinates/results/summary.json`, `source_manifest.json` | 778 source points, 663 main points, 662 usable GPS; finite recorded GPS precision |
| Paired nominal–GPS refit | `results/gps_summary.csv`, `experiments/gps/results/context_gain_summary.csv` | 662 identical points, frozen outer/inner memberships; sqrt(mean point/repeat squared losses), fixed-fit block bootstrap |
| Main Figures 2 and 3 | `figures/figure2_controls.png`, `figures/figure3_metric_dependence.png` | Saved figures from the clean manuscript draft; no sample-location map included |
| GPS supplementary figure | `experiments/gps/results/SOC_GPS_center_comparison.png` and `.pdf` | 32 paired conditions and conditional 95% block-bootstrap intervals |

Canonical summaries contain numerical source records for manuscript tables; they are not copies of Word table formatting. Main tables and GPS table use different RMSE aggregation conventions and must not be combined without labelling them.
