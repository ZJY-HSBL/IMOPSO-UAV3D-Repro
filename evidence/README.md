# Verified reproduction evidence

**Previous seed-42 unnormalized MC-HV values are superseded.** This folder contains the corrected Eq.(10) normalized results.

## Traceable evidence

- [24 runs: HV, raw objective means, sampled/continuous feasibility](rigorous_three_seed_runs.csv)
- [Paired three-seed differences and exploratory bootstrap intervals](rigorous_paired_comparisons.csv)
- [Frozen calibration bounds and full run protocol](rigorous_protocol.json)
- [Paper Table 2: 32 min/max/mean rows](paper_table2_reference.csv)
- [Direct comparison of all 32 paper means with this reproduction](paper_vs_reproduction_means.csv)
- [Full reproduction audit report](REPRODUCTION_AUDIT.md)
- [Scenario 1 HV (real seed-42 archive)](scenario_1_hv.svg)
- [Scenario 1 paths (real seed-42 archive)](scenario_1_paths.svg)
- [Scenario 2 HV (real seed-42 archive)](scenario_2_hv.svg)
- [Scenario 2 paths (real seed-42 archive)](scenario_2_paths.svg)

The four SVG plots were updated from verified seed-42 NPZ arrays after baseline fixes. They are not synthetic replicas of the figures in the paper.

Three-seed normalized HV means:

| Scenario | IMOPSO | MOPSO | NSGA-II | PESA-II |
|---|---:|---:|---:|---:|
| 1 | 0.84719 | 0.63256 | 0.97617 | 0.85833 |
| 2 | 0.98311 | 0.98333 | 0.97889 | 0.97914 |

Our experiments **do not establish** the claimed universal superiority of IMOPSO, nor do they numerically reproduce the paper's Table 2. The paper does not fully specify required parameters.

The source code and local-download ZIP include the 24 per-run decision/objective archives and HV iteration history, which are the canonical basis for the figures and calculations.
