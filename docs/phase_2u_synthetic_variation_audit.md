# Phase 2U Synthetic Variation Audit

Dataset audited: `data/synthetic/phase_2q`

Sample count: 500

Audit command:

```powershell
python -m synthetic.audit_synthetic_variation --dataset data/synthetic/phase_2q --output artifacts/analysis/phase_2u_variation_audit
```

The audit produced local `summary.json` and `report.md` outputs under `artifacts/analysis/phase_2u_variation_audit`.

## Measurement Ranges

| Measurement | Min | Max | Mean | Std | Range |
| --- | ---: | ---: | ---: | ---: | ---: |
| height_cm | 150.50 | 204.80 | 177.79 | 15.74 | 54.30 |
| weight_kg | 45.40 | 129.70 | 86.22 | 24.05 | 84.30 |
| chest_cm | 75.30 | 129.90 | 101.87 | 15.95 | 54.60 |
| waist_cm | 55.10 | 125.00 | 91.79 | 20.27 | 69.90 |
| hip_cm | 75.30 | 134.80 | 105.49 | 16.85 | 59.50 |
| shoulder_cm | 35.00 | 59.90 | 47.66 | 7.21 | 24.90 |
| inseam_cm | 65.20 | 95.00 | 80.90 | 8.43 | 29.80 |
| sleeve_cm | 50.00 | 75.00 | 62.07 | 7.24 | 25.00 |
| neck_cm | 30.00 | 50.00 | 39.99 | 5.71 | 20.00 |
| thigh_cm | 40.10 | 79.90 | 59.76 | 11.51 | 39.80 |
| calf_cm | 28.00 | 54.90 | 41.00 | 7.74 | 26.90 |

## Audit Findings

No low-variation warnings were emitted.

No outlier warnings were emitted.

No high-coupling correlation warnings were emitted.

No missing measurement fields were found, and no non-numeric measurement values were found.

The current 500-sample dataset has good broad numeric coverage across the configured measurement ranges. This supports the Phase 2T conclusion that the current bottleneck is probably not only model family choice.

## Generation Controls

Phase 2U adds optional renderer `variation_controls` to the Phase 2G rigged mesh config. The controls define named body-shape profiles such as `slim`, `average`, `broad`, and `curvy`, with profile-specific measurement range overrides.

These controls are currently disabled in the example config to preserve existing generation behavior. A future generation phase can enable them to intentionally balance or broaden specific body-shape regions instead of relying only on uniform sampling from global measurement ranges.

## Recommendation

For the next synthetic generation phase, keep the dataset size moderate and enable profile-aware sampling deliberately. Focus on improving target signal for measurements that remained weak in Phase 2T, especially `weight_kg`, `thigh_cm`, `sleeve_cm`, and `neck_cm`. The audit suggests the labels already span wide ranges, so the next gain likely comes from better shape-to-image correspondence, profile balancing, and render diversity rather than simply switching lightweight regressors.
