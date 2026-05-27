# Phase 3Y Label Geometry Alignment

Phase 3Y audited whether the synthetic labels for chest, waist, hip, and thigh are visibly expressed in the rendered front/side geometry.

## Inputs

- Dataset: `data/synthetic/phase_3t`
- Samples: 1000
- Band feature version: `silhouette_geometry_v6_bands`
- Output artifacts: `artifacts/phase_3y_label_geometry_alignment/`

Artifacts produced:

- `label_geometry_correlations.json`
- `label_geometry_correlations.csv`
- `label_geometry_correlations.md`
- `monotonicity_checks.csv`
- `ambiguity_pairs.csv`
- `deformation_realism_summary.md`
- visual contact sheets under `visual_contact_sheets/`

## Alignment Summary

| Target | Alignment Grade | Dominant Geometry Channel | Best Proxy | Abs Corr | Monotonic |
| --- | --- | --- | --- | ---: | --- |
| chest_cm | good_alignment | front_and_side_combined | `chest_band_03_y40_norm_width_depth_product` | 0.8514 | true |
| waist_cm | good_alignment | side_depth_dominant | `waist_band_01_y46_side_norm_depth` | 0.8238 | true |
| hip_cm | good_alignment | front_and_side_combined | `hip_band_03_y68_norm_ellipse_circumference_proxy` | 0.7502 | true |
| thigh_cm | partial_alignment | front_and_side_combined | `thigh_band_00_y68_norm_width_depth_product` | 0.6293 | true |

Chest, waist, and hip labels are visibly expressed in the geometry according to localized band proxies. Thigh has a weaker but still monotonic signal.

## Bucket Checks

Low/mid/high label buckets generally show increasing geometry proxies for the best local measurements.

Examples:

- Chest: raw ellipse proxy at `chest_band_03_y40` increases from `1.4587` to `1.5042` to `1.5446`.
- Waist: raw ellipse proxy at `waist_band_01_y46` increases from `1.6167` to `1.6479` to `1.6847`.
- Hip: raw ellipse proxy at `hip_band_03_y68` increases from `0.6412` to `0.6594` to `0.6765`.
- Thigh: raw ellipse proxy at `thigh_band_00_y68` increases from `0.6432` to `0.6613` to `0.6727`.

This argues against a total label/geometry disconnect for these targets.

## Ambiguity Findings

The audit still found samples with similar localized geometry but large label differences. Top examples:

| Target | Sample A | Sample B | Label Difference |
| --- | --- | --- | ---: |
| thigh_cm | sample_000175 | sample_000363 | 21.30 |
| thigh_cm | sample_000061 | sample_000592 | 35.10 |
| hip_cm | sample_000566 | sample_000831 | 38.70 |
| waist_cm | sample_000189 | sample_000263 | 34.90 |
| waist_cm | sample_000752 | sample_000770 | 38.00 |

These cases explain why strong correlations do not automatically become low MAE. The labels move geometry in the right direction overall, but there are still collisions where different labels produce similar local silhouettes.

## Visual Diagnostics

Contact sheets were generated for each target:

- low/mid/high label buckets
- ambiguous pairs with similar geometry and different labels
- same-height examples with different target labels

These local artifacts are intentionally not committed.

## Interpretation

Phase 3Y changes the diagnosis slightly from Phase 3X:

- The target labels are not random with respect to rendered geometry.
- Chest, waist, and hip have good monotonic label-to-geometry alignment.
- Thigh alignment is weaker, likely because hip/upper-leg geometry overlaps in the current silhouette.
- The bottleneck is not simply fixed band location.
- The bottleneck is more likely collision/noise in how labels map to visible geometry, plus insufficient independent local deformation.

## Recommendation

The next fix should happen in the synthetic generator or label pipeline, not by adding another generic model.

Recommended next phase:

- Add renderer-side measurement probes for chest, waist, hip, and thigh.
- Compare generated labels against geometry-derived measurements from the rendered mesh or silhouette.
- Consider generating labels from measured geometry instead of only from sampled parameter formulas.
- Strengthen local deformation controls so increasing a label produces a clearer, more independent geometry change at the matching body region.
- Preserve the current modeling baselines as benchmarks until label/geometry collisions are reduced.
