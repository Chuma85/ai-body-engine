# Phase 3U Measurement Signal Audit

Phase 3U audited whether the Phase 3T rendered front/side images contain enough visible geometry signal to learn the measurement labels.

## Inputs

- Dataset: `data/synthetic/phase_3t`
- Samples: 1000
- Split: 800 train / 100 val / 100 test
- Feature extractor: `silhouette_geometry_v5_hybrid`
- Optional prediction source: `artifacts/deep/phase_3t_dual_branch_augmented/predictions_test.csv`
- Audit output: `artifacts/phase_3u_measurement_signal_audit/`

Generated audit artifacts include:

- `signal_correlations.json`
- `signal_correlations.csv`
- `signal_correlations.md`
- `ambiguous_pairs.csv`
- `per_target_error_analysis.csv`
- `visual_audit_summary.md`
- contact sheets under `contact_sheets/`

## Feature Signal Summary

The strongest correlations show that torso circumference-like targets have learnable image signal, while some length/metadata-like targets do not.

| Target | Max Abs Corr | Weak Signal | Likely Label/Image Mismatch | Best Feature Group | Stronger View |
| --- | ---: | --- | --- | --- | --- |
| chest_cm | 0.8565 | false | false | combined_hybrid | side |
| waist_cm | 0.8290 | false | false | raw_scale_camera | side |
| hip_cm | 0.7725 | false | false | combined_hybrid | side |
| thigh_cm | 0.6603 | false | false | raw_scale_camera | side |
| weight_kg | 0.6451 | false | false | raw_scale_camera | side |
| shoulder_cm | 0.6147 | false | false | raw_scale_camera | front |
| calf_cm | 0.5880 | false | false | raw_scale_camera | side |
| inseam_cm | 0.0962 | true | true | normalized_shape | side |
| height_cm | 0.0936 | true | true | normalized_shape | front |
| neck_cm | 0.0799 | true | true | normalized_shape | front |
| sleeve_cm | 0.0737 | true | true | normalized_shape | front |

The strongest learnable targets are chest, waist, hip, thigh, weight, shoulder, and calf. The weakest targets are height, inseam, neck, and sleeve. These weak targets have no strong relationship to the extracted visible silhouette features, which means the model is likely being asked to infer labels that the current rendering/feature setup does not expose reliably.

## Ambiguity Checks

The ambiguous-pair audit found samples with similar extracted silhouettes but very different labels. Top examples included:

| Target | Sample A | Sample B | Label Difference |
| --- | --- | --- | ---: |
| weight_kg | sample_000012 | sample_000681 | 64.60 |
| neck_cm | sample_000035 | sample_000928 | 18.00 |
| shoulder_cm | sample_000551 | sample_000673 | 21.90 |
| inseam_cm | sample_000300 | sample_000388 | 26.30 |
| sleeve_cm | sample_000031 | sample_000352 | 20.20 |
| waist_cm | sample_000551 | sample_000673 | 47.70 |
| height_cm | sample_000291 | sample_000514 | 50.00 |

These pairs are not proof of wrong labels by themselves, but they show where the visible feature representation cannot separate materially different labels. Height and inseam are especially suspicious because the renderer normalizes framing and full-body visibility, so raw pixel height is not a stable proxy for real height.

## CNN Error Analysis

The optional Phase 3T CNN predictions were available, so the audit also summarized test-set errors:

| Target | CNN MAE | Median Abs Error | Max Abs Error |
| --- | ---: | ---: | ---: |
| weight_kg | 15.6551 | 13.0943 | 46.0030 |
| height_cm | 13.4613 | 12.8512 | 28.0680 |
| waist_cm | 10.3083 | 8.5849 | 38.6289 |
| hip_cm | 8.1786 | 6.6057 | 31.8369 |
| inseam_cm | 7.5757 | 7.3418 | 15.5452 |
| chest_cm | 7.2353 | 6.3493 | 21.8563 |
| thigh_cm | 6.1600 | 5.1151 | 15.7922 |
| sleeve_cm | 6.1329 | 5.7719 | 13.2207 |
| shoulder_cm | 5.1676 | 4.5987 | 15.0417 |
| neck_cm | 5.1645 | 5.3255 | 10.1935 |
| calf_cm | 4.8061 | 4.4960 | 12.2609 |

The CNN struggles most with weight, height, waist, hip, and inseam. Some of those targets have strong silhouette correlations in the hand-engineered features, but height and inseam do not. That suggests the CNN is not only capacity-limited; it is also limited by label-image alignment and the available visual cues.

## Visual Audit Outputs

Contact sheets were generated for:

- lowest/highest waist samples
- lowest/highest chest samples
- lowest/highest hip samples
- lowest/highest inseam samples
- best and worst CNN prediction examples per target

These sheets are local artifacts and intentionally not committed. They should be used for manual inspection before changing labels, renderer deformation, or model architecture.

## Interpretation

Phase 3U points to a measurement-signal bottleneck rather than a simple dataset-size bottleneck.

What looks learnable:

- Chest, waist, hip, thigh, shoulder, calf, and some weight signal are visible in the current front/side silhouettes.
- Side-view features often dominate torso and lower-body targets.
- Front-view features matter most for shoulder.

What looks weak or mismatched:

- Height and inseam are not strongly represented after camera/framing normalization and full-body rendering.
- Neck and sleeve labels appear weakly coupled to visible geometry.
- Ambiguous pairs show that some large label differences are not separable with the current image features.

Likely bottlenecks:

- Some measurement labels are generated independently enough that visible mesh deformation does not fully reflect them.
- Height/inseam/sleeve/neck need either stronger geometric deformation, explicit landmark/pose supervision, or metadata/camera scale calibration.
- CNN performance is unlikely to improve dramatically until those targets have clearer image signal.

## Recommendation

Before a larger CNN phase, add a label-render alignment phase. Recommended next work:

- Audit and tighten the renderer deformation mapping for height, inseam, sleeve, and neck.
- Add explicit scale/calibration metadata if height is expected from images.
- Add target-specific landmark or geometry probes in the renderer for labels that are not visible in silhouettes.
- Keep Phase 3L clean ridge as the current benchmark anchor until the weak targets become visibly learnable.
