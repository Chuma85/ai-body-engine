# Phase 3V Target-Specific Measurement Strategy

Phase 3V separates measurements by learnability instead of relying on one global MAE. This follows Phase 3U, which showed that some labels have visible silhouette signal while others are weakly aligned with the rendered front/side geometry.

## Why Global MAE Is Misleading

The same model output contains different measurement types:

- Torso and limb circumference targets that are at least partly visible in front/side silhouettes.
- Length or landmark-like targets that are weakly represented by current silhouette features.
- User-known or calibration-dependent values such as height.
- Weight, which is a mass proxy rather than a direct tailoring measurement.

A single global MAE can improve or regress for reasons that do not reflect product readiness. Phase 3V therefore reports target groups separately.

## Target Groups

Config: `training/configs/target_strategy_phase_3v.json`

| Group | Targets | Product Interpretation |
| --- | --- | --- |
| `silhouette_learnable` | chest_cm, waist_cm, hip_cm, thigh_cm, shoulder_cm, calf_cm | Can be modeled from front/side silhouettes, but still needs manual confirmation above 3 cm MAE. |
| `landmark_or_proportion_required` | height_cm, inseam_cm, neck_cm, sleeve_cm | Requires explicit landmarks, proportions, better render-label alignment, or manual confirmation. |
| `manual_or_user_input` | height_cm | Ask the user or use calibrated metadata instead of inferring from current silhouettes. |
| `mass_proxy_uncertain` | weight_kg | Use only as a coarse fit proxy; not a final tailoring measurement. |

## Grouped Benchmark Results

Artifacts were written under `artifacts/phase_3v_target_specific_strategy/`.

| Group | Best Run | Group MAE | Gate |
| --- | --- | ---: | --- |
| all_targets | phase_3l_clean_ridge | 6.5780 | global_mae_mixes_target_types |
| silhouette_learnable | phase_3t_raw_scale_camera__ridge | 5.3132 | research_only |
| landmark_or_proportion_required | phase_3l_clean_ridge | 6.4949 | landmark_or_manual_required |
| manual_or_user_input | phase_3l_clean_ridge | 6.7535 | manual_or_user_input_required |
| mass_proxy_uncertain | phase_3l_clean_ridge | 12.7613 | coarse_proxy_only |

The most important split is that the silhouette-learnable group performs better than the global average, but still does not meet the 3-5 cm assisted/manual-confirmation gate. This means the current silhouette baseline is useful for research prioritization, not final measurements.

## Per-Target Recommendations

| Target | Best Run | Best MAE | Gate | Recommendation |
| --- | --- | ---: | --- | --- |
| height_cm | phase_3l_clean_ridge | 6.7535 | manual_or_user_input_required | Use explicit user input or calibrated metadata. |
| weight_kg | phase_3l_clean_ridge | 12.7613 | coarse_proxy_only | Use only as a coarse fit/body-shape proxy. |
| chest_cm | phase_3t_raw_scale_camera__ridge | 5.4918 | research_only | Visually learnable, but not reliable enough yet. |
| waist_cm | phase_3t_raw_scale_camera__ridge | 5.9736 | research_only | Visually learnable, but not reliable enough yet. |
| hip_cm | phase_3t_raw_scale_camera__ridge | 6.8213 | research_only | Visually learnable, but not reliable enough yet. |
| shoulder_cm | phase_3l_clean_ridge | 3.2922 | assisted_manual_confirmation | Candidate for assisted sizing with manual confirmation. |
| inseam_cm | phase_3n_camera_jitter_only__combined_hybrid_without_area_ratios__random_forest | 7.0061 | landmark_or_manual_required | Needs landmarks/proportion strategy or manual confirmation. |
| sleeve_cm | phase_3n_background_only__raw_scale_camera__elasticnet | 4.9167 | landmark_or_manual_required | Needs landmarks/proportion strategy or manual confirmation. |
| neck_cm | phase_3n_camera_jitter_only__combined_hybrid_without_area_ratios__random_forest | 4.8304 | landmark_or_manual_required | Needs landmarks/proportion strategy or manual confirmation. |
| thigh_cm | phase_3t_raw_scale_camera__ridge | 5.7711 | research_only | Visually learnable, but not reliable enough yet. |
| calf_cm | phase_3t_raw_scale_camera__ridge | 4.3981 | assisted_manual_confirmation | Candidate for assisted sizing with manual confirmation. |

## Product Behavior

Recommended current behavior:

- Treat all generated predictions as research-only by default.
- For shoulder and calf, allow assisted-sizing experiments only with explicit manual confirmation.
- For chest, waist, hip, and thigh, continue improving visual signal before product exposure.
- For height, ask the user or use calibrated capture metadata.
- For inseam, sleeve, and neck, do not present final estimates until a landmark/proportion strategy exists.
- For weight, avoid presenting it as a tailoring measurement; treat it as a coarse body-shape proxy if used at all.

## Technical Recommendation

The next phase should not simply train a larger all-target CNN. The evidence points toward a target-specific measurement stack:

- Keep a silhouette model for chest, waist, hip, thigh, shoulder, and calf.
- Add landmark/proportion supervision for inseam, sleeve, neck, and height-related reasoning.
- Add explicit user height or camera calibration metadata.
- Evaluate silhouette-learnable targets independently from landmark/manual targets.

The benchmark anchor remains Phase 3L clean ridge for global MAE, while Phase 3T raw-scale Ridge is the best current silhouette-group candidate.
