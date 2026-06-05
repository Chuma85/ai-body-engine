# Phase 3H-K Clean vs Mobile-Realism Curriculum Strategy

Phase 3H-K compares the clean safer synthetic Phase 3H-I dataset against the mobile-realistic Phase 3H-J dataset and proposes a cautious curriculum strategy for the next model iteration.

This phase is analysis and manifest planning only. It does not generate Blender images, does not delete or modify datasets, and does not claim real-world validation.

## Dataset Inputs

| Dataset | Path | Labels | PNGs | Metadata | Views |
| --- | --- | ---: | ---: | --- | --- |
| Clean coupled synthetic | `data/synthetic/phase_3h_i_coupled_1000` | `1000` | `3000` | present | front, side, back |
| Mobile-realistic synthetic | `data/synthetic/phase_3h_j_mobile_realism_1000` | `1000` | `3000` | present | front, side, back |

Both datasets are synthetic-only. Phase 3H-J keeps `synthetic_labels=true` and `real_world_validated=false`.

## Audit Summary

| Dataset | Strict Audit | Warnings | Errors | Flagged Samples | Clipped Views |
| --- | --- | ---: | ---: | ---: | ---: |
| Phase 3H-I clean | passed | `0` | `0` | `0` | `0` |
| Phase 3H-J mobile realism | passed | `0` | `0` | `0` | `0` |

The available audit reports show both datasets are usable for synthetic benchmarking. The mobile-realism dataset is harder but not invalid: clipping remains at zero and the strict audit passes.

## Correlation Comparison

| Target | Phase 3H-I Clean | Feature | Phase 3H-J Mobile | Feature |
| --- | ---: | --- | ---: | --- |
| `height_cm` | `0.4036` | `front_projection_column_height_std` | `0.3558` | `front_projection_column_height_std` |
| `chest_cm` | `0.4720` | `side_projection_column_height_mean` | `0.3564` | `mean_projection_column_height_mean` |
| `waist_cm` | `0.3926` | `front_vertical_projection_bin_01` | `0.2595` | `mean_raw_mask_area_ratio` |
| `hip_cm` | `0.5607` | `side_horizontal_projection_bin_02` | `0.4305` | `back_crop_offset_y` |
| `shoulder_cm` | `0.5468` | `front_vertical_projection_bin_03` | `0.4995` | `front_neck_width_ratio` |
| `inseam_cm` | `0.4108` | `front_projection_column_height_std` | `0.3630` | `front_torso_area_ratio` |

The mobile-realism features are weaker for every target, especially waist and hip. This is expected because camera distance, framing shift, lighting, background tone, and small body yaw perturb the silhouette-derived features that the current ridge baseline relies on.

## Benchmark Comparison

| Target | Phase 3H-I MAE | Phase 3H-J MAE | Delta |
| --- | ---: | ---: | ---: |
| `height_cm` | `3.0953` | `3.2723` | `+0.1771` |
| `chest_cm` | `1.5471` | `1.7146` | `+0.1675` |
| `waist_cm` | `1.4792` | `1.9691` | `+0.4899` |
| `hip_cm` | `1.6474` | `1.7190` | `+0.0716` |
| `shoulder_cm` | `0.9236` | `0.9750` | `+0.0515` |
| `inseam_cm` | `1.7992` | `1.8934` | `+0.0941` |

| Dataset | Best Model | Overall Mean MAE |
| --- | --- | ---: |
| Phase 3H-I clean | `ridge` | `1.7486 cm` |
| Phase 3H-J mobile realism | `ridge` | `1.9239 cm` |

Phase 3H-J is worse by `+0.1753 cm` overall. It remains within the intended `1-3 cm` synthetic benchmark range, but it should be treated as a harder robustness set rather than a replacement for clean training data.

## Interpretation

Mobile realism is harder because it intentionally weakens several assumptions in the current image-feature baseline:

- framing jitter changes raw body scale and crop offsets;
- camera height and body yaw perturb silhouette proportions;
- lighting and background variation reduce contrast stability;
- safer framing makes the body occupy less of the full image;
- the current feature stack is still silhouette-heavy and therefore sensitive to small camera/view changes.

The clean Phase 3H-I dataset preserves stronger label-to-feature correlations. The mobile Phase 3H-J dataset is better for robustness pressure and mobile capture readiness, but training only on the harder distribution costs accuracy in this synthetic benchmark.

## Recommended Curriculum

Use a staged curriculum instead of replacing clean data with mobile-realistic data immediately:

1. Stage 1: train the baseline on clean synthetic Phase 3H-I to learn stable shape-key label geometry.
2. Stage 2: fine-tune or retrain with a mixed manifest containing all clean Phase 3H-I rows plus the Phase 3H-J mobile-realistic training subset.
3. Stage 3: evaluate on a held-out Phase 3H-J mobile-realistic subset to measure robustness to phone-like capture variation.
4. Stage 4: prepare a small real mobile-photo validation set later before making production or real-world accuracy claims.

This strategy preserves the clean signal while adding mobile robustness pressure. The next model iteration should report both clean synthetic MAE and mobile-realistic holdout MAE.

## Manifest Outputs

The optional manifest builder writes CSV/JSON references only. It does not copy PNGs.

```bash
python scripts/build_phase_3h_k_curriculum_manifest.py
```

Output directory:

```text
artifacts/phase_3h_k_curriculum_manifest
```

Generated files:

- `clean_train_manifest.csv`: all `1000` Phase 3H-I clean rows for Stage 1.
- `mobile_realism_train_manifest.csv`: `800` Phase 3H-J mobile rows for Stage 2 adaptation.
- `mixed_curriculum_manifest.csv`: `1800` rows combining clean Stage 1 rows and mobile Stage 2 rows.
- `evaluation_manifest.csv`: `200` held-out Phase 3H-J rows for Stage 3 evaluation.
- `summary.json`: dataset summaries, manifest paths, row counts, and strategy notes.

The artifact directory is ignored/local and should not be committed.

## Guardrails

- Do not use archived old mannequin datasets.
- Do not claim real-world validation.
- Do not commit generated PNGs, CSV manifests, metadata, dataset folders, benchmark artifacts, or model binaries.
- Do not run Blender or generate new images for Phase 3H-K.
