# Phase 3H-M View Ablation Benchmark

## Purpose

Phase 3H-M measures how much the front, side, and back views contribute to the current synthetic body-measurement baseline. It uses the completed mobile-realistic synthetic dataset from Phase 3H-J and compares single-view, two-view, and three-view training strategies against the same deterministic holdout split.

This phase does not run Blender, generate new PNGs, modify datasets, use archived old mannequin datasets, or claim real-world validation.

## Dataset

- Dataset: `data/synthetic/phase_3h_j_mobile_realism_1000`
- Labels: `1000`
- PNGs: `3000`
- Views: `images/front`, `images/side`, `images/back`
- Metadata: `metadata.json`
- Synthetic labels: `true`
- Real-world validated: `false`

## Benchmark Setup

- Script: `scripts/run_phase_3h_m_view_ablation_benchmark.py`
- Output folder: `artifacts/phase_3h_m_view_ablation_benchmark`
- Seed: `42`
- Test size: `0.2`
- Train samples: `800`
- Test samples: `200`
- Models: current baseline regressors from `scripts/train_blend_dataset_baseline.py`
- Best model for every view combination: `ridge`

The benchmark extracts the existing silhouette and projection features for only the selected views in each ablation. It writes metrics and comparison files only under the ignored `artifacts/` tree.

## View Combinations

| View combination | Views |
| --- | --- |
| `front` | front |
| `side` | side |
| `back` | back |
| `front_side` | front, side |
| `front_back` | front, back |
| `side_back` | side, back |
| `front_side_back` | front, side, back |

## Results

| Rank | View combination | Best model | Overall MAE | Delta vs front+side+back | Height | Chest | Waist | Hip | Shoulder | Inseam |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `front_side` | ridge | 1.8900 | -0.0339 | 3.1875 | 1.7222 | 1.8927 | 1.7159 | 0.9946 | 1.8270 |
| 2 | `front` | ridge | 1.9066 | -0.0173 | 3.1338 | 1.8664 | 1.9079 | 1.7338 | 0.9889 | 1.8089 |
| 3 | `front_back` | ridge | 1.9174 | -0.0065 | 3.1903 | 1.8153 | 1.9532 | 1.7148 | 0.9899 | 1.8411 |
| 4 | `front_side_back` | ridge | 1.9239 | 0.0000 | 3.2723 | 1.7146 | 1.9691 | 1.7190 | 0.9750 | 1.8934 |
| 5 | `back` | ridge | 1.9603 | 0.0364 | 3.1038 | 2.0167 | 2.0495 | 1.7505 | 1.0451 | 1.7962 |
| 6 | `side_back` | ridge | 1.9746 | 0.0507 | 3.2389 | 1.8756 | 2.0927 | 1.7629 | 1.0153 | 1.8622 |
| 7 | `side` | ridge | 1.9812 | 0.0573 | 3.1029 | 1.8982 | 2.2035 | 1.8039 | 1.0846 | 1.7942 |

## Back-View Contribution

Adding the back view to `front_side` did not improve overall holdout performance in this benchmark. The overall MAE changed from `1.8900 cm` for `front_side` to `1.9239 cm` for `front_side_back`, a `+0.0339 cm` MAE increase.

Targets improved by adding back to front+side:

- `chest_cm`: `-0.0076 cm`
- `shoulder_cm`: `-0.0196 cm`

Targets worsened by adding back to front+side:

- `height_cm`: `+0.0848 cm`
- `waist_cm`: `+0.0764 cm`
- `hip_cm`: `+0.0031 cm`
- `inseam_cm`: `+0.0664 cm`

Back-only training was not the weakest result, but it ranked fifth overall at `1.9603 cm`. It helped more than side-only for overall MAE, but it did not beat front-only or any front-paired combination.

## Interpretation

For the current silhouette/projection feature baseline, `front_side` is enough for the strongest overall mobile-realistic synthetic holdout result. The back view still carries useful shoulder and chest signal, but the added features appear to introduce enough noise or redundancy to worsen height, waist, and inseam on the ridge baseline.

The result does not mean back photos should be discarded permanently. It means the current baseline feature/model stack does not extract net overall value from back view when front and side are already present. A future learned model, stronger feature selection, target-specific model routing, or confidence-gated back-view features may recover back-view value without hurting the other measurements.

## Recommendation

Use `front_side` as the default lightweight benchmark configuration for the next synthetic-only baseline iteration, while preserving `front_side_back` as the full-data reference and as a candidate for target-specific models.

Recommended next strategy:

- Train the main baseline on front+side features for overall MAE optimization.
- Track a target-specific shoulder/chest experiment that can optionally include back-view features.
- Keep collecting and validating back views in the dataset so later model classes can use them.
- Do not make production or real-world accuracy claims until a small real mobile-photo validation set exists.

## Limitations

- `synthetic_labels=true`
- `real_world_validated=false`
- Results are from synthetic mobile-realistic Blender renders only.
- The benchmark uses the existing deterministic train/test split and current baseline regressors, not a new model architecture.
- No Blender rerendering or new image generation was performed in Phase 3H-M.
