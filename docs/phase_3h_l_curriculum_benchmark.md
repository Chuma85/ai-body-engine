# Phase 3H-L Curriculum Training Benchmark

Phase 3H-L benchmarks clean-only, mobile-realism-only, and mixed clean plus mobile-realistic synthetic training against the same Phase 3H-J mobile-realistic holdout set.

This phase does not run Blender, does not generate new PNG images, and does not modify the existing datasets. It remains synthetic-only and does not claim real-world validation.

## Dataset Inputs

| Dataset | Path | Labels | PNGs | Role |
| --- | --- | ---: | ---: | --- |
| Clean synthetic | `data/synthetic/phase_3h_i_coupled_1000` | `1000` | `3000` | clean-only training and mixed training |
| Mobile-realistic synthetic | `data/synthetic/phase_3h_j_mobile_realism_1000` | `1000` | `3000` | mobile-only training, mixed training, and mobile holdout evaluation |

Both datasets have `labels.csv`, `metadata.json`, and `images/front`, `images/side`, `images/back`.

## Benchmark Setup

Curriculum manifests are built by:

```bash
python scripts/build_phase_3h_k_curriculum_manifest.py
```

Benchmark command:

```bash
python scripts/run_phase_3h_l_curriculum_benchmark.py
```

Output:

```text
artifacts/phase_3h_l_curriculum_benchmark
```

The benchmark reuses the existing Phase 3H image-feature baseline logic and current supported regressors:

- `ridge`
- `random_forest`
- `knn`

No unrelated model architecture is introduced. The current pipeline does not support true sequential fine-tuning for these scikit-learn baselines, so the mixed curriculum is implemented as a mixed retraining benchmark.

## Evaluation Split

The Phase 3H-K manifests define:

- clean train: `1000` Phase 3H-I rows
- mobile train: `800` Phase 3H-J rows
- mixed train: `1800` rows
- mobile holdout evaluation: `200` Phase 3H-J rows

Every strategy is evaluated against the same `200`-row mobile-realistic holdout manifest.

## Results

| Strategy | Training Rows | Best Model | Mobile Holdout Overall MAE |
| --- | ---: | --- | ---: |
| clean-only | `1000` | `random_forest` | `2.3202 cm` |
| mobile-realism-only | `800` | `ridge` | `1.8606 cm` |
| mixed curriculum | `1800` | `ridge` | `1.8633 cm` |

Mobile-realism-only is the best strategy on the mobile holdout. Mixed curriculum is very close, trailing mobile-only by `0.0027 cm`, and improves over clean-only by `0.4569 cm`.

## MAE By Target

| Target | Clean Only | Mobile Only | Mixed Curriculum |
| --- | ---: | ---: | ---: |
| `height_cm` | `3.2921` | `2.9206` | `2.9128` |
| `chest_cm` | `2.5102` | `1.8536` | `1.8583` |
| `waist_cm` | `2.6851` | `1.9717` | `2.0470` |
| `hip_cm` | `2.1462` | `1.7300` | `1.7010` |
| `shoulder_cm` | `1.3367` | `0.9465` | `0.9378` |
| `inseam_cm` | `1.9510` | `1.7411` | `1.7230` |

Mixed curriculum improves over mobile-only on height, hip, shoulder, and inseam. It is slightly worse on chest and waist, with the largest target regression on waist at `+0.0753 cm`. No target worsens significantly using the `0.25 cm` guardrail.

## Comparison Against Phase 3H-I And Phase 3H-J

Existing full-dataset benchmarks:

- Phase 3H-I clean benchmark overall MAE: `1.7486 cm`
- Phase 3H-J mobile-realism benchmark overall MAE: `1.9239 cm`

Phase 3H-L evaluates a different question: how training sources perform on a fixed mobile-realistic holdout. On that holdout:

- clean-only training is least robust: `2.3202 cm`
- mobile-only training is best: `1.8606 cm`
- mixed curriculum is almost tied with mobile-only: `1.8633 cm`

This means Phase 3H-I remains the strongest clean synthetic benchmark, but Phase 3H-J-style data is necessary for mobile-holdout robustness.

## Interpretation

Clean-only training transfers poorly to mobile-realistic holdout images because it does not see camera distance, framing, lighting, background, and small body-yaw perturbations during training.

Mobile-only training performs best on the mobile holdout because train and evaluation distributions match. Mixed training is nearly identical overall and may be preferable when the next model needs to preserve clean synthetic behavior while adding mobile robustness. The slight waist/chest regression should be monitored in the next iteration.

## Recommended Next Step

Use mobile-realism-only as the current best mobile-holdout baseline, but keep mixed curriculum as the safer next-training candidate if the next milestone requires performance across both clean and mobile-realistic distributions.

The next benchmark should evaluate both:

- clean holdout performance, to ensure the mixed strategy does not forget clean synthetic geometry;
- mobile holdout performance, to preserve the Phase 3H-L robustness result.

Before any product-facing accuracy claims, add a small real mobile-photo validation set. This phase remains synthetic-only with `synthetic_labels=true` and `real_world_validated=false`.

## Artifacts

Generated benchmark artifacts are ignored/local and should not be committed:

- `artifacts/phase_3h_l_curriculum_benchmark/metrics.json`
- `artifacts/phase_3h_l_curriculum_benchmark/comparison.csv`
- `artifacts/phase_3h_l_curriculum_benchmark/summary.json`
