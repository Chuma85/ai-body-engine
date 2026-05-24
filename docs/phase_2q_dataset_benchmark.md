# Phase 2Q Dataset Benchmark

Phase 2Q scaled the corrected Blender synthetic renderer from the 100-sample Phase 2K dataset to a 500-sample dataset under `data/synthetic/phase_2q`.

## Render

- Command: `blender --background --python synthetic/blender/scripts/render_parametric_body.py -- --config synthetic/blender/configs/phase_2g_rigged_mesh_config.example.json --output data/synthetic/phase_2q --num-samples 500`
- Render time: 1:09:56.8428009, about 69.95 minutes
- Output files:
  - Front PNGs: 500
  - Side PNGs: 500
  - Labels CSV: `data/synthetic/phase_2q/labels/labels.csv`

## Validation

`python -m synthetic.validate_synthetic_dataset --dataset data/synthetic/phase_2q`

- Valid: True
- Samples complete: 500
- Front PNGs: 500
- Side PNGs: 500
- Label rows: 500

## Manifest

`python -m synthetic.build_dataset_manifest --dataset data/synthetic/phase_2q`

- Rows: 500
- Train: 400
- Val: 50
- Test: 50

## Improved Silhouette Baseline

`python -m training.train_image_feature_baseline --dataset data/synthetic/phase_2q --output artifacts/baselines/phase_2q`

- Train overall MAE: 8.2487
- Val overall MAE: 9.3730
- Test overall MAE: 9.5982

## Comparison

`python -m training.analyze_baseline_errors --runs artifacts/baselines/phase_2m artifacts/baselines/phase_2n artifacts/baselines/phase_2p artifacts/baselines/phase_2q --output artifacts/analysis/phase_2q`

- Phase 2Q improves over Phase 2M by 1.5390 test MAE.
- Phase 2Q is 1.2719 test MAE worse than Phase 2P.
- Important caveat: Phase 2M, Phase 2N, and Phase 2P artifacts were benchmarked on the 100-sample Phase 2K split, while Phase 2Q is a new 500-sample dataset and split. The comparison is useful for trend tracking, but it is not a strict same-test-set comparison.
- Phase 2Q has the best reported result for `sleeve_cm` and `neck_cm` among the compared local artifacts.
- Weak targets remain `weight_kg`, `thigh_cm`, and `hip_cm`.

## Visual Spot-Check

Manually inspected:

- `sample_000001_front.png` and `sample_000001_side.png`
- `sample_000250_front.png` and `sample_000250_side.png`
- `sample_000500_front.png` and `sample_000500_side.png`

Notes:

- Front views are square-on, centered, and full-body framed.
- Side views are clean profile views, centered, and full-body framed.
- The neutral separated-arm pose remains consistent with Phase 2J/2P measurement-friendly rendering.

## Test

`python -m pytest`

- Result: 78 passed
