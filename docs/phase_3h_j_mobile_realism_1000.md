# Phase 3H-J Mobile Realism 1000-Sample Blender Dataset

Phase 3H-J is completed locally. It extends the Phase 3H-I safe-framed coupled Blender dataset with conservative mobile-camera realism while preserving synthetic label traceability and front/side/back view separation.

This phase does not claim real-world validation. Generated labels remain synthetic, `synthetic_labels=true`, and `real_world_validated=false`.

## Dataset

Final merged dataset path:

```text
data/synthetic/phase_3h_j_mobile_realism_1000
```

Final structure:

```text
labels.csv
metadata.json
images/front
images/side
images/back
```

Final counts:

- Labels: `1000`
- PNG images: `3000`
- Views: `front`, `side`, `back`
- Metadata: present
- Clipped views: `0`

Generated dataset files are ignored/local and should not be committed.

## Realism Settings

Phase 3H-J keeps the Phase 3H-I coupled shape-key labels and safe framing, then adds deterministic per-sample/per-view mobile capture jitter:

```json
{
  "mobile_realism": true,
  "safe_framing_scale": 1.34,
  "distance_jitter": 0.015,
  "camera_height_jitter": 0.015,
  "body_rotation_jitter": 2.0,
  "lighting_jitter": 0.08,
  "background_jitter": 0.04,
  "phone_framing_jitter": 0.006
}
```

Metadata requirements:

- `mobile_realism=true`
- `variation_source=shape_keys_safe_range_plus_mobile_realism`
- `label_generation_mode=shape_key_coupled_synthetic`
- `label_formula_version=shape_key_coupled_synthetic_v2_wide_safe_range`
- `synthetic_labels=true`
- `real_world_validated=false`

## Recovery Note

The first full clean generation produced all four chunks and merged the final dataset, but chunk `000001_000250` contained flat black renders from sample 58/59 onward. The failed images were detected before benchmark use. The repair replaced only `chunk_000001_000250`; chunks `000251_000500`, `000501_000750`, and `000751_001000` were preserved. The final dataset was then rebuilt from the four completed chunks and validated with `--no-render`.

## Commands

Smoke:

```bash
python scripts/verify_phase_3h_j_mobile_realism_1000.py --samples 25 --batch-size 25 --smoke --force
```

Full clean generation:

```bash
python scripts/verify_phase_3h_j_mobile_realism_1000.py --force --batch-size 250
```

Chunk repair used after the first generated chunk contained invalid flat renders:

```bash
python scripts/generate_blend_dataset.py --source blend --blend-file assets/body_meshes/base_body_scene.blend --out artifacts/phase_3h_j_mobile_realism_generation_batches/chunk_000001_000250 --samples 250 --seed 42 --start-index 1 --blender-executable "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" --shape-key-range 0.24 --label-formula-version shape_key_coupled_synthetic_v2_wide_safe_range --label-measurement-scale 2.0 --safe-framing-scale 1.34 --distance-jitter 0.015 --camera-height-jitter 0.015 --body-rotation-jitter 2.0 --lighting-jitter 0.08 --background-jitter 0.04 --phone-framing-jitter 0.006 --view-subdirs --mobile-realism --overwrite
```

Final merge and benchmark from completed chunks:

```bash
python scripts/verify_phase_3h_j_mobile_realism_1000.py --resume --batch-size 250
```

Final no-render validation:

```bash
python scripts/verify_phase_3h_j_mobile_realism_1000.py --no-render
```

`--no-render` and `--reuse-existing` never call Blender and never regenerate PNGs. They require the final merged dataset path to exist first.

## Smoke Result

- Labels: `25`
- PNG images: `75`
- Metadata: present
- Front/side/back folders: present
- Clipped views: `0`
- Benchmark: skipped

## Full Audit Result

Audit output:

```text
artifacts/phase_3h_j_mobile_realism_1000_audit
```

Final strict audit:

- Strict audit passed: `True`
- Warnings: `0`
- Errors: `0`
- Flagged samples: `0`
- Clipped views: `0`

## Correlation Result

Correlation output:

```text
artifacts/phase_3h_j_mobile_realism_label_visual_correlation
```

Strongest visual correlation by target:

- `height_cm`: `0.3558` via `front_projection_column_height_std`
- `chest_cm`: `0.3564` via `mean_projection_column_height_mean`
- `waist_cm`: `0.2595` via `mean_raw_mask_area_ratio`
- `hip_cm`: `0.4305` via `back_crop_offset_y`
- `shoulder_cm`: `0.4995` via `front_neck_width_ratio`
- `inseam_cm`: `0.3630` via `front_torso_area_ratio`

Weak targets below `0.25`: none.

## Benchmark Result

Benchmark output:

```text
artifacts/phase_3h_j_mobile_realism_blend_baseline
```

Best model: `ridge`

Overall mean MAE: `1.9239 cm`

MAE by target:

- `height_cm`: `3.2723`
- `chest_cm`: `1.7146`
- `waist_cm`: `1.9691`
- `hip_cm`: `1.7190`
- `shoulder_cm`: `0.9750`
- `inseam_cm`: `1.8934`

Train/test split: `800/200`

## Comparison To Phase 3H-I

Phase 3H-I overall mean MAE: `1.7486 cm`

Phase 3H-J overall mean MAE: `1.9239 cm`

Delta: `+0.1753 cm`

Target deltas vs Phase 3H-I:

- `height_cm`: worsened by `0.1771`
- `chest_cm`: worsened by `0.1675`
- `waist_cm`: worsened by `0.4899`
- `hip_cm`: worsened by `0.0716`
- `shoulder_cm`: worsened by `0.0515`
- `inseam_cm`: worsened by `0.0941`

Phase 3H-J is slightly worse than Phase 3H-I, as expected for a harder mobile-realism dataset, but remains in the intended synthetic benchmark range for this phase.

## Limitations

- This remains synthetic-only Blender data and is not real-world validated.
- Mobile realism is conservative: no distracting room objects, clothing, camera blur, device sensor noise, or real user capture artifacts yet.
- The baseline feature pipeline is still silhouette-heavy, so background and framing realism can reduce apparent correlation for some targets.
- Generated PNG/CSV/metadata files and benchmark artifacts remain ignored/local.
