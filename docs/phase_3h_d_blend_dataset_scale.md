# Phase 3H-D Blend Dataset Scale Verification

Phase 3H-D verifies that the `.blend` dataset workflow remains stable beyond the 3-sample smoke run. This phase generates and audits 250 samples only. It does not train a model.

## Why 250 Before 1000

Two hundred fifty samples are large enough to expose repeated render, camera, label, and variation issues while staying small enough to debug if Blender rendering slows down or fails. A clean 250-sample run is a readiness checkpoint before considering a larger 1000-sample generation.

## Dataset

Target output:

```text
data/synthetic/phase_3h_blend_250
```

Expected image count:

```text
250 samples * 3 views = 750 PNG images
```

The generated dataset remains a verification dataset, not the active training dataset. The active training dataset marker remains `data/synthetic/phase_3t`.

## Generation Command

```powershell
python scripts\generate_blend_dataset.py --source blend --blend-file assets\body_meshes\base_body_scene.blend --out data\synthetic\phase_3h_blend_250 --samples 250 --seed 42
```

If Blender is not on `PATH`, pass the executable:

```powershell
python scripts\generate_blend_dataset.py --source blend --blend-file assets\body_meshes\base_body_scene.blend --out data\synthetic\phase_3h_blend_250 --samples 250 --seed 42 --blender-executable "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
```

## Audit Command

```powershell
python scripts\audit_blend_dataset.py --dataset data\synthetic\phase_3h_blend_250 --out artifacts\phase_3h_blend_250_audit --expected-samples 250 --strict
```

## One-Step Verification

```powershell
python scripts\verify_phase_3h_d_blend_dataset_scale.py
```

The verifier discovers Blender from `PATH` or the standard Windows Blender Foundation install folder. It generates the dataset when missing, refuses to overwrite an existing dataset unless `--overwrite` is passed, runs the strict audit, checks all required output files, and prints a summary.

## Expected Outputs

Dataset files:

```text
data/synthetic/phase_3h_blend_250/labels.csv
data/synthetic/phase_3h_blend_250/metadata.json
data/synthetic/phase_3h_blend_250/images/*.png
```

Audit artifacts:

```text
artifacts/phase_3h_blend_250_audit/audit_report.json
artifacts/phase_3h_blend_250_audit/audit_summary.md
artifacts/phase_3h_blend_250_audit/sample_contact_sheet.png
artifacts/phase_3h_blend_250_audit/label_distribution_summary.csv
artifacts/phase_3h_blend_250_audit/flagged_samples.csv
```

## Passing Audit Meaning

A passing strict audit means:

- `labels.csv` and `metadata.json` exist.
- 250 rows are present.
- 750 front/side/back images exist.
- Images open successfully.
- Front, side, and back views differ enough to catch duplicated cameras.
- Important measurement labels are numeric and vary across samples.
- Shape-key metadata is present, with `variation_source=shape_keys_safe_range` and a positive `shape_key_count`.

## What Blocks Training

Do not train from this dataset if strict audit fails, if images are missing or blank, if front/side/back views are near-identical, if important label columns are all identical, or if variation metadata reports `static_blend_mesh` when body-shape variation is required.

This phase is readiness validation only. Model training remains out of scope.

## Manual Contact Sheet Review

Open:

```text
artifacts/phase_3h_blend_250_audit/sample_contact_sheet.png
```

Inspect several rows manually. Each row should show a coherent front, side, and back render for the same sample. Repeated identical views, clipped bodies, blank frames, or washed-out silhouettes should block training until corrected.
