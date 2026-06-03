# Phase 3H-C Blend Dataset Audit

Phase 3H-C adds a quality-control audit for Blender-generated synthetic Body AI datasets. It does not train a model. The audit exists to catch unusable renders, camera mistakes, missing files, schema drift, and label distributions that are too weak for training.

## Generate A Small Dataset

Use the Phase 3H-B blend-file generation workflow:

```powershell
python scripts\generate_blend_dataset.py --source blend --blend-file assets\body_meshes\base_body_scene.blend --out data\synthetic\phase_3h_blend --samples 3 --seed 42 --blender-executable "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" --overwrite
```

The generated labels are synthetic and not real-world validated. The current `.blend` reports `variation_source=shape_keys_safe_range` when shape-key variation is active. If a future scene reports `variation_source=static_blend_mesh`, treat it as a warning that the dataset is not suitable for serious training variation yet.

## Run The Audit

```powershell
python scripts\audit_blend_dataset.py --dataset data\synthetic\phase_3h_blend --out artifacts\phase_3h_blend_audit --expected-samples 3
```

Strict mode:

```powershell
python scripts\audit_blend_dataset.py --dataset data\synthetic\phase_3h_blend --out artifacts\phase_3h_blend_audit --expected-samples 3 --strict
```

Non-strict mode reports warnings but only fails when the dataset is unusable. Strict mode also fails on blank renders, required schema problems, all-identical important measurement columns, and near-identical front/side/back views.

## Audit Outputs

The audit writes:

```text
artifacts/phase_3h_blend_audit/
  audit_report.json
  audit_summary.md
  sample_contact_sheet.png
  label_distribution_summary.csv
  flagged_samples.csv
```

`audit_report.json` is the machine-readable result. `audit_summary.md` is the human-readable checkpoint. `sample_contact_sheet.png` shows front, side, and back views side by side for several samples. `label_distribution_summary.csv` contains min, max, mean, and standard deviation for key measurements. `flagged_samples.csv` lists per-sample image or view issues.

## What The Audit Checks

The audit validates:

- Dataset folder, `labels.csv`, `metadata.json`, and `images/` exist.
- Every row has front, side, and back image paths.
- Referenced image files exist and can be opened as PNGs.
- Image dimensions are consistent.
- Renders are not blank, near-blank, extremely dark, or washed out.
- A simple non-background pixel heuristic sees a body-like foreground.
- Front, side, and back views differ enough to catch camera duplication.
- Required label columns exist.
- Important measurement columns are numeric.
- Important measurements have variation across samples.
- Safe measurement ranges are respected.
- Metadata includes `variation_source` and `shape_key_count`.

## Interpreting Warnings

Warnings indicate risk, not always failure. Examples:

- `static_blend_mesh`: render output exists, but shape variation is not active.
- `near_identical_views`: cameras may be duplicated or pointed at the same view.
- `missing_body_silhouette`: the body may be too faint, too small, or absent.
- Low or zero label variation: training would likely learn weak or misleading relationships.

## What Blocks Training

Do not train on the dataset when:

- Required files or image paths are missing.
- `labels.csv` or `metadata.json` schema is invalid.
- Front, side, and back views are near-identical.
- Blank or unreadable renders are present.
- Important measurement columns are all identical.
- `variation_source=static_blend_mesh` for a dataset intended to teach body-shape variation.
- Labels are treated as real-world validated. They are still synthetic/generated unless a future validation phase proves otherwise.
