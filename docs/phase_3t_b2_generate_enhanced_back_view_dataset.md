# Phase 3T-B2: Generate Enhanced Back-View Dataset

## Purpose

Phase 3T-B2 creates a repeatable path for a new enhanced Phase 3T synthetic dataset with front, side, and back views.

- Keep `data/synthetic/phase_3t` intact as the legacy front/side dataset.
- Generate enhanced output into `data/synthetic/phase_3t_enhanced`.
- Use the enhanced dataset later for back-view model/input experiments.
- Front + side remains the minimum legacy scan set.
- Front + side + back is the enhanced scan set.
- This phase does not claim real-world accuracy improvement.

## Target Structure

```text
data/synthetic/phase_3t_enhanced/
  images/
    front/
    side/
    back/
  labels/
    labels.csv
  manifest.csv
```

`data/synthetic/phase_3t_enhanced/` is a local generated artifact path and is ignored by Git.

## Naming Convention

All views for a sample use the same sample id:

```text
sample_000001_front.png
sample_000001_side.png
sample_000001_back.png
```

## Same-Sample Consistency

Front, side, and back images for each `sample_id` must be generated from the same body parameters, morphology, pose, lighting, skin tone, and label row.

Back view must not be generated as an unrelated body variation. The wrapper calls the existing generator once per sample and renders all enabled views from the same `BodyProfile`.

## Generation Commands

Lightweight enhanced dataset generation:

```powershell
python scripts\generate_phase_3t_enhanced_back_view.py --overwrite
```

Tiny smoke generation into a temp or scratch folder:

```powershell
python scripts\generate_phase_3t_enhanced_back_view.py --smoke --output-dir $env:TEMP\ai-body-engine-phase-3t-b2-smoke --overwrite
```

Explicit sample count:

```powershell
python scripts\generate_phase_3t_enhanced_back_view.py --sample-count 1000 --output-dir data\synthetic\phase_3t_enhanced --overwrite
```

Blender config path for future renderer-based generation:

```powershell
blender --background --python synthetic/blender/scripts/render_parametric_body.py -- --config synthetic/blender/configs/phase_3t_enhanced_back_view_config.example.json
```

The Blender config includes `views: ["front", "side", "back"]` and targets `data/synthetic/phase_3t_enhanced`.

## Commit Policy

- Commit scripts/configs/docs/source/tests.
- Do not commit large generated image datasets unless explicitly approved.
- Generated enhanced datasets can remain local artifacts.
- `manifest.csv` and `labels.csv` should be committed only if the generated dataset is intentionally tiny and approved.
- The default enhanced output path is ignored by `.gitignore`.

## Validation Commands

Verify enhanced dataset structure and sample alignment:

```powershell
python -m synthetic.validate_synthetic_dataset --dataset data\synthetic\phase_3t_enhanced
python -m synthetic.build_dataset_manifest --dataset data\synthetic\phase_3t_enhanced --require-back
```

Verify front/side/back image folders exist:

```powershell
Get-ChildItem data\synthetic\phase_3t_enhanced\images\front
Get-ChildItem data\synthetic\phase_3t_enhanced\images\side
Get-ChildItem data\synthetic\phase_3t_enhanced\images\back
```

Verify generated labels and manifest:

```powershell
Get-ChildItem data\synthetic\phase_3t_enhanced\labels\labels.csv
Get-ChildItem data\synthetic\phase_3t_enhanced\manifest.csv
```

Verify legacy front/side compatibility remains documented and supported:

```powershell
python scripts\verify_phase_3t_optional_back_view.py
```

## Expected Metadata

Enhanced `labels.csv` and `manifest.csv` rows should include:

- `front_image_path`
- `side_image_path`
- `back_image_path`
- `has_front=true`
- `has_side=true`
- `has_back=true`
- `capture_views=front,side,back`
- `minimum_scan_views=front,side`
- `enhanced_scan_views=front,side,back`

## Remaining Scope

This phase does not train a new model, update inference to require back view, or claim real-world accuracy. Future phases can compare front/side and front/side/back inputs once the enhanced synthetic dataset is available and validated.
