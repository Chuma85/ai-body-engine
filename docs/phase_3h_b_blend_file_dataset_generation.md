# Phase 3H-B Blender Blend-File Dataset Generation

Phase 3H-B adds a supported `.blend` source mode for synthetic Body AI measurement dataset generation. The workflow opens a prepared Blender scene, requires named front/side/back cameras, renders PNG images, and writes training labels plus metadata.

## Blend File Location

The current local default is:

```text
assets/body_meshes/base_body_scene.blend
```

The CLI also accepts an explicit path, so a future canonical location such as this can be used without code changes:

```text
assets/body-ai/blender/base_body_scene.blend
```

Large mesh and `.blend` files remain local assets by repository policy. Do not commit licensed or generated body meshes unless the asset license and storage policy explicitly allow it.

## Required Blender Scene Contract

The `.blend` scene must include:

- One or more human mesh objects.
- `FrontCam`, `SideCam`, and `BackCam` camera objects, unless overridden with the CLI camera-name options.
- Camera framing set correctly in Blender. The workflow verifies camera presence, but visual orientation and framing still need render QA.

If any required camera is missing, the Blender script fails with a clear error naming the missing camera.

## Command

Small sample run:

```powershell
python scripts\generate_blend_dataset.py --source blend --blend-file assets\body_meshes\base_body_scene.blend --out data\synthetic\phase_3h_blend --samples 3 --seed 42 --overwrite
```

Dry run:

```powershell
python scripts\generate_blend_dataset.py --source blend --blend-file assets\body_meshes\base_body_scene.blend --out data\synthetic\phase_3h_blend --samples 3 --seed 42 --dry-run
```

If Blender is not on `PATH`, pass the executable explicitly:

```powershell
python scripts\generate_blend_dataset.py --blend-file assets\body_meshes\base_body_scene.blend --out data\synthetic\phase_3h_blend --samples 3 --seed 42 --blender-executable "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" --overwrite
```

## Output Structure

The generated dataset uses a flat image folder for this blend-file source mode:

```text
data/synthetic/phase_3h_blend/
  images/
    sample_000001_front.png
    sample_000001_side.png
    sample_000001_back.png
  labels.csv
  metadata.json
```

`labels.csv` includes:

- `sample_id`
- `front_image`
- `side_image`
- `back_image`
- `height_cm`
- `chest_cm`
- `waist_cm`
- `hip_cm`
- `shoulder_cm`
- `inseam_cm`
- `source_blend_file`
- `variation_source`
- `camera_set`
- `seed`
- `label_source`
- `synthetic_labels`
- `real_world_validated`

## Labels And Variation

True body measurements are not derived from the `.blend` mesh yet. Labels are generated through the existing synthetic label generator and are marked:

```json
{
  "synthetic_labels": true,
  "real_world_validated": false
}
```

If shape keys exist in the `.blend`, the renderer applies deterministic safe-range shape-key variation and records `variation_source=shape_keys_safe_range`. If no shape keys exist, it still renders the available scene and records `variation_source=static_blend_mesh`.

Static mesh limitation:

```text
TODO: true body shape variation requires shape keys, parametric mesh controls, or multiple body meshes.
```

Optional pose variation is available with `--pose-variation-degrees`, but the default is `0.0` to keep the master scene stable.

## Validation

The wrapper validates:

- The `.blend` file exists and has a `.blend` suffix.
- Blender is available before a real render is launched.
- `labels.csv`, `metadata.json`, and `images/` are created.
- `labels.csv` row count matches `--samples`.
- Every row has existing front, side, and back image paths.
- Labels are marked synthetic and not real-world validated.

Normal unit tests do not require Blender. A full render smoke test should be run locally when Blender is installed.
