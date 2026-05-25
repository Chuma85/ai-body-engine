# Phase 2V Recovery: Checkpoint-Safe Rendering

Phase 2V attempted to render a controlled 1000-sample dataset under `data/synthetic/phase_2v`. The long Blender process ran for more than two hours and did not finish cleanly. A partial output remained with rendered PNG pairs, but no `labels/labels.csv`.

The missing labels file made the partial dataset invalid: the validator requires every front/side image pair to have a matching label row, and every label row to have matching front/side image files. PNGs without labels cannot be safely consumed by the manifest builder or training loader.

## What Changed

The Blender renderer now writes labels incrementally:

- `labels/labels.csv` is created at the start of rendering.
- A label row is appended immediately after each sample's front and side PNGs are rendered.
- Label writes are flushed and fsynced so interrupted renders preserve completed rows.
- The existing one-shot behavior remains the default for normal runs.

The renderer also supports explicit resume behavior:

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" --background --python .\synthetic\blender\scripts\render_parametric_body.py -- --config .\synthetic\blender\configs\phase_2v_controlled_variation_config.example.json --output .\data\synthetic\phase_2v --num-samples 1000 --resume
```

With `--resume`, the renderer:

- skips samples that already have front PNG, side PNG, and a label row
- backfills a label row for existing front/side PNG pairs that are missing labels
- renders missing samples
- avoids creating duplicate label rows

Batch controls are also available for safer long runs:

```powershell
--start-index 501 --num-samples 500 --append-labels
```

The random number sequence is advanced for skipped indices, so batch rendering can preserve the deterministic sample sequence.

## Smoke Test

A 5-sample recovery smoke render was run:

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" --background --python .\synthetic\blender\scripts\render_parametric_body.py -- --config .\synthetic\blender\configs\phase_2g_rigged_mesh_config.example.json --output .\data\synthetic\phase_2v_recovery_smoke --num-samples 5
```

Validation passed:

```text
Valid: True
Samples complete: 5
Front PNGs: 5
Side PNGs: 5
Label rows: 5
```

A resume smoke on the same output skipped all 5 completed samples and validation still reported exactly 5 label rows.

## Safe Retry Plan

Do not treat the interrupted `data/synthetic/phase_2v` output as complete. Before retrying Phase 2V benchmarking, run the renderer with `--resume` so existing image pairs can be checkpointed into labels and missing samples can be completed. After rendering, run validation before building a manifest or training:

```powershell
python -m synthetic.validate_synthetic_dataset --dataset data/synthetic/phase_2v
```

Only proceed to manifest, audit, and ridge benchmarking after validation reports 1000 complete samples.
