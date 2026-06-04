# Phase 3H-I Coupled 1000 Dataset Workflow

Phase 3H-I scales the successful Phase 3H-H coupled Blender workflow from 250 samples to 1000 samples. It keeps the dataset synthetic-only while widening safe rendered shape-key variation and using a v2 synthetic label scale to reduce low-label-variation warnings.

## Purpose

Scaling from 250 to 1000 samples gives the silhouette baseline a larger train/test split and a better check on whether Phase 3H-G/3H-H shape-key-coupled labels remain learnable beyond a small dataset. Phase 3H-H remains the baseline comparison because it proved the coupled-label workflow on 250 samples.

## Dataset And Artifacts

```text
data/synthetic/phase_3h_i_coupled_1000
artifacts/phase_3h_i_coupled_1000_audit
artifacts/phase_3h_i_label_visual_correlation
artifacts/phase_3h_i_blend_baseline
```

The dataset uses view-specific image folders:

```text
data/synthetic/phase_3h_i_coupled_1000/images/front
data/synthetic/phase_3h_i_coupled_1000/images/side
data/synthetic/phase_3h_i_coupled_1000/images/back
```

Generated dataset files, rendered PNGs, CSV outputs, trained model artifacts, and binary outputs are local artifacts and are not committed.

## Widened Variation

Phase 3H-I keeps `variation_source=shape_keys_safe_range` and widens the renderer shape-key range from the Phase 3H-H default `0.15` to `0.24`. Because labels are normalized against the chosen range, Phase 3H-I also uses:

```text
label_formula_version=shape_key_coupled_synthetic_v2_wide_safe_range
label_measurement_scale=2.0
```

This preserves deterministic shape-key-coupled synthetic labels while increasing label spread enough to make low-label-variation warnings less likely. The measurements remain clamped to the existing safe synthetic ranges.

## Commands

One-step verification, generation, audit, correlation, and retraining:

```powershell
python scripts\verify_phase_3h_i_coupled_1000.py
```

Resume generation without deleting completed batches:

```powershell
python scripts\verify_phase_3h_i_coupled_1000.py --resume --start-index 501 --batch-size 250
```

If an interrupted batch folder exists, replace only the incomplete chunk while preserving completed chunks:

```powershell
python scripts\verify_phase_3h_i_coupled_1000.py --resume --force --start-index 501 --batch-size 250
```

Smoke verification:

```powershell
python scripts\verify_phase_3h_i_coupled_1000.py --samples 25 --batch-size 25 --smoke
```

Smoke mode writes `data/synthetic/phase_3h_i_coupled_smoke`, generates 25 samples and 75 PNGs, validates `labels.csv`, `metadata.json`, and view folders, and skips audit/correlation/training unless `--run-benchmark` is explicitly supplied.

Equivalent generation command:

```powershell
python scripts\generate_blend_dataset.py --source blend --blend-file assets\body_meshes\base_body_scene.blend --out data\synthetic\phase_3h_i_coupled_1000 --samples 1000 --seed 42 --shape-key-range 0.24 --label-formula-version shape_key_coupled_synthetic_v2_wide_safe_range --label-measurement-scale 2.0 --view-subdirs
```

Strict audit:

```powershell
python scripts\audit_blend_dataset.py --dataset data\synthetic\phase_3h_i_coupled_1000 --out artifacts\phase_3h_i_coupled_1000_audit --expected-samples 1000 --strict
```

Visual correlation audit:

```powershell
python scripts\audit_blend_label_visual_correlation.py --dataset data\synthetic\phase_3h_i_coupled_1000 --out artifacts\phase_3h_i_label_visual_correlation --target-columns height_cm chest_cm waist_cm hip_cm shoulder_cm inseam_cm --min-abs-correlation 0.25
```

Retraining:

```powershell
python scripts\train_blend_dataset_baseline.py --dataset data\synthetic\phase_3h_i_coupled_1000 --out artifacts\phase_3h_i_blend_baseline --seed 42 --test-size 0.2 --target-columns height_cm chest_cm waist_cm hip_cm shoulder_cm inseam_cm --strict-audit-required --audit-report artifacts\phase_3h_i_coupled_1000_audit\audit_report.json
```

## Expected Verification

The verifier checks:

- `labels.csv`, `metadata.json`, and `images/front`, `images/side`, `images/back`
- 1000 label rows and 3000 PNGs
- no missing images
- non-identical front/side/back views
- all six measurement targets
- shape-key-coupled metadata and traceability
- weak visual-correlation targets below `0.25`
- low-label-variation warnings separately from strict audit errors
- strict audit pass before benchmark training

## Result

Current local status after smoke verification:

```text
1000-sample final dataset: not merged yet
Completed full batches: 1-250 and 251-500
Interrupted batch: 501-750, partial PNGs only
Smoke dataset: data/synthetic/phase_3h_i_coupled_smoke
Smoke result: 25 labels, 75 PNGs, front/side/back folders verified
Audit result: pending full resumed generation
Benchmark result: pending full resumed generation
Comparison to Phase 3H-H: pending full resumed generation
```

## Limitations

Phase 3H-I is synthetic-only Blender validation:

```text
synthetic_labels=true
real_world_validated=false
```

It does not claim real-world measurement accuracy, production tailoring accuracy, cloth simulation accuracy, or FashionApp try-on readiness.

## Next Recommended Phase

If Phase 3H-I keeps strict audit passing, removes most low-label-variation warnings, avoids weak targets below `0.25`, and improves or holds Phase 3H-H benchmark quality, the next phase should inspect qualitative samples and consider a controlled 1000-sample model-selection or feature-ablation pass before any real-world validation work.
