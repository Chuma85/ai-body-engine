# Phase 3H-H Coupled 250 Retraining Workflow

Phase 3H-H regenerates the 250-sample Blender dataset with the Phase 3H-G shape-key-coupled label formula, then runs strict audit, label-visual correlation audit, and baseline retraining.

The old Phase 3H-D/3H-E outputs remain useful because they provide the before/after comparison: Phase 3H-E measured model performance when labels were weakly tied to rendered shape-key variation.

## Dataset And Artifacts

```text
data/synthetic/phase_3h_h_coupled_250
artifacts/phase_3h_h_coupled_250_audit
artifacts/phase_3h_h_label_visual_correlation
artifacts/phase_3h_h_blend_baseline
```

Generated files are local artifacts and are not committed.

## Commands

Generate:

```powershell
python scripts\generate_blend_dataset.py --source blend --blend-file assets\body_meshes\base_body_scene.blend --out data\synthetic\phase_3h_h_coupled_250 --samples 250 --seed 42
```

Strict audit:

```powershell
python scripts\audit_blend_dataset.py --dataset data\synthetic\phase_3h_h_coupled_250 --out artifacts\phase_3h_h_coupled_250_audit --expected-samples 250 --strict
```

Visual correlation audit:

```powershell
python scripts\audit_blend_label_visual_correlation.py --dataset data\synthetic\phase_3h_h_coupled_250 --out artifacts\phase_3h_h_label_visual_correlation --target-columns height_cm chest_cm waist_cm hip_cm shoulder_cm inseam_cm --min-abs-correlation 0.25
```

Retrain:

```powershell
python scripts\train_blend_dataset_baseline.py --dataset data\synthetic\phase_3h_h_coupled_250 --out artifacts\phase_3h_h_blend_baseline --seed 42 --test-size 0.2 --target-columns height_cm chest_cm waist_cm hip_cm shoulder_cm inseam_cm --strict-audit-required --audit-report artifacts\phase_3h_h_coupled_250_audit\audit_report.json
```

One-step verification:

```powershell
python scripts\verify_phase_3h_h_coupled_250_retrain.py
```

## Interpreting Correlation

Improved label-visual correlation means the deterministic silhouette features now move with synthetic measurement labels. Targets below `0.25` should be considered weakly visually learnable and should block scaling.

## Interpreting MAE

Compare the new baseline against Phase 3H-E:

```text
Phase 3H-E overall mean MAE: 13.1239 cm
```

Lower MAE on the coupled dataset indicates the model can better learn the synthetic label signal. It does not prove real-world accuracy.

## Before Scaling To 1000 Samples

Do not scale until:

- strict audit passes,
- no important target remains below the visual-correlation threshold,
- retraining improves materially versus Phase 3H-E,
- labels remain `shape_key_coupled_synthetic`,
- `synthetic_labels=true` and `real_world_validated=false` remain explicit.

This phase is still synthetic-only and not production tailoring accuracy.
