# Phase K.2 Gender-Aware Measurement Targets

Phase K.2 adds the `gender_measurement_schema_v1` measurement schema for male, female, unisex, and unspecified body profiles.

## Profile Types

- `male`
- `female`
- `unisex`
- `unspecified`

## Target Availability

Shared tailoring targets are available for every profile type. Male-specific targets are available only for `male` profiles, and female-specific targets are available only for `female` profiles. Dataset loaders and candidate training skip profile-incompatible target requirements instead of forcing blank male/female-specific fields to exist on every row.

## Expanded Targets

The shared schema includes expanded tailoring fields such as abdomen, stomach, wrist, thigh, knee, calf, ankle, and `sleeve_shoulder_to_wrist_cm`. Female-specific targets include bust, high bust, underbust, bust-point, shoulder-to-bust, and waist-to-hip fields. Male-specific targets include jacket length, trouser rise, and across-back fields.

## Synthetic Generation

The lightweight synthetic generator now emits deterministic male and female body variants, records `gender_measurement_schema_v1`, and writes capture variation metadata for camera angle, camera distance, body rotation, and phone framing.

## Training Boundary

This phase updates schemas, loaders, candidate training, vision training, and evaluation metrics only. It does not promote a model, replace production artifacts, or change live production model selection.
