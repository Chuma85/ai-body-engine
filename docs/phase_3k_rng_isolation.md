# Phase 3K RNG Isolation

Phase 3K separates synthetic body generation randomness from render-realism randomness. This makes future experiments cleaner: a dataset can keep the exact same body measurements while changing lighting, background, camera jitter, and material brightness, or it can change body distribution while holding render style fixed.

## Behavior Added

- `body_seed` controls body-shape profile selection, skin tone ID, pose variation, and measurement values.
- `render_seed` controls render-only realism variation, including background color/brightness, lighting strength, skin-tone brightness multiplier, and camera jitter.
- Existing configs that only define `random_seed` remain compatible. The renderer resolves both RNG streams from `random_seed` unless `body_seed` or `render_seed` is explicitly provided.
- `labels.csv` now records `body_seed`, `render_seed`, `render_realism_enabled`, and `render_realism_version` as optional metadata columns.

The new example config is:

```text
synthetic/blender/configs/phase_3k_rng_isolation_config.example.json
```

## Smoke Verification

Two 5-sample smoke datasets were rendered:

- `data/synthetic/phase_3k_smoke_a`
  - `body_seed=42`
  - `render_seed=314159`
- `data/synthetic/phase_3k_smoke_b`
  - `body_seed=42`
  - `render_seed=271828`

Both validator runs passed:

```text
Valid: True
Samples complete: 5
Front PNGs: 5
Side PNGs: 5
Label rows: 5
```

Measurement-column comparison between smoke A and smoke B:

```text
MeasurementRowsMatch: True
MismatchCount: 0
BodySeeds: 42
RenderSeedA: 314159
RenderSeedB: 271828
```

The first front-image file hashes differed between smoke A and B, confirming render appearance changed while labels stayed fixed.

## Recommendation

Use `body_seed` and `render_seed` in the next realism-enabled dataset phase so image-style ablations do not accidentally change the body-measurement distribution. The next useful step is a matched-control dataset experiment: same `body_seed`, multiple render realism profiles, then compare ridge and CNN behavior on equivalent body labels.
