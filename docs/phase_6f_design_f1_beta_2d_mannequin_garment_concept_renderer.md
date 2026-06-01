# Phase 6F-Design-F1: Beta 2D Mannequin/Garment Concept Renderer

## Purpose

Phase 6F-Design-F1 implements the first real beta visual renderer path for the graphical fitting asset pipeline.

The renderer creates a safe 2D mannequin/garment concept preview that helps a customer understand the selected design direction on a body/mannequin representation. It is mobile-friendly graphical fitting concept metadata, not production-grade cloth simulation, not real-world fitting accuracy, and not a real user scan/photo render.

This phase keeps the existing `demo_synthetic` renderer mode working and adds a distinct `beta_2d_concept` provider mode for beta concept preview assets.

## Input Contract

The beta 2D concept renderer consumes the existing `FittingAssetGenerationRequest` contract:

- Body profile snapshot: `BodyProfileSnapshot` with body profile ID, measurement result reference, measurement summary, morphology tags, validation flags, and warnings.
- Mannequin/body morphology reference: `mannequin_reference_id` and morphology tags from the body profile snapshot.
- Selected generated design option: `GeneratedDesignOption` with title, style description, garment details, color direction, fit direction, and asset references.
- Garment type: derived from design option details such as `Garment type: dress` when present.
- Color palette: derived from `color_direction`.
- Style direction: derived from the selected design option title and style description.
- Fit preference: derived from `fit_direction`.
- Render mode/provider: `renderer_provider=beta_2d_concept` and `preview_kind=beta_2d_mannequin_concept`.

No real user scan/photo media is required or used.

## Output Contract

The renderer returns the existing fitting asset result shape:

- `FittingAssetGenerationResult` with `renderer_provider=beta_2d_concept`.
- At least one `FittingPreviewAsset` containing fitting preview asset metadata.
- An asset manifest through `FittingAssetManifest` with body profile, mannequin, design option, garment layer refs, warnings, confidence metadata, and quality metadata.
- A deterministic `concept://fitting-preview/{sessionId}/{optionId}` asset URI.
- A deterministic thumbnail/preview reference under the same `concept://` namespace.
- `renderStatus=generated`.
- `previewKind=beta_2d_mannequin_concept`.
- Warnings including "Beta 2D concept preview only.", "Not production-grade cloth simulation.", and "Maker review required.".
- The shared beta disclaimer.

The output is concept metadata. It may later be upgraded to SVG/string asset content or server-generated PNG/WebP files, but this phase does not generate fake real rendered clothing.

## Renderer Behavior

The `Beta2DConceptRendererProvider` behavior is deterministic:

- Builds concept preview metadata from the body profile snapshot, mannequin reference, selected design option, garment type, color direction, fit direction, and garment asset references.
- Uses body profile/mannequin proportions when available, including measurement keys such as chest, waist, hip, height, and shoulder width.
- Uses garment design metadata to influence concept labels such as garment type, primary color, style description, and fit direction.
- Returns a `concept://` asset URI instead of an external image URL.
- Marks `uses_real_scan_media` as `False`.
- Marks `validated_renderer` as `False`.
- Includes render notes stating that the concept uses body profile/mannequin metadata only, uses no real user scan/photo media, and performs no cloth physics simulation.

The renderer does not use real user photos, does not run cloth physics, does not store private user image assets, and does not claim real-world fit accuracy.

## FashionApp Compatibility

FashionApp Phase 6F-Design-D2 can consume the returned fitting asset metadata through the existing fields:

- `fitting_preview_assets`
- `asset_manifest`
- `preview_asset_references`
- `renderer_provider`
- `render_status`
- `preview_kind`
- `asset_uri`
- `thumbnail_url`
- `warnings`
- `beta_disclaimer`

Mobile clients should treat `concept://` and `demo://` references as beta concept/demo references. If a reference is not an `http(s)` image URL, the mobile display should show the beta/demo concept panel using the manifest metadata, warnings, and beta disclaimer.

## Future Upgrade Path

The beta 2D concept renderer can evolve in stages:

- Server-side SVG/PNG output generated from the same metadata contract.
- Signed storage URLs for private generated preview assets.
- Three.js or Blender mannequin snapshots for richer visual previews.
- Validated renderer provider integration with monitored failures, provider credentials outside source control, and storage/deletion controls.
- Cloth simulation only after body profile accuracy, garment pattern data, maker feedback, render QA, privacy/storage/deletion gates, and outcome validation are complete.

Until those gates pass, this renderer remains a beta concept preview and must not be described as validated graphical fitting.
