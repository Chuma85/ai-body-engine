# Phase 6F-Design-D1: Graphical Fitting Asset Pipeline Foundation

## Product Flow

Phase 6F-Design-D1 adds the backend foundation for graphical fitting preview assets after the Phase 6F design-agent workflow:

1. A body scan creates measurements, a body profile snapshot, and a mannequin reference through the existing Body AI measurement path.
2. The design agent creates or refines structured garment design options.
3. A fitting preview request starts the graphical fitting asset pipeline for a selected design option.
4. The renderer/asset pipeline generates preview asset metadata and an asset manifest.
5. FashionApp mobile displays a returned preview asset when available, or a beta placeholder/pending/failed state when a render is not available.
6. Customer approval and maker review remain required before any production handoff.

This phase is a contract and safe demo/synthetic pipeline foundation. It does not claim final production-grade cloth simulation, real-world fit accuracy, or validated production rendering.

## Pipeline Architecture

The intended internal architecture is:

- Body/mannequin profile adapter: converts `BodyProfileSnapshot`, measurement result references, scan references, and mannequin references into renderer-safe metadata.
- Garment design option adapter: maps a selected `GeneratedDesignOption` and its asset references into garment layer references.
- Fitting asset request builder: creates a `FittingAssetGenerationRequest` with body, mannequin, design option, preview kind, provider mode, and beta safety metadata.
- Renderer provider interface: exposes a narrow `generate(request)` boundary so renderer implementations can be swapped without changing the API response contract.
- Demo/synthetic renderer provider: returns deterministic `demo://` asset metadata and manifest content for safe beta testing without real scan media.
- Future real renderer provider: can call a validated external/local renderer once privacy, storage, deletion, and accuracy gates are complete.
- Asset metadata store/manifest: groups preview assets, body/design references, warning metadata, quality metadata, and renderer status in a `FittingAssetManifest`.
- Fitting preview result enrichment: attaches `fitting_preview_assets` and `asset_manifest` to the existing `VirtualFittingResult` while preserving legacy summary fields.

## Asset Contract

The fitting asset contract is represented by `FittingPreviewAsset`, `FittingAssetManifest`, `FittingAssetGenerationRequest`, and `FittingAssetGenerationResult`.

Core fields include:

- `asset_id`
- `asset_type`
- `preview_kind`
- `render_status`
- `renderer_provider`
- `image_url` or `asset_uri`
- `thumbnail_url`
- `manifest_uri`
- `body_profile_ref`
- `mannequin_ref`
- `design_option_ref`
- `garment_layer_refs`
- `warnings`
- `confidence_metadata`
- `quality_metadata`
- `beta_disclaimer`
- `created_at`

The manifest groups asset IDs, embedded asset metadata, body/mannequin/design references, warnings, renderer status, provider mode, and beta disclaimer language so FashionApp can consume one stable metadata object.

## Renderer Provider Modes

Supported provider modes are:

- `demo_synthetic`: deterministic safe beta mode. It returns `demo://fitting-preview/{sessionId}/{optionId}` references, a manifest, and warnings that the result is beta-only and maker-reviewed.
- `local_placeholder`: local/generated placeholder mode for environments that want placeholder metadata while avoiding external render calls.
- `external_renderer`: reserved for a future validated renderer provider. Current unsupported use should degrade to a structured unavailable result.
- `unavailable`: failure/unavailable mode. It returns structured warnings/errors and no generated image assets.

Supported render statuses are `pending`, `generated`, `failed`, `unavailable`, and `demo_placeholder`.

Supported preview kinds include `fit_summary_overlay`, `mannequin_design_concept`, and `beta_placeholder`.

## Beta Safety Constraints

Phase 6F-Design-D1 explicitly preserves these constraints:

- No production-grade cloth simulation claim is made.
- No real user scan/photo use is allowed until privacy, storage, and deletion gates pass.
- Graphical preview output can remain beta/demo until a renderer is validated.
- Maker review remains required before pattern, cutting, or production decisions.
- Customer approval does not automatically start production.
- Preview assets are metadata contracts in this phase; they are not proof of real-world measurement accuracy.

## FashionApp Integration Expectations

FashionApp should consume fitting preview asset metadata and not assume a final image render always exists.

Mobile should:

- Show the beta disclaimer on fitting preview panels.
- Handle `pending`, `failed`, `unavailable`, and `demo_placeholder` render statuses.
- Render a returned preview asset when `image_url` or `asset_uri` is available.
- Fall back to a beta placeholder panel when only manifest/status metadata is available.
- Keep customer approval and maker review as separate required steps.

Web remains compatible because the existing fitting preview endpoint and result fields are preserved. New clients can read `fitting_preview_assets` and `asset_manifest`; older clients can continue reading `preview_asset_references`, `fit_summary`, caution notes, and confidence metadata.

## Remaining Gaps Before Validated Graphical Fitting

- Add durable asset metadata persistence and signed/CDN URL strategy.
- Add explicit consent, privacy, storage, retention, and deletion gates for real scan/photo media.
- Integrate a validated renderer provider with monitored failures and provider credentials outside source control.
- Validate garment geometry, body/mannequin alignment, fabric behavior, and measurement tolerances against reviewed data.
- Add maker-facing review workflows in FashionApp before any production handoff.
- Define an API versioning policy once external FashionApp clients depend on asset metadata fields.

