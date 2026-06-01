# Phase 6F-Design-E: Renderer Provider Selection Plan

## Current State

Phase 6F-Design-D1 established the graphical fitting asset pipeline foundation:

- Graphical fitting asset contracts exist through `FittingPreviewAsset`, `FittingAssetManifest`, `FittingAssetGenerationRequest`, and `FittingAssetGenerationResult`.
- FashionApp can consume fitting asset metadata from `fitting_preview_assets` and `asset_manifest`, while older clients can continue using the legacy preview references.
- The `demo_synthetic` renderer exists and returns deterministic `demo://` metadata for safe beta preview flows.
- `local_placeholder`, `external_renderer`, and `unavailable` provider modes exist as contract-level modes, with unsupported real renderer paths degrading to structured unavailable metadata.
- No validated production-grade renderer exists yet.
- No production-grade cloth simulation, real-world fit accuracy, or validated customer measurement outcome claim is enabled by this phase.

Phase 6F-Design-E decides the real renderer path before implementation. The decision must keep the demo/synthetic mode working, avoid real user scan/photo-derived assets until privacy gates are approved, and preserve FashionApp's ability to display beta preview metadata safely.

## Renderer Options Comparison

| Option | What It Can Do | Implementation Complexity | Infrastructure Cost | Mobile Usefulness | Accuracy Risk | Privacy/Storage Risk | Beta Suitability | Production Suitability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2D silhouette/garment overlay renderer | Generate a body-profile or mannequin-referenced 2D preview with garment concept layers, color direction, and simple fit/ease annotations. Can produce SVG/PNG-style concept assets or metadata for FashionApp. | Low to moderate. Can reuse body profile measurements, mannequin references, design option metadata, and existing asset manifest contracts. | Low. Can run in-process or as a lightweight job with static asset output. | High. Fast to render, easy to cache, small files, and predictable on mobile. | Moderate if users interpret it as fit proof. Must remain clearly labeled as beta concept visualization. | Low when using synthetic/mannequin references. Higher only if real scan/photo-derived silhouettes are introduced, which is not approved yet. | Strong. Best first beta path because it creates useful visual feedback without pretending to solve cloth physics. | Limited by itself. Useful as a preview and maker communication aid, not a validated fit engine. |
| Three.js/WebGL mannequin snapshot renderer | Render a mannequin/design concept snapshot from body/mannequin references and simple garment layers. Can support richer camera angles and asset generation for mobile preview. | Moderate. Requires geometry assets, browser/server snapshot handling, deterministic camera/framing, and asset capture. | Low to moderate. Client-side render is cheap; server snapshot adds queue/storage needs. | High if snapshots are pre-generated; medium if relying on live client rendering across devices. | Moderate to high if mannequin/body alignment is overread. Does not validate garment pattern or fabric behavior by default. | Moderate. Geometry/mannequin assets and generated thumbnails need access control if user-specific. | Good as Stage 2 after the 2D concept renderer, especially for more expressive previews. | Moderate only after QA, asset governance, and body/mannequin alignment validation. Not enough for production-grade fitting claims alone. |
| Server-side Blender renderer | Generate higher-fidelity mannequin snapshots, procedural body renders, and future garment concept scenes using existing Blender knowledge in the repo. | High. Needs render job orchestration, container/runtime support, timeout handling, deterministic assets, and monitoring. | Moderate to high. CPU/GPU render workers, storage, and queueing are likely needed. | High if it outputs mobile-ready PNG/WebP thumbnails; live rendering is too slow for mobile UX. | High if positioned as fit simulation. Blender snapshots can look convincing without proving fabric or measurement accuracy. | Moderate to high. Generated user-specific assets need signed URLs, retention, deletion, and access controls. | Good as a higher-fidelity beta step after lightweight concepts prove the flow. | Potentially useful for production preview assets after validation, but still not a validated cloth simulation by itself. |
| External cloth simulation provider | Use a specialized try-on, cloth simulation, or rendering service for garment/body previews. | High. Requires vendor evaluation, API integration, data mapping, credential management, legal/privacy review, and failure handling. | Variable to high. Usually usage-based and may need premium compute. | High if provider returns optimized mobile assets. | Very high until body profile accuracy, garment pattern data, fabric metadata, and provider outputs are validated against outcomes. | High. External transfer of body, scan, garment, or derived assets requires privacy, retention, deletion, and contract approvals. | Weak for the initial beta. It is too risky before internal contracts and validation data mature. | Possible later only after provider validation, privacy approval, QA, and outcome feedback. |
| Hybrid staged approach | Start with safe 2D/mannequin concepts, add Three.js or Blender snapshots, then evaluate validated cloth simulation/provider options when the data and governance gates exist. | Moderate overall because each stage is scoped and reversible. | Starts low and grows only when the product signal justifies it. | High. Mobile gets stable preview assets early and better fidelity later. | Managed. Each stage keeps disclaimers and avoids claiming more than it can prove. | Managed. Demo assets can stay `demo://`; private generated assets move to signed URLs when needed. | Strongest. It balances product value, engineering risk, and beta safety. | Best long-term path because production readiness can be gated by evidence instead of renderer ambition. |

## Recommended Beta Renderer Path

Use a hybrid staged approach.

### Stage 1: 2D Silhouette/Garment Overlay Or Mannequin/Design Concept Renderer

Purpose:

- Provide a mobile-friendly graphical preview for FashionApp.
- Create safe beta concept visualization using body profile/mannequin references and selected design option metadata.
- Generate preview asset metadata, and optionally SVG/PNG concept assets, through the existing `FittingAssetGenerationResult` and manifest contract.
- Keep `demo_synthetic` available for deterministic tests and demo flows.
- Avoid any production-grade fit claim.

Stage 1 should not store or display real user scan/photo-derived assets. It should use safe mannequin references, body-profile metadata, synthetic/demo assets, or generated concept assets that do not expose raw scan media.

### Stage 2: Three.js Or Blender Mannequin Snapshot Renderer

Purpose:

- Improve visual richness with mannequin-based garment previews.
- Produce generated preview images and thumbnails for mobile.
- Preserve the current renderer provider boundary so FashionApp still consumes the same asset metadata contract.
- Keep warnings, confidence metadata, quality metadata, and beta disclaimers attached to every generated asset.

Three.js is preferable when fast iteration and lightweight snapshots are enough. Blender is preferable when the product needs more controlled lighting, camera, or batch-rendered asset generation. Both remain beta visualization paths until validated.

### Stage 3: Validated Cloth Simulation Or External Provider

Only evaluate cloth simulation or an external renderer provider after:

- Body profile accuracy improves and is validated against manual measurement comparison.
- Garment pattern data exists and maps cleanly to renderer inputs.
- A maker feedback loop exists for fit and construction review.
- Privacy, storage, retention, and deletion gates are approved.
- Fit/render QA data exists for visual quality and outcome tracking.

Until those gates pass, cloth simulation/provider output must remain beta/internal-preview only and must not be described as validated real-world fit accuracy.

## Real Render Service Contract

A future renderer provider should implement the existing `RendererProvider.generate(request)` boundary and return `FittingAssetGenerationResult`.

Expected input contract:

- `bodyProfileSnapshot`: the body profile snapshot ID, measurement result reference, measurement summary, morphology tags, validation flags, and safety warnings.
- `mannequinRef`: the mannequin reference ID or generated mannequin asset reference associated with the body profile.
- `selectedDesignOption`: the selected `GeneratedDesignOption`, including style description, garment details, fit direction, color direction, and asset references.
- `garmentMetadata`: garment type, layer references, fabric direction when known, color/material concept metadata, and any future pattern or construction references.
- `renderMode`: requested preview kind and renderer provider mode, such as `mannequin_design_concept`, `fit_summary_overlay`, beta placeholder, lightweight concept, or higher-fidelity snapshot.
- `safetyMetadata`: maker review requirement, beta disclaimer, real-world validation flag, and whether real scan media is allowed. For this phase, real scan media is not allowed.

Expected output contract:

- `FittingAssetGenerationResult` with `request_id`, `preview_kind`, `render_status`, `renderer_provider`, warnings, errors, confidence metadata, quality metadata, and beta disclaimer.
- `FittingAssetManifest` with asset IDs, body profile reference, mannequin reference, design option reference, garment layer references, warnings, provider mode, and render status.
- Preview image and thumbnail metadata using `image_url`, `thumbnail_url`, `asset_uri`, and `manifest_uri`.
- Warnings and confidence metadata that state beta limitations, whether a renderer was validated, whether real scan media was used, and whether maker review is required.
- A beta disclaimer that makes clear the preview is not production-grade cloth simulation and is not a real-world fit accuracy claim.

## Storage And Privacy Gates

Generated preview assets need a storage and CDN strategy before real renderer output is broadly enabled:

- Generated private assets should use signed URLs rather than public permanent URLs.
- Asset manifests should distinguish demo assets, generated mannequin/concept assets, and any future private user-derived assets.
- Real scan/photo-derived assets cannot be stored, displayed, transferred, or rendered until retention, deletion, consent, and access-control policy is verified.
- Demo and synthetic assets can remain `demo://` while the beta renderer contract is validated.
- Provider credentials, storage credentials, and CDN signing secrets must stay outside the repo.
- Render logs should avoid raw scan/photo payloads and should redact provider request data where user-specific body details are not needed for debugging.

## Validation Gates Before Production-Grade Fitting Claims

The product must keep "not production-grade cloth simulation" language until validation data supports a narrower, audited claim.

Before any production-grade fitting or real-world fit accuracy claim, the workflow needs:

- Manual measurement comparison across representative body profiles.
- Maker review feedback on garment fit, ease, construction feasibility, and required alterations.
- Garment outcome feedback after real fittings, returns, adjustments, or maker corrections.
- Visual render QA covering body/mannequin alignment, garment placement, pose/framing, clipping, layer order, colors, and thumbnails.
- Error tracking for renderer failures, provider timeouts, missing asset references, storage failures, and unavailable render states.
- Customer approval tracking that stays separate from maker production approval.
- Evidence that privacy, retention, deletion, consent, and storage policies are operating correctly for private generated assets.

Approval of a visual preview must not automatically start production. Maker review and approved measurement handling remain required.

## Implementation Recommendation

Recommended next implementation phase:

**Phase 6F-Design-F1: Beta 2D Mannequin/Garment Concept Renderer in ai-body-engine**

This future phase should:

- Generate safe preview assets or SVG/PNG concept metadata through the existing fitting asset pipeline.
- Use `BodyProfileSnapshot` and `mannequin_reference_id` without storing raw real scan/photo assets.
- Use selected `GeneratedDesignOption` metadata, including garment details, fit direction, color direction, and asset references.
- Return preview asset metadata to FashionApp through `FittingAssetGenerationResult` and `FittingAssetManifest`.
- Preserve `demo_synthetic`, `local_placeholder`, `external_renderer`, and `unavailable` provider modes.
- Keep the experience beta/disclaimer-based.
- Avoid validated fit, measurement accuracy, or cloth simulation claims until the validation gates above pass.

## Remaining Gaps Before Real Validated Graphical Fitting

- Durable asset storage, CDN delivery, and signed URL support for generated preview images.
- Renderer job orchestration, retries, monitoring, timeout handling, and structured failure metadata.
- Privacy approval for user-derived scan/photo/mannequin assets, including retention and deletion workflows.
- Garment metadata beyond high-level design options, especially pattern geometry, fabric behavior, construction constraints, and maker adjustments.
- Body/mannequin alignment QA and traceable validation against measurements and maker-reviewed outcomes.
- Customer approval and maker production review flows that prevent visual preview approval from being treated as production approval.
