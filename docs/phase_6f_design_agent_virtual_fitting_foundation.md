# Phase 6F-Design-A: Design Agent and Virtual Fitting Foundation

## Product Flow

Phase 6F-Design-A adds the backend foundation for the post-scan design/styling and remote virtual fitting workflow:

1. A customer scan is completed by the existing Body AI measurement pipeline.
2. Measurements are returned with confidence, source, interval, caveat, and review metadata.
3. A user-specific body profile/mannequin snapshot is available or derived from the measurement result reference.
4. The customer requests AI design/styling help by creating a design session with garment preferences.
5. The design agent generates structured garment design options.
6. The customer iterates by sending refinement prompts against a selected option or session preference set.
7. A virtual fitting preview is requested against the user-specific body/mannequin profile.
8. The customer approves a final design option.
9. A production brief is generated for maker handoff.

This phase implements workflow contracts and a deterministic demo/synthetic execution path. It does not claim production cloth simulation, production garment fit accuracy, or legal/privacy approval for broad live scan usage.

## Core Backend Modules

- Measurement inference: existing measurement contracts remain in `training/measurements/body_ai_inference.py` and `training/measurements/measurement_result_schema.py`.
- Measurement snapshot/body profile: existing snapshot validation remains in `training/measurements/measurement_snapshot_store.py`; the design workflow introduces a lightweight `BodyProfileSnapshot` schema for fitting/design orchestration.
- Design styling agent: `app/services/design_agent_virtual_fitting.py` defines a deterministic `DesignStylingAgent` that accepts user preferences and returns structured design options.
- Virtual fitting engine: `app/services/design_agent_virtual_fitting.py` defines a `VirtualFittingEngine` service boundary that returns a safe beta/demo preview result contract.
- Orchestration layer: `DesignSessionService` creates sessions, generates options, refines options, requests fitting previews, approves final options, and generates production briefs.
- API contracts: `app/schemas/design_agent_virtual_fitting.py` defines Pydantic request/response models and enums for the workflow.
- API routes: `app/api/routes/design_agent_virtual_fitting.py` exposes the phase endpoint family under `/v1/body-ai/design-sessions`.

## Responsibility Split

`ai-body-engine` owns backend contracts and service orchestration for:

- measurement and measurement-result references,
- body profile/mannequin snapshot references,
- design session state,
- design option generation orchestration,
- design refinement orchestration,
- virtual fitting preview orchestration,
- customer approval state,
- production brief generation.

`CUSTOM-FASHION-MARKETPLACE` should own:

- customer, maker, and admin UI workflows,
- authentication and authorization,
- consent and account-level privacy prompts,
- marketplace order state,
- maker review screens,
- asset presentation and customer approval UI,
- calls into the `ai-body-engine` API contracts.

## Beta Safety Language

The fitting preview is beta/internal-preview unless separately production-validated. This foundation intentionally includes warning and disclaimer fields in sessions, fitting results, and production briefs:

- no production accuracy claim is made for virtual fitting previews,
- maker review remains required before production pattern or cutting decisions,
- real user scans must not be used outside the currently approved privacy/storage scope,
- demo/synthetic mode remains supported for safe testing and integration development,
- measurement results retain synthetic-calibrated and real-world-validation metadata from the existing Body AI pipeline.

## Target API Flow

The following endpoint family is implemented:

- `POST /v1/body-ai/design-sessions`
  - Creates a design session from a measurement result reference, inline measurement result payload, or body profile snapshot.
- `POST /v1/body-ai/design-sessions/:id/generate`
  - Generates structured design options from the session preferences and body profile snapshot.
- `POST /v1/body-ai/design-sessions/:id/refine`
  - Creates a refined design option variation from an option and prompt.
- `POST /v1/body-ai/design-sessions/:id/fitting-preview`
  - Creates a beta/demo virtual fitting preview result contract for a chosen design option.
- `POST /v1/body-ai/design-sessions/:id/approve`
  - Approves a selected option and optionally stores maker production notes.
- `GET /v1/body-ai/design-sessions/:id`
  - Fetches current session state.
- `GET /v1/body-ai/design-sessions/:id/production-brief`
  - Returns a structured maker handoff brief after approval.

## Domain Models

The phase introduces the following core models:

- `DesignSession`
- `DesignPreferenceInput`
- `GeneratedDesignOption`
- `DesignRefinementRequest`
- `BodyProfileSnapshot`
- `VirtualFittingRequest`
- `VirtualFittingResult`
- `ProductionBrief`
- `ApprovalState`
- `DesignSessionStatus`

The models support user/session identifiers, scan and measurement references, mannequin/body profile references, garment preferences, generated design metadata, optional asset references, beta fitting preview references, confidence/warning metadata, approval state, and maker production notes.

## Integration Notes

The current implementation uses in-memory design session storage because this phase is a backend/service contract foundation. A persistent repository can be added behind `DesignSessionService` without changing API consumers.

The virtual fitting pathway returns a deterministic synthetic preview asset reference such as `synthetic://virtual-fitting/...`. A future renderer can replace `VirtualFittingEngine.preview` while preserving the `VirtualFittingResult` response shape.

The design agent returns deterministic structured concepts. A future image generation or styling model provider can replace `DesignStylingAgent.generate_options` and `DesignStylingAgent.refine_option` while preserving the session and option contracts.

## Remaining Gaps Before FashionApp Integration

- Persist design sessions and production briefs in a durable store.
- Add authentication, authorization, tenant/user scoping, and audit logging.
- Add formal consent gates for real user scan use in design/fitting previews.
- Connect generated visual assets to an approved storage and CDN strategy.
- Integrate a validated garment render/cloth simulation backend when available.
- Define maker review UI requirements and production-readiness gates in `CUSTOM-FASHION-MARKETPLACE`.
- Add versioned API compatibility policy once external consumers depend on the contracts.

