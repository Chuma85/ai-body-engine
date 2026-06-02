# Phase 6F-Design-G: Asset Storage And Signed Preview URL Plan

## Current Asset State

The current ai-body-engine graphical fitting path supports beta concept metadata, not real stored render output.

- concept:// beta assets exist for the `beta_2d_concept` renderer.
- Demo/synthetic metadata exists through `demo://fitting-preview/...` and synthetic design option references.
- No real generated SVG, PNG, image, or render asset storage is active yet.
- No real user scan/photo asset is uploaded or stored by this phase.
- FashionApp mobile can display beta concept metadata safely through `concept://` and beta warning fields.
- Current graphical fitting output remains internal-preview/beta and is not production-grade cloth simulation.

## Future Real Asset Flow

The intended future real asset flow is:

1. A renderer generates SVG, PNG, or render preview output from an approved body/mannequin profile and selected garment design.
2. ai-body-engine stores the generated asset in private object storage.
3. The storage layer returns a private object key.
4. ai-body-engine stores the object key and asset metadata as permanent truth.
5. ai-body-engine creates a signed, expiring preview URL only when an authorized app/user context requests a preview.
6. FashionApp receives the signed URL or safe asset metadata.
7. Mobile displays the preview only when the asset is safe, approved for the viewer, and still inside its URL expiry window.
8. A retention/deletion job removes expired, deleted, or policy-revoked assets.
9. Maker/customer approval gates remain active and do not imply production readiness.

## Storage Provider Options

| Provider | Private bucket support | Signed URL support | Lifecycle/deletion support | Cost/complexity | Beta suitability |
| --- | --- | --- | --- | --- | --- |
| AWS S3 | Strong private bucket and IAM support | Mature presigned URL support | Mature lifecycle rules, object versioning, and deletion APIs | Moderate cost, higher IAM complexity | Strong choice when AWS is already part of the stack |
| Cloudflare R2 | Private buckets with S3-compatible access | S3-compatible signed URL patterns and public/private controls | Lifecycle and deletion support, with operational details to validate | Often lower egress cost, moderate integration complexity | Recommended beta candidate for S3-compatible storage with cost control |
| Supabase Storage | Private buckets and auth-aware access patterns | Signed URL support through Supabase APIs | Deletion APIs and policy-driven access; lifecycle automation depends on setup | Low to moderate complexity if Supabase is already used | Good beta choice for app teams already using Supabase |
| Local development storage | Local/private by process convention only | Can stub signed URL metadata but should not be treated as real signing | Manual cleanup or local test cleanup only | Lowest cost and complexity | Suitable only for synthetic/demo assets and tests |
| Future CDN/private media service | Depends on provider | Usually supports signed cookies, signed URLs, or tokenized URLs | Depends on provider | Higher integration and vendor review complexity | Useful later when preview delivery needs production media controls |

## Recommended Beta Storage Path

Use a private S3-compatible bucket for beta storage planning, preferably Cloudflare R2 or AWS S3 after vendor/security review.

- Object keys should be scoped by environment, session id, and asset id.
- Signed URLs should use short expiry windows.
- User-derived assets must never be placed in a public bucket.
- Local/dev storage is allowed only for synthetic/demo assets.
- Real provider credentials must stay outside the repo and outside logs.
- `concept://` remains the beta concept mode until real generated assets are approved.

Example object key shape:

```text
{environment}/design-sessions/{sessionId}/fitting-assets/{assetId}.{extension}
```

## Asset Metadata Contract

Future stored fitting assets should carry metadata fields like:

- `assetId`
- `storageProvider`
- `storageBucket`
- `objectKey`
- `contentType`
- `byteSize`
- `checksum` or hash when available
- `signedUrl`
- `signedUrlExpiresAt`
- `createdAt`
- `retentionExpiresAt`
- `deletionStatus`
- `usesRealScanMedia`
- `privacyGateStatus`
- `rendererProvider`
- `renderStatus`
- `previewKind`
- `warnings`

The Python schema foundation adds snake_case contract fields for these same concepts through `FittingAssetStorageMetadata`, `SignedPreviewUrl`, `AssetDeletionState`, `PrivacyGateStatus`, `StorageProviderKind`, and `StoredFittingAssetReference`.

Storage metadata is optional on `FittingPreviewAsset` and `FittingAssetManifest`. This keeps beta `concept://` and demo metadata outputs valid without requiring storage.

## Signed URL Safety Rules

- Signed URLs must expire.
- Signed URLs are not stored as permanent truth.
- Object keys are stored; signed URLs are regenerated as needed.
- Never log signed URLs.
- Never expose bucket credentials.
- Signed URLs are returned only to an authorized app/user context.
- Signed URLs should use short expiry windows for preview use.
- No real user-derived render assets are allowed until consent, privacy, retention, deletion, and access-control gates pass.
- The current planner can produce deterministic signed URL metadata stubs only; it does not generate real provider signatures.

## Retention/Deletion Policy

- Demo assets can be ephemeral.
- Beta user-derived assets require an explicit retention policy before storage.
- A deletion request must remove stored objects, not only application metadata.
- Generated assets should be linked to scan, session, fitting result, design option, and production brief ids where applicable.
- Deletion status must be auditable through states such as `active`, `delete_requested`, `deleted`, and `deletion_failed`.
- Failed deletion must be reported and retried.
- Expired assets should be removed or made inaccessible according to the retention policy.

## Privacy Gates

- Real scan/photo-derived assets are blocked until approval.
- `usesRealScanMedia=false` for the beta concept-only renderer.
- `usesRealScanMedia=true` requires consent, retention policy, deletion policy, and access controls.
- No public URLs are allowed for real user-derived assets.
- Maker review may see preview summaries without exposing unauthorized scan/photo media.
- Real scan/photo media is not used by `beta_2d_concept` or `concept://` outputs.

## FashionApp Integration Expectations

- FashionApp should display signed `http(s)` image URLs only when provided and safe.
- `concept://` remains beta concept mode.
- Signed URLs should be treated as expiring.
- Mobile should handle expired URLs and request a refresh later.
- Maker review should show preview summary and warnings without exposing unauthorized scan/photo media.
- FashionApp should preserve beta disclaimers and maker review warnings next to any graphical fitting preview.

## Next Implementation Phase

Recommended next phase:

Phase 6F-Design-H1 - Storage Abstraction + Signed URL Contract Stub.

That phase should:

- Add a storage provider interface.
- Add a local/dev storage stub.
- Add a signed URL result model or response wrapper.
- Keep real provider credentials outside the repo.
- Generate signed URL metadata only for synthetic/demo assets first.
- Add refresh behavior for expired signed URLs.
- Add deletion audit events before real user-derived SVG/PNG/render assets are stored.

## Explicit Safety Position

This phase is a storage/signed URL plan and safety contract foundation only. It does not upload real files, does not connect to AWS S3, Cloudflare R2, Supabase Storage, or a CDN, does not commit secrets, and does not enable production-grade cloth simulation or real-world fit accuracy claims.
