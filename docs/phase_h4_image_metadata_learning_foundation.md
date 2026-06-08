# Phase H.4 Image And Metadata Learning Foundation

Phase H.4 adds the image ingestion and multimodal dataset foundation needed for a future image+metadata Body AI training phase.

This phase does not train a model, promote a model, replace production, modify live inference, or change API behavior.

## Image Ingestion Architecture

The entrypoint is `training.datasets.multimodal_verified_dataset.MultimodalVerifiedDataset`.

It builds on the H.1 verified dataset loader and adds:

- `ImageResolver`
- `ImagePreprocessor`
- view-aware multimodal sample assembly
- image dataset readiness reporting

The H.1 loader is used in inspection mode so missing or unreadable images can be reported instead of aborting the entire dataset.

## Image Resolution

`ImageResolver` supports:

- local image paths
- storage keys when a local `storage_root` mapping is supplied
- future signed URL references

Signed URLs are recognized but not downloaded in H.4. This avoids network access and keeps this phase as dataset preparation only.

## Image Validation

For each required view, the dataset validates:

- front image exists
- side image exists
- back image exists
- image reference points to a file
- image can be opened by Pillow

Validation tracks:

- missing images
- broken references
- unresolved storage keys
- future signed URL references
- unreadable images

## Image Preprocessing

`ImagePreprocessor` performs:

- EXIF orientation normalization
- RGB conversion
- aspect-ratio preserving resize with padding
- float tensor conversion
- optional normalization to `0.0..1.0`

Preprocessing metadata records:

- original image size
- processed image size
- source mode
- processed mode
- resize mode
- normalization range
- aspect-ratio policy
- orientation normalization status

## View-Aware Multimodal Dataset Schema

Each multimodal sample keeps front, side, and back views separate:

```json
{
  "frontImage": {},
  "sideImage": {},
  "backImage": {},
  "poseMetadata": {},
  "validationMetadata": {},
  "verificationMetadata": {},
  "confidenceMetadata": {},
  "eligibilityMetadata": {},
  "finalApprovedMeasurements": {},
  "lineage": {
    "ai_estimate": {},
    "customer_edit": {},
    "maker_adjustment": {},
    "final_approved": {}
  },
  "correctionDeltas": {},
  "readinessState": "multimodal_ready"
}
```

The views are not treated as interchangeable. Lineage is preserved and no training target is flattened into image input features.

## Image Dataset Report

`write_report(output_dir)` writes:

- `image_dataset_report.json`

The report includes:

- image coverage
- missing image counts
- unreadable image counts
- view distribution
- preprocessing success rate
- broken reference count
- readiness counts
- dataset readiness

## Readiness States

- `metadata_only`: one or more required images are missing, unresolved, unreadable, or not preprocessed.
- `image_ready`: required images preprocess successfully, but metadata or lineage is incomplete.
- `multimodal_ready`: front, side, and back images preprocess successfully and required metadata, final approved measurements, and lineage are present.

## Known Limitations

- No vision model is trained in H.4.
- Signed URL downloads are not implemented.
- Storage keys require an explicit local storage-root mapping.
- Preprocessing prepares in-memory tensors; it does not write preprocessed image artifacts.
- The preprocessing policy is intentionally simple and should be revisited before GPU training.
- Dataset privacy, consent, retention, duplicate-subject, and holdout rules remain required before large-scale image training.

## Readiness For H.5

H.5 can use this foundation to:

- decide the final tensor shape and normalization policy
- add a persisted image cache if needed
- combine image tensors with metadata features
- train an image+metadata candidate model
- keep the production model and live API unchanged until an explicit promotion phase

## Verification

```powershell
python -m pytest tests/test_multimodal_verified_dataset.py
python -m pytest
git diff --check
```
