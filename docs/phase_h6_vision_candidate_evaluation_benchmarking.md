# Phase H.6 Vision Candidate Evaluation And Benchmarking

Phase H.6 evaluates the Phase H.5 vision multimodal candidate against the production-style verified baseline and the Phase H.2 metadata candidate. It writes a promotion recommendation, but it does not promote any model, replace production inference, or change live API behavior.

## Entry Point

```bash
python -m training.evaluate_vision_candidate_model \
  --dataset <verified-dataset-root> \
  --metadata-candidate-model <metadata-candidate-dir>/model.json \
  --vision-candidate-model <vision-candidate-dir>/vision_model.json \
  --output <evaluation-output-dir>
```

The evaluator loads:

- verified dataset records
- H.2 metadata candidate artifact
- H.5 vision candidate metadata and weights
- H.4 multimodal tensors for front, side, and back images

The same deterministic split is used for all comparisons.

## Benchmark Methodology

The benchmark reports MAE in centimeters for:

- production-style baseline estimator
- metadata candidate
- vision multimodal candidate

Metrics include overall MAE and per-measurement MAE for:

- `chest_cm`
- `waist_cm`
- `hip_cm`
- `shoulder_cm`
- `sleeve_cm`
- `inseam_cm`
- `neck_cm`

The baseline is a verified train-split mean estimator. It is a comparison anchor, not a production promotion artifact.

## View Contribution Analysis

View contribution is measured by masking image branches during evaluation:

- front only
- front + side
- front + side + back

The metadata branch remains enabled so the report isolates the incremental effect of adding side and back image branches to the already available metadata context.

`backViewHelped` is true when the front + side + back test MAE is lower than front + side test MAE.

## Ablation Analysis

Ablations are evaluation-time branch masks; no retraining occurs:

- `metadata_only`: image branches zeroed, metadata enabled
- `images_only`: image branches enabled, metadata zeroed
- `images_metadata`: all branches enabled

This shows whether the candidate is relying more on image tensors, metadata, or their fused representation.

## Confidence Calibration

The confidence report groups actual test error by exported confidence tier:

- high confidence
- medium confidence
- low confidence
- unknown

The calibration gate checks whether available confidence buckets are monotonic, meaning higher confidence should not have higher actual error than lower confidence buckets.

## Leakage Audit

The vision leakage audit blocks promotion if:

- final approved measurements are configured as input features
- correction deltas are configured as input features
- customer, maker, final, or lineage values appear as input features
- the model does not report `pixelsConsumed: true`
- test errors are suspiciously low

Final approved measurements remain labels only.

## Split Integrity

The split audit checks that these identifiers do not cross train, validation, and test splits:

- `profileId`
- `scanSessionId`
- `orderId`

Duplicate identifiers across splits block promotion.

## Promotion Gate Logic

Possible recommendations:

- `promote_candidate`
- `do_not_promote`
- `needs_more_data`
- `leakage_risk`
- `regression_detected`
- `confidence_not_calibrated`

Promotion eligibility requires all of the following:

- vision candidate beats the production baseline
- vision candidate beats the metadata candidate
- no leakage risk
- valid split integrity
- no per-measurement regression
- enough test records
- confidence calibration is acceptable

The evaluator is intentionally conservative. Any blocker prevents promotion.

## Output Artifacts

- `vision_candidate_evaluation_metrics.json`
- `vision_candidate_benchmark_report.md`
- `vision_ablation_report.json`
- `vision_view_contribution_report.json`
- `vision_confidence_calibration_report.json`
- `vision_promotion_recommendation.json`

## Next Steps

H.7 should use these reports to decide whether the vision candidate deserves deeper evaluation on a larger verified holdout set. Promotion still requires a separate explicit phase with production artifact handling, deployment review, and live inference compatibility checks.
