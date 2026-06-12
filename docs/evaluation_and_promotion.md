# Evaluation And Promotion

Evaluation and promotion are separate gates.

Completing training creates an `evaluation_pending` model. It never creates a production model automatically.

## Evaluation Reports

`reports/evaluation_report.json` tracks:

- MAE
- RMSE
- confidence metrics
- measurement-specific accuracy
- benchmark comparison
- regression analysis
- previous production comparison

The report includes:

- `promotion_gate.auto_promoted: false`
- `promotion_gate.requires_explicit_approval: true`

## Promotion Decisions

Promotion decision statuses are:

- `candidate`
- `approved_for_production`
- `rejected`
- `archived`

Approval requires a reviewer identity. Approval sets model status to `approved`; it does not replace production.

## Production Activation

Production activation requires:

- an existing model
- an approved promotion decision for that model
- an explicit promotion call

The previous production model is retained in `previous_production_models`.

## Rollback

Rollback requires a target model already present in `previous_production_models`.

Rollback records:

- source production model
- restored model
- reviewer/operator
- timestamp
- reason

Rollback updates production tracking but preserves promotion and rollback history.
