# Presentation component factorial experiment

## Question

Determine whether the candidate 2.3.0 affinity and processing components improve
held-out presentation performance independently or only in combination, before
spending the full release-training budget.

## Frozen design

- Fit the existing presentation logistic combiner separately for each cell of a
  2 x 2 factorial: public/new affinity crossed with public/new processing.
- Use the same presentation training rows, filtering, ordering, seed, and fitting
  code in all four cells.
- Cache affinity, with-flank processing, and no-flank processing features with
  input and predictor hashes. Do not recompute mixed cells: assemble them from
  the two pure public/public and new/new held-out score passes.
- Preserve row-level held-out predictions for both with-flank and no-flank modes,
  per-sample and per-length metrics, fitted coefficients, provenance, and plots.
- The directional screen uses the available 8-network affinity-control ensemble.
  It is not a substitute for evaluating the final 35-architecture affinity grid.
- The directional processing candidate is a fixed, equal-network ensemble of the
  representative legacy 5-aa and boundary models. The public no-flank predictor
  remains fixed in this first screen so that the processing contrast isolates the
  with-flank architecture.

## Candidate choice and anti-overfitting rule

Use a fixed 50/50 legacy/boundary mixture; do not tune mixture weights on the
held-out comparison cohort. Choose the boundary radius before presentation
evaluation using the already recorded processing results. Prefer radius 5 for
precision-recall performance, unless the equal mixture introduces a material
controlled AUROC or concordance regression relative to legacy alone; in that
case use radius 4.

## Decision gate

A component advances only if its factorial main effect improves macro AUPRC and
macro PPV@N in the relevant presentation mode without a meaningful macro AUROC,
micro-metric, per-length, or per-sample regression. A beneficial interaction may
advance the new/new stack even when one main effect is neutral. If a candidate
regresses, retain the corresponding public component as the safe fallback.

The final release decision still requires the full 2.3.0 affinity and processing
grids, an untouched confirmatory cohort, external-tool joins, and calibration.
This screen is for architecture and resource allocation, not a release claim.
