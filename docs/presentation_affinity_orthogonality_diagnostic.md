# Presentation affinity/processing orthogonality diagnostic

## Question

Does the new affinity predictor contain held-out presentation signal that is
conditional on both the exact public affinity predictor and the selected new
processing predictor?

## Frozen design

- Reuse the exact training rows and held-out predictions from the completed
  presentation component factorial.
- Fit one unmodified scikit-learn `lbfgs` logistic regression using three
  features: public affinity score, new affinity score, and new processing
  score.
- Use no feature selection, coefficient constraints, blend search, or
  hyperparameter search.
- Compare against public affinity plus new processing, new affinity plus new
  processing, the fixed 10/18 + 8/18 affinity-score blend, and exact public
  2.2.
- Preserve fitted coefficients, full joinable row-level predictions, overall,
  per-sample, and per-length metrics, score correlations, and input hashes.

## Interpretation

This is a conditional-information diagnostic, not a release model: the current
`Class1PresentationPredictor` format accepts one affinity feature. A positive
new-affinity coefficient and held-out improvement would justify preserving both
affinity families in the full candidate or extending the model format. Failure
to improve would argue that the screening affinity ensemble is redundant for
presentation even if it improves affinity-only metrics.
