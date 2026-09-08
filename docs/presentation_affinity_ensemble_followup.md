# Presentation affinity-ensemble follow-up

## Question

Does retaining the exact public 2.2 affinity predictor while adding the eight
new affinity-screen networks recover the public predictor's presentation AUPRC
and preserve the new predictor's PPV@N gain when paired with the selected
legacy-plus-radius-5 processing ensemble?

The affinity and processing components represent different biological stages,
but their numerical scores are not assumed to be statistically independent.
This experiment measures complementarity at the end-to-end presentation level.

## Frozen design

- Reuse the presentation training rows, release holdout, component predictions,
  random seed, and selected processing ensemble from the completed 2 x 2
  component factorial.
- Compare three affinity alternatives with the same new processing scores:
  exact public 2.2, the eight-network new screen, and a fixed hybrid of both.
- Define the hybrid at the presentation-feature level as the model-count-weighted
  mean of the two `affinity_score` values: 10/18 public plus 8/18 new. Each
  predictor first resolves a multiallelic example to its own best allele. This
  is therefore a deployable-score screening blend, not numerically identical
  to merging all networks before best-allele resolution. It deliberately avoids
  merging networks trained against incompatible allele pseudosequence tables.
- Fit the ordinary two-feature presentation combiner separately for each
  alternative. Do not tune the public/new ratio or add model-specific features.
- Evaluate with and without flanks against the exact public 2.2 presentation
  predictor, preserving row-level predictions, per-sample and per-length
  metrics, fitted weights, input hashes, and plotting inputs.

## Decision rule

Promote the hybrid over public affinity plus new processing only if it improves
both macro AUPRC and macro PPV@N without a meaningful macro/micro, per-sample,
or per-length regression. A tradeoff is evidence for affinity-model diversity
in the eventual full ensemble, not a reason to tune the blend on this holdout.

This is a screening follow-up. The release decision still requires the full
35-architecture x 4-fold affinity candidate and an untouched final evaluation.
