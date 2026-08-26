# Release training recipe compatibility

The MHCflurry 2.3 release recipe uses the published 2.1.x/2.2.x scientific
configuration as its baseline. The 2.1.x and 2.2.x public model downloads use
the same affinity, processing, and presentation generation recipes.

The only training-hyperparameter exception currently supported by direct
held-out evidence is the Class I pan-allele affinity minibatch increase from
128 to 1024. A release-candidate sweep found that 1024 improved affinity
generalization; it is therefore retained. Other prediction-affecting recipe
changes are reverted unless and until they have isolated held-out evidence.

The layer-by-layer comparison, framework-equation audit, discrepancy register,
and controlled experiment plan are in
{doc}`release_neural_hyperparameter_audit`.

## Exact release settings

| Stage | Setting | Published 2.1.x | 2.3 release recipe | Status |
|---|---|---:|---:|---|
| Affinity | minibatch | 128 | 1024 | Retained: held-out improvement |
| Affinity | maximum epochs | 5000 | 5000 | Restored |
| Affinity | early-stop `min_delta` | 0 | 0 | Restored |
| Affinity | validation interval | every epoch | every epoch | Restored |
| Affinity | random-negative pool | fresh each epoch | fresh each epoch | Explicitly pinned to 1 |
| Affinity | LSUV variance target | post-activation Dense output | post-activation Dense output | Restored and explicit |
| Affinity | RMSprop equations | Keras | Keras | Restored and explicit; PyTorch selectable for ablation |
| Affinity calibration | peptides per length | 100,000 | 100,000 | Restored |
| Processing | minibatch | 512 | 512 | Restored |
| Processing | held-out samples per fold | 10 | 10 | Restored |
| Processing | initializer | Glorot uniform, zero bias | Glorot uniform, zero bias | Restored and explicit; former Kaiming behavior selectable for ablation |
| Processing | Adam equations | Keras | Keras | Restored and explicit; PyTorch selectable for ablation |
| Processing | decoy candidates retained | 2 per hit | 2 per hit | Unchanged |
| Presentation | decoys per hit | 2 | 2 | Restored |
| Presentation | training-row sample fraction | 0.1 | 0.1 | Restored |
| Presentation | excluded PMIDs | 31844290, 31495665, 31154438 | Same three studies | Restored; PMID 31154438 is the frozen sample holdout |
| Presentation | with-flanks processing input | `short_flanks` | `short_flanks` | Restored; 5 aa on each side |
| Presentation | logistic solver | L-BFGS, 100 iterations | L-BFGS, 100 iterations | Restored |
| Presentation calibration | peptides per length | 10,000 | 10,000 | Restored |

The affinity architecture grid, processing architecture grid, loss functions,
optimizers, learning rates, dropout, regularization, early-stop patience,
peptide lengths, affinity-random-negative distribution, fold counts, and model
selection minima/maxima otherwise match the published recipe.

## Execution changes that remain

The maintained implementation differs from the historical implementation in
ways that do not intentionally change the scientific objective:

- PyTorch replaces TensorFlow/Keras, and fixed BLOSUM62 expansion is performed
  by a frozen on-device embedding rather than host numpy code.
- Keras-compatible RMSprop and Adam update equations, including epsilon
  placement, are explicit and equation-tested. The native PyTorch equations
  remain selectable through ``optimizer_implementation`` for controlled
  experiments, but are not the compatibility default.
- Affinity LSUV measures post-activation variance as the historical Keras
  implementation did. ``data_dependent_initialization_target`` records this
  choice and permits a pre-activation ablation.
- Processing Glorot initializers and zero biases are explicit. The rejected
  port's Kaiming/fan-in behavior remains selectable as
  ``kaiming_uniform_fan_in`` for an ablation.
- Validation splits use Keras' exact boundary calculation: training rows are
  ``floor(N * (1 - validation_split))`` and the tail is validation.
- A fixed master seed and derived per-fit seeds replace entropy-derived random
  state. Exact peptide identities and trained weights consequently need not
  reproduce historical TensorFlow runs even when distributions match.
- Proteome decoys are sampled lazily from the same candidate universe instead
  of materializing the full peptide table. This preserves the sampling
  distribution while changing the seeded identity stream.
- Worker counts, prediction batches, feature chunks, and calibration work
  chunks may be autosized. They are execution controls, not model
  hyperparameters. The release workflow fails instead of silently shrinking a
  configured training minibatch.
- Release training and evaluation default to eager execution and `highest`
  float32 matmul precision. `torch.compile` and reduced matmul precision remain
  opt-in because their effects on a newly trained trajectory have not been
  isolated empirically.
- Allele names and genotypes are normalized with the maintained sequence-aware
  code, and the frozen release holdout is excluded before training.

## Inference and calibration behavior

Runtime prediction batch sizes now default to capacity-aware `auto` with OOM
retry. This changes partitioning, not the prediction formula. Presentation
feature chunking likewise avoids materializing large peptide-by-genotype
tables while preserving the minimum-affinity and tie-order semantics.

Presentation percentile calibration uses adaptive score-quantile bins instead
of uniform bins over `[0, 1]`. This is a deliberate correctness fix: uniform
bins collapse compressed logistic scores and can destroy rank resolution. It
changes presentation percentile outputs, but not raw affinity, processing, or
presentation scores. Affinity percentile bins retain their published
log-spaced IC50 definition.

## Decoy semantics

Affinity training generates random amino-acid peptides as synthetic negatives.
Processing and presentation use unobserved peptides sampled from proteins in
the reference proteome. Presentation therefore does not use the affinity-style
synthetic amino-acid negative generator. A stronger affinity predictor can
change which processing decoys are selected, but it does not by itself explain
a presentation regression: the presentation combiner must still be evaluated
end to end on the frozen holdout.
