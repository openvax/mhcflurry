# MHCflurry 2.3.0

MHCflurry 2.3.0 provides updated presentation models, reproducible training and
model comparison workflows, and shared percentile calibration across the
PyTorch predictors. Python 3.10 or newer is required.

## Install and upgrade

```shell
pip install --upgrade "mhcflurry==2.3.0"
mhcflurry downloads fetch models_class1_presentation
```

Pinning the version also handles installations that used a higher-numbered
development build. Existing 2.2.x model directories remain loadable. Software
and model downloads have separate provenance; a software upgrade alone does
not replace a user-specified model directory.

## Models and evaluation

The new full presentation model uses the 2023 training-data snapshot and the
recorded `final-2.3.0-candidate-v2` recipe. Its with-flank processing component
is an equal mixture of short-flank and cleavage-boundary networks. The
separate long-flank processing ensemble is not the presentation component.

Evaluate complete presentation predictions separately from affinity and
processing diagnostics. Full-model improvements do not imply improvements in
every component. Comparisons must use identical positive/negative rows and
shared overlap exclusions; external-predictor training overlap may remain
unknown. The release assets include the model comparison and provenance. See the
[full presentation evaluation](docs/release_model_evaluation.md) for metrics,
paired intervals and component tradeoffs.

## Prediction and calibration

- Prediction progress is written to stderr so redirected stdout contains valid CSV.
- The `mhcflurry` command groups prediction, download, training and evaluation
  commands. Existing standalone `mhcflurry-*` entry points remain supported.
- CPU, CUDA and Apple Silicon execution use PyTorch. Automatic worker and
  prediction-batch planning respects caller overrides and device capacity.
- Affinity, processing and presentation share percentile-calibration methods.
  New compact mappings use independent background data; existing histogram
  calibrations retain their original semantics when loaded.
- Model manifests, allele pseudosequences, saved calibration and prediction
  weights travel together. Calibration changes do not alter raw predictions.

## Training and reproducibility

- Streaming pretraining validation now reverses concentration inequalities
  when converting to the decreasing regression-target scale. This fixes future
  validation/retry decisions; it does not modify the selected released weights.
- Release workflows record source, data, configuration and artifact checksums;
  validate holdout exclusions; and preserve resumable training stages.
- Processing preparation uses matched negatives with unique assignments,
  bounded pool expansion and deterministic replay.
- Affinity and processing training can retain checkpoint alternatives for
  development. Inference downloads contain the selected model, while the
  training archive preserves alternative states and selection evidence.
- Evaluation supports saved predictions, paired uncertainty estimates and
  external NetMHCpan/MixMHCpred scores on explicitly shared rows.

See the [training recipe](docs/release_training_recipe.md),
[evaluation guide](docs/evaluation.md), and
[release workflow](scripts/release/README.md) for the supported commands.
