# Training models

Most users should start with the released models. Train a model when you have
new measurements, need a controlled experiment, or are preparing a release.
Always evaluate a trained ensemble on held-out data before using it for
scientific conclusions.

## Choose a workflow

| Data and goal | Command |
|---|---|
| One or a few well-covered alleles | `mhcflurry class1-train-allele-specific-models` |
| Measurements spanning many alleles | `mhcflurry class1-train-pan-allele-models` |
| Full retrain, selection, calibration, and evaluation | `mhcflurry train pan-allele-release` |

The low-level commands fit candidate models. Production ensembles also require
model selection and percentile calibration; use the release workflow when you
need the complete pipeline.

## Small allele-specific example

The training command accepts a YAML list of architectures. This single,
four-member architecture is suitable for learning the workflow; it is not a
replacement for the released ensemble search:

Fetch the curated example data:

```shell
mhcflurry downloads fetch data_curated
```

```yaml
- activation: tanh
  dropout_probability: 0.0
  early_stopping: true
  layer_sizes: [8]
  locally_connected_layers: []
  loss: custom:mse_with_inequalities
  max_epochs: 500
  minibatch_size: 16384
  n_models: 4
  output_activation: sigmoid
  patience: 20
  peptide_amino_acid_encoding: BLOSUM62
  random_negative_affinity_max: 50000.0
  random_negative_affinity_min: 20000.0
  random_negative_constant: 25
  random_negative_rate: 0.0
  validation_split: 0.1
```

Save it as `hyperparameters.yaml`, then run:

```shell
mhcflurry class1-train-allele-specific-models \
    --data "$(mhcflurry downloads path data_curated)/curated_training_data.csv.bz2" \
    --hyperparameters hyperparameters.yaml \
    --allele 'HLA-A*02:01' \
    --min-measurements-per-allele 75 \
    --out-models-dir models
```

The output directory is a complete predictor. Keep its manifest, metadata, and
weights together; pass the directory to prediction commands with `--models`.

## Prepare training data

Affinity training tables use one row per measurement. Allele-specific training
requires `allele`, `peptide`, `measurement_value`, and `measurement_type`.
Use `measurement_type` to identify quantitative or qualitative measurements.
The optional `measurement_inequality` column records `=`, `<`, or `>`; if the
column is omitted, all measurements are treated as equalities.

Alleles must be sequence-resolved MHC class I names. Allele-specific training
does not require a pseudosequence table unless the hyperparameters enable
cross-allele pretraining with `pretrain_min_points`.

The curated data used in the example and for released models is the
`data_curated` download bundle. Use `mhcflurry downloads path data_curated` to
locate it.

## Pan-allele and release training

Pan-allele training additionally needs an allele pseudosequence table and a
model-selection step. Each training allele must resolve to a key in that table;
rows for alleles without a matching pseudosequence are excluded. Start with
the command help and the maintained training recipes rather than copying
individual flags from an old run:

```shell
mhcflurry class1-train-pan-allele-models --help
mhcflurry train pan-allele-release --help
```

The release workflow can run locally or on a configured remote backend. It
records source and workflow provenance, resumes completed phases, compares the
candidate with public models, and copies review artifacts back to the control
machine. Deployment is opt-in.

Leave worker counts, GPU packing, DataLoader workers, and prediction batches on
`auto` initially. See {doc}`configuration` before pinning resource values and
the [maintained training scripts](https://github.com/openvax/mhcflurry/tree/master/scripts/training)
for the current release recipe.

## Evaluate before use

Run `mhcflurry eval compare-models` against a public baseline and inspect the
summary metrics and plots. Prediction-affecting changes need held-out evidence,
not only a successful unit-test run. See {doc}`evaluation` for the evaluation
workflow.
