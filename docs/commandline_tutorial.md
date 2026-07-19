(commandline_tutorial)=

# Command-line tutorial

(downloading)=

## Downloading models

Most users will use pre-trained MHCflurry models that we release. These models
are distributed separately from the pip package and may be downloaded with the
{ref}`mhcflurry-downloads <ref-mhcflurry-downloads>` tool:

```shell
$ mhcflurry-downloads fetch models_class1_presentation
```

Files downloaded with {ref}`mhcflurry-downloads <ref-mhcflurry-downloads>` are stored in a platform-specific
directory. To get the path to downloaded data, you can use:

```{command-output} mhcflurry-downloads path models_class1_presentation
:nostderr:
```

We also release a number of other "downloads," such as curated training data and some
experimental models. To see what's available and what you have downloaded, run
`mhcflurry-downloads info`.

Most users need only `models_class1_presentation`, which includes both the
binding-affinity and antigen-processing components.


## Generating predictions

The {ref}`mhcflurry-predict <ref-mhcflurry-predict>` command generates predictions for individual peptides
(see the next section for how to scan protein sequences for epitopes). By
default it will use the pre-trained models you downloaded above. Other
models can be used by specifying the `--models` argument.

Running:

```{command-output} mhcflurry-predict --alleles HLA-A0201 HLA-A0301 --peptides SIINFEKL SIINFEKD SIINFEKQ --out /tmp/predictions.csv
:nostderr:
```

results in a file like this:

```{command-output} cat /tmp/predictions.csv
```

The binding predictions are given as predicted affinities in nM in the
`mhcflurry_affinity` column. Lower values indicate stronger binders. A commonly used
threshold for peptides with a reasonable chance of being immunogenic is 500 nM.

The `mhcflurry_affinity_percentile` gives the percentile of the affinity
prediction among a large number of random peptides tested on that allele (range
0 - 100). Lower is stronger. Two percent is a commonly-used threshold.

The last two columns give the antigen processing and presentation scores,
respectively. These range from 0 to 1 with higher values indicating more
favorable processing or presentation.

```{note}
The processing predictor is experimental. It models allele-independent
effects that influence whether a
peptide will be detected in a mass spec experiment. The presentation score is
a simple logistic regression model that combines the (log) binding affinity
prediction with the processing score to give a composite prediction. The resulting
prediction may be useful for prioritizing potential epitopes, but no
thresholds have been established for what constitutes a "high enough"
presentation score.
```

In most cases you'll want to specify the input as a CSV file instead of passing
peptides and alleles as commandline arguments. If you're relying on the
processing or presentation scores, you may also want to pass the upstream and
downstream sequences of the peptides from their source proteins for potentially more
accurate cleavage prediction. See the {ref}`mhcflurry-predict <ref-mhcflurry-predict>` docs.


## Scanning protein sequences for predicted MHC I ligands

Starting in version 1.6.0, MHCflurry supports scanning proteins for MHC-binding
peptides using the `mhcflurry-predict-scan` command.

We'll generate predictions across `example.fasta`, a FASTA file with two short
sequences:

```{literalinclude} /example.fasta
```

Here's a `mhcflurry-predict-scan` invocation using a 100 nM affinity threshold:

```shell
$ mhcflurry-predict-scan example.fasta \
    --alleles HLA-A*02:01 \
    --threshold-affinity 100
```

See the {ref}`mhcflurry-predict-scan <ref-mhcflurry-predict-scan>` docs for more options.


## Fitting your own models

If you have your own data and want to fit your own MHCflurry models, you have
a few options. If you have data for only one or a few MHC I alleles, the best
approach is to use the
{ref}`mhcflurry-class1-train-allele-specific-models <ref-mhcflurry-class1-train-allele-specific-models>` command to fit an
"allele-specific" predictor, in which separate neural networks are used for
each allele.

To call {ref}`mhcflurry-class1-train-allele-specific-models <ref-mhcflurry-class1-train-allele-specific-models>` you'll need some
training data. The data we use for our released predictors can be downloaded with
{ref}`mhcflurry-downloads <ref-mhcflurry-downloads>`:

```shell
$ mhcflurry-downloads fetch data_curated
```

It looks like this:

```{command-output} bzcat "$(mhcflurry-downloads path data_curated)/curated_training_data.csv.bz2" | head -n 3
:shell:
:nostderr:
```

Here's an example invocation to fit a predictor:

```shell
$ mhcflurry-class1-train-allele-specific-models \
    --data curated_training_data.csv.bz2 \
    --hyperparameters hyperparameters.yaml \
    --min-measurements-per-allele 75 \
    --out-models-dir models
```

The `hyperparameters.yaml` file gives the list of neural network architectures
to train models for. Here's an example specifying a single architecture:

```yaml
- activation: tanh
  dense_layer_l1_regularization: 0.0
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

The available hyperparameters for binding predictors are defined in
{class}`~mhcflurry.Class1NeuralNetwork`. To see exactly how
these are used you will need to read the source code.

The output directory is a complete predictor and can be passed to prediction
commands with `--models`. Its `manifest.csv` records the component models; do
not copy individual weight files out of that directory.

To fit pan-allele models like the ones released with MHCflurry, you can use
a similar tool, {ref}`mhcflurry-class1-train-pan-allele-models <ref-mhcflurry-class1-train-pan-allele-models>`. You'll probably
also want to take a look at the scripts used to generate the production models,
which are available in the *downloads-generation* directory in the MHCflurry
repository. See the scripts in the *models_class1_pan* subdirectory to see how the
fitting and model selection was done for models currently distributed with MHCflurry.

Released ensembles evaluate many architectures, but smaller searches can be
trained with substantially fewer resources.


## Evaluating trained models

After fitting a model, compare it with a released predictor before using it as a
default. A local comparison and diagnostic PDF take two commands:

```shell
$ mhcflurry eval compare-models \
    --a results/new_run/ \
    --b public \
    --out results/new_run/eval_comparison/

$ mhcflurry eval plot-comparison \
    --input results/new_run/eval_comparison/ \
    --summary-pdf results/new_run/eval_comparison/plots/model_comparison_figures.pdf
```

The {doc}`evaluation` guide explains the output layers, saved-prediction schema,
paper-style figures, and remote release behavior.

## Using older allele-specific models

MHCflurry still distributes the allele-specific predictors described in the
2018 paper. Download them and pass their model directory explicitly:

```shell
$ mhcflurry-downloads fetch models_class1
$ mhcflurry-predict \
    --alleles HLA-A0201 HLA-A0301 \
    --peptides SIINFEKL SIINFEKD SIINFEKQ \
    --models "$(mhcflurry-downloads path models_class1)/models" \
    --out /tmp/predictions.csv
```

Use the current pan-allele presentation bundle unless you specifically need
these historical models.

## Configuration and command reference

See {doc}`configuration` for prediction batches, hardware autosizing,
reproducibility, and unified command aliases. The complete generated argument
reference is in {doc}`commandline_tools`.
