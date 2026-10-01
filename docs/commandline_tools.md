# Command-line reference

This page documents the options of the prediction, training, calibration,
evaluation, and helper commands. Workflow namespaces with many specialized
subcommands (`mhcflurry eval`, `mhcflurry train`, `mhcflurry pseudosequences`)
are listed here; run a subcommand's `--help` for its options. If you are new to
MHCflurry, start with the {ref}`tutorial <commandline_tutorial>` and return here
to look up specific arguments.

Commands are grouped in the order a project usually needs them: prediction
first, then training, calibration, and evaluation, with release and helper
workflows last. Run `mhcflurry <command> --help` for the same information in a
terminal; set `NO_COLOR=1` to disable colored help. Historical `mhcflurry-*`
script names remain available; see {doc}`configuration`.

## Prediction and data

See {doc}`model_downloads` for bundle descriptions, local paths, and selecting
older weights with `--model-release`.

```{eval-rst}
.. _ref-mhcflurry-predict:

.. autoprogram:: mhcflurry.cli.predict_command:parser
    :prog: mhcflurry predict

.. _ref-mhcflurry-predict-scan:

.. autoprogram:: mhcflurry.cli.predict_scan_command:parser
    :prog: mhcflurry predict-scan

.. _ref-mhcflurry-downloads:

.. autoprogram:: mhcflurry.cli.downloads_command:parser
    :prog: mhcflurry downloads
```

## Training and model selection

See {doc}`training` for choosing between these commands. Training commands fit
candidate models; selection commands choose the ensemble members to keep.

```{eval-rst}
.. _ref-mhcflurry-class1-train-allele-specific-models:

.. autoprogram:: mhcflurry.cli.train_allele_specific_models_command:parser
    :prog: mhcflurry class1-train-allele-specific-models

.. _ref-mhcflurry-class1-select-allele-specific-models:

.. autoprogram:: mhcflurry.cli.select_allele_specific_models_command:parser
    :prog: mhcflurry class1-select-allele-specific-models

.. _ref-mhcflurry-class1-train-pan-allele-models:

.. autoprogram:: mhcflurry.cli.train_pan_allele_models_command:parser
    :prog: mhcflurry class1-train-pan-allele-models

.. _ref-mhcflurry-class1-select-pan-allele-models:

.. autoprogram:: mhcflurry.cli.select_pan_allele_models_command:parser
    :prog: mhcflurry class1-select-pan-allele-models

.. _ref-mhcflurry-class1-train-processing-models:

.. autoprogram:: mhcflurry.cli.train_processing_models_command:parser
    :prog: mhcflurry class1-train-processing-models

.. _ref-mhcflurry-class1-select-processing-models:

.. autoprogram:: mhcflurry.cli.select_processing_models_command:parser
    :prog: mhcflurry class1-select-processing-models

.. _ref-mhcflurry-class1-train-presentation-models:

.. autoprogram:: mhcflurry.cli.train_presentation_models_command:parser
    :prog: mhcflurry class1-train-presentation-models
```

## Percentile calibration

Released affinity and presentation predictors ship with percentile
calibration; released processing predictors do not. Calibrate custom models, or
a released processing predictor when you need processing percentiles. See
{doc}`shared_percent_rank_transforms` for examples and background
requirements. Calibration writes into `--models-dir`; use a copy to preserve
an existing calibration.

```{eval-rst}
.. _ref-mhcflurry-calibrate-percentile-ranks:

.. autoprogram:: mhcflurry.cli.calibrate_percentile_ranks_command:parser
    :prog: mhcflurry calibrate-percentile-ranks
```

(ref-mhcflurry-eval)=

## Evaluation

See {doc}`evaluation` for the workflow and output layout.

(ref-mhcflurry-eval-artifacts)=

### Metrics and figures

These commands cover the common path: compute metrics, render diagnostics, and
optionally produce paper-style figures. The shorter `mhcflurry compare-models`,
`mhcflurry plot-model-comparison`, and `mhcflurry paper-figures` forms remain
available for existing scripts.

```{eval-rst}
.. _ref-mhcflurry-compare-models:

.. autoprogram:: mhcflurry.cli.compare_models:parser
    :prog: mhcflurry eval compare-models

.. _ref-mhcflurry-plot-model-comparison:

.. autoprogram:: mhcflurry.cli.plot_model_comparison:parser
    :prog: mhcflurry eval plot-comparison

.. _ref-mhcflurry-paper-figures-run:

.. autoprogram:: mhcflurry.cli.eval_command:make_paper_figures_run_parser()
    :prog: mhcflurry eval paper-figures run

.. _ref-mhcflurry-paper-figures:

.. autoprogram:: mhcflurry.cli.paper_figures:parser
    :prog: mhcflurry eval paper-figures render
```

### Other evaluation workflows

`mhcflurry eval` also contains specialized workflows for release experiments
and external-predictor comparisons. Run the subcommand's `--help` for its
options.

```{command-output} mhcflurry eval --help
:nostderr:
```

(ref-mhcflurry-train)=

## Release and research workflows

`mhcflurry train` groups the release pipeline and research workflows used to
produce the published models. Most need a source checkout; see
{doc}`training` and {doc}`maintainers`. Run the subcommand's `--help` for its
options.

```{command-output} mhcflurry train --help
:nostderr:
```

## Helpers

These commands support release training and data preparation rather than
prediction. They match the `Helpers` group in `mhcflurry --help`. The generated
hyperparameter grids feed the maintained training scripts; see the
[training pipeline README](https://github.com/openvax/mhcflurry/tree/master/scripts/training)
for how each grid is used. {doc}`training_provenance` shows how the mass-spec
reassignment command excludes evaluation peptide–MHC pairs and source samples.

```{eval-rst}
.. _ref-mhcflurry-class1-generate-training-hyperparameters:

.. autoprogram:: mhcflurry.cli.generate_training_hyperparameters:make_parser()
    :prog: mhcflurry class1-generate-training-hyperparameters

.. _ref-mhcflurry-class1-reassign-mass-spec-training-data:

.. autoprogram:: mhcflurry.cli.reassign_mass_spec_training_data:make_parser()
    :prog: mhcflurry class1-reassign-mass-spec-training-data
```

### `mhcflurry pseudosequences`

```{note}
`mhcflurry pseudosequences` is a shell-helper CLI for the
pseudosequence CSV registry. It has its own subcommands
(`filename`, `path`, `list`, `legacy`); run
`mhcflurry pseudosequences --help` for the full argument forms.
```
