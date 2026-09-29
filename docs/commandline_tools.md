# Command-line reference

This page gives generated help for the main commands and workflow namespaces.
Use a workflow's subcommand help for its complete options. If you are new to MHCflurry or are
choosing a workflow, start with the {ref}`tutorial <commandline_tutorial>` and
return here to look up specific arguments.

MHCflurry provides a unified `mhcflurry` command while retaining the
historical `mhcflurry-*` names. Both forms use the same implementation. See
{doc}`configuration` for the naming convention, {doc}`evaluation` for the
evaluation workflow, and the generated argument reference below for every
option.

Prediction help uses a compact usage line, spaced options and terminal colors.
Redirected help stays plain text; set `NO_COLOR=1` to disable colors in a terminal.
Browse and select weight releases with `mhcflurry downloads releases`,
`mhcflurry downloads list`, and `mhcflurry predict --model-release RELEASE`.
See {doc}`model_downloads` for bundle descriptions, local paths, and a worked
comparison with historical weights. Use {doc}`shared_percent_rank_transforms`
for compact percentile calibration.

## Prediction and data

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

## Calibration

See {doc}`shared_percent_rank_transforms` for examples and background
requirements. Calibration writes into `--models-dir`; use a copy to preserve
an existing calibration.

```{eval-rst}
.. _ref-mhcflurry-calibrate-percentile-ranks:

.. autoprogram:: mhcflurry.cli.calibrate_percentile_ranks_command:parser
    :prog: mhcflurry calibrate-percentile-ranks
```

## Class I training and selection

```{eval-rst}
.. _ref-mhcflurry-train:
```

### `mhcflurry train`

`mhcflurry train` groups release-training workflows. It is a namespace command;
run `mhcflurry train --help` or the concrete subcommand help for the complete
argument list.

```{command-output} mhcflurry train --help
:nostderr:
```

The release workflow delegates to the maintained release script:

```console
$ mhcflurry train pan-allele-release --help
```

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

## Evaluation and figures

```{eval-rst}
.. _ref-mhcflurry-eval:
```

### `mhcflurry eval`

`mhcflurry eval` groups model comparison, diagnostic plotting, reusable score
generation, and paper-style figure rendering. It is a namespace command; run
the concrete subcommand help for the complete argument list.

```{command-output} mhcflurry eval --help
:nostderr:
```

```{eval-rst}
.. _ref-mhcflurry-eval-artifacts:
```

### Evaluation and Plotting Artifacts

The commands deliberately separate reusable metrics from rendering. See
{doc}`evaluation` for the output map, saved-prediction schema, paper-figure
workflow, and external-predictor integration.

```{eval-rst}
.. _ref-mhcflurry-compare-models:

.. autoprogram:: mhcflurry.cli.compare_models:parser
    :prog: mhcflurry compare-models

.. _ref-mhcflurry-plot-model-comparison:

.. autoprogram:: mhcflurry.cli.plot_model_comparison:parser
    :prog: mhcflurry plot-model-comparison

.. _ref-mhcflurry-paper-figures:

.. autoprogram:: mhcflurry.cli.paper_figures:parser
    :prog: mhcflurry paper-figures
```

Prefer the namespaced `mhcflurry eval ...` form in new automation. Compatibility
shortcuts remain available for existing scripts.

## Pseudosequence registry helper

```{note}
`mhcflurry pseudosequences` is a shell-helper CLI for the
pseudosequence CSV registry. It has its own subcommands
(`filename`, `path`, `list`, `legacy`); run
`mhcflurry pseudosequences --help` for the full argument forms.
```
