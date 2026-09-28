# Command-line reference

This page lists every command and option. If you are new to MHCflurry or are
choosing a workflow, start with the {ref}`tutorial <commandline_tutorial>` and
return here to look up specific arguments.

MHCflurry 2.3.0 provides a unified `mhcflurry` command while retaining the
historical `mhcflurry-*` names. Both forms use the same implementation. See
{doc}`configuration` for the naming convention, {doc}`evaluation` for the
evaluation workflow, and the generated argument reference below for every
option.

Prediction help uses a compact usage line, spaced options and terminal colors.
Redirected help stays plain text; set `NO_COLOR=1` to disable colors in a terminal.
Run `mhcflurry downloads info` to inspect available bundles and the active cache,
or `mhcflurry downloads path models_class1_presentation` for a fetched bundle's
directory. Model bundles are versioned separately from the Python package.
`downloads info` lists the active catalogue, not every version of each bundle.
For the historical catalogue, run:

```shell
MHCFLURRY_DOWNLOADS_CURRENT_RELEASE=2.2.0 mhcflurry downloads info
```

The public 2.1.5, 2.2.0 and 2.2.1 packages used the same 2020 model archives,
registered under catalogue `2.2.0`. Catalogue `2.3.0` selects the newly trained
weights. Older GitHub tags such as `pre-2.0` in download URLs identify where
an archive was uploaded; they do not imply that the active package is a prerelease.

| Bundle name | Purpose |
| --- | --- |
| `models_class1_presentation` | Full presentation predictor, including affinity, processing and their combiner. The default for prediction. |
| `models_class1_pan` | Selected pan-allele binding-affinity ensemble. |
| `models_class1_processing` | Standalone antigen-processing ensembles. |
| `models_class1` | Legacy 2018 allele-specific affinity models; not the current general-purpose class-I predictor. |
| `*_unselected` | Candidate networks before ensemble selection. |
| `*_variants`, `*_with_mass_spec`, `*_no_mass_spec`, `*_minimal` | Historical experimental configurations, training variants or small subsets. |
| `data_*`, `allele_sequences`, `analysis_predictor_info` | Data and supporting resources rather than model weights. |

The presentation bundle contains its own affinity and processing components.
The standalone bundles can therefore show `NO` under `DOWNLOADED?` while
presentation prediction is fully installed.

To compare the current and historical public weights on one input CSV:

```shell
mhcflurry downloads fetch --release 2.3.0 models_class1_presentation
mhcflurry downloads fetch --release 2.2.0 models_class1_presentation
MHCFLURRY_DOWNLOADS_CURRENT_RELEASE=2.3.0 mhcflurry predict eval.csv --no-flanking --out new.csv
MHCFLURRY_DOWNLOADS_CURRENT_RELEASE=2.2.0 mhcflurry predict eval.csv --no-flanking --out old.csv
```

This compares weight bundles using the installed code. To select an arbitrary
presentation predictor, pass `--models /path/to/models_class1_presentation/models`.
Use the same input rows for both runs. Omit `--no-flanking` from both commands to
compare with native flanks supplied in the CSV. Reproducing a historical package's
behavior also requires that package in a separate environment; see
{doc}`release_model_evaluation` for the completed stable-version comparison.

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

```console
$ mhcflurry train --help
usage: mhcflurry train <subcommand> [args]

Subcommands:
  pan-allele-release  Run the retrain/evaluate/plot/release workflow.
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

## Evaluation and figures (new in 2.3.0)

```{eval-rst}
.. _ref-mhcflurry-eval:
```

### `mhcflurry eval`

`mhcflurry eval` groups model comparison, diagnostic plotting, reusable score
generation, and paper-style figure rendering. It is a namespace command; run
the concrete subcommand help for the complete argument list.

```console
$ mhcflurry eval --help
usage: mhcflurry eval <subcommand> [args]

Subcommands:
  compare-models                 Compare two model ensembles.
  plot-comparison                Render diagnostic plots from compare output.
  paper-figures render           Render paper figures from saved inputs.
  paper-figures score-predictions
                                 Derive score tables from saved predictions.
  paper-figures run              Compare, render paper figures, and write PDFs.
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
