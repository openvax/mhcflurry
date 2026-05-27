# Command-line reference

See also the {ref}`tutorial <commandline_tutorial>`.

Starting in 2.3.0, MHCflurry installs a unified `mhcflurry` parent
command whose subcommands share one help surface (`mhcflurry --help`).
Every historical `mhcflurry-*` console script is also reachable as
`mhcflurry <subcommand>` (`mhcflurry-predict` ↔ `mhcflurry predict`,
`mhcflurry-class1-train-pan-allele-models` ↔
`mhcflurry class1-train-pan-allele-models`, etc.). Both forms run the
same underlying entry point; the legacy `mhcflurry-*` scripts remain
installed as compat shims and are not changing.

The two commands new in 2.3.0 — `mhcflurry compare-models` and
`mhcflurry plot-model-comparison` — only have the parent-command form.

## Prediction and data

```{eval-rst}
.. _ref-mhcflurry-predict:

.. autoprogram:: mhcflurry.predict_command:parser
    :prog: mhcflurry predict

.. _ref-mhcflurry-predict-scan:

.. autoprogram:: mhcflurry.predict_scan_command:parser
    :prog: mhcflurry predict-scan

.. _ref-mhcflurry-downloads:

.. autoprogram:: mhcflurry.downloads_command:parser
    :prog: mhcflurry downloads
```

## Calibration

```{eval-rst}
.. _ref-mhcflurry-calibrate-percentile-ranks:

.. autoprogram:: mhcflurry.calibrate_percentile_ranks_command:parser
    :prog: mhcflurry calibrate-percentile-ranks
```

## Class1 training and selection

```{eval-rst}
.. _ref-mhcflurry-class1-train-allele-specific-models:

.. autoprogram:: mhcflurry.train_allele_specific_models_command:parser
    :prog: mhcflurry class1-train-allele-specific-models

.. _ref-mhcflurry-class1-select-allele-specific-models:

.. autoprogram:: mhcflurry.select_allele_specific_models_command:parser
    :prog: mhcflurry class1-select-allele-specific-models

.. _ref-mhcflurry-class1-train-pan-allele-models:

.. autoprogram:: mhcflurry.train_pan_allele_models_command:parser
    :prog: mhcflurry class1-train-pan-allele-models

.. _ref-mhcflurry-class1-select-pan-allele-models:

.. autoprogram:: mhcflurry.select_pan_allele_models_command:parser
    :prog: mhcflurry class1-select-pan-allele-models

.. _ref-mhcflurry-class1-train-processing-models:

.. autoprogram:: mhcflurry.train_processing_models_command:parser
    :prog: mhcflurry class1-train-processing-models

.. _ref-mhcflurry-class1-select-processing-models:

.. autoprogram:: mhcflurry.select_processing_models_command:parser
    :prog: mhcflurry class1-select-processing-models

.. _ref-mhcflurry-class1-train-presentation-models:

.. autoprogram:: mhcflurry.train_presentation_models_command:parser
    :prog: mhcflurry class1-train-presentation-models
```

## Model comparison (new in 2.3.0)

```{eval-rst}
.. _ref-mhcflurry-compare-models:

.. autoprogram:: mhcflurry.cli.compare_models:parser
    :prog: mhcflurry compare-models

.. _ref-mhcflurry-plot-model-comparison:

.. autoprogram:: mhcflurry.cli.plot_model_comparison:parser
    :prog: mhcflurry plot-model-comparison
```

## Pseudosequence registry helper

```{note}
`mhcflurry pseudosequences` is a shell-helper CLI for the
pseudosequence CSV registry. It has its own subcommands
(`filename`, `path`, `list`, `legacy`); run
`mhcflurry pseudosequences --help` for the full argument forms.
```
