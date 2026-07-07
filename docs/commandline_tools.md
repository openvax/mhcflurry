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

The evaluation commands new in 2.3.0 are grouped under
`mhcflurry eval`. The older top-level forms (`mhcflurry compare-models`,
`mhcflurry plot-model-comparison`, and `mhcflurry paper-figures`) remain
available as compatibility shortcuts.

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

```{eval-rst}
.. _ref-mhcflurry-calibrate-percentile-ranks:

.. autoprogram:: mhcflurry.cli.calibrate_percentile_ranks_command:parser
    :prog: mhcflurry calibrate-percentile-ranks
```

## Class I training and selection

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

.. autoprogram:: mhcflurry.cli.eval_command:parser
    :prog: mhcflurry eval

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

Prefer the namespaced form in new automation:

```shell
mhcflurry eval compare-models --a results/new_run --b public --out results/eval
mhcflurry eval plot-comparison --input results/eval
mhcflurry eval paper-figures render --comparison-dir results/eval --out results/eval/plots/paper_figures
mhcflurry eval paper-figures run --a results/new_run --b public --out results/eval
```

The evaluation namespace is also the home for the 2023-notebook-style figure
inputs. Use `mhcflurry eval paper-figures score-predictions` to turn saved
benchmark prediction tables into reusable `accuracy_scores.*.csv` caches, then
use `mhcflurry eval paper-figures render` for the publication-style panels.
`paper-figures run` composes the local MHCflurry comparison and rendering steps;
external predictor binaries such as NetMHCpan and MixMHCpred still need to
produce saved prediction columns before they enter this pipeline.

## Pseudosequence registry helper

```{note}
`mhcflurry pseudosequences` is a shell-helper CLI for the
pseudosequence CSV registry. It has its own subcommands
(`filename`, `path`, `list`, `legacy`); run
`mhcflurry pseudosequences --help` for the full argument forms.
```
