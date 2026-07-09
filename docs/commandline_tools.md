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

Release-training orchestration is grouped under `mhcflurry train`. The
`pan-allele-release` workflow is a source-checkout command that delegates to
the maintained release script, so it can provision remote machines, run
evaluation/plots remotely, sync artifacts back, and optionally deploy model
archives without duplicating orchestration logic in Python.

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

The plotting commands are layered so expensive prediction and metric work can
be cached and reused:

| Layer | Command | Reads | Writes | Purpose |
|---|---|---|---|---|
| Metrics | `mhcflurry eval compare-models` | A candidate run and a baseline run or public release | `eval_comparison/` CSV/JSON metrics, `release_summary.csv`, `release_summary.md` | Produce reusable evaluation data without importing matplotlib. |
| Diagnostics | `mhcflurry eval plot-comparison` | `eval_comparison/` | `eval_comparison/plots/`, optional `model_comparison_figures.pdf` | Render release-review ROC/PR/scatter/delta plots. |
| Score cache | `mhcflurry eval paper-figures score-predictions` | Saved benchmark prediction table | `accuracy_scores.multiallelic.csv` or `accuracy_scores.monoallelic.csv` | Cache per-sample/per-allele AUC and PPV tables for repeated figure runs. |
| Paper figures | `mhcflurry eval paper-figures render` | `eval_comparison/`, score cache, saved prediction tables, optional metadata/artwork | SVG/PDF/PNG panels, `paper_figures.pdf`, `manifest.csv`, `missing_inputs.md` | Render publication-style panels and report unavailable optional inputs. |
| Local composition | `mhcflurry eval paper-figures run` | Candidate/baseline model directories | A fresh comparison plus diagnostic and paper figures | One-command local eval-to-figures path for already-trained models. |

Saved prediction tables use a small canonical schema: `hit`, a grouping column
(`sample_id` for multiallelic or `allele` / `hla` for monoallelic), optional
peptide metadata, and one numeric score column per predictor. External tools
such as NetMHCpan or MixMHCpred should be run separately and registered as
additional numeric columns. Score direction is explicit: built-in predictor
names have defaults, while custom predictor columns require
`predictor_info.csv` rows with `predictor` and `higher_is_better`.

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

Prefer the namespaced form in new automation:

```shell
mhcflurry eval compare-models --a results/new_run --b public --out results/eval
mhcflurry eval plot-comparison --input results/eval
mhcflurry eval paper-figures render --comparison-dir results/eval --out results/eval/plots/paper_figures
mhcflurry eval paper-figures run --a results/new_run --b public --out results/eval
```

The `paper-figures` subcommands use the artifact contract above. External
predictor binaries such as NetMHCpan and MixMHCpred still need to produce saved
prediction columns before they enter this pipeline.

For release-style training plus remote evaluation/plotting, use the training
namespace:

```shell
mhcflurry train pan-allele-release \
    --run-dir results/release-run \
    --release 2.3.0 \
    --backend brev-provision
```

Deployment is not part of the default path. Add `--deploy-mode dry-run`,
`draft`, or `publish` only when you want to package/upload model artifacts.

## Pseudosequence registry helper

```{note}
`mhcflurry pseudosequences` is a shell-helper CLI for the
pseudosequence CSV registry. It has its own subcommands
(`filename`, `path`, `list`, `legacy`); run
`mhcflurry pseudosequences --help` for the full argument forms.
```
