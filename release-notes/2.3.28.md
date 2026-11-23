# MHCflurry 2.3.28

Each command now has one supported spelling. The duplicate names below still
work, with a deprecation notice, and will be removed in 2.4.0. Predictions are
unchanged.

## Console scripts (#526)

- `mhcflurry-class1-train-*`, `mhcflurry-class1-select-*` and
  `mhcflurry-calibrate-percentile-ranks` print a one-line notice on stderr and
  then run as before. Use `mhcflurry class1-train-pan-allele-models`,
  `mhcflurry calibrate-percentile-ranks` and so on.
- `mhcflurry-pseudosequences` is removed. Use `mhcflurry pseudosequences`.
- `mhcflurry-downloads`, `mhcflurry-predict` and `mhcflurry-predict-scan` are
  unchanged and remain supported.

## Evaluation commands (#512)

- `mhcflurry compare-models`, `mhcflurry plot-model-comparison` and
  `mhcflurry paper-figures` print a one-line notice on stderr and then run as
  before. Use `mhcflurry eval compare-models`, `mhcflurry eval plot-comparison`
  and `mhcflurry eval paper-figures render`.
- `mhcflurry eval plot-model-comparison` is removed. It was an alias for
  `mhcflurry eval plot-comparison`.
- `mhcflurry.cli.main.DEPRECATED_SUBCOMMANDS` maps each deprecated top-level
  name to its replacement.

## Python modules (#505)

Importing these wrappers now emits a `DeprecationWarning`:

- `mhcflurry.predict_command`, `mhcflurry.downloads_command` and the other nine
  top-level `mhcflurry.*_command` modules. Import `mhcflurry.cli.*_command`.
- `mhcflurry.local_parallelism`. Import `mhcflurry.parallelism`.
