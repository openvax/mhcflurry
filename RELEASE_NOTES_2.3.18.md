# MHCflurry 2.3.18

The training, model-selection and percentile-calibration commands now share
one definition each of `--verbosity`, `--debug`, `--out-models-dir` and
`--max-epochs` (new public helpers in `mhcflurry.common`) instead of eight
copies. The only visible difference is that `mhcflurry-class1-train-presentation-models --help`
now describes `--verbosity` as "Verbosity. Default: 1" like the other commands.

The prediction models and numerical prediction behavior are unchanged.
