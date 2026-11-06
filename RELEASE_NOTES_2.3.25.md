# MHCflurry 2.3.25

`mhcflurry --help` no longer lists the "Shortcuts for eval subcommands" group.
`compare-models`, `plot-model-comparison` and `paper-figures` duplicated
`mhcflurry eval compare-models`, `eval plot-model-comparison` and
`eval paper-figures`, which the help already lists under "Evaluation and
figures" (#512). The top-level names still dispatch, so existing scripts keep
working; they are recorded in `mhcflurry.cli.main.UNLISTED_SUBCOMMANDS`.
