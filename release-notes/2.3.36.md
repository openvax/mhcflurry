# MHCflurry 2.3.36

Shared CLI flag helpers and consistent `--allele`/`--alleles` handling (#504):

- Added import-light public helpers `add_data_arg`, `add_models_dir_arg`, `add_hyperparameters_arg`, `add_out_arg`, and `add_alleles_arg` to `mhcflurry.cli_args` (re-exported from `mhcflurry.common`) and wired them across all CLI parsers defining `--data`, `--models-dir`, `--hyperparameters`, `--out`, and `--allele`/`--alleles`.
- Standardized `--allele` and `--alleles` as interchangeable aliases across `predict`, `predict-scan`, `class1-train-allele-specific-models`, `class1-select-allele-specific-models`, `calibrate-percentile-ranks`, and `eval allele-capabilities`. Existing invocations and parsed attribute names are unchanged.

Model weights, catalogue identities, and predictions are unchanged.
