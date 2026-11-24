# MHCflurry 2.3.34

CLI completion and help-styling polish (#514, #515, #511):

- **Fast tab completion for all subcommands (#515):** Extracted import-light CLI argument helpers into `mhcflurry/cli_args.py` (re-exported from `mhcflurry.common` and `mhcflurry.cluster_parallelism`) and lightweight parser builders for the 8 training, selection, and calibration commands into `mhcflurry/cli/parsers.py`, and deferred heavy top-level imports in `eval_command.py`, `generate_training_hyperparameters.py`, and `reassign_mass_spec_training_data.py`. Every listed `mhcflurry` subcommand and deprecated `mhcflurry-*` console script now tab-completes its flags in ~35–40 ms without importing `torch`, `numpy`, `pandas`, `sklearn`, `scipy`, `matplotlib`, `mhcgnomes`, `tqdm`, or `yaml`.
- **Consistent subtle help styling (#514):** Switched all remaining CLI argument parsers to `HelpArgumentParser` (and styled custom `eval`/`train` help screens via `style_help()`), and dropped green (`32`) status coloring in `mhcflurry downloads` tables so normal status values print plain while keeping dim (`2`) and yellow (`33`) for absent/warning states.
- **Docs cleanup (#511):** Updated `docs/commandline_tools.md` to remove the stale note claiming most `mhcflurry train` commands require a source checkout after PR #537 removed the 23 unpackaged script dispatchers.

Model weights, catalogue identities, and predictions are unchanged.
