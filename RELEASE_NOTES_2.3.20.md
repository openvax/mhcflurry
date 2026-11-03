# MHCflurry 2.3.20

Command-line options now complete with Tab, and each Tab press stays fast.

- Every console script (`mhcflurry` and the 12 `mhcflurry-*` scripts) supports
  shell completion through [argcomplete](https://github.com/kislyuk/argcomplete),
  now a dependency. Turn it on once with `activate-global-python-argcomplete`,
  or per command with `eval "$(register-python-argcomplete mhcflurry)"`.
- `mhcflurry` completes every subcommand name. For `predict`, `predict-scan`,
  `downloads` and `pseudosequences` it also completes their options, importing
  only that subcommand's module; a Tab press takes 45–55 ms. The other
  subcommands complete by name only: their modules import PyTorch or pandas
  at startup, or, for most of `train` and `eval`, run scripts from a source
  checkout. The training, selection and calibration `mhcflurry-*` scripts do
  complete their options, at the cost of importing PyTorch (2–3 seconds per
  Tab press).
- Importing `mhcflurry.downloads` no longer reads the download catalogue
  (`downloads.yml`). It is read, and the download directory configured, the
  first time they are needed, so a warning about incompatible downloads now
  appears then rather than at import. Calling `configure()` still re-reads
  `MHCFLURRY_DOWNLOADS_DIR`, `MHCFLURRY_DOWNLOADS_CURRENT_RELEASE` and
  `MHCFLURRY_DATA_DIR`.
- `mhcflurry.pseudosequences` no longer imports pandas at import time, and its
  parser is available from the new `make_parser()`.
- `mhcflurry.cli.downloads_command` imports tqdm and `urllib.request` only when
  fetching. Its `TqdmUpTo` progress-bar class is replaced by
  `download_progress_hook()`.

Help text, command output, model weights and predictions are unchanged.
