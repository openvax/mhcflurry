# MHCflurry 2.3.10

This maintenance release reorganizes the documentation around the common
prediction workflow, corrects places where the docs disagreed with the code, and
makes the documented fast test tier reliable. Library behavior and model weights
are unchanged; default weights remain 2.3.0 and predictions are unchanged.

Documentation reading order:

- The landing page and sidebar lead with installation and the two prediction
  tutorials, then model selection, then training and evaluation. Calibration and
  training-sample audits are under **Advanced topics**; the detailed 2.3.0
  comparison is nested under its summary page.
- The tutorials stay on prediction and end with a short list of next steps.
  Instructions for the older allele-specific models moved to the model
  downloads page and now pass `--affinity-only`.
- The training and evaluation guides present the common path first. Processing
  data policy and specialized evaluation workflows follow it in labeled
  sections. The source-checkout requirement now appears where `mhcflurry train`
  first does.
- Maintainer pages are grouped as current **Training internals** and
  **Controlled experiments**, and `maintainers.md` introduces each one.

Consistency with the code:

- The command-line reference follows the pipeline order and documents
  evaluation commands under the recommended `mhcflurry eval ...` names,
  including `eval paper-figures run`. Its parser is now the public
  `make_paper_figures_run_parser`.
- `class1-generate-training-hyperparameters` and
  `class1-reassign-mass-spec-training-data` are documented, and every
  reassignment option now has help text.
- The docs no longer say that all released predictors are calibrated: released
  affinity and presentation predictors are, released processing predictors are
  not.
- The `api_coverage` build guard derives its class list from `mhcflurry.__all__`.
- `mhcflurry --help` lists commands in pipeline order, places the `eval`
  compatibility shortcuts last, and mentions the historical script names once.

Tests (#475, #477):

- The fast tier (`-m "not slow and not downloads"`) needs only the default
  presentation bundle, as `docs/testing.md` now states. Tests that need other
  bundles or the pinned `2.2.0` catalogue are marked `downloads`, and two modules
  no longer read bundles at import time.
- A seed-derivation unit test no longer needs a download.
- The affinity-predictor tests report a missing bundle instead of failing on a
  `None` predictor.
- Seven test modules whose `startup()`/`cleanup()` fixtures never ran, because
  of a missing `@`, now run them.
