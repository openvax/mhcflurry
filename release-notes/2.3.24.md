# MHCflurry 2.3.24

CI now tests the 2.3.0 models, the package's default release, instead of
pinning the 2.2.0 catalogue.

- The test workflow no longer sets `MHCFLURRY_DOWNLOADS_CURRENT_RELEASE`, so
  it follows `current-release` in `mhcflurry/downloads.yml`. The documentation
  workflow no longer pins 2.2.0 on pull requests.
- The download cache key hashes `downloads.yml` and the workflow file, and has
  no restore-keys. `fetch` skips any bundle whose directory exists, so a
  fallback cache could have kept weights from another catalogue.
- Tests whose expected values were recorded from the 2020 public weights now
  load catalogue 2.2.0 explicitly, and CI fetches its `models_class1`,
  `models_class1_pan` and `models_class1_presentation` bundles. These tests
  compare TF/Keras predictions, which exist only for those weights:
  `test_released_master_predictions.py`,
  `test_released_presentation_highscore_rows.py` and
  `test_selected_peptides_csv.py`.
- The torch baseline regression test keeps one baseline per catalogue (#489).
  The June 2026 baseline is now `torch_baseline_predictions_2.2.0.json`, with
  its numbers unchanged. A new 2.3.0 baseline scores the same 8 peptides with
  the 2.3.0 bundles. It agrees with `mhcflurry predict --model-release 2.3.0`
  within 2.2e-7. Its largest differences from the 2.2.0 baseline, 0.179 in
  processing with flanks and 0.203 in presentation score, are the ones
  reported in #489. Run `python -m test.test_torch_baseline_regression
  RELEASE` to write a baseline; for 2.2.0 it reproduces the June file exactly.
- `docs/testing.md` and `CONTRIBUTING.md` describe the new fetch commands.

The prediction models and numerical prediction behavior are unchanged.
