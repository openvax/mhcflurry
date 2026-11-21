# MHCflurry 2.3.31

Maintenance and packaging cleanup release:

- Fix `NumpyJSONEncoder.default()` for NumPy 2.x (`numpy.integer`, `numpy.floating`, `numpy.bool_`) and replace `yaml.unsafe_load` with `yaml.safe_load` (#536).
- Prevent `mhcflurry-calibrate-percentile-ranks` from overwriting `GENERATE.sh`, delete unreferenced `mhcflurry/scoring.py`, and deduplicate optimizer/provenance helpers (#536).
- Remove 23 broken-in-wheel CLI script dispatchers from `cli/eval_command.py` and `cli/train_command.py`, remove dead `custom_loss.py` and `dla_evaluation.py` modules, move test/benchmark/dataset-generation utilities out of the wheel, and drop unused `ahocorasick-rs` from runtime dependencies (#537).

Model weights, catalogue identities, and predictions are unchanged.
