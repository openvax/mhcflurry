# MHCflurry 2.3.33

Default model loading and release validation improvements (#521):

- `Class1ProcessingPredictor.load()` now falls back to the default presentation bundle's `processing_predictor_with_flanks` when standalone `models_class1_processing` is not installed and no `MHCFLURRY_DEFAULT_CLASS1_PROCESSING_MODELS_DIR` override is set. All three default predictors (`Class1PresentationPredictor.load()`, `Class1AffinityPredictor.load()`, and `Class1ProcessingPredictor.load()`) now work out of the box with only `mhcflurry downloads fetch models_class1_presentation`.
- `Class1AffinityPredictor.load()`'s presentation-bundle fallback now loads `models_class1_presentation/models/affinity_predictor` directly instead of instantiating the full presentation predictor, avoiding unnecessary processing-model loads and honoring `max_models` and `optimization_level`.
- `mhcflurry eval compare-models` falls back to `models_class1_presentation/models/affinity_predictor` when standalone `models_class1_pan` is absent, and CI/test fixtures load embedded presentation components directly instead of fetching redundant standalone archives.
- Release provenance validation (`scripts/release/validate_release_provenance.py` / `deploy_trained_models.sh`) now compares manifest-selected component weight SHA-256 digests between standalone exports and the presentation bundle, failing on unexpected mismatches (with `--allow-processing-with-flanks-mismatch` for intentional hybrids such as the 2.3.0 short-flank/boundary presentation component vs. the standalone 15-aa long-flank diagnostic ensemble).
- Standalone `models_class1_pan` and `models_class1_processing` catalogue entries remain available (`default: false`) for backwards compatibility; existing predictions and model weights are unchanged.
