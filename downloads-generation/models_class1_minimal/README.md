# Class I allele-specific models (minimal: ensemble size of 1)

This download contains "minimal" MHC Class I MHCflurry predictors consisting
of a single model per allele. These predictors are expected to have slightly
lower accuracy than the standard ensembles (models_class1) but are small and
fast. Useful for testing.

To download these models and set them as the default predictor, run:

```
$ mhcflurry-downloads fetch models_class1_minimal
$ export MHCFLURRY_DEFAULT_CLASS1_MODELS=$(mhcflurry-downloads path models_class1_minimal)/models
```