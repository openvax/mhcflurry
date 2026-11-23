# MHCflurry 2.3.26

Removes dead code and catalogue entries. Predictions are unchanged.

## Download catalogue

`mhcflurry downloads` no longer offers 15 bundles whose archives were never
uploaded, so every remaining catalogue URL resolves (#523):

- The 1.7.0 stanza is gone. It was a draft of 2.0.0 and never a package
  release; `mhcflurry downloads releases models_class1_presentation` no longer
  lists it. Use `--release 2.0.0` for the same archives.
- 1.6.0 drops `models_class1_pan_unselected` and `data_evaluation`. 1.5.0 drops
  `models_class1_pan_variants`, `data_mass_spec_benchmark`,
  `data_mass_spec_annotated` and `data_curated`. `models_class1_pan_refined`
  is removed from 1.5.0 and 1.4.0 and from the bundle list.

Maintainers can run `python scripts/release/check_download_urls.py` to check
every catalogue URL after editing `mhcflurry/downloads.yml`.

## Deprecated Keras-era aliases

These aliases now warn on every use and will be removed in 2.4.0 (#524):

| alias | use instead |
|---|---|
| `mhcflurry.common.configure_tensorflow` | `configure_pytorch` |
| `Class1NeuralNetwork.keras_network_cache_key` | `model_cache_key` |
| `Class1NeuralNetwork.fit_generator` | `fit_streaming_batches` |
| `Loss.get_keras_loss()` | the `loss` attribute |
| `data_dependent_weights_initialization.get_activations` | `get_activations_pytorch` |
| `worker_init(keras_backend=...)` | `backend=` |

`configure_tensorflow` previously warned only for `tensorflow-*` backend names.
`mhcflurry.parallelism.run_single_worker_torch_compile_warmup` is removed; call
`run_single_worker_resource_probe`.

## Removed repository scripts

Not part of the installed package (#529, #525):

- `scripts/training/assemble_affinity_checkpoint.py` and
  `verify_processing_boundary_compatibility.py`, which nothing referenced.
- The compatibility shims `scripts/training/plot_loss_curves.py` and
  `release_exact/generate_hyperparameters.py`, `.base.py`, `.variants.py` and
  `reassign_mass_spec_training_data.py`. Use `mhcflurry train plot-loss-curves`,
  `mhcflurry class1-generate-training-hyperparameters` and
  `mhcflurry class1-reassign-mass-spec-training-data`. The reassign shim
  ignored `--exclude-pmhcs`, `--exclude-source-samples` and `--sample-aliases`;
  the command honours them.
- `downloads-generation/models_class1_kim_benchmark/`, the generator for a
  download that was never published.
