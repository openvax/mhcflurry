# MHCflurry documentation

MHCflurry predicts MHC class I binding affinity, antigen processing, and peptide
presentation.

## Start here

- {doc}`intro` installs MHCflurry and makes a first prediction.
- {doc}`commandline_tutorial` scores peptides and scans proteins from the shell.
- {doc}`python_tutorial` does the same from Python.

## Choosing and trusting models

- {doc}`model_downloads` lists the available weights and shows how to select an
  older release.
- {doc}`release_model_evaluation` compares the released weights with other
  predictors on identical data.

## Training your own models

- {doc}`training` fits custom models from your own measurements.
- {doc}`evaluation` compares a trained model with a released one.

## Reference

- {doc}`commandline_tools` documents the command-line options.
- {doc}`api` documents every Python class and method.
- {doc}`configuration` covers hardware autosizing, environment overrides, and
  reproducibility.

Contributors and release maintainers can start with {doc}`maintainers`.

```{toctree}
:maxdepth: 2
:caption: Getting started
:hidden:

intro
commandline_tutorial
python_tutorial
```

```{toctree}
:maxdepth: 2
:caption: User guides
:hidden:

model_downloads
release_model_evaluation
training
evaluation
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

commandline_tools
api
configuration
```

```{toctree}
:maxdepth: 2
:caption: Advanced topics
:hidden:

shared_percent_rank_transforms
training_provenance
```

```{toctree}
:maxdepth: 2
:caption: Contributors and maintainers
:hidden:

maintainers
testing
development
orchestrator
auto_sizing_audit
```

```{toctree}
:maxdepth: 2
:caption: Training internals
:hidden:

release_training_recipe
release_neural_hyperparameter_audit
final_230_candidate_experiment
affinity_dual_checkpoint_workflow
processing_preparation_acceleration
probabilistic_processing_matching
presentation_percentile_calibration
saved_candidate_evaluation
```

```{toctree}
:maxdepth: 2
:caption: Controlled experiments
:hidden:

processing_cleavage_boundary_experiment
processing_kernel_sweep
processing_hyperparameter_campaign
exact_public_data_experiment
```
