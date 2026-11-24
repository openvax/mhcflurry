# MHCflurry documentation

MHCflurry predicts MHC class I binding affinity, antigen processing, and peptide
presentation.

## Start here

- {doc}`intro` installs MHCflurry and makes a first prediction.
- {doc}`commandline_tutorial` scores peptides and scans proteins from the shell.
- {doc}`python_tutorial` does the same from Python.
- {doc}`docker` runs the same tools from a published image, without installing
  anything on the host.

## User guides

- {doc}`model_downloads` lists the available weights, manages the local data
  cache, and shows how to select an older release.
- {doc}`release_model_evaluation` compares the released weights with other
  predictors on identical data.
- {doc}`dla_validation` reports DLA support, training evidence, and the limits
  of the exploratory canine evaluation.
- {doc}`cross_species_validation` shows how to predict on newly sequenced
  species and pocket mutants via `mhcseqs`, and reports zero-shot cross-species
  validation benchmarks.
- {doc}`training` fits custom models from your own measurements.
- {doc}`evaluation` compares a trained model with a released one.

## Reference

- {doc}`commandline_tools` documents the command-line options.
- {doc}`api` summarizes the public Python API and lists every module.
- {doc}`configuration` covers hardware autosizing, cache and environment
  overrides, and reproducibility.
- {doc}`changelog` summarizes release highlights and links to per-version notes.

## Advanced topics

- {doc}`shared_percent_rank_transforms` explains custom percentile calibration.
- {doc}`training_provenance` checks whether evaluation samples overlap any
  compared model's training data.

Contributors and release maintainers can start with {doc}`maintainers`.

```{toctree}
:maxdepth: 2
:caption: Getting started
:hidden:

intro
commandline_tutorial
python_tutorial
docker
```

```{toctree}
:maxdepth: 2
:caption: User guides
:hidden:

model_downloads
release_model_evaluation
dla_validation
cross_species_validation
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
changelog
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
