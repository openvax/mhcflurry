# MHCflurry documentation

MHCflurry predicts which peptides are likely to be displayed on the cell surface
by MHC class I molecules.

## Start here

- {doc}`intro` — install MHCflurry and make your first prediction.
- {doc}`commandline_tutorial` — score peptides and scan protein sequences from the shell.
- {doc}`python_tutorial` — run the same workflows from Python.
- {doc}`docker` — run MHCflurry from a prebuilt container image with bundled weights.

## User guides

- {doc}`model_downloads` — browse available model weights, manage your local cache,
  and select historical weight releases.
- {doc}`cross_species_validation` — predict on newly sequenced vertebrate species
  and engineered MHC mutants using the optional
  [`mhcseqs`](https://github.com/pirl-unc/mhcseqs) sequence database.
- {doc}`release_model_evaluation` — compare MHCflurry's released `2.3.0` weights
  against earlier releases and external predictors.
- {doc}`training` — train custom models on your own binding or mass-spectrometry data.
- {doc}`evaluation` — compare a custom model against a released one.

## Research & validation

- {doc}`whitepaper_25x_cross_species` — unified MHCflurry 2.5 technical whitepaper
  ([download PDF](_static/whitepaper/mhcflurry_25x_cross_species_whitepaper.pdf))
  covering cross-species Class I accuracy across 223 vertebrate species, audited
  cattle and dog benchmarks, and an exploratory Class I → Class II test.
- {doc}`dla_validation` — canine (`DLA`) allele support, training-data overlap audit,
  and evaluation limits.
- {doc}`experiments` — individual dated benchmark logs and ablation studies.

## Reference

- {doc}`commandline_tools` — command-line options for every subcommand.
- {doc}`api` — public Python API overview and full module reference.
- {doc}`configuration` — automatic GPU/CPU sizing, environment variables, and
  reproducible seeds.
- {doc}`changelog` — release highlights and links to per-version release notes.

## Advanced topics

- {doc}`shared_percent_rank_transforms` — custom percentile-rank calibration.
- {doc}`training_provenance` — verify whether evaluation samples overlap a model's
  training data.

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
cross_species_validation
release_model_evaluation
training
evaluation
```

```{toctree}
:maxdepth: 2
:caption: Research & validation
:hidden:

whitepaper_25x_cross_species
dla_validation
experiments
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
