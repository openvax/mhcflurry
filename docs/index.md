# MHCflurry documentation

MHCflurry predicts which peptides are likely to be displayed on the cell surface
by MHC class I molecules.

## Start here

- {doc}`intro` — install MHCflurry and make your first prediction.
- {doc}`commandline_tutorial` — score peptides and scan protein sequences from the shell.
- {doc}`python_tutorial` — run the same workflows from Python.
- {doc}`docker` — run MHCflurry from a prebuilt container image with bundled weights.

## User guides

- {doc}`interpreting_predictions` — understand each output score, pick the right
  column to rank by, and choose practical cutoffs.
- {doc}`scanning_proteins` — scan protein FASTA files and whole proteomes,
  filter hits, and avoid common CLI pitfalls.
- {doc}`model_downloads` — browse available model weights, manage your local cache,
  and select historical weight releases.

## Reference

- {doc}`commandline_tools` — command-line options for every subcommand.
- {doc}`api` — public Python API overview and full module reference.
- {doc}`configuration` — automatic GPU/CPU sizing, environment variables, and
  reproducible seeds.
- {doc}`changelog` — release highlights and links to per-version release notes.

## Advanced topics

- {doc}`cross_species_validation` — predict on newly sequenced vertebrate species
  and engineered MHC mutants using the optional
  [`mhcseqs`](https://github.com/pirl-unc/mhcseqs) sequence database.
- {doc}`training` — train custom models on your own binding or mass-spectrometry data.
- {doc}`evaluation` — compare a custom model against a released one.
- {doc}`release_model_evaluation` — compare MHCflurry's released `2.3.0` weights
  against earlier releases and external predictors.
- {doc}`shared_percent_rank_transforms` — custom percentile-rank calibration.
- {doc}`training_provenance` — verify whether evaluation samples overlap a model's
  training data.
- {doc}`experiments` — experiment log, cross-species technical whitepaper (web
  & PDF), canine/bovine audits, and dated benchmark studies.

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

interpreting_predictions
scanning_proteins
model_downloads
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

cross_species_validation
training
evaluation
release_model_evaluation
shared_percent_rank_transforms
training_provenance
experiments
```

```{toctree}
:maxdepth: 2
:caption: Contributors and maintainers
:hidden:

maintainers
testing
development
```
