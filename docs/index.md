# MHCflurry documentation

MHCflurry predicts MHC class I binding affinity, antigen processing, and peptide
presentation. Start with the introduction for installation and a first
prediction, then choose either the command-line or Python tutorial. The
reference pages are intentionally exhaustive and are best used to look up a
specific option after you know which workflow you need.

## Start here

- {doc}`intro` explains the three prediction types and gets you to a first
  result.
- {doc}`commandline_tutorial` covers peptide prediction and protein scanning.
- {doc}`python_tutorial` shows the same predictors through the Python API.

## Common next steps

- {doc}`training` explains custom model fitting and release-style retraining.
- {doc}`evaluation` compares trained models and builds diagnostic or
  publication-style figures.
- {doc}`configuration` explains automatic hardware planning, expert overrides,
  and reproducibility.
- {doc}`commandline_tools` and {doc}`api` are the complete references.

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

training
evaluation
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

commandline_tools
configuration
api
```

```{toctree}
:maxdepth: 2
:caption: Contributors and maintainers
:hidden:

testing
development
orchestrator
maintainers
```
