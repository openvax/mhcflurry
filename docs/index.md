# MHCflurry documentation

MHCflurry predicts MHC class I binding affinity, antigen processing, and peptide
presentation. Start with the introduction for installation and a first
prediction, then choose a task-oriented guide below.

## Start here

- {doc}`intro` explains the three prediction types and gets you to a first
  result.
- {doc}`commandline_tutorial` covers peptide prediction, protein scanning, and
  model training.
- {doc}`python_tutorial` shows the same predictors through the Python API.

## Common next steps

- {doc}`evaluation` compares trained models and builds diagnostic or
  publication-style figures.
- {doc}`configuration` explains automatic hardware planning, runtime defaults,
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
