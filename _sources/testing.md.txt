# Testing

Use focused tests while iterating and the full suite before merge or release.
The default pytest command runs unit, training, command-level, and downloaded
model checks; it is not a fast unit-only loop.

## Quick local feedback

From a checkout, first source the development environment:

```shell
$ source develop.sh
```

Run lint plus focused unit tests while iterating:

```shell
$ ./lint.sh
$ python -m pytest -q test/test_amino_acid.py test/test_random_negative_peptides.py
```

To run the broad fast tier, fetch the default presentation bundle once, then
skip the tests marked as slow or as needing other download bundles:

```shell
$ mhcflurry downloads fetch models_class1_presentation
$ python -m pytest -q test -m "not slow and not downloads"
```

The fast tier uses the presentation bundle because prediction commands, the
tutorial examples and the default predictors all need it. Any catalogue
release works.

When working on training internals, add the directly affected files rather
than jumping immediately to the full suite. Useful examples:

```shell
$ python -m pytest -q test/test_class1_affinity_training_data.py
$ python -m pytest -q test/test_pytorch_regressions.py
$ python -m pytest -q test/test_train_pan_allele_models_command.py::test_pretrain_network_input_iterator_compact_torch_indices
```

## Full verification

CI pins download catalogue `2.2.0` for historical model/data regression fixtures.
Prepare the same fixtures before running the full suite locally:

```shell
export MHCFLURRY_DOWNLOADS_CURRENT_RELEASE=2.2.0
mhcflurry downloads fetch data_curated data_mass_spec_annotated models_class1 \
    models_class1_presentation models_class1_processing models_class1_pan allele_sequences
```

Use a dedicated test shell, or unset the variable afterward to return ordinary
prediction to the default weights. The 2.3.0 prediction documentation is also
checked separately against the current presentation bundle.

Before calling a release-branch change complete, run:

```shell
$ ./lint.sh
$ python -m pytest test/
```

If the run is unexpectedly slow, ask pytest for the slowest tests:

```shell
$ python -m pytest -q test --durations=25 --durations-min=0.5
```

On macOS, prefer `python -m pytest` over the generated `pytest` console script
so PyTorch can see MPS accelerators.

Tests default to CPU; accelerator-specific cases opt in explicitly. Set
`MHCFLURRY_TEST_ACCELERATORS=mps` (or `gpu` for CUDA) to require that coverage
instead of skipping it when the device is unavailable.

## What the full suite covers

The full suite includes:

* pure unit tests for encoding, losses, random-negative planning, and argument
  resolution;
* small neural-network training tests that verify numerical behavior;
* command-level subprocess tests that train, select, and calibrate tiny
  predictors end-to-end; and
* public-model smoke tests that require cached MHCflurry download bundles.

The slowest tests are usually small integration tests that do real model work:

* `test/test_train_pan_allele_models_command.py` runs serial,
  parallel, and cluster-shaped pan-allele train/select command flows.
* `test/test_train_processing_models_command.py` trains and selects
  processing models.
* `test/test_class1_neural_network.py` contains full training behavior
  checks such as inequality handling, early stopping, and learned motif
  recovery.
* public-model tests load cached MHCflurry download bundles and run
  prediction smoke checks.

Mark new tests according to their cost. Keep small deterministic logic in
unit tests, and reserve end-to-end command or training checks for behavior
that cannot be covered at a narrower level.

## Markers

`slow`
: Tests that are too expensive for the fast local loop. These are
  usually small training jobs or benchmark-style checks.

`integration`
: End-to-end command or training tests that exercise multiple modules
  through the public CLI/API.

`downloads`
: Tests that require locally cached MHCflurry download bundles other than
  the default presentation bundle, or that compare against a specific
  catalogue (CI pins `2.2.0`). These tests should not fetch from the
  network; missing bundles should fail or skip with an instruction to run
  `mhcflurry downloads fetch` outside pytest. Load bundles in a fixture or
  test body, not at import time, so deselection works without them.
