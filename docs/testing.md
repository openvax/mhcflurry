# Testing

The default pytest command is a full test suite. It is not a fast
unit-only loop.

The suite intentionally covers several layers:

* pure unit tests for encoding, losses, random-negative planning, and
  argument resolution;
* small neural-network training tests that verify numerical behavior;
* command-level subprocess tests that train, select, and calibrate tiny
  predictors end-to-end;
* public-model smoke tests that require cached MHCflurry download bundles.

That mix is useful before merging a release branch, but it makes a plain
`pytest test/` run take many minutes on a laptop.

## Quick local feedback

From a checkout, first source the development environment:

```shell
$ source develop.sh
```

Run lint plus focused unit tests while iterating:

```shell
$ ./lint.sh
$ pytest -q test/test_amino_acid.py test/test_random_negative_peptides.py
```

To run the broad fast tier, skip the tests marked as slow integration,
cached-bundle, or benchmark checks:

```shell
$ pytest -q test -m "not slow and not downloads"
```

When working on training internals, add the directly affected files rather
than jumping immediately to the full suite. Useful examples:

```shell
$ pytest -q test/test_class1_affinity_training_data.py
$ pytest -q test/test_pytorch_regressions.py
$ pytest -q test/test_train_pan_allele_models_command.py::test_pretrain_network_input_iterator_compact_torch_indices
```

## Full verification

Before calling a release-branch change complete, run:

```shell
$ ./lint.sh
$ pytest test/
```

If the run is unexpectedly slow, ask pytest for the slowest tests:

```shell
$ pytest -q test --durations=25 --durations-min=0.5
```

## Current slow buckets

The slowest tests are usually not pure unit tests. They are small
integration tests that do real model work:

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
: Tests that require locally cached MHCflurry download bundles. These
  tests should not fetch from the network; missing bundles should fail
  or skip with an instruction to run `mhcflurry-downloads fetch`
  outside pytest.
