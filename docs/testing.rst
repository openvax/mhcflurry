Testing
=======

The default pytest command is a full test suite. It is not a fast
unit-only loop.

The suite intentionally covers several layers:

* pure unit tests for encoding, losses, random-negative planning, and
  argument resolution;
* small neural-network training tests that verify numerical behavior;
* command-level subprocess tests that train, select, and calibrate tiny
  predictors end-to-end;
* downloaded-model and public-model smoke tests.

That mix is useful before merging a release branch, but it makes a plain
``pytest test/`` run take many minutes on a laptop.

Quick local feedback
--------------------

From a checkout, first source the development environment:

.. code-block:: shell

    $ source develop.sh

Run lint plus focused unit tests while iterating:

.. code-block:: shell

    $ ./lint.sh
    $ pytest -q test/test_amino_acid.py test/test_random_negative_peptides.py

When working on training internals, add the directly affected files rather
than jumping immediately to the full suite. Useful examples:

.. code-block:: shell

    $ pytest -q test/test_class1_affinity_training_data.py
    $ pytest -q test/test_pytorch_regressions.py
    $ pytest -q test/test_train_pan_allele_models_command.py::test_pretrain_network_input_iterator_compact_torch_indices

Full verification
-----------------

Before calling a release-branch change complete, run:

.. code-block:: shell

    $ ./lint.sh
    $ pytest test/

If the run is unexpectedly slow, ask pytest for the slowest tests:

.. code-block:: shell

    $ pytest -q test --durations=25 --durations-min=0.5

Current slow buckets
--------------------

The slowest tests are usually not pure unit tests. They are small
integration tests that do real model work:

* ``test/test_train_pan_allele_models_command.py`` runs serial,
  parallel, and cluster-shaped pan-allele train/select command flows.
* ``test/test_train_processing_models_command.py`` trains and selects
  processing models.
* ``test/test_class1_neural_network.py`` contains full training behavior
  checks such as inequality handling, early stopping, and learned motif
  recovery.
* downloaded-model tests load public model bundles and run prediction
  smoke checks.

Mark new tests according to their cost. Keep small deterministic logic in
unit tests, and reserve end-to-end command or training checks for behavior
that cannot be covered at a narrower level.
